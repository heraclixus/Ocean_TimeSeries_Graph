import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .models import (
    NXROLinearModel,
    NXROMemoryLinearModel,
    NXROMemoryResModel,
    NXROMemoryAttentionModel,
    NXROMemoryGraphModel,
    NXROMemoryDecayModel,
    NXROMemoryGRUModel,
    NXROROModel,
    NXRORODiagModel,
    NXROResModel,
    NXROResidualMixModel,
    NXRONeuralODEModel,
    NXROBilinearModel,
    NXROAttentiveModel,
    NXROGraphModel,
    NXROGraphPyGModel,
    NXROTransformerModel,
    PureNeuralODEModel,
    PureTransformerModel,
    NXRODeepLearnableGCN,
    build_edge_index_from_corr,
)
from .data import get_dataloaders as _get_dataloaders_raw
from graph_construction import build_xro_coupling_graph, normalize_with_self_loops, get_or_build_xro_graph, get_or_build_stat_knn_graph


def _get_dataloaders_with_val(**kwargs):
    """Wrapper around get_dataloaders that always returns (dl_train, dl_val_or_none, dl_test, var_order).

    When val_slice is None, dl_val is None and behaviour is unchanged from the
    original 3-tuple return.
    """
    result = _get_dataloaders_raw(**kwargs)
    if len(result) == 4:
        dl_train, dl_val, dl_test, var_order = result
    else:
        dl_train, dl_test, var_order = result
        dl_val = None
    return dl_train, dl_val, dl_test, var_order


# Keep original name for backward compatibility: callers that don't use val_slice
# continue to work unchanged.
def get_dataloaders(**kwargs):
    """Backward-compatible wrapper: returns (dl_train, dl_test, var_order) when no val_slice."""
    result = _get_dataloaders_raw(**kwargs)
    return result


def _is_improved(curr_value: float, best_value: float, min_delta: float) -> bool:
    return curr_value < (best_value - min_delta)


def _run_memory_epoch(model, dl, loss_fn, device: str, rollout_k: int,
                      train: bool = False, opt=None, regularizer_fn=None) -> dict:
    model.train(train)
    total_one_step, total_objective, count = 0.0, 0.0, 0
    dt = 1.0 / 12.0

    for batch in tqdm(dl, disable=not train):
        if rollout_k > 1:
            x_in, t_in, x_next, x_seq = batch
            x_seq = x_seq.to(device)
        else:
            x_in, t_in, x_next = batch

        x_in = x_in.to(device)
        t_in = t_in.to(device)
        x_next = x_next.to(device)

        # Allow memory models to train with memory_depth=0 by wrapping Markov batches
        # into a length-1 history window.
        if x_in.dim() == 2:
            x_hist = x_in.unsqueeze(1)
            t_hist = t_in.unsqueeze(1)
        else:
            x_hist = x_in
            t_hist = t_in

        dxdt = model(x_hist, t_hist)
        x_hat = x_hist[:, -1, :] + dxdt * dt
        base_loss = loss_fn(x_hat, x_next)
        loss = base_loss
        n_loss_terms = 1

        if rollout_k > 1:
            x_buf = torch.cat([x_hist[:, 1:, :], x_hat.unsqueeze(1)], dim=1)
            t_next = t_hist[:, -1:] + dt
            t_buf = torch.cat([t_hist[:, 1:], t_next], dim=1)
            for step_idx in range(1, rollout_k):
                dxdt_k = model(x_buf, t_buf)
                x_roll = x_buf[:, -1, :] + dxdt_k * dt
                loss = loss + loss_fn(x_roll, x_seq[:, step_idx])
                n_loss_terms += 1
                x_buf = torch.cat([x_buf[:, 1:, :], x_roll.unsqueeze(1)], dim=1)
                t_buf = torch.cat([t_buf[:, 1:], (t_buf[:, -1:] + dt)], dim=1)

        objective_loss = loss / float(n_loss_terms)

        if regularizer_fn is not None:
            loss = loss + regularizer_fn(model)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        batch_size = x_hist.size(0)
        total_one_step += base_loss.item() * batch_size
        total_objective += objective_loss.item() * batch_size
        count += batch_size

    denom = max(count, 1)
    return {
        'one_step_rmse': (total_one_step / denom) ** 0.5,
        'objective_rmse': (total_objective / denom) ** 0.5,
    }


def _train_memory_model(model, dl_train, dl_test, n_epochs: int, lr: float,
                        weight_decay: float, device: str, rollout_k: int,
                        tag: str, regularizer_fn=None,
                        early_stop_patience: Optional[int] = 200,
                        early_stop_min_delta: float = 1e-4,
                        dl_val=None):
    """Train a memory-aware model with early stopping.

    When ``dl_val`` is provided, early stopping and model selection use the
    validation loader instead of the test loader, preventing test-set leakage
    into architecture/hyperparameter selection.
    """
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Use validation loader for model selection if available, else fall back to test
    dl_select = dl_val if dl_val is not None else dl_test

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    train_one_step_hist, test_one_step_hist = [], []
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, n_epochs + 1):
        train_stats = _run_memory_epoch(
            model, dl_train, loss_fn, device=device, rollout_k=rollout_k,
            train=True, opt=opt, regularizer_fn=regularizer_fn,
        )
        select_stats = _run_memory_epoch(
            model, dl_select, loss_fn, device=device, rollout_k=rollout_k,
            train=False, opt=None, regularizer_fn=None,
        )
        train_rmse = train_stats['objective_rmse']
        select_rmse = select_stats['objective_rmse']
        train_hist.append(float(train_rmse))

        # If val split active, also compute test stats for reporting (but NOT selection)
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_stats = _run_memory_epoch(
                model, dl_test, loss_fn, device=device, rollout_k=rollout_k,
                train=False, opt=None, regularizer_fn=None,
            )
            test_hist.append(float(test_stats['objective_rmse']))
            test_one_step_hist.append(float(test_stats['one_step_rmse']))
        else:
            test_hist.append(float(select_rmse))

        train_one_step_hist.append(float(train_stats['one_step_rmse']))

        if _is_improved(float(select_rmse), float(best_rmse), early_stop_min_delta):
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if rollout_k > 1:
            msg = (f"[{tag}] Epoch {epoch:03d} | "
                   f"train rollout RMSE: {train_rmse:.4f} | ")
            if dl_val is not None:
                msg += f"val RMSE: {select_rmse:.4f} | test RMSE: {test_stats['objective_rmse']:.4f}"
            else:
                msg += f"test rollout RMSE: {select_rmse:.4f}"
            msg += (f" | train 1-step RMSE: {train_stats['one_step_rmse']:.4f} | "
                    f"select 1-step RMSE: {select_stats['one_step_rmse']:.4f}")
            print(msg)
        else:
            if dl_val is not None:
                print(f"[{tag}] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | "
                      f"val RMSE: {select_rmse:.4f} | test RMSE: {test_stats['objective_rmse']:.4f}")
            else:
                print(f"[{tag}] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")
        if early_stop_patience is not None and early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_early = True
            select_label = "val" if dl_val is not None else "test"
            print(
                f"[{tag}] Early stopping at epoch {epoch:03d} | "
                f"best epoch: {best_epoch:03d} | best {select_label} RMSE: {float(best_rmse):.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "train_rmse_one_step": train_one_step_hist,
        "test_rmse_one_step": test_one_step_hist,
        "selection_metric": "rollout_rmse" if rollout_k > 1 else "one_step_rmse",
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
        "stopped_early": bool(stopped_early),
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": float(early_stop_min_delta),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, best_rmse, history


def train_nxro_linear(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    early_stop_patience: Optional[int] = 200,
    early_stop_min_delta: float = 1e-4,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-Linear model (variants 1, 1a).

    Args:
        L_basis_init: If None, random init (variant 1). If provided, warm-start (variant 1a).
        pretrained_state_dict: Optional state dict to load weights from (e.g. for two-stage training).
        exclude_vars: Optional list of variable names to exclude (e.g., ['WWV']).
        include_only_vars: If specified, filter to only these variables (for two-stage training).
        val_start/val_end: If set, use validation split for early stopping instead of test.
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = NXROLinearModel(n_vars=n_vars, k_max=k_max, L_basis_init=L_basis_init).to(device)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5  # RMSE

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if _is_improved(float(select_rmse), float(best_rmse), early_stop_min_delta):
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if dl_val is not None:
            print(f"Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")
        if early_stop_patience is not None and early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_early = True
            select_label = "val" if dl_val is not None else "test"
            print(
                f"[Linear] Early stopping at epoch {epoch:03d} | "
                f"best epoch: {best_epoch:03d} | best {select_label} RMSE: {float(best_rmse):.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
        "stopped_early": bool(stopped_early),
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": float(early_stop_min_delta),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_memory_linear(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    memory_depth: int = 3,
    k_max: int = 2,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    freeze_instantaneous: bool = False,
    freeze_lagged: bool = False,
    init_lagged_zero: bool = True,
    early_stop_patience: Optional[int] = 200,
    early_stop_min_delta: float = 1e-4,
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        memory_depth=memory_depth,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    n_vars = len(var_order)
    model = NXROMemoryLinearModel(
        n_vars=n_vars,
        memory_depth=memory_depth,
        k_max=k_max,
        L_basis_init=L_basis_init,
        freeze_instantaneous=freeze_instantaneous,
        freeze_lagged=freeze_lagged,
        init_lagged_zero=init_lagged_zero,
    ).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    model, best_rmse, history = _train_memory_model(
        model, dl_train, dl_test, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, device=device, rollout_k=rollout_k,
        tag='MemoryLinear',
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    return model, var_order, best_rmse, history


def train_nxro_memory_res(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    memory_depth: int = 3,
    k_max: int = 2,
    hidden: int = 64,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    freeze_instantaneous: bool = False,
    freeze_lagged: bool = False,
    init_lagged_zero: bool = True,
    early_stop_patience: Optional[int] = 200,
    early_stop_min_delta: float = 1e-4,
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        memory_depth=memory_depth,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    n_vars = len(var_order)
    model = NXROMemoryResModel(
        n_vars=n_vars,
        memory_depth=memory_depth,
        k_max=k_max,
        hidden=hidden,
        L_basis_init=L_basis_init,
        freeze_instantaneous=freeze_instantaneous,
        freeze_lagged=freeze_lagged,
        init_lagged_zero=init_lagged_zero,
    ).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    model, best_rmse, history = _train_memory_model(
        model, dl_train, dl_test, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, device=device, rollout_k=rollout_k,
        tag='MemoryRes',
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    return model, var_order, best_rmse, history


def train_nxro_memory_attentive(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    memory_depth: int = 3,
    k_max: int = 2,
    d: int = 32,
    n_heads: int = 1,
    dropout: float = 0.0,
    mask_mode: str = 'th_only',
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    freeze_instantaneous: bool = False,
    freeze_lagged: bool = False,
    init_lagged_zero: bool = True,
    early_stop_patience: Optional[int] = 200,
    early_stop_min_delta: float = 1e-4,
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        memory_depth=memory_depth,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    n_vars = len(var_order)
    model = NXROMemoryAttentionModel(
        n_vars=n_vars,
        memory_depth=memory_depth,
        k_max=k_max,
        d=d,
        n_heads=n_heads,
        dropout=dropout,
        mask_mode=mask_mode,
        L_basis_init=L_basis_init,
        freeze_instantaneous=freeze_instantaneous,
        freeze_lagged=freeze_lagged,
        init_lagged_zero=init_lagged_zero,
    ).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    model, best_rmse, history = _train_memory_model(
        model, dl_train, dl_test, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, device=device, rollout_k=rollout_k,
        tag='MemoryAttention',
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    return model, var_order, best_rmse, history


def train_nxro_memory_graph(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    memory_depth: int = 3,
    k_max: int = 2,
    use_fixed_graph: bool = True,
    graph_mode: str = 'agg_spatial',
    learned_l1_lambda: float = 0.0,
    stat_knn_method: Optional[str] = None,
    stat_knn_top_k: int = 2,
    stat_knn_source: Optional[str] = None,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    freeze_instantaneous: bool = False,
    freeze_lagged: bool = False,
    init_lagged_zero: bool = True,
    early_stop_patience: Optional[int] = 200,
    early_stop_min_delta: float = 1e-4,
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        memory_depth=memory_depth,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )
    n_vars = len(var_order)

    adj_init = None
    if stat_knn_method:
        data_source = stat_knn_source or 'data/XRO_indices_oras5_train.csv'
        A_stat, _ = get_or_build_stat_knn_graph(
            data_path=data_source,
            train_start=train_start,
            train_end=train_end,
            var_order=var_order,
            method=stat_knn_method,
            top_k=stat_knn_top_k,
        )
        adj_init = normalize_with_self_loops(A_stat)
    else:
        try:
            A_xro, _ = get_or_build_xro_graph(
                nc_path=nc_path,
                train_start=train_start,
                train_end=train_end,
                var_order=var_order,
            )
            adj_init = normalize_with_self_loops(A_xro)
        except Exception:
            adj_init = torch.eye(n_vars)

    model = NXROMemoryGraphModel(
        n_vars=n_vars,
        memory_depth=memory_depth,
        k_max=k_max,
        use_fixed_graph=use_fixed_graph,
        adj_init=adj_init,
        graph_mode=graph_mode,
        L_basis_init=L_basis_init,
        freeze_instantaneous=freeze_instantaneous,
        freeze_lagged=freeze_lagged,
        init_lagged_zero=init_lagged_zero,
    ).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    def regularizer_fn(curr_model):
        if curr_model.use_fixed_graph or learned_l1_lambda <= 0:
            return 0.0
        return learned_l1_lambda * torch.relu(curr_model.A_param).sum()

    model, best_rmse, history = _train_memory_model(
        model, dl_train, dl_test, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, device=device, rollout_k=rollout_k,
        tag='MemoryGraph', regularizer_fn=regularizer_fn,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    return model, var_order, best_rmse, history


def train_nxro_memory_decay(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    memory_depth: int = 3,
    k_max: int = 2,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    freeze_instantaneous: bool = False,
    early_stop_patience: Optional[int] = 200,
    early_stop_min_delta: float = 1e-4,
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        memory_depth=memory_depth,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    n_vars = len(var_order)
    model = NXROMemoryDecayModel(
        n_vars=n_vars,
        memory_depth=memory_depth,
        k_max=k_max,
        L_basis_init=L_basis_init,
        freeze_instantaneous=freeze_instantaneous,
    ).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    model, best_rmse, history = _train_memory_model(
        model, dl_train, dl_test, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, device=device, rollout_k=rollout_k,
        tag='MemoryDecay',
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    return model, var_order, best_rmse, history


def train_nxro_memory_gru(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    memory_depth: int = 3,
    k_max: int = 2,
    gru_hidden: int = 32,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    freeze_instantaneous: bool = False,
    early_stop_patience: Optional[int] = 200,
    early_stop_min_delta: float = 1e-4,
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        memory_depth=memory_depth,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    n_vars = len(var_order)
    model = NXROMemoryGRUModel(
        n_vars=n_vars,
        memory_depth=memory_depth,
        k_max=k_max,
        gru_hidden=gru_hidden,
        L_basis_init=L_basis_init,
        freeze_instantaneous=freeze_instantaneous,
    ).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    model, best_rmse, history = _train_memory_model(
        model, dl_train, dl_test, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, device=device, rollout_k=rollout_k,
        tag='MemoryGRU',
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    return model, var_order, best_rmse, history


def train_nxro_ro(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    warmstart_init_dict: dict = None,
    freeze_flags: dict = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-RO model (variants 2, 2a, 2a-Fix*).

    Args:
        warmstart_init_dict: Dict with 'L_basis_init', 'W_T_init', 'W_H_init' for warm-start
        freeze_flags: Dict with 'freeze_linear', 'freeze_ro' flags
        pretrained_state_dict: Optional state dict to load weights from.
        exclude_vars: Optional list of variable names to exclude (e.g., ['WWV']).
        include_only_vars: If specified, filter to only these variables (for two-stage training).
        val_start/val_end: If set, use validation split for early stopping instead of test.
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    
    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max}
    if warmstart_init_dict:
        model_kwargs.update(warmstart_init_dict)
    if freeze_flags:
        model_kwargs.update(freeze_flags)
    
    model = NXROROModel(**model_kwargs).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[RO] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[RO] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_rodiag(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    warmstart_init_dict: dict = None,
    freeze_flags: dict = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-RO+Diag model (variants 3, 3a, 3a-Fix*).

    Args:
        warmstart_init_dict: Dict with init parameters for warm-start
        freeze_flags: Dict with 'freeze_linear', 'freeze_ro', 'freeze_diag' flags
        pretrained_state_dict: Optional state dict to load weights from.
        exclude_vars: Optional list of variable names to exclude (e.g., ['WWV']).
        include_only_vars: If specified, filter to only these variables (for two-stage training).
        val_start/val_end: If set, use validation split for early stopping instead of test.
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    
    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max}
    if warmstart_init_dict:
        model_kwargs.update(warmstart_init_dict)
    if freeze_flags:
        model_kwargs.update(freeze_flags)
    
    model = NXRORODiagModel(**model_kwargs).to(device)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[RO+Diag] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[RO+Diag] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def _jacobian_fro_estimate(model, x: torch.Tensor, t_years: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """Hutchinson estimator for ||J||_F^2 where J = d f / d x.

    Returns a scalar tensor.
    """
    B, V = x.shape
    total = 0.0
    for _ in range(num_samples):
        v = torch.randint_like(x, low=0, high=2, dtype=torch.long)
        v = v.float().mul_(2).sub_(1)  # Rademacher {-1,1}
        x_req = x.detach().requires_grad_(True)
        f = model(x_req, t_years)
        s = (f * v).sum()
        g = torch.autograd.grad(s, x_req, retain_graph=False, create_graph=False)[0]
        total = total + (g * g).sum(dim=1).mean()  # ||J^T v||^2 average over batch
    return total / num_samples


def _divergence_estimate(model, x: torch.Tensor, t_years: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """Hutchinson estimator for trace(J) where J = d f / d x.

    Returns mean absolute divergence to penalize non-zero divergence.
    """
    B, V = x.shape
    total = 0.0
    for _ in range(num_samples):
        v = torch.randint_like(x, low=0, high=2, dtype=torch.long)
        v = v.float().mul_(2).sub_(1)
        x_req = x.detach().requires_grad_(True)
        f = model(x_req, t_years)
        s = (f * v).sum()
        g = torch.autograd.grad(s, x_req, retain_graph=False, create_graph=False)[0]
        total = total + (g * v).sum(dim=1).mean()
    return total.abs() / num_samples


def train_nxro_neural_phys(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    hidden: int = 64,
    depth: int = 2,
    dropout: float = 0.1,
    allow_cross: bool = False,
    mask_mode: str = 'th_only',
    jac_reg: float = 1e-4,
    div_reg: float = 0.0,
    noise_std: float = 0.0,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = NXRONeuralODEModel(n_vars=n_vars, k_max=k_max, hidden=hidden, depth=depth,
                               dropout=dropout, allow_cross=allow_cross, mask_mode=mask_mode).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict for neural_phys model.")
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            if train and noise_std > 0:
                x_t = x_t + noise_std * torch.randn_like(x_t)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            base = loss_fn(x_hat, x_next)
            reg = 0.0
            if jac_reg > 0:
                reg = reg + jac_reg * _jacobian_fro_estimate(model, x_t, t_y)
            if div_reg > 0:
                reg = reg + div_reg * _divergence_estimate(model, x_t, t_y)
            loss = base + reg
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += base.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[NeuralPhys] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[NeuralPhys] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_graph_pyg(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    top_k: int = 2,
    hidden: int = 16,
    dropout: float = 0.0,
    use_gat: bool = False,
    stat_knn_method: Optional[str] = None,
    stat_knn_source: Optional[str] = None,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 10,
    resume_from_checkpoint: Optional[str] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    disable_seasonal_gate: bool = False,
):
    # dataloaders for training
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test
    # Build edge_index from chosen graph prior
    def edge_index_from_adjacency(A: torch.Tensor, k: int) -> torch.Tensor:
        V = A.shape[0]
        A_use = A.clone()
        A_use.fill_diagonal_(0.0)
        edges = []
        for i in range(V):
            vals, idx = torch.topk(A_use[i], k=min(k, V - 1))
            for j in idx.tolist():
                if i != j and A_use[i, j] > 0:
                    edges.append([i, j])
                    edges.append([j, i])
        if len(edges) == 0:
            return torch.empty(2, 0, dtype=torch.long, device=A.device)
        return torch.tensor(edges, dtype=torch.long, device=A.device).T

    try:
        if stat_knn_method:
            data_source = stat_knn_source or 'data/XRO_indices_oras5_train.csv'
            A_stat, _ = get_or_build_stat_knn_graph(data_path=data_source, train_start=train_start, train_end=train_end,
                                                   var_order=var_order, method=stat_knn_method, top_k=top_k)
            edge_index = edge_index_from_adjacency(A_stat.to(device), top_k)
        else:
            # XRO-based adjacency, then prune to top_k
            A_xro, _ = get_or_build_xro_graph(nc_path=nc_path, train_start=train_start, train_end=train_end, var_order=var_order)
            edge_index = edge_index_from_adjacency(A_xro.to(device), top_k)
    except Exception:
        # Fallback: empirical Pearson correlation
        # Use _load_obs_with_cesm2_support to handle CESM2 variable renaming (e.g., ENSO -> Nino34)
        from nxro.data import _load_obs_with_cesm2_support
        ds = _load_obs_with_cesm2_support(nc_path).sel(time=slice(train_start, train_end))
        X_np = []
        for v in var_order:
            X_np.append(ds[v].values)
        X = torch.tensor(np.stack(X_np, axis=-1), dtype=torch.float32)
        corr = torch.corrcoef(X.T)
        edge_index = build_edge_index_from_corr(corr, top_k=top_k).to(device)

    n_vars = len(var_order)
    model = NXROGraphPyGModel(n_vars=n_vars, k_max=k_max, edge_index=edge_index, hidden=hidden, dropout=dropout, use_gat=use_gat, disable_seasonal_gate=disable_seasonal_gate).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    start_epoch = 1

    # Resume from checkpoint if provided
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"[GraphPyG] Resuming from checkpoint: {resume_from_checkpoint}")
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_rmse = ckpt.get('best_rmse', float('inf'))
        best_state = ckpt.get('best_state', None)
        train_hist = ckpt.get('train_hist', [])
        test_hist = ckpt.get('test_hist', [])
        val_hist = ckpt.get('val_hist', [])
        print(f"  Resumed from epoch {start_epoch - 1}, best_rmse={best_rmse:.4f}")

    for epoch in range(start_epoch, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[GraphPyG] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[GraphPyG] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

        # Save checkpoint periodically
        if checkpoint_dir and epoch % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            tag = 'gat' if use_gat else 'gcn'
            ckpt_path = os.path.join(checkpoint_dir, f'graphpyg_{tag}_k{k_max}_epoch{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_rmse': best_rmse,
                'best_state': best_state,
                'train_hist': train_hist,
                'test_hist': test_hist,
                'val_hist': val_hist,
                'var_order': var_order,
            }, ckpt_path)
            print(f"  [Checkpoint] Saved to: {ckpt_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_graph(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    use_fixed_graph: bool = True,
    learned_l1_lambda: float = 0.0,
    stat_knn_method: Optional[str] = None,
    stat_knn_top_k: int = 2,
    stat_knn_source: Optional[str] = None,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    warmstart_init_dict: dict = None,
    freeze_flags: dict = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-Graph model (variants 5b, 5b-WS, 5b-FixL).

    Args:
        val_start/val_end: If set, use validation split for early stopping instead of test.
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    # Choose adjacency prior
    adj_init = None
    if stat_knn_method:
        # Statistical KNN from CSV (or NC if provided), then normalize
        data_source = stat_knn_source or 'data/XRO_indices_oras5_train.csv'
        A_stat, _ = get_or_build_stat_knn_graph(data_path=data_source, train_start=train_start, train_end=train_end,
                                               var_order=var_order, method=stat_knn_method, top_k=stat_knn_top_k)
        adj_init = normalize_with_self_loops(A_stat)
    else:
        try:
            A_xro, _ = get_or_build_xro_graph(nc_path=nc_path, train_start=train_start, train_end=train_end, var_order=var_order)
            adj_init = normalize_with_self_loops(A_xro)
        except Exception:
            adj_init = torch.eye(n_vars)

    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max, 'use_fixed_graph': use_fixed_graph, 'adj_init': adj_init}
    if warmstart_init_dict:
        model_kwargs.update(warmstart_init_dict)
    if freeze_flags:
        model_kwargs.update(freeze_flags)
    
    # Build model (fixed vs learned). If learned, initialize with adj_init and regularize with L1.
    model = NXROGraphModel(**model_kwargs).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            base_loss = loss_fn(x_hat, x_next)
            loss = base_loss
            # L1 sparsity penalty for learned adjacency
            if (not model.use_fixed_graph) and learned_l1_lambda > 0:
                A_pos = torch.relu(model.A_param)
                loss = loss + learned_l1_lambda * A_pos.sum()
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += base_loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[Graph] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[Graph] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_neural(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    hidden: int = 64,
    depth: int = 2,
    dropout: float = 0.0,
    allow_cross: bool = False,
    mask_mode: str = 'th_only',
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = NXRONeuralODEModel(n_vars=n_vars, k_max=k_max, hidden=hidden, depth=depth,
                               dropout=dropout, allow_cross=allow_cross, mask_mode=mask_mode).to(device)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[NeuralODE] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[NeuralODE] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_pure_neural_ode(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 64,
    depth: int = 2,
    dropout: float = 0.0,
    use_time: bool = False,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train a pure Neural ODE model with NO physical priors.

    This is a baseline model that learns dynamics entirely from data:
    - No seasonal linear operator
    - No XRO structure
    - Just a pure MLP: dX/dt = G_θ(X) or G_θ([X, t])

    Args:
        nc_path: path to ORAS5 data
        train_start, train_end: training period
        test_start, test_end: test period
        n_epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        weight_decay: L2 regularization
        hidden: MLP hidden size
        depth: number of hidden layers
        dropout: dropout rate
        use_time: if True, include normalized time as input feature
        device: 'cpu' or 'cuda'
        rollout_k: multi-step rollout for training loss
        extra_train_nc_paths: additional training data
        pretrained_state_dict: optional state dict to load
        exclude_vars: variables to exclude
        include_only_vars: variables to include
        val_start/val_end: If set, use validation split for early stopping instead of test.

    Returns:
        model, var_order, best_rmse, history
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = PureNeuralODEModel(
        n_vars=n_vars, hidden=hidden, depth=depth,
        dropout=dropout, use_time=use_time
    ).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict for PureNeuralODE.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[PureNeuralODE] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[PureNeuralODE] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_attentive(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    d: int = 32,
    dropout: float = 0.0,
    mask_mode: str = 'th_only',
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    warmstart_init_dict: dict = None,
    freeze_flags: dict = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    disable_seasonal_gate: bool = False,
):
    """Train NXRO-Attentive model (variants 5a, 5a-WS, 5a-FixL).

    Args:
        val_start/val_end: If set, use validation split for early stopping instead of test.
        disable_seasonal_gate: If True, disable the seasonal gate α(t) (for ablation).
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)

    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max, 'd': d, 'dropout': dropout, 'mask_mode': mask_mode,
                    'disable_seasonal_gate': disable_seasonal_gate}
    if warmstart_init_dict:
        model_kwargs.update(warmstart_init_dict)
    if freeze_flags:
        model_kwargs.update(freeze_flags)

    model = NXROAttentiveModel(**model_kwargs).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[Attentive] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[Attentive] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_bilinear(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    n_channels: int = 2,
    rank: int = 2,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = NXROBilinearModel(n_vars=n_vars, k_max=k_max, n_channels=n_channels, rank=rank).to(device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict for bilinear model.")
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[Bilinear] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[Bilinear] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_res(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    hidden: int = 64,
    res_reg: float = 1e-4,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    warmstart_init_dict: dict = None,
    freeze_flags: dict = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-Res model (variants 4, 4a).

    Args:
        warmstart_init_dict: Dict with 'L_basis_init' for warm-start
        freeze_flags: Dict with 'freeze_linear' flag
        pretrained_state_dict: Optional state dict to load weights from.
        exclude_vars: Optional list of variable names to exclude (e.g., ['WWV']).
        include_only_vars: If specified, filter to only these variables (for two-stage training).
        hidden: MLP hidden layer size (default 64).
        val_start/val_end: If set, use validation split for early stopping instead of test.
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    
    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max, 'hidden': hidden}
    if warmstart_init_dict:
        model_kwargs.update(warmstart_init_dict)
    if freeze_flags:
        model_kwargs.update(freeze_flags)
    
    model = NXROResModel(**model_kwargs).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            base_loss = loss_fn(x_hat, x_next)
            # Residual regularization: L2 on last layer weights
            res_pen = 0.0
            for name, p in model.residual.named_parameters():
                if 'weight' in name:
                    res_pen = res_pen + (p**2).mean()
            loss = base_loss + res_reg * res_pen
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += base_loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[Res] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[Res] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_resmix(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    hidden: int = 64,
    alpha_init: float = 0.1,
    alpha_learnable: bool = False,
    alpha_max: float = 0.5,
    res_reg: float = 1e-4,
    device: str = 'cpu',
    extra_train_nc_paths=None,
    warmstart_init_dict: dict = None,
    freeze_flags: dict = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-ResidualMix model (variants 5d, 5d-WS, 5d-Fix*).

    Args:
        val_start/val_end: If set, use validation split for early stopping instead of test.
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    
    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max, 'hidden': hidden,
                   'alpha_init': alpha_init, 'alpha_learnable': alpha_learnable,
                   'alpha_max': alpha_max, 'dropout': 0.0}
    if warmstart_init_dict:
        model_kwargs.update(warmstart_init_dict)
    if freeze_flags:
        model_kwargs.update(freeze_flags)
    
    model = NXROResidualMixModel(**model_kwargs).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for x_t, t_y, x_next in tqdm(dl, disable=not train):
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            base_loss = loss_fn(x_hat, x_next)
            # Residual regularization: L2 on residual weights; small alpha if learnable
            res_pen = 0.0
            for name, p in model.residual.named_parameters():
                if 'weight' in name:
                    res_pen = res_pen + (p**2).mean()
            alpha_pen = 0.0
            if alpha_learnable:
                alpha_val = model.alpha()
                alpha_pen = (alpha_val**2)
            loss = base_loss + res_reg * (res_pen + alpha_pen)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += base_loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[ResMix] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[ResMix] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_res_fullxro(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    res_reg: float = 1e-4,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    xro_init_dict: dict = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-Res-FullXRO model (variant 4b): Frozen full XRO + trainable MLP.

    All XRO components (L, RO, Diag) are frozen. Only residual MLP is trainable.

    Args:
        xro_init_dict: REQUIRED dict with 'L_basis', 'W_T', 'W_H', 'B_diag', 'C_diag' from XRO
        exclude_vars: Optional list of variable names to exclude (e.g., ['WWV']).
        include_only_vars: If specified, filter to only these variables (for two-stage training).
        val_start/val_end: If set, use validation split for early stopping instead of test.
    """
    from .models import NXROResFullXROModel

    assert xro_init_dict is not None, "Variant 4b requires XRO initialization!"
    assert all(k in xro_init_dict for k in ['L_basis', 'W_T', 'W_H', 'B_diag', 'C_diag']), \
        "xro_init_dict must contain all XRO components!"

    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    
    model = NXROResFullXROModel(
        n_vars=n_vars,
        k_max=k_max,
        hidden=64,
        L_basis_xro=xro_init_dict['L_basis'],
        W_T_xro=xro_init_dict['W_T'],
        W_H_xro=xro_init_dict['W_H'],
        B_diag_xro=xro_init_dict['B_diag'],
        C_diag_xro=xro_init_dict['C_diag'],
    ).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            base_loss = loss_fn(x_hat, x_next)
            # Residual regularization
            res_pen = 0.0
            for name, p in model.residual.named_parameters():
                if 'weight' in name:
                    res_pen = res_pen + (p**2).mean()
            loss = base_loss + res_reg * res_pen
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += base_loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[Res-FullXRO] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[Res-FullXRO] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_transformer(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_max: int = 2,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-Transformer model.

    Args:
        nc_path: Path to NetCDF data file
        train_start, train_end: Training period
        test_start, test_end: Test period
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: L2 regularization
        k_max: Maximum harmonic order for seasonal features
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        device: Device to train on
        rollout_k: K-step rollout for training
        extra_train_nc_paths: Additional training data paths
        pretrained_state_dict: Optional state dict to load weights from (for two-stage training)
        exclude_vars: Optional list of variable names to exclude (e.g., ['WWV']).
        include_only_vars: If specified, filter to only these variables (for two-stage training).
        val_start/val_end: If set, use validation split for early stopping instead of test.

    Returns:
        model: Trained model
        var_order: Variable order
        best_rmse: Best test RMSE achieved
        history: Training history dict
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = NXROTransformerModel(
        n_vars=n_vars, 
        k_max=k_max, 
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[Transformer] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[Transformer] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_pure_transformer(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    use_time: bool = False,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train a pure Transformer model with NO physical priors.

    This is a baseline model that learns dynamics entirely from data:
    - No seasonal Fourier features
    - No XRO structure
    - Just raw state variables processed through Transformer

    Args:
        nc_path: Path to NetCDF data file
        train_start, train_end: Training period
        test_start, test_end: Test period
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: L2 regularization
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        use_time: if True, include normalized time as input feature
        device: Device to train on
        rollout_k: K-step rollout for training
        extra_train_nc_paths: Additional training data paths
        pretrained_state_dict: Optional state dict to load weights from
        exclude_vars: Optional list of variable names to exclude
        include_only_vars: If specified, filter to only these variables
        val_start/val_end: If set, use validation split for early stopping instead of test.

    Returns:
        model: Trained model
        var_order: Variable order
        best_rmse: Best test RMSE achieved
        history: Training history dict
    """
    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = PureTransformerModel(
        n_vars=n_vars, 
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_time=use_time
    ).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict for PureTransformer.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dl_val is not None:
            print(f"[PureTransformer] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f}")
        else:
            print(f"[PureTransformer] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history


def train_nxro_deep_gcn(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_start: str = '1979-01',
    train_end: str = '2022-12',
    test_start: str = '2023-01',
    test_end: Optional[str] = None,
    n_epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    k_max: int = 2,
    hidden: int = 64,
    n_layers: int = 3,
    dropout: float = 0.0,
    use_layer_norm: bool = True,
    l1_lambda: float = 0.0,
    use_cosine_scheduler: bool = True,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
    exclude_vars: Optional[list] = None,
    include_only_vars: Optional[list] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Train NXRO-DeepLearnableGCN model.

    This is the best performing model from hyperparameter search:
    - 3-layer GCN with hidden=64
    - Layer normalization
    - Learnable adjacency with zero L1 regularization
    - Cosine annealing LR scheduler

    Achieves val_rmse=0.4633, beating previous best of 0.4974 (6.9% improvement).

    Args:
        nc_path: Path to NetCDF data file
        train_start, train_end: Training period
        test_start, test_end: Test period
        n_epochs: Number of training epochs (default 2000)
        batch_size: Batch size
        lr: Learning rate (default 1e-3)
        weight_decay: L2 regularization (default 1e-5)
        k_max: Number of Fourier harmonics for seasonal coupling
        hidden: Hidden dimension for GCN layers (default 64)
        n_layers: Number of GCN layers (default 3)
        dropout: Dropout rate (default 0.0)
        use_layer_norm: Whether to use layer normalization (default True)
        l1_lambda: L1 regularization on adjacency matrix (default 0.0)
        use_cosine_scheduler: Whether to use cosine annealing LR (default True)
        device: Device to train on
        rollout_k: K-step rollout for training
        extra_train_nc_paths: Additional training data paths
        L_basis_init: Optional initial values for L_basis
        pretrained_state_dict: Optional state dict to load weights from
        exclude_vars: Optional list of variable names to exclude
        include_only_vars: If specified, filter to only these variables
        val_start/val_end: If set, use validation split for early stopping instead of test.

    Returns:
        model: Trained model
        var_order: Variable order
        best_rmse: Best test RMSE achieved
        history: Training history dict
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR

    val_slice = (val_start, val_end) if val_start else None
    dl_train, dl_val, dl_test, var_order = _get_dataloaders_with_val(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        val_slice=val_slice,
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
        exclude_vars=exclude_vars,
        include_only_vars=include_only_vars,
    )

    # Use val for model selection if available, else test
    dl_select = dl_val if dl_val is not None else dl_test

    n_vars = len(var_order)
    model = NXRODeepLearnableGCN(
        n_vars=n_vars,
        k_max=k_max,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        L_basis_init=L_basis_init,
    ).to(device)
    
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print("Loaded pretrained state dict for DeepLearnableGCN.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    scheduler = None
    if use_cosine_scheduler:
        scheduler = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
                x_seq = x_seq.to(device)
            else:
                x_t, t_y, x_next = batch
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            t_y = t_y.to(device)
            dt = 1.0 / 12.0
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            
            # L1 regularization on adjacency
            if l1_lambda > 0:
                loss = loss + l1_lambda * torch.abs(model.A_param).mean()
            
            if rollout_k > 1:
                x_roll = x_hat
                for k in range(1, rollout_k):
                    t_next = t_y + k / 12.0
                    dxdt_k = model(x_roll, t_next)
                    x_roll = x_roll + dxdt_k * dt
                    loss = loss + loss_fn(x_roll, x_seq[:, k])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * x_t.size(0)
            count += x_t.size(0)
        return (total / max(count, 1)) ** 0.5

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    train_hist, test_hist, val_hist = [], [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        select_rmse = run_epoch(dl_select, train=False)
        train_hist.append(float(train_rmse))
        if dl_val is not None:
            val_hist.append(float(select_rmse))
            test_rmse = run_epoch(dl_test, train=False)
            test_hist.append(float(test_rmse))
        else:
            test_hist.append(float(select_rmse))
        if scheduler:
            scheduler.step()
        if select_rmse < best_rmse:
            best_rmse = select_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0] if scheduler else lr
            if dl_val is not None:
                print(f"[DeepGCN] Epoch {epoch:04d} | train RMSE: {train_rmse:.4f} | val RMSE: {select_rmse:.4f} | test RMSE: {test_rmse:.4f} | lr: {lr_now:.6f}")
            else:
                print(f"[DeepGCN] Epoch {epoch:04d} | train RMSE: {train_rmse:.4f} | test RMSE: {select_rmse:.4f} | lr: {lr_now:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {
        "train_rmse": train_hist,
        "test_rmse": test_hist,
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_test_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "epochs_completed": len(train_hist),
    }
    if val_hist:
        history["val_rmse"] = val_hist
        history["selection_on"] = "val"
    else:
        history["selection_on"] = "test"
    return model, var_order, best_rmse, history
