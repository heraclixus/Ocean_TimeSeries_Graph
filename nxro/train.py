from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .models import (
    NXROLinearModel,
    NXROROModel,
    NXRORODiagModel,
    NXROResModel,
    NXROResidualMixModel,
    NXRONeuralODEModel,
    NXROBilinearModel,
    NXROAttentiveModel,
    NXROGraphModel,
    NXROGraphPyGModel,
    build_edge_index_from_corr,
)
from .data import get_dataloaders
from graph_construction import build_xro_coupling_graph, normalize_with_self_loops, get_or_build_xro_graph, get_or_build_stat_knn_graph


def train_nxro_linear(
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
    L_basis_init: Optional[torch.Tensor] = None,
    pretrained_state_dict: Optional[dict] = None,
):
    """Train NXRO-Linear model (variants 1, 1a).
    
    Args:
        L_basis_init: If None, random init (variant 1). If provided, warm-start (variant 1a).
        pretrained_state_dict: Optional state dict to load weights from (e.g. for two-stage training).
    """
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

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
                # Multi-step rollout starting from x_hat to match x_seq
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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    """Train NXRO-RO model (variants 2, 2a, 2a-Fix*).
    
    Args:
        warmstart_init_dict: Dict with 'L_basis_init', 'W_T_init', 'W_H_init' for warm-start
        freeze_flags: Dict with 'freeze_linear', 'freeze_ro' flags
        pretrained_state_dict: Optional state dict to load weights from.
    """
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[RO] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    """Train NXRO-RO+Diag model (variants 3, 3a, 3a-Fix*).
    
    Args:
        warmstart_init_dict: Dict with init parameters for warm-start
        freeze_flags: Dict with 'freeze_linear', 'freeze_ro', 'freeze_diag' flags
        pretrained_state_dict: Optional state dict to load weights from.
    """
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[RO+Diag] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

    n_vars = len(var_order)
    model = NXRONeuralODEModel(n_vars=n_vars, k_max=k_max, hidden=hidden, depth=depth,
                               dropout=dropout, allow_cross=allow_cross, mask_mode=mask_mode).to(device)
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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[NeuralPhys] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    # dataloaders for training
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )
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
        import xarray as xr
        ds = xr.open_dataset(nc_path).sel(time=slice(train_start, train_end))
        X_np = []
        for v in var_order:
            X_np.append(ds[v].values)
        X = torch.tensor(np.stack(X_np, axis=-1), dtype=torch.float32)
        corr = torch.corrcoef(X.T)
        edge_index = build_edge_index_from_corr(corr, top_k=top_k).to(device)

    n_vars = len(var_order)
    model = NXROGraphPyGModel(n_vars=n_vars, k_max=k_max, edge_index=edge_index, hidden=hidden, dropout=dropout, use_gat=use_gat).to(device)
    
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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[GraphPyG] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    """Train NXRO-Graph model (variants 5b, 5b-WS, 5b-FixL)."""
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )
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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[Graph] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[NeuralODE] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    """Train NXRO-Attentive model (variants 5a, 5a-WS, 5a-FixL)."""
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

    n_vars = len(var_order)
    
    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max, 'd': d, 'dropout': dropout, 'mask_mode': mask_mode}
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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[Attentive] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

    n_vars = len(var_order)
    model = NXROBilinearModel(n_vars=n_vars, k_max=k_max, n_channels=n_channels, rank=rank).to(device)
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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[Bilinear] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
    res_reg: float = 1e-4,
    device: str = 'cpu',
    rollout_k: int = 1,
    extra_train_nc_paths=None,
    warmstart_init_dict: dict = None,
    freeze_flags: dict = None,
    pretrained_state_dict: Optional[dict] = None,
):
    """Train NXRO-Res model (variants 4, 4a).
    
    Args:
        warmstart_init_dict: Dict with 'L_basis_init' for warm-start
        freeze_flags: Dict with 'freeze_linear' flag
        pretrained_state_dict: Optional state dict to load weights from.
    """
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

    n_vars = len(var_order)
    
    # Merge warm-start and freeze parameters
    model_kwargs = {'n_vars': n_vars, 'k_max': k_max}
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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[Res] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    """Train NXRO-ResidualMix model (variants 5d, 5d-WS, 5d-Fix*)."""
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        extra_train_nc_paths=extra_train_nc_paths,
    )

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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[ResMix] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
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
):
    """Train NXRO-Res-FullXRO model (variant 4b): Frozen full XRO + trainable MLP.
    
    All XRO components (L, RO, Diag) are frozen. Only residual MLP is trainable.
    
    Args:
        xro_init_dict: REQUIRED dict with 'L_basis', 'W_T', 'W_H', 'B_diag', 'C_diag' from XRO
    """
    from .models import NXROResFullXROModel
    
    assert xro_init_dict is not None, "Variant 4b requires XRO initialization!"
    assert all(k in xro_init_dict for k in ['L_basis', 'W_T', 'W_H', 'B_diag', 'C_diag']), \
        "xro_init_dict must contain all XRO components!"
    
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
        extra_train_nc_paths=extra_train_nc_paths,
    )

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
    train_hist, test_hist = [], []
    for epoch in range(1, n_epochs + 1):
        train_rmse = run_epoch(dl_train, train=True)
        test_rmse = run_epoch(dl_test, train=False)
        train_hist.append(float(train_rmse))
        test_hist.append(float(test_rmse))
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[Res-FullXRO] Epoch {epoch:03d} | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_rmse": train_hist, "test_rmse": test_hist}
    return model, var_order, best_rmse, history
