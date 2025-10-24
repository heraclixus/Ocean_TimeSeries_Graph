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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
    )

    n_vars = len(var_order)
    model = NXROLinearModel(n_vars=n_vars, k_max=k_max).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
    )

    n_vars = len(var_order)
    model = NXROROModel(n_vars=n_vars, k_max=k_max).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
    )

    n_vars = len(var_order)
    model = NXRORODiagModel(n_vars=n_vars, k_max=k_max).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
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
    device: str = 'cpu',
    rollout_k: int = 1,
):
    # dataloaders for training
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
    )
    # build edge_index from train correlation
    import xarray as xr
    import torch
    ds = xr.open_dataset(nc_path).sel(time=slice(train_start, train_end))
    X_np = []
    for v in var_order:
        X_np.append(ds[v].values)
    X = torch.tensor(np.stack(X_np, axis=-1), dtype=torch.float32)
    corr = torch.corrcoef(X.T)
    edge_index = build_edge_index_from_corr(corr, top_k=top_k).to(device)

    n_vars = len(var_order)
    model = NXROGraphPyGModel(n_vars=n_vars, k_max=k_max, edge_index=edge_index, hidden=hidden, dropout=dropout, use_gat=use_gat).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
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
    device: str = 'cpu',
    rollout_k: int = 1,
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
    )
    n_vars = len(var_order)
    model = NXROGraphModel(n_vars=n_vars, k_max=k_max, use_fixed_graph=use_fixed_graph).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
    )

    n_vars = len(var_order)
    model = NXROAttentiveModel(n_vars=n_vars, k_max=k_max, d=d, dropout=dropout, mask_mode=mask_mode).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
        rollout_k=rollout_k,
    )

    n_vars = len(var_order)
    model = NXROResModel(n_vars=n_vars, k_max=k_max).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for batch in tqdm(dl, disable=not train):
            if rollout_k > 1:
                x_t, t_y, x_next, x_seq = batch
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
):
    dl_train, dl_test, var_order = get_dataloaders(
        nc_path=nc_path,
        train_slice=(train_start, train_end),
        test_slice=(test_start, test_end),
        batch_size=batch_size,
    )

    n_vars = len(var_order)
    model = NXROResidualMixModel(n_vars=n_vars, k_max=k_max, hidden=hidden,
                                 alpha_init=alpha_init, alpha_learnable=alpha_learnable,
                                 alpha_max=alpha_max, dropout=0.0).to(device)
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

