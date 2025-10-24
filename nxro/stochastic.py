import numpy as np
import torch
import xarray as xr
import pandas as pd
from typing import Tuple, List


@torch.no_grad()
def compute_residuals_series(model, ds: xr.Dataset, var_order: List[str], device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-step residuals on a dataset given a deterministic model.

    Returns:
        residuals: [T-1, V] numpy array of residuals e_t = x_{t+1} - (x_t + f(x_t,t) dt)
        months: [T-1] numpy array of calendar months (1..12) for each residual at time t
    """
    ncycle = 12
    dt = 1.0 / ncycle
    time_index = pd.DatetimeIndex(ds.time.values)
    years = time_index.year + (time_index.month - 1) / 12.0
    T = len(time_index)
    n_vars = len(var_order)
    residuals = np.zeros((T - 1, n_vars), dtype=np.float32)
    months = time_index.month.values[:-1].astype(np.int32)

    model.eval()
    for t in range(T - 1):
        x_t = np.stack([ds[v].isel(time=t).item() for v in var_order], axis=-1).astype(np.float32)
        x_tp1 = np.stack([ds[v].isel(time=t + 1).item() for v in var_order], axis=-1).astype(np.float32)
        x = torch.from_numpy(x_t[None, :]).to(device)
        t_year = torch.tensor([float(years[t])], dtype=torch.float32, device=device)
        dxdt = model(x, t_year)
        x_hat = x + dxdt * dt
        e_t = x_tp1 - x_hat.squeeze(0).cpu().numpy()
        residuals[t] = e_t
    return residuals, months


def fit_seasonal_ar1_from_residuals(residuals: np.ndarray, months: np.ndarray, a1_clip: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Fit month-wise AR(1) parameters a1(m), sigma(m) per variable via OLS on e_t = a1(m) e_{t-1} + eps.

    Args:
        residuals: [T-1, V]
        months: [T-1] with values 1..12 corresponding to residual at time t
        a1_clip: clamp |a1| <= a1_clip for stability
    Returns:
        a1: [12, V]
        sigma: [12, V]
    """
    Tm1, V = residuals.shape
    a1 = np.zeros((12, V), dtype=np.float32)
    sigma = np.zeros((12, V), dtype=np.float32)
    eps = 1e-8
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        # need previous residuals, so drop first index if it is 0
        idx = idx[idx > 0]
        if idx.size == 0:
            continue
        prev_idx = idx - 1
        E_prev = residuals[prev_idx]  # [N, V]
        E_curr = residuals[idx]       # [N, V]
        # OLS per variable
        num = (E_prev * E_curr).sum(axis=0)
        den = (E_prev * E_prev).sum(axis=0) + eps
        a = num / den
        a = np.clip(a, -a1_clip, a1_clip)
        a1[m - 1] = a.astype(np.float32)
        # residual std
        resid = E_curr - E_prev * a
        s = np.sqrt(np.maximum((resid * resid).mean(axis=0), 0.0))
        sigma[m - 1] = s.astype(np.float32)
    return a1, sigma


class SeasonalAR1Noise:
    """Seasonal AR(1) noise with month-wise parameters per variable.

    a1: [12, V], sigma: [12, V]
    """

    def __init__(self, a1: torch.Tensor, sigma: torch.Tensor):
        assert a1.shape == sigma.shape and a1.dim() == 2
        self.a1 = a1  # [12, V]
        self.sigma = sigma  # [12, V]

    def step(self, xi_prev: torch.Tensor, month_idx: int) -> torch.Tensor:
        """Advance noise one step given previous state and current calendar month (1..12).

        Args:
            xi_prev: [V] tensor
            month_idx: int in [1..12]
        Returns:
            xi: [V]
        """
        m = month_idx - 1
        a = self.a1[m]
        s = self.sigma[m]
        eps = torch.randn_like(xi_prev)
        return a * xi_prev + s * eps


@torch.no_grad()
def nxro_reforecast_stochastic(model, init_ds: xr.Dataset, n_month: int, var_order: list,
                               noise_model: SeasonalAR1Noise, n_members: int = 100, device: str = 'cpu') -> xr.Dataset:
    """Stochastic reforecast with seasonal AR(1) noise.

    Returns xr.Dataset with variables, dims: ['lead','init','member'] and lead attrs.
    """
    ncycle = 12
    dt = 1.0 / ncycle
    n_vars = len(var_order)
    n_init = len(init_ds.time)

    # [V, L, I, M]
    out = np.zeros((n_vars, n_month, n_init, n_members), dtype=np.float32)

    model.eval()
    for i in range(n_init):
        X0 = np.stack([init_ds[v].isel(time=i).item() for v in var_order], axis=-1).astype(np.float32)
        base_date = pd.to_datetime(str(init_ds.time.values[i]))
        for m in range(n_members):
            x = torch.from_numpy(X0[None, :]).to(device)
            xi = torch.zeros(x.shape[1], dtype=torch.float32, device=device)
            out[:, 0, i, m] = x.squeeze(0).cpu().numpy()
            for t_step in range(1, n_month):
                current_dt = base_date + pd.DateOffset(months=t_step - 1)
                month_idx = int(current_dt.month)
                t_year = torch.tensor([float(current_dt.year + (current_dt.month - 1) / 12.0)],
                                      dtype=torch.float32, device=device)
                dxdt = model(x, t_year)
                xi = noise_model.step(xi, month_idx)
                x = x + dxdt * dt + xi.unsqueeze(0)
                out[:, t_step, i, m] = x.squeeze(0).cpu().numpy()

    coords = {
        'ranky': np.arange(1, n_vars + 1),
        'lead': np.arange(0, n_month),
        'init': ('init', init_ds.time.values),
        'member': np.arange(1, n_members + 1),
    }
    da = xr.DataArray(out, dims=['ranky', 'lead', 'init', 'member'], coords=coords)
    da['lead'].attrs['units'] = 'months'
    da['lead'].attrs['long_name'] = 'Lead'
    ds = xr.Dataset({var: da.sel(ranky=k + 1).drop('ranky') for k, var in enumerate(var_order)})
    return ds


