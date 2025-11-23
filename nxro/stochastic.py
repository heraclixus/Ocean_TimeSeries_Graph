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


def fit_seasonal_arp_from_residuals(residuals: np.ndarray, months: np.ndarray, p: int = 1, a_clip: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Fit month-wise AR(p) parameters per variable via OLS.
    
    e_t = a_1(m) e_{t-1} + ... + a_p(m) e_{t-p} + eps
    
    Args:
        residuals: [T-1, V]
        months: [T-1] with values 1..12 corresponding to residual at time t
        p: lag order
        a_clip: clamp coeffs (applied to sum of abs or similar? For now just used if needed, simple OLS usually stable enough for small p)
    Returns:
        coeffs: [12, V, p] - AR coefficients [a_1, ..., a_p] for each month and variable
        sigma: [12, V] - residual std dev
    """
    if p == 1:
        # Fallback to AR(1) optimized impl but reshape output
        a1, sigma = fit_seasonal_ar1_from_residuals(residuals, months, a_clip)
        return a1[..., None], sigma

    Tm1, V = residuals.shape
    coeffs = np.zeros((12, V, p), dtype=np.float32)
    sigma = np.zeros((12, V), dtype=np.float32)
    eps = 1e-8
    
    # Pre-compute lag matrices for all t to avoid re-doing it per month
    # We can just do it per month by indexing
    
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        # We need p previous residuals
        idx = idx[idx >= p]
        if idx.size == 0:
            continue
            
        # Construct Y and X
        # Y: [N, V] -> E_curr
        Y = residuals[idx] # [N, V]
        N = Y.shape[0]
        
        # X: [N, V, p] -> lags
        # We need to solve for each variable v: y_v = X_v @ a_v
        # y_v: [N]
        # X_v: [N, p]
        
        # Gather lags
        X_lags = []
        for lag in range(1, p + 1):
            X_lags.append(residuals[idx - lag]) # [N, V]
        
        # Iterate over variables to solve OLS per variable
        for v in range(V):
            y_v = Y[:, v] # [N]
            # Stack lags for this variable: [N, p]
            X_v = np.stack([lag_arr[:, v] for lag_arr in X_lags], axis=1) 
            
            # OLS: (X'X)^-1 X'y
            # Add regularization?
            XtX = X_v.T @ X_v # [p, p]
            Xty = X_v.T @ y_v # [p]
            
            # Add small ridge for stability
            XtX = XtX + np.eye(p) * eps
            
            try:
                a_v = np.linalg.solve(XtX, Xty) # [p]
            except np.linalg.LinAlgError:
                # Fallback to zeros or similar if singular
                a_v = np.zeros(p)
            
            # Optional clipping? For p>1 simpler clipping is hard.
            # Let's trust OLS for now, or just clip values individually (naive)
            # a_v = np.clip(a_v, -a_clip, a_clip) 
            
            coeffs[m-1, v, :] = a_v.astype(np.float32)
            
            # Compute residuals of AR fit
            y_pred = X_v @ a_v
            resid = y_v - y_pred
            s = np.sqrt(np.maximum((resid * resid).mean(), 0.0))
            sigma[m-1, v] = s.astype(np.float32)
            
    return coeffs, sigma


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


class SeasonalARPNoise:
    """Seasonal AR(p) noise with month-wise parameters per variable.
    
    coeffs: [12, V, p]
    sigma: [12, V]
    """
    
    def __init__(self, coeffs: torch.Tensor, sigma: torch.Tensor):
        self.coeffs = coeffs # [12, V, p]
        self.sigma = sigma   # [12, V]
        self.p = coeffs.shape[2]
        
    def step(self, xi_history: torch.Tensor, month_idx: int) -> torch.Tensor:
        """Advance noise one step.
        
        Args:
            xi_history: [V, p] tensor where column 0 is lag-1, col 1 is lag-2, ...
            month_idx: int in [1..12]
        Returns:
            xi_new: [V]
        """
        m = month_idx - 1
        # Get coeffs for this month: [V, p]
        ac = self.coeffs[m] 
        s = self.sigma[m]
        
        # AR prediction: sum_k a_k * xi_{t-k}
        # xi_history[:, k] is xi_{t-(k+1)}
        # element-wise dot product along p dim
        pred = (ac * xi_history).sum(dim=1) # [V]
        
        eps = torch.randn_like(pred)
        return pred + s * eps


def ar1_log_likelihood(residuals: torch.Tensor, a1: torch.Tensor, log_sigma: torch.Tensor, 
                       months: torch.Tensor, dt: float = 1.0/12.0) -> torch.Tensor:
    """
    Compute AR(1) log-likelihood for seasonal red noise model.
    
    Args:
        residuals: [T, n_vars] - model residuals
        a1: [n_vars] - lag-1 autocorrelation per variable
        log_sigma: [n_vars, 12] - log seasonal std dev (in log space for positivity)
        months: [T] - month indices (0-11 or 1-12, will be normalized to 0-11)
        dt: float - time step (default 1/12 for monthly data)
    
    Returns:
        log_likelihood: scalar tensor
    """
    T, n_vars = residuals.shape
    
    # Ensure month indices are 0-11
    months_idx = months.long()
    if months_idx.max() >= 12:
        months_idx = months_idx - 1  # Convert 1-12 to 0-11
    
    # Get sigma (positive)
    sigma = torch.exp(log_sigma)  # [n_vars, 12]
    
    # Get sigma for each timestep
    sigma_t = sigma[:, months_idx].T  # [T, n_vars]
    
    # Compute AR(1) innovations
    # r_t = r_{t-1} * a1 + innovation
    r_lag = torch.cat([torch.zeros(1, n_vars, device=residuals.device), 
                       residuals[:-1]], dim=0)  # [T, n_vars]
    innovations = residuals - a1[None, :] * r_lag  # [T, n_vars]
    
    # AR(1) innovation variance
    # Var(innovation) = dt^2 * (1 - a1^2) * sigma^2
    var_innovation = dt**2 * (1 - a1**2)[None, :] * (sigma_t**2)  # [T, n_vars]
    
    # Log-likelihood (Gaussian)
    # log N(innovation; 0, var) = -0.5 * [log(2*pi*var) + innovation^2/var]
    log_2pi_var = torch.log(2 * torch.pi * var_innovation + 1e-8)  # Stability
    squared_error = innovations**2 / (var_innovation + 1e-8)
    
    log_prob = -0.5 * (log_2pi_var + squared_error)
    
    return log_prob.sum()  # Sum over all timesteps and variables


def fit_noise_from_simulations(obs_path: str, sim_paths: List[str], var_order: List[str], 
                              train_period: slice) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit noise parameters from simulation-observation differences.
    
    Uses differences between observations (ORAS5) and simulations (ERA5, GODAS, etc.)
    to estimate noise distribution independently of model residuals.
    
    Args:
        obs_path: Path to observation NetCDF (ORAS5)
        sim_paths: List of simulation NetCDF paths (ERA5, GODAS, etc.)
        var_order: Variable names
        train_period: Time slice for training period
    
    Returns:
        a1: [12, n_vars] - AR(1) autocorrelation from combined differences
        sigma: [12, n_vars] - Seasonal std dev from combined differences
    """
    import xarray as xr
    import pandas as pd
    
    # Load full observations (don't restrict to train_period yet)
    obs_ds_full = xr.open_dataset(obs_path)
    
    # Get train period times for filtering
    obs_train = obs_ds_full.sel(time=train_period)
    train_times = set(obs_train.time.values)
    
    print(f"  Observation dataset: {len(obs_ds_full.time)} total timesteps")
    print(f"  Training period: {train_period} ({len(train_times)} timesteps)")
    print(f"  Scanning {len(sim_paths)} simulation files for overlaps...")
    
    all_diffs = []
    all_months = []
    loaded_count = 0
    
    for sim_path in sim_paths:
        try:
            sim_ds = xr.open_dataset(sim_path)
            
            # Convert simulation times to comparable format
            # Handle cftime objects (non-standard calendars like noleap)
            def to_comparable_date(time_val):
                """Convert any time type to (year, month) tuple for comparison."""
                if hasattr(time_val, 'year') and hasattr(time_val, 'month'):
                    return (time_val.year, time_val.month)
                else:
                    dt = pd.Timestamp(time_val)
                    return (dt.year, dt.month)
            
            # Create year-month tuples for comparison
            obs_dates = {to_comparable_date(t): t for t in obs_ds_full.time.values}
            sim_dates = {to_comparable_date(t): t for t in sim_ds.time.values}
            train_dates = {to_comparable_date(t): t for t in train_times}
            
            # Find common year-month pairs
            common_date_keys = set(obs_dates.keys()) & set(sim_dates.keys()) & set(train_dates.keys())
            
            if len(common_date_keys) == 0:
                continue  # Skip without warning (expected for many files)
            
            # Get actual time values for selection
            obs_times_select = [obs_dates[key] for key in sorted(common_date_keys)]
            sim_times_select = [sim_dates[key] for key in sorted(common_date_keys)]
            
            # Align datasets using actual time coordinates
            obs_aligned = obs_ds_full.sel(time=obs_times_select)
            sim_aligned = sim_ds.sel(time=sim_times_select)
            
            # Compute difference (obs - sim)
            diff_array = []
            for v in var_order:
                if v in obs_aligned and v in sim_aligned:
                    diff_vals = obs_aligned[v].values - sim_aligned[v].values
                    diff_array.append(diff_vals)
                else:
                    print(f"  Warning: Variable {v} missing in {sim_path}, skipping file")
                    break  # Skip this file entirely if any variable is missing
            
            if len(diff_array) == len(var_order):
                diff_array = np.stack(diff_array, axis=-1)  # [T, n_vars]
                # Extract months from the sorted common date keys
                months = np.array([key[1] for key in sorted(common_date_keys)], dtype=np.int32)
                
                all_diffs.append(diff_array)
                all_months.append(months)
                loaded_count += 1
                if loaded_count == 1:
                    print(f"  ✓ First successful load: {len(diff_array)} samples from {sim_path}")
                elif loaded_count <= 5 or loaded_count % 10 == 0:  # Print first 5, then every 10th
                    print(f"  [{loaded_count}] Loaded {len(diff_array)} samples")
            
        except Exception as e:
            print(f"  Error loading {sim_path}: {e}")
            continue
    
    if len(all_diffs) == 0:
        print(f"  ERROR: No simulation data overlaps with training period {train_period}")
        print(f"  Tried {len(sim_paths)} simulation files")
        print(f"  Training period: {train_period}")
        print(f"  Available obs times: {len(obs_ds_full.time)} timesteps")
        raise ValueError("No simulation data could be loaded!")
    
    # Concatenate all differences
    combined_diffs = np.concatenate(all_diffs, axis=0)  # [T_total, n_vars]
    combined_months = np.concatenate(all_months)
    
    print(f"  ✓ Successfully loaded {loaded_count}/{len(sim_paths)} simulation datasets")
    print(f"  ✓ Total samples from simulations: {len(combined_diffs)}")
    print(f"  ✓ Skipped {len(sim_paths) - loaded_count} files (no overlap with training period)")
    
    # Fit AR(1) from combined simulation-observation differences
    a1, sigma = fit_seasonal_ar1_from_residuals(combined_diffs, combined_months)
    
    return a1, sigma


def train_noise_stage2(model, train_ds: xr.Dataset, var_order: List[str],
                      a1_init: np.ndarray, sigma_init: np.ndarray,
                      n_epochs: int = 100, lr: float = 1e-3, 
                      device: str = 'cpu', verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2 training: Optimize noise parameters using likelihood objective.
    
    Given a trained deterministic model, optimize (a1, sigma) to maximize likelihood
    of observed data under the stochastic model.
    
    Args:
        model: Trained NXRO model (frozen)
        train_ds: Training dataset
        var_order: Variable names
        a1_init: [12, n_vars] - initial lag-1 autocorrelation (from post-hoc fit)
        sigma_init: [12, n_vars] - initial seasonal std dev (from post-hoc fit)
        n_epochs: Number of optimization epochs
        lr: Learning rate
        device: Device
        verbose: Print progress
    
    Returns:
        a1_optimized: [12, n_vars] - optimized autocorrelation
        sigma_optimized: [12, n_vars] - optimized seasonal std dev
    """
    # Compute residuals once (model is frozen)
    if verbose:
        print("Computing residuals from trained model...")
    
    residuals_np, months_np = compute_residuals_series(model, train_ds, var_order, device=device)
    residuals = torch.from_numpy(residuals_np).to(device)
    months = torch.from_numpy(months_np - 1).to(device)  # Convert to 0-11
    
    # Get month indices for reshaping a1, sigma
    # Current a1_init, sigma_init are [12, n_vars] from OLS per month
    # We need [n_vars] for a1 (shared across months) and [n_vars, 12] for sigma
    
    # Initialize parameters
    a1_shared = torch.tensor(a1_init.mean(axis=0), dtype=torch.float32, device=device)
    a1 = torch.nn.Parameter(a1_shared)  # [n_vars]
    
    # Sigma varies by month
    log_sigma = torch.nn.Parameter(
        torch.tensor(np.log(sigma_init.T + 1e-6), dtype=torch.float32, device=device)
    )  # [n_vars, 12]
    
    # Optimizer
    optimizer = torch.optim.Adam([a1, log_sigma], lr=lr)
    
    # Training loop
    if verbose:
        print(f"Optimizing noise parameters for {n_epochs} epochs...")
    
    best_nll = float('inf')
    best_a1 = None
    best_log_sigma = None
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute negative log-likelihood
        log_likelihood = ar1_log_likelihood(residuals, a1, log_sigma, months)
        loss = -log_likelihood
        
        # Optional: Regularization to keep close to initial values
        # reg_loss = 0.01 * torch.sum((a1 - a1_shared)**2)
        # loss = loss + reg_loss
        
        loss.backward()
        optimizer.step()
        
        # Constraint: a1 should be in (0, 1)
        with torch.no_grad():
            a1.clamp_(0.01, 0.99)
        
        # Track best
        if loss.item() < best_nll:
            best_nll = loss.item()
            best_a1 = a1.detach().clone()
            best_log_sigma = log_sigma.detach().clone()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, NLL={loss.item():.4f}")
    
    # Return best parameters
    a1_opt = best_a1.cpu().numpy()  # [n_vars]
    sigma_opt = torch.exp(best_log_sigma).cpu().numpy()  # [n_vars, 12]
    
    # Expand a1 to [12, n_vars] for compatibility with existing code
    a1_opt_expanded = np.tile(a1_opt[None, :], (12, 1))  # [12, n_vars]
    
    if verbose:
        print(f"Stage 2 training complete. Best NLL: {best_nll:.4f}")
        print(f"  a1 range: [{a1_opt.min():.3f}, {a1_opt.max():.3f}]")
        print(f"  sigma range: [{sigma_opt.min():.3f}, {sigma_opt.max():.3f}]")
    
    return a1_opt_expanded, sigma_opt.T  # Return as [12, n_vars]


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

@torch.no_grad()
def nxro_reforecast_stochastic_arp(model, init_ds: xr.Dataset, n_month: int, var_order: list,
                                  noise_model: SeasonalARPNoise, n_members: int = 100, device: str = 'cpu') -> xr.Dataset:
    """Stochastic reforecast with seasonal AR(p) noise."""
    ncycle = 12
    dt = 1.0 / ncycle
    n_vars = len(var_order)
    n_init = len(init_ds.time)
    p = noise_model.p

    # [V, L, I, M]
    out = np.zeros((n_vars, n_month, n_init, n_members), dtype=np.float32)

    model.eval()
    for i in range(n_init):
        X0 = np.stack([init_ds[v].isel(time=i).item() for v in var_order], axis=-1).astype(np.float32)
        base_date = pd.to_datetime(str(init_ds.time.values[i]))
        for m_ens in range(n_members):
            x = torch.from_numpy(X0[None, :]).to(device)
            
            # xi history: [V, p]. Init with zeros (mean of noise)
            xi_history = torch.zeros(n_vars, p, dtype=torch.float32, device=device)
            
            out[:, 0, i, m_ens] = x.squeeze(0).cpu().numpy()
            for t_step in range(1, n_month):
                current_dt = base_date + pd.DateOffset(months=t_step - 1)
                month_idx = int(current_dt.month)
                t_year = torch.tensor([float(current_dt.year + (current_dt.month - 1) / 12.0)],
                                      dtype=torch.float32, device=device)
                
                dxdt = model(x, t_year)
                
                # Get new noise
                xi_new = noise_model.step(xi_history, month_idx) # [V]
                
                # Update history: shift right, put new at 0
                # BUT: step expects xi_history where col 0 is lag 1.
                # So we shift: old lag 1 becomes lag 2.
                # [lag1, lag2, ...] -> [new, lag1, ...]
                if p > 1:
                    xi_history = torch.cat([xi_new.unsqueeze(1), xi_history[:, :-1]], dim=1)
                else:
                    xi_history = xi_new.unsqueeze(1)
                
                x = x + dxdt * dt + xi_new.unsqueeze(0)
                out[:, t_step, i, m_ens] = x.squeeze(0).cpu().numpy()

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
