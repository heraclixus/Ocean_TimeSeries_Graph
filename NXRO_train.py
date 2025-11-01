import os
import warnings
warnings.filterwarnings("ignore")
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import re
import hashlib

from nxro.train import (
    train_nxro_linear,
    train_nxro_ro,
    train_nxro_rodiag,
    train_nxro_res,
    train_nxro_res_fullxro,
    train_nxro_neural,
    train_nxro_bilinear,
    train_nxro_attentive,
    train_nxro_graph,
    train_nxro_graph_pyg,
    train_nxro_neural_phys,
    train_nxro_resmix,
)
from utils.xro_utils import calc_forecast_skill, nxro_reforecast, plot_forecast_plume, evaluate_stochastic_ensemble
from nxro.stochastic import compute_residuals_series, fit_seasonal_ar1_from_residuals, SeasonalAR1Noise, nxro_reforecast_stochastic


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_observed_nino34(obs_ds: xr.Dataset, out_path: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    obs_ds['Nino34'].plot(ax=ax, c='black')
    ax.set_title('Observed Nino3.4 SSTA')
    plt.savefig(out_path, dpi=300)
    plt.close()


def pick_sample_inits(ds: xr.Dataset, n: int = 3) -> list:
    time_index = pd.to_datetime(ds.time.values)
    T = len(time_index)
    if T == 0:
        return []
    candidates = [min(12, T - 1), T // 2, max(T - 12, 0)]
    uniq = sorted(set(candidates))
    out = []
    for i in uniq[:n]:
        out.append(f"{time_index[i].year:04d}-{time_index[i].month:02d}")
    return out


def simulate_nxro_longrun(model, X0_ds: xr.Dataset, var_order: list, nyear: int = 100, device: str = 'cpu') -> xr.Dataset:
    """Deterministic long-run simulation for seasonal stddev plots.

    Starts from X0_ds first time step; monthly Euler stepping without noise.
    """
    ncycle = 12
    dt = 1.0 / ncycle
    start_time = pd.to_datetime(str(X0_ds.time.values[0]))
    n_months = nyear * ncycle
    time_index = pd.date_range(start=start_time, periods=n_months, freq='MS')
    years = time_index.year + (time_index.month - 1) / 12.0

    X0 = np.stack([X0_ds[v].isel(time=0).item() for v in var_order], axis=-1).astype(np.float32)
    x = torch.from_numpy(X0[None, :]).to(device)  # [1, n_vars]
    n_vars = x.shape[1]
    out = np.zeros((n_months, n_vars), dtype=np.float32)
    out[0] = x.squeeze(0).cpu().numpy()

    model.eval()
    with torch.no_grad():
        for t in range(1, n_months):
            t_year = torch.tensor([float(years[t-1])], dtype=torch.float32, device=device)
            dxdt = model(x, t_year)
            x = x + dxdt * dt
            out[t] = x.squeeze(0).detach().cpu().numpy()

    ds = xr.Dataset({var: (['time'], out[:, i]) for i, var in enumerate(var_order)}, coords={'time': time_index})
    return ds


def plot_seasonal_sync(train_ds: xr.Dataset, sim_ds: xr.Dataset, sel_var: str, out_path: str) -> None:
    stddev_obs = train_ds.groupby('time.month').std('time')
    stddev_sim = sim_ds.groupby('time.month').std('time')

    plt.plot(stddev_obs.month, stddev_obs[sel_var], c='black', label='ORAS5')
    plt.plot(stddev_sim.month, stddev_sim[sel_var], c='green', label='NXRO-Linear')
    plt.legend()
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel(f'{sel_var} seasonal standard deviation (℃)')
    plt.xlabel('Calendar Month')
    plt.title('Seasonal synchronization')
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_skill_curves(acc_ds: xr.Dataset, rmse_ds: xr.Dataset, sel_var: str, out_prefix: str, label: str) -> None:
    # ACC
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    acc_ds[sel_var].plot(ax=ax, label=label, c='green', lw=2)
    ax.set_ylabel('Correlation')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0.2, 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend()
    plt.savefig(f'{out_prefix}_acc.png', dpi=300)
    plt.close()

    # RMSE
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    rmse_ds[sel_var].plot(ax=ax, label=label, c='green', lw=2)
    ax.set_ylabel('RMSE (℃)')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0., 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend()
    plt.savefig(f'{out_prefix}_rmse.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train NXRO models')
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'ro', 'rodiag', 'res', 'res_fullxro', 'neural', 'neural_phys', 'resmix', 'bilinear', 'attentive', 'graph', 'graph_pyg', 'all'])
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--gat', action='store_true')
    parser.add_argument('--res_reg', type=float, default=1e-4)
    parser.add_argument('--nc_path', type=str, default='data/XRO_indices_oras5.nc')
    parser.add_argument('--train_start', type=str, default='1979-01')
    parser.add_argument('--train_end', type=str, default='2022-12')
    parser.add_argument('--test_start', type=str, default='2023-01')
    parser.add_argument('--test_end', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--k_max', type=int, default=2)
    parser.add_argument('--jac_reg', type=float, default=1e-4)
    parser.add_argument('--div_reg', type=float, default=0.0)
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--alpha_init', type=float, default=0.1)
    parser.add_argument('--alpha_learnable', action='store_true')
    parser.add_argument('--alpha_max', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--stochastic', action='store_true', help='Enable stochastic reforecast outputs')
    parser.add_argument('--members', type=int, default=100, help='Number of ensemble members if stochastic')
    parser.add_argument('--rollout_k', type=int, default=1, help='K-step rollout loss if >1')
    parser.add_argument('--test', action='store_true', help='Save checkpoint as *_best_test.pt (best on test)')
    parser.add_argument('--extra_train_nc', type=str, nargs='*', default=None,
                        help='Additional NetCDFs for training only (test stays ORAS5).\n'
                             'Usage: pass one or more paths (space-separated).\n'
                             'You can also pass a single token "auto" or provide the flag with no paths to auto-use all data/*_preproc.nc.')
    # Graph options
    parser.add_argument('--graph_learned', action='store_true', help='Use learnable adjacency in NXRO-Graph (L1 sparsity)')
    parser.add_argument('--graph_l1', type=float, default=0.0, help='L1 lambda for learned adjacency')
    parser.add_argument('--graph_stat_method', type=str, default=None,
                        choices=[None, 'pearson', 'spearman', 'mi', 'xcorr_max'],
                        help='Statistical interaction method for KNN prior (if set)')
    parser.add_argument('--graph_stat_topk', type=int, default=2, help='Top-k neighbors for statistical KNN')
    parser.add_argument('--graph_stat_source', type=str, default='data/XRO_indices_oras5_train.csv',
                        help='Data source (CSV or NC) for statistical KNN prior')
    # Warm-start and freezing arguments
    parser.add_argument('--warm_start', type=str, default=None,
                        help='Path to XRO fit file (NetCDF) for warm-start initialization')
    parser.add_argument('--freeze', type=str, default=None,
                        help='Comma-separated list of components to freeze: linear, ro, diag (e.g., "linear,ro")')
    args = parser.parse_args()

    ensure_dir('results')
    device = args.device
    
    # Helper function to parse freeze argument
    def parse_freeze_arg(freeze_str):
        """Parse freeze argument into boolean flags."""
        if freeze_str is None:
            return {'freeze_linear': False, 'freeze_ro': False, 'freeze_diag': False}
        components = [c.strip().lower() for c in freeze_str.split(',')]
        return {
            'freeze_linear': 'linear' in components,
            'freeze_ro': 'ro' in components,
            'freeze_diag': 'diag' in components,
        }
    
    # Helper function to load XRO fit and extract init parameters
    def load_xro_init(xro_path, k_max, include_ro=False, include_diag=False):
        """Load XRO fit and extract initialization parameters."""
        if xro_path is None:
            return {}
        from utils.xro_utils import init_nxro_from_xro
        import xarray as xr
        xro_fit_ds = xr.open_dataset(xro_path)
        init_dict = init_nxro_from_xro(xro_fit_ds, k_max=k_max, 
                                       include_ro=include_ro, include_diag=include_diag)
        # Rename keys to match model __init__ parameters
        init_params = {}
        if 'L_basis' in init_dict:
            init_params['L_basis_init'] = init_dict['L_basis']
        if 'W_T' in init_dict:
            init_params['W_T_init'] = init_dict['W_T']
            init_params['W_H_init'] = init_dict['W_H']
        if 'B_diag' in init_dict:
            init_params['B_diag_init'] = init_dict['B_diag']
            init_params['C_diag_init'] = init_dict['C_diag']
        return init_params
    
    freeze_flags = parse_freeze_arg(args.freeze)
    
    # Helper function to generate variant suffix for filename
    def get_variant_suffix():
        """Generate filename suffix based on warm-start and freeze settings."""
        if args.warm_start is None and args.freeze is None:
            # Base variant (random init, no freezing)
            return ''
        
        suffix_parts = []
        
        # Check if warm-start is used
        if args.warm_start is not None:
            # Check freeze settings
            if args.freeze is None:
                # Warm-start all, train all
                suffix_parts.append('ws')
            else:
                # Warm-start with freezing
                freeze_components = [c.strip().lower() for c in args.freeze.split(',')]
                
                # Determine freeze pattern
                has_linear = 'linear' in freeze_components
                has_ro = 'ro' in freeze_components
                has_diag = 'diag' in freeze_components
                
                if has_linear and has_ro and has_diag:
                    suffix_parts.append('fixPhysics')
                elif has_ro and has_diag and not has_linear:
                    suffix_parts.append('fixNL')
                elif has_linear and not has_ro and not has_diag:
                    suffix_parts.append('fixL')
                elif has_ro and not has_linear and not has_diag:
                    suffix_parts.append('fixRO')
                elif has_diag and not has_linear and not has_ro:
                    suffix_parts.append('fixDiag')
                else:
                    # Custom combination
                    suffix_parts.append('fix' + '_'.join(sorted(freeze_components)))
        
        return '_' + '_'.join(suffix_parts) if suffix_parts else ''
    
    variant_suffix = get_variant_suffix()

    # Auto-discover helper for *_preproc.nc under data dir of nc_path
    def discover_preprocessed_all(nc_path_base):
        base_dir = os.path.dirname(nc_path_base) or 'data'
        try:
            import glob
            cands = sorted(glob.glob(os.path.join(base_dir, 'XRO_indices_*_preproc.nc')))
            return cands
        except Exception:
            return []

    # Only use preprocessed climate files: map provided paths to *_preproc.nc or skip
    def resolve_preprocessed_paths(paths):
        if paths is None:
            return None
        # Support special cases: empty list => auto; or token 'auto' present => auto
        if isinstance(paths, (list, tuple)) and len(paths) == 0:
            auto = discover_preprocessed_all(args.nc_path)
            print(f"Auto-using {len(auto)} preprocessed files from data/: {auto}")
            return auto or None
        if isinstance(paths, (list, tuple)) and any(str(p).lower() == 'auto' for p in paths):
            auto = discover_preprocessed_all(args.nc_path)
            print(f"Auto-using {len(auto)} preprocessed files from data/: {auto}")
            return auto or None
        resolved = []
        for p in paths:
            if not isinstance(p, str):
                continue
            if p.endswith('_preproc.nc'):
                if os.path.exists(p):
                    resolved.append(p)
                else:
                    print(f"[WARN] Preprocessed file not found: {p} (skipping)")
            elif p.endswith('.nc'):
                root, ext = os.path.splitext(p)
                cand = f"{root}_preproc.nc"
                if os.path.exists(cand):
                    print(f"Using preprocessed file for {p}: {cand}")
                    resolved.append(cand)
                else:
                    print(f"[WARN] Skipping non-preprocessed file {p}; expected {cand}")
            else:
                print(f"[WARN] Skipping non-NetCDF path: {p}")
        return resolved

    args.extra_train_nc = resolve_preprocessed_paths(args.extra_train_nc)
    # Build suffix tag for filenames when extra training data are used
    # Simple suffix: just "_extra_data" if any extra data is present
    def build_extra_tag(extra_paths):
        if not extra_paths:
            return ''
        return '_extra_data'

    extra_tag = build_extra_tag(args.extra_train_nc)
    # For figures, use the same suffix as extra_tag for consistency
    fig_suffix = extra_tag

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    obs_ds = xr.open_dataset(args.nc_path)
    train_ds = obs_ds.sel(time=slice(args.train_start, args.train_end))

    # Observed plot
    plot_observed_nino34(obs_ds, out_path='results/NXRO_observed_Nino34.png')

    def run_linear():
        base_dir = 'results/linear'
        ensure_dir(base_dir)
        
        # Load warm-start parameters if provided (only needs linear for variant 1)
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=False, include_diag=False) if args.warm_start else None
        L_basis_init = warmstart_params.get('L_basis_init') if warmstart_params else None
        
        model, var_order, best_rmse, history = train_nxro_linear(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc,
            L_basis_init=L_basis_init,
        )
        # Plot training curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Linear training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_linear_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save weights (optionally with best test epoch suffix)
        best_epoch = int(np.argmin(history['test_rmse'])) + 1
        lin_save = f'{base_dir}/nxro_linear{variant_suffix}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_linear{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': model.state_dict(), 'var_order': var_order}, lin_save)
        print(f"✓ Saved to: {lin_save}")

        # Reforecast for skills (deterministic; stochastic optional)
        NXRO_fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device=device)
        acc_NXRO = calc_forecast_skill(NXRO_fcst, obs_ds, metric='acc', is_mv3=True,
                                       by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO = calc_forecast_skill(NXRO_fcst, obs_ds, metric='rmse', is_mv3=True,
                                        by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO, rmse_NXRO, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_linear{fig_suffix}', label='NXRO-Linear')
        if args.stochastic:
            resid, months = compute_residuals_series(model, train_ds, var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_fcst_m = nxro_reforecast_stochastic(model, init_ds=obs_ds, n_month=21, var_order=var_order,
                                                     noise_model=noise, n_members=args.members, device=device)
            # Save noise params and forecasts
            np.savez(f'{base_dir}/nxro_linear_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': model.state_dict(), 'var_order': var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_linear_stochastic{extra_tag}.pt')
            NXRO_fcst_m.to_netcdf(f'{base_dir}/NXRO_linear_stochastic_forecasts{extra_tag}.nc')
            # Skills on ensemble mean
            NXRO_fcst_m_mean = NXRO_fcst_m.mean('member')
            acc_lin_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_lin_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                                by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_lin_stoc, rmse_lin_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_linear_stochastic{fig_suffix}', label='NXRO-Linear (stochastic mean)')
            # Plume plots
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_fcst, NXRO_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_linear_plume', fig_suffix=fig_suffix)
            # Ensemble evaluation
            evaluate_stochastic_ensemble(NXRO_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_linear_stochastic_eval{extra_tag}')

        # Seasonal synchronization via long-run deterministic simulation
        sim_ds = simulate_nxro_longrun(model, X0_ds=train_ds, var_order=var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_linear_seasonal_synchronization{fig_suffix}.png')

    def run_ro():
        base_dir = 'results/ro'
        ensure_dir(base_dir)
        
        # Load warm-start parameters if provided (needs linear + RO for variant 2)
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=True, include_diag=False) if args.warm_start else None
        
        # Filter freeze_flags to only include linear and ro (no diag for this model)
        freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k != 'freeze_diag'}
        
        ro_model, ro_var_order, ro_best_rmse, ro_history = train_nxro_ro(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc,
            warmstart_init_dict=warmstart_params,
            freeze_flags=freeze_flags_filtered,
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(ro_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(ro_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-RO training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_ro_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        ro_best_epoch = int(np.argmin(ro_history['test_rmse'])) + 1
        ro_save = f'{base_dir}/nxro_ro{variant_suffix}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_ro{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': ro_model.state_dict(), 'var_order': ro_var_order}, ro_save)
        print(f"✓ Saved to: {ro_save}")
        # Skills & seasonal sync
        NXRO_ro_fcst = nxro_reforecast(ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device=device)
        acc_NXRO_ro = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_ro = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_ro, rmse_NXRO_ro, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_ro{fig_suffix}', label='NXRO-RO')
        if args.stochastic:
            resid, months = compute_residuals_series(ro_model, train_ds, ro_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_ro_fcst_m = nxro_reforecast_stochastic(ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_ro_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_ro_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_ro_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': ro_model.state_dict(), 'var_order': ro_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_ro_stochastic{extra_tag}.pt')
            NXRO_ro_fcst_m.to_netcdf(f'{base_dir}/NXRO_ro_stochastic_forecasts{extra_tag}.nc')
            NXRO_ro_fcst_m_mean = NXRO_ro_fcst_m.mean('member')
            acc_ro_stoc = calc_forecast_skill(NXRO_ro_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_ro_stoc = calc_forecast_skill(NXRO_ro_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_ro_stoc, rmse_ro_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_ro_stochastic{fig_suffix}', label='NXRO-RO (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_ro_fcst, NXRO_ro_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_ro_plume', fig_suffix=fig_suffix)
        ro_sim_ds = simulate_nxro_longrun(ro_model, X0_ds=train_ds, var_order=ro_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, ro_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_ro_seasonal_synchronization{fig_suffix}.png')

    def run_rodiag():
        base_dir = 'results/rodiag'
        ensure_dir(base_dir)
        
        # Load warm-start parameters if provided (needs all: L, RO, Diag for variant 3)
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=True, include_diag=True) if args.warm_start else None
        
        rd_model, rd_var_order, rd_best_rmse, rd_history = train_nxro_rodiag(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc,
            warmstart_init_dict=warmstart_params,
            freeze_flags=freeze_flags,
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rd_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rd_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-RO+Diag training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_rodiag_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        rd_best_epoch = int(np.argmin(rd_history['test_rmse'])) + 1
        rd_save = f'{base_dir}/nxro_rodiag{variant_suffix}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_rodiag{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': rd_model.state_dict(), 'var_order': rd_var_order}, rd_save)
        print(f"✓ Saved to: {rd_save}")
        # Skills & seasonal sync
        NXRO_rd_fcst = nxro_reforecast(rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device=device)
        acc_NXRO_rd = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_rd = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_rd, rmse_NXRO_rd, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_rodiag{fig_suffix}', label='NXRO-RO+Diag')
        if args.stochastic:
            resid, months = compute_residuals_series(rd_model, train_ds, rd_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_rd_fcst_m = nxro_reforecast_stochastic(rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_rd_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_rodiag_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_rodiag_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': rd_model.state_dict(), 'var_order': rd_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_rodiag_stochastic{extra_tag}.pt')
            NXRO_rd_fcst_m.to_netcdf(f'{base_dir}/NXRO_rodiag_stochastic_forecasts{extra_tag}.nc')
            NXRO_rd_fcst_m_mean = NXRO_rd_fcst_m.mean('member')
            acc_rd_stoc = calc_forecast_skill(NXRO_rd_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_rd_stoc = calc_forecast_skill(NXRO_rd_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_rd_stoc, rmse_rd_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_rodiag_stochastic{fig_suffix}', label='NXRO-RO+Diag (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_rd_fcst, NXRO_rd_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_rodiag_plume', fig_suffix=fig_suffix)
        rd_sim_ds = simulate_nxro_longrun(rd_model, X0_ds=train_ds, var_order=rd_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rd_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_rodiag_seasonal_synchronization{fig_suffix}.png')

    def run_res():
        base_dir = 'results/res'
        ensure_dir(base_dir)
        
        # Load warm-start parameters if provided (only needs linear for variant 4/4a)
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=False, include_diag=False) if args.warm_start else None
        
        # Filter freeze_flags to only include linear (no ro, no diag for this model)
        freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
        
        rs_model, rs_var_order, rs_best_rmse, rs_history = train_nxro_res(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc,
            warmstart_init_dict=warmstart_params,
            freeze_flags=freeze_flags_filtered,
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rs_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rs_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Res training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_res_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        rs_best_epoch = int(np.argmin(rs_history['test_rmse'])) + 1
        rs_save = f'{base_dir}/nxro_res{variant_suffix}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_res{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': rs_model.state_dict(), 'var_order': rs_var_order}, rs_save)
        print(f"✓ Saved to: {rs_save}")
        # Skills & seasonal sync
        NXRO_rs_fcst = nxro_reforecast(rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device=device)
        acc_NXRO_rs = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_rs = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_rs, rmse_NXRO_rs, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_res{fig_suffix}', label='NXRO-Res')
        if args.stochastic:
            resid, months = compute_residuals_series(rs_model, train_ds, rs_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_rs_fcst_m = nxro_reforecast_stochastic(rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_rs_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_res_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_res_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': rs_model.state_dict(), 'var_order': rs_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_res_stochastic{extra_tag}.pt')
            NXRO_rs_fcst_m.to_netcdf(f'{base_dir}/NXRO_res_stochastic_forecasts{extra_tag}.nc')
            NXRO_rs_fcst_m_mean = NXRO_rs_fcst_m.mean('member')
            acc_res_stoc = calc_forecast_skill(NXRO_rs_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_res_stoc = calc_forecast_skill(NXRO_rs_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                                by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_res_stoc, rmse_res_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_res_stochastic{fig_suffix}', label='NXRO-Res (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_rs_fcst, NXRO_rs_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_res_plume', fig_suffix=fig_suffix)
        rs_sim_ds = simulate_nxro_longrun(rs_model, X0_ds=train_ds, var_order=rs_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rs_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_res_seasonal_synchronization{fig_suffix}.png')

    def run_res_fullxro():
        """Run variant 4b: Frozen full XRO + trainable residual MLP."""
        assert args.warm_start is not None, "Variant 4b (res_fullxro) requires --warm_start argument!"
        
        base_dir = 'results/res_fullxro'
        ensure_dir(base_dir)
        
        # Load XRO fit and extract ALL components (no renaming to _init suffix)
        xro_init = load_xro_init(args.warm_start, k_max=args.k_max, include_ro=True, include_diag=True)
        # Remove _init suffix for res_fullxro (it expects L_basis, not L_basis_init)
        xro_init_dict = {k.replace('_init', ''): v for k, v in xro_init.items()}
        
        rs_fullxro_model, rs_fullxro_var_order, rs_fullxro_best_rmse, rs_fullxro_history = train_nxro_res_fullxro(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc,
            xro_init_dict=xro_init_dict
        )
        
        # Training curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rs_fullxro_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rs_fullxro_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Res-FullXRO training (Variant 4b)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_res_fullxro_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        # Save model
        rs_fullxro_save = f'{base_dir}/nxro_res_fullxro_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_res_fullxro_best{extra_tag}.pt'
        torch.save({'state_dict': rs_fullxro_model.state_dict(), 'var_order': rs_fullxro_var_order}, rs_fullxro_save)
        
        # Forecast skill
        NXRO_rs_fullxro_fcst = nxro_reforecast(rs_fullxro_model, init_ds=obs_ds, n_month=21, var_order=rs_fullxro_var_order, device=device)
        acc_NXRO_rs_fullxro = calc_forecast_skill(NXRO_rs_fullxro_fcst, obs_ds, metric='acc', is_mv3=True,
                                                  by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_rs_fullxro = calc_forecast_skill(NXRO_rs_fullxro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                                   by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_rs_fullxro, rmse_NXRO_rs_fullxro, sel_var='Nino34', 
                         out_prefix=f'{base_dir}/NXRO_res_fullxro{fig_suffix}', label='NXRO-Res-FullXRO')
        
        # Seasonal synchronization
        rs_fullxro_sim_ds = simulate_nxro_longrun(rs_fullxro_model, X0_ds=train_ds, var_order=rs_fullxro_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rs_fullxro_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_res_fullxro_seasonal_synchronization{fig_suffix}.png')
        
        print(f"✓ Variant 4b (res_fullxro) complete: {rs_fullxro_save}")

    def run_neural():
        base_dir = 'results/neural'
        ensure_dir(base_dir)
        nn_model, nn_var_order, nn_best_rmse, nn_history = train_nxro_neural(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only', device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(nn_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(nn_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-NeuralODE training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_neural_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        nn_best_epoch = int(np.argmin(nn_history['test_rmse'])) + 1
        nn_save = f'{base_dir}/nxro_neural_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_neural_best{extra_tag}.pt'
        torch.save({'state_dict': nn_model.state_dict(), 'var_order': nn_var_order}, nn_save)
        # Skills & seasonal sync
        NXRO_nn_fcst = nxro_reforecast(nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device=device)
        acc_NXRO_nn = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_nn = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_nn, rmse_NXRO_nn, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_neural{fig_suffix}', label='NXRO-NeuralODE')
        if args.stochastic:
            resid, months = compute_residuals_series(nn_model, train_ds, nn_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_nn_fcst_m = nxro_reforecast_stochastic(nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_nn_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_neural_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_neural_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': nn_model.state_dict(), 'var_order': nn_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_neural_stochastic{extra_tag}.pt')
            NXRO_nn_fcst_m.to_netcdf(f'{base_dir}/NXRO_neural_stochastic_forecasts{extra_tag}.nc')
            NXRO_nn_fcst_m_mean = NXRO_nn_fcst_m.mean('member')
            acc_nn_stoc = calc_forecast_skill(NXRO_nn_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_nn_stoc = calc_forecast_skill(NXRO_nn_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_nn_stoc, rmse_nn_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_neural_stochastic{fig_suffix}', label='NXRO-NeuralODE (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_nn_fcst, NXRO_nn_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_neural_plume', fig_suffix=fig_suffix)
        nn_sim_ds = simulate_nxro_longrun(nn_model, X0_ds=train_ds, var_order=nn_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, nn_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_neural_seasonal_synchronization{extra_tag}.png')

    def run_neural_phys():
        base_dir = 'results/neural_phys'
        ensure_dir(base_dir)
        np_model, np_var_order, np_best_rmse, np_history = train_nxro_neural_phys(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only',
            jac_reg=args.jac_reg, div_reg=args.div_reg, noise_std=args.noise_std, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(np_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(np_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-NeuralODE (PhysReg) training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_neural_phys_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        np_best_epoch = int(np.argmin(np_history['test_rmse'])) + 1
        np_save = f'{base_dir}/nxro_neural_phys_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_neural_phys_best{extra_tag}.pt'
        torch.save({'state_dict': np_model.state_dict(), 'var_order': np_var_order}, np_save)
        # Skills & seasonal sync
        NXRO_np_fcst = nxro_reforecast(np_model, init_ds=obs_ds, n_month=21, var_order=np_var_order, device=device)
        acc_NXRO_np = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_np = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_np, rmse_NXRO_np, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_neural_phys{fig_suffix}', label='NXRO-NeuralODE (PhysReg)')
        np_sim_ds = simulate_nxro_longrun(np_model, X0_ds=train_ds, var_order=np_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, np_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_neural_phys_seasonal_synchronization{extra_tag}.png')
        if args.stochastic:
            resid, months = compute_residuals_series(np_model, train_ds, np_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_np_fcst_m = nxro_reforecast_stochastic(np_model, init_ds=obs_ds, n_month=21, var_order=np_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_np_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_neural_phys_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_neural_phys_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': np_model.state_dict(), 'var_order': np_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_neural_phys_stochastic{extra_tag}.pt')
            NXRO_np_fcst_m.to_netcdf(f'{base_dir}/NXRO_neural_phys_stochastic_forecasts{extra_tag}.nc')
            NXRO_np_fcst_m_mean = NXRO_np_fcst_m.mean('member')
            acc_np_stoc = calc_forecast_skill(NXRO_np_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_np_stoc = calc_forecast_skill(NXRO_np_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_np_stoc, rmse_np_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_neural_phys_stochastic{fig_suffix}', label='NXRO-NeuralODE (PhysReg, stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_np_fcst, NXRO_np_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_neural_phys_plume', fig_suffix=fig_suffix)

    def run_resmix():
        base_dir = 'results/resmix'
        ensure_dir(base_dir)
        
        # Load warm-start parameters if provided (needs all: L, RO, Diag for variant 5d)
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=True, include_diag=True) if args.warm_start else None
        
        rx_model, rx_var_order, rx_best_rmse, rx_history = train_nxro_resmix(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            hidden=64, alpha_init=args.alpha_init, alpha_learnable=args.alpha_learnable,
            alpha_max=args.alpha_max, res_reg=args.res_reg, device=device,
            extra_train_nc_paths=args.extra_train_nc,
            warmstart_init_dict=warmstart_params,
            freeze_flags=freeze_flags,
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rx_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rx_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-ResidualMix training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_resmix_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        rx_best_epoch = int(np.argmin(rx_history['test_rmse'])) + 1
        rx_save = f'{base_dir}/nxro_resmix{variant_suffix}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_resmix{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': rx_model.state_dict(), 'var_order': rx_var_order}, rx_save)
        print(f"✓ Saved to: {rx_save}")
        # Skills & seasonal sync
        NXRO_rx_fcst = nxro_reforecast(rx_model, init_ds=obs_ds, n_month=21, var_order=rx_var_order, device=device)
        acc_NXRO_rx = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_rx = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_rx, rmse_NXRO_rx, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_resmix{fig_suffix}', label='NXRO-ResidualMix')
        rx_sim_ds = simulate_nxro_longrun(rx_model, X0_ds=train_ds, var_order=rx_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rx_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_resmix_seasonal_synchronization{extra_tag}.png')
        if args.stochastic:
            resid, months = compute_residuals_series(rx_model, train_ds, rx_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_rx_fcst_m = nxro_reforecast_stochastic(rx_model, init_ds=obs_ds, n_month=21, var_order=rx_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_rx_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_resmix_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_resmix_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': rx_model.state_dict(), 'var_order': rx_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_resmix_stochastic{extra_tag}.pt')
            NXRO_rx_fcst_m.to_netcdf(f'{base_dir}/NXRO_resmix_stochastic_forecasts{extra_tag}.nc')
            NXRO_rx_fcst_m_mean = NXRO_rx_fcst_m.mean('member')
            acc_rx_stoc = calc_forecast_skill(NXRO_rx_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_rx_stoc = calc_forecast_skill(NXRO_rx_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_rx_stoc, rmse_rx_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_resmix_stochastic{extra_tag}', label='NXRO-ResidualMix (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_rx_fcst, NXRO_rx_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_resmix_plume', fig_suffix=fig_suffix)

    def run_bilinear():
        base_dir = 'results/bilinear'
        ensure_dir(base_dir)
        bl_model, bl_var_order, bl_best_rmse, bl_history = train_nxro_bilinear(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            n_channels=2, rank=2, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(bl_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(bl_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Bilinear training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_bilinear_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        bl_best_epoch = int(np.argmin(bl_history['test_rmse'])) + 1
        bl_save = f'{base_dir}/nxro_bilinear_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_bilinear_best{extra_tag}.pt'
        torch.save({'state_dict': bl_model.state_dict(), 'var_order': bl_var_order}, bl_save)
        # Skills & seasonal sync
        NXRO_bl_fcst = nxro_reforecast(bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device=device)
        acc_NXRO_bl = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_bl = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_bl, rmse_NXRO_bl, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_bilinear{fig_suffix}', label='NXRO-Bilinear')
        if args.stochastic:
            resid, months = compute_residuals_series(bl_model, train_ds, bl_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_bl_fcst_m = nxro_reforecast_stochastic(bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_bl_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_bilinear_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_bilinear_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': bl_model.state_dict(), 'var_order': bl_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_bilinear_stochastic{extra_tag}.pt')
            NXRO_bl_fcst_m.to_netcdf(f'{base_dir}/NXRO_bilinear_stochastic_forecasts{extra_tag}.nc')
            NXRO_bl_fcst_m_mean = NXRO_bl_fcst_m.mean('member')
            acc_bl_stoc = calc_forecast_skill(NXRO_bl_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_bl_stoc = calc_forecast_skill(NXRO_bl_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_bl_stoc, rmse_bl_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_bilinear_stochastic{extra_tag}', label='NXRO-Bilinear (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_bl_fcst, NXRO_bl_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_bilinear_plume', fig_suffix=fig_suffix)
        bl_sim_ds = simulate_nxro_longrun(bl_model, X0_ds=train_ds, var_order=bl_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, bl_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_bilinear_seasonal_synchronization{extra_tag}.png')

    def run_attentive():
        base_dir = 'results/attentive'
        ensure_dir(base_dir)
        
        # Load warm-start parameters if provided (only needs linear for variant 5a)
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=False, include_diag=False) if args.warm_start else None
        
        # Filter freeze_flags to only include linear (no ro, no diag for this model)
        freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
        
        at_model, at_var_order, at_best_rmse, at_history = train_nxro_attentive(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            d=32, dropout=0.1, mask_mode='th_only', device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc,
            warmstart_init_dict=warmstart_params,
            freeze_flags=freeze_flags_filtered,
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(at_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(at_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-AttentiveCoupling training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_attentive_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        # Save
        at_best_epoch = int(np.argmin(at_history['test_rmse'])) + 1
        at_save = f'{base_dir}/nxro_attentive{variant_suffix}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_attentive{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': at_model.state_dict(), 'var_order': at_var_order}, at_save)
        print(f"✓ Saved to: {at_save}")
        # Skills & seasonal sync
        NXRO_at_fcst = nxro_reforecast(at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device=device)
        acc_NXRO_at = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_at = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_at, rmse_NXRO_at, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_attentive{fig_suffix}', label='NXRO-AttentiveCoupling')
        if args.stochastic:
            resid, months = compute_residuals_series(at_model, train_ds, at_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_at_fcst_m = nxro_reforecast_stochastic(at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_at_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_attentive_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/nxro_attentive_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': at_model.state_dict(), 'var_order': at_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_attentive_stochastic{extra_tag}.pt')
            NXRO_at_fcst_m.to_netcdf(f'{base_dir}/NXRO_attentive_stochastic_forecasts{extra_tag}.nc')
            NXRO_at_fcst_m_mean = NXRO_at_fcst_m.mean('member')
            acc_at_stoc = calc_forecast_skill(NXRO_at_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_at_stoc = calc_forecast_skill(NXRO_at_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_at_stoc, rmse_at_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_attentive_stochastic{extra_tag}', label='NXRO-AttentiveCoupling (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_at_fcst, NXRO_at_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_attentive_plume', fig_suffix=fig_suffix)
        at_sim_ds = simulate_nxro_longrun(at_model, X0_ds=train_ds, var_order=at_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, at_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_attentive_seasonal_synchronization{extra_tag}.png')

    def run_graph():
        # Load warm-start parameters if provided (only needs linear for variant 5b)
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=False, include_diag=False) if args.warm_start else None
        
        # Filter freeze_flags to only include linear (no ro, no diag for this model)
        freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
        
        gr_model, gr_var_order, gr_best_rmse, gr_history = train_nxro_graph(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            use_fixed_graph=(not args.graph_learned),
            learned_l1_lambda=args.graph_l1,
            stat_knn_method=args.graph_stat_method,
            stat_knn_top_k=args.graph_stat_topk,
            stat_knn_source=args.graph_stat_source,
            warmstart_init_dict=warmstart_params,
            freeze_flags=freeze_flags_filtered,
            device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc
        )
        # Build graph tag for filenames
        graph_kind = f"stat_{args.graph_stat_method}_k{args.graph_stat_topk}" if args.graph_stat_method else "xro"
        graph_mode = "learned" if args.graph_learned else "fixed"
        l1_tag = f"_l1{args.graph_l1}" if args.graph_learned and args.graph_l1 > 0 else ""
        graph_tag = f"_{graph_mode}_{graph_kind}{l1_tag}"
        base_dir = f"results/graph/{graph_mode}_{graph_kind}{l1_tag}"
        ensure_dir(base_dir)
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(gr_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(gr_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'NXRO-Graph training ({graph_mode}, {graph_kind})')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_graph{graph_tag}{fig_suffix}_training_curves.png', dpi=300)
        plt.close()
        # Save
        gr_best_epoch = int(np.argmin(gr_history['test_rmse'])) + 1
        gr_save = f'{base_dir}/nxro_graph{graph_tag}{variant_suffix}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_graph{graph_tag}{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': gr_model.state_dict(), 'var_order': gr_var_order}, gr_save)
        print(f"✓ Saved to: {gr_save}")
        # Skills & seasonal sync
        NXRO_gr_fcst = nxro_reforecast(gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device=device)
        acc_NXRO_gr = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_gr = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_gr, rmse_NXRO_gr, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_graph{graph_tag}{fig_suffix}', label='NXRO-Graph')
        if args.stochastic:
            resid, months = compute_residuals_series(gr_model, train_ds, gr_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_gr_fcst_m = nxro_reforecast_stochastic(gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order,
                                                         noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_gr_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_graph{graph_tag}_stochastic_eval{extra_tag}')
            np.savez(f'{base_dir}/NXRO_graph{graph_tag}_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': gr_model.state_dict(), 'var_order': gr_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/NXRO_graph{graph_tag}_stochastic{extra_tag}.pt')
            NXRO_gr_fcst_m.to_netcdf(f'{base_dir}/NXRO_graph{graph_tag}_stochastic_forecasts{extra_tag}.nc')
            NXRO_gr_fcst_m_mean = NXRO_gr_fcst_m.mean('member')
            acc_gr_stoc = calc_forecast_skill(NXRO_gr_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_gr_stoc = calc_forecast_skill(NXRO_gr_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_gr_stoc, rmse_gr_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_graph{graph_tag}_stochastic{fig_suffix}', label='NXRO-Graph (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_gr_fcst, NXRO_gr_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_graph{graph_tag}_plume', fig_suffix=fig_suffix)
        gr_sim_ds = simulate_nxro_longrun(gr_model, X0_ds=train_ds, var_order=gr_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, gr_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_graph{graph_tag}_seasonal_synchronization{fig_suffix}.png')

    def run_graph_pyg():
        gp_model, gp_var_order, gp_best_rmse, gp_history = train_nxro_graph_pyg(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            top_k=args.top_k, hidden=16, dropout=0.1, use_gat=args.gat, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc
        )
        tag2 = 'gat' if args.gat else 'gcn'
        ktag = f"k{args.top_k}"
        base_dir = f'results/graphpyg/{tag2}_{ktag}'
        ensure_dir(base_dir)
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(gp_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(gp_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'NXRO-GraphPyG ({tag2}, {ktag}) training')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_training_curves.png', dpi=300)
        plt.close()
        # Save
        gp_best_epoch = int(np.argmin(gp_history['test_rmse'])) + 1
        gp_save = f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}_best_test{extra_tag}.pt' if args.test else f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}_best{extra_tag}.pt'
        torch.save({'state_dict': gp_model.state_dict(), 'var_order': gp_var_order}, gp_save)
        # Skills & seasonal sync
        NXRO_gp_fcst = nxro_reforecast(gp_model, init_ds=obs_ds, n_month=21, var_order=gp_var_order, device=device)
        acc_NXRO_gp = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_gp = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_gp, rmse_NXRO_gp, sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}', label=f'NXRO-GraphPyG ({tag2.upper()})')
        if args.stochastic:
            resid, months = compute_residuals_series(gp_model, train_ds, gp_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_gp_fcst_m = nxro_reforecast_stochastic(gp_model, init_ds=obs_ds, n_month=21, var_order=gp_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            evaluate_stochastic_ensemble(NXRO_gp_fcst_m, obs_ds, var='Nino34', out_prefix=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{extra_tag}_stochastic_eval')
            np.savez(f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}{extra_tag}_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': gp_model.state_dict(), 'var_order': gp_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}{extra_tag}_stochastic.pt')
            NXRO_gp_fcst_m.to_netcdf(f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{extra_tag}_stochastic_forecasts.nc')
            NXRO_gp_fcst_m_mean = NXRO_gp_fcst_m.mean('member')
            acc_gp_stoc = calc_forecast_skill(NXRO_gp_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_gp_stoc = calc_forecast_skill(NXRO_gp_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_gp_stoc, rmse_gp_stoc, sel_var='Nino34',
                              out_prefix=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_stochastic', label=f'NXRO-GraphPyG ({tag2.upper()} stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_gp_fcst, NXRO_gp_fcst_m, obs_ds, init_dates, fname_prefix=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}_plume', fig_suffix=fig_suffix)
        gp_sim_ds = simulate_nxro_longrun(gp_model, X0_ds=train_ds, var_order=gp_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, gp_sim_ds, sel_var='Nino34', out_path=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{extra_tag}_seasonal_synchronization.png')

    if args.model in ('linear', 'all'):
        run_linear()
    if args.model in ('ro', 'all'):
        run_ro()
    if args.model in ('rodiag', 'all'):
        run_rodiag()
    if args.model in ('res', 'all'):
        run_res()
    if args.model == 'res_fullxro':
        run_res_fullxro()
    if args.model in ('neural', 'all'):
        run_neural()
    if args.model in ('neural_phys', 'all'):
        run_neural_phys()
    if args.model in ('resmix', 'all'):
        run_resmix()
    if args.model in ('bilinear', 'all'):
        run_bilinear()
    if args.model in ('attentive', 'all'):
        run_attentive()
    if args.model in ('graph', 'all'):
        run_graph()
    if args.model in ('graph_pyg', 'all'):
        run_graph_pyg()


if __name__ == '__main__':
    main()


