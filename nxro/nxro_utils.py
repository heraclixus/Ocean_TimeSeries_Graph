"""
NXRO Utility Functions

Shared helper functions for NXRO training, evaluation, and plotting.
Used by both NXRO_train.py and NXRO_train_out_of_sample.py.
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from utils.xro_utils import calc_forecast_skill


def _uses_history_interface(model) -> bool:
    return bool(hasattr(model, 'n_lags'))


def simulate_nxro_longrun(model, X0_ds: xr.Dataset, var_order: list, nyear: int = 100, device: str = 'cpu') -> xr.Dataset:
    """Deterministic long-run simulation for seasonal stddev plots."""
    ncycle = 12
    dt = 1.0 / ncycle
    start_time = pd.to_datetime(str(X0_ds.time.values[0]))
    n_months = nyear * ncycle
    time_index = pd.date_range(start=start_time, periods=n_months, freq='MS')
    years = np.asarray(time_index.year + (time_index.month - 1) / 12.0, dtype=np.float32)
    memory_depth = int(getattr(model, 'memory_depth', 0) or 0)
    uses_history = _uses_history_interface(model)

    X_all = np.stack([X0_ds[v].values for v in var_order], axis=-1).astype(np.float32)
    if memory_depth >= X_all.shape[0]:
        raise ValueError(
            f"Not enough history for memory_depth={memory_depth} with only {X_all.shape[0]} timesteps."
        )

    if not uses_history:
        x = torch.from_numpy(X_all[0:1]).to(device)
        n_vars = x.shape[1]
    else:
        x_hist = torch.from_numpy(X_all[:memory_depth + 1][None, ...]).to(device)
        n_vars = x_hist.shape[-1]
    out = np.zeros((n_months, n_vars), dtype=np.float32)
    out[0] = (x if not uses_history else x_hist[:, -1, :]).squeeze(0).cpu().numpy()

    model.eval()
    with torch.no_grad():
        if uses_history:
            lag_offsets = (
                torch.arange(memory_depth, -1, -1, dtype=torch.float32, device=device)
                .unsqueeze(0) * dt
            )
        for t in range(1, n_months):
            if not uses_history:
                t_year = torch.tensor([float(years[t - 1])], dtype=torch.float32, device=device)
                dxdt = model(x, t_year)
                x = x + dxdt * dt
                out[t] = x.squeeze(0).detach().cpu().numpy()
            else:
                current_time = torch.tensor([[float(years[t - 1])]], dtype=torch.float32, device=device)
                t_hist = current_time - lag_offsets
                dxdt = model(x_hist, t_hist)
                x_new = x_hist[:, -1, :] + dxdt * dt
                x_hist = torch.cat([x_hist[:, 1:, :], x_new.unsqueeze(1)], dim=1)
                out[t] = x_new.squeeze(0).detach().cpu().numpy()

    ds = xr.Dataset({var: (['time'], out[:, i]) for i, var in enumerate(var_order)}, coords={'time': time_index})
    return ds


def plot_seasonal_sync(train_ds: xr.Dataset, sim_ds: xr.Dataset, sel_var: str, out_path: str, model_label: str = 'NXRO') -> None:
    """Plot seasonal synchronization comparison."""
    stddev_obs = train_ds.groupby('time.month').std('time')
    stddev_sim = sim_ds.groupby('time.month').std('time')

    plt.plot(stddev_obs.month, stddev_obs[sel_var], c='black', label='ORAS5')
    plt.plot(stddev_sim.month, stddev_sim[sel_var], c='green', label=model_label)
    plt.legend()
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel(f'{sel_var} seasonal standard deviation (C)')
    plt.xlabel('Calendar Month')
    plt.title('Seasonal synchronization')
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_observed_nino34(obs_ds: xr.Dataset, out_path: str, train_end: str = None, test_start: str = None) -> None:
    """Plot observed Nino3.4 with optional train/test split."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    obs_ds['Nino34'].plot(ax=ax, c='black', label='Observed')
    
    if train_end and test_start:
        train_end_date = pd.to_datetime(train_end)
        test_start_date = pd.to_datetime(test_start)
        ax.axvline(train_end_date, color='red', linestyle='--', alpha=0.7, label=f'Train end ({train_end})')
        ax.axvline(test_start_date, color='blue', linestyle='--', alpha=0.7, label=f'Test start ({test_start})')
    
    ax.set_title('Observed Nino3.4 SSTA')
    ax.legend()
    plt.savefig(out_path, dpi=300)
    plt.close()


def pick_sample_inits(ds: xr.Dataset, n: int = 3) -> list:
    """Pick sample initialization dates from dataset."""
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


def load_all_eval_datasets(exclude_vars=None):
    """Load all available evaluation datasets.
    
    Args:
        exclude_vars: Optional list of variable names to exclude (e.g., ['WWV']).
    """
    datasets = {}
    
    # Primary dataset (always present)
    ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    if exclude_vars:
        for var in exclude_vars:
            if var in ds.data_vars:
                ds = ds.drop_vars(var)
    datasets['ORAS5'] = ds
    
    # Find all preprocessed datasets
    all_nc_files = glob.glob('data/XRO_indices_*_preproc.nc')
    for nc_file in all_nc_files:
        basename = os.path.basename(nc_file)
        dataset_name = basename.replace('XRO_indices_', '').replace('_preproc.nc', '').upper()
        if dataset_name and dataset_name != 'ORAS5':
            try:
                ds = xr.open_dataset(nc_file)
                if exclude_vars:
                    for var in exclude_vars:
                        if var in ds.data_vars:
                            ds = ds.drop_vars(var)
                datasets[dataset_name] = ds
                print(f"  Loaded {dataset_name}: {nc_file}")
            except Exception as e:
                print(f"  Warning: Could not load {nc_file}: {e}")
    
    return datasets


def evaluate_on_all_datasets(fcst, datasets, eval_period, sel_var='Nino34'):
    """Evaluate forecast on all datasets, return aggregated metrics."""
    results = {}
    
    for ds_name, obs_ds in datasets.items():
        try:
            acc = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=eval_period)
            rmse = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=eval_period)
        except Exception as e:
            print(f"    Warning: Could not evaluate on {ds_name}: {e}")
            continue
        
        results[ds_name] = {
            'acc': acc,
            'rmse': rmse,
        }
        
        mean_acc = float(np.nanmean(acc[sel_var].values))
        mean_rmse = float(np.nanmean(rmse[sel_var].values))
        print(f"    {ds_name}: ACC={mean_acc:.3f}, RMSE={mean_rmse:.3f}")
    
    return results


def evaluate_on_all_datasets_dual(fcst, datasets, train_period, test_period, sel_var='Nino34'):
    """Evaluate forecast on all datasets with both train and test periods."""
    results = {}
    
    for ds_name, obs_ds in datasets.items():
        try:
            acc_train = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        except Exception as e:
            print(f"    Warning: Could not evaluate on {ds_name}: {e}")
            continue
        
        results[ds_name] = {
            'acc_train': acc_train,
            'rmse_train': rmse_train,
            'acc_test': acc_test,
            'rmse_test': rmse_test,
        }
        
        mean_acc_train = float(np.nanmean(acc_train[sel_var].values))
        mean_rmse_train = float(np.nanmean(rmse_train[sel_var].values))
        mean_acc_test = float(np.nanmean(acc_test[sel_var].values))
        mean_rmse_test = float(np.nanmean(rmse_test[sel_var].values))
        print(f"    {ds_name}: Train ACC={mean_acc_train:.3f}, RMSE={mean_rmse_train:.3f} | "
              f"Test ACC={mean_acc_test:.3f}, RMSE={mean_rmse_test:.3f}")
    
    return results


def plot_skill_curves_multi_dataset(all_results, sel_var, out_prefix, label):
    """Plot skills across multiple datasets."""
    if len(all_results) <= 1:
        return
    
    # Check if results have train/test split
    has_dual = 'acc_train' in list(all_results.values())[0]
    
    if has_dual:
        # Dual plot (train and test)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        for ds_name, results in all_results.items():
            results['acc_train'][sel_var].plot(ax=axes[0], label=f'{ds_name}', marker='o', markersize=3)
        axes[0].set_ylabel('Correlation', fontsize=11)
        axes[0].set_xlabel('Forecast lead (months)', fontsize=11)
        axes[0].set_title(f'{label}: ACC (Train Period)', fontsize=12)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([1., 21])
        axes[0].set_ylim([0.2, 1.])
        
        for ds_name, results in all_results.items():
            results['acc_test'][sel_var].plot(ax=axes[1], label=f'{ds_name}', marker='s', markersize=3, linestyle='--')
        axes[1].set_ylabel('Correlation', fontsize=11)
        axes[1].set_xlabel('Forecast lead (months)', fontsize=11)
        axes[1].set_title(f'{label}: ACC (Test Period)', fontsize=12)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([1., 21])
        axes[1].set_ylim([0.2, 1.])
        
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_acc_multi_dataset.png', dpi=300)
        plt.close()
        
        # RMSE
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        for ds_name, results in all_results.items():
            results['rmse_train'][sel_var].plot(ax=axes[0], label=f'{ds_name}', marker='o', markersize=3)
        axes[0].set_ylabel('RMSE (C)', fontsize=11)
        axes[0].set_xlabel('Forecast lead (months)', fontsize=11)
        axes[0].set_title(f'{label}: RMSE (Train Period)', fontsize=12)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([1., 21])
        axes[0].set_ylim([0., 1.])
        
        for ds_name, results in all_results.items():
            results['rmse_test'][sel_var].plot(ax=axes[1], label=f'{ds_name}', marker='s', markersize=3, linestyle='--')
        axes[1].set_ylabel('RMSE (C)', fontsize=11)
        axes[1].set_xlabel('Forecast lead (months)', fontsize=11)
        axes[1].set_title(f'{label}: RMSE (Test Period)', fontsize=12)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([1., 21])
        axes[1].set_ylim([0., 1.])
        
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_rmse_multi_dataset.png', dpi=300)
        plt.close()
    else:
        # Single period plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        for ds_name, results in all_results.items():
            results['acc'][sel_var].plot(ax=ax, label=f'{ds_name}', marker='o', markersize=3)
        
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'{label}: ACC (Multi-Dataset)', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1., 21])
        ax.set_ylim([0.2, 1.])
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_acc_multi_dataset.png', dpi=300)
        plt.close()
        
        # RMSE
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        for ds_name, results in all_results.items():
            results['rmse'][sel_var].plot(ax=ax, label=f'{ds_name}', marker='o', markersize=3)
        
        ax.set_ylabel('RMSE (C)', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'{label}: RMSE (Multi-Dataset)', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1., 21])
        ax.set_ylim([0., 1.])
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_rmse_multi_dataset.png', dpi=300)
        plt.close()


def plot_skill_curves(acc_ds: xr.Dataset, rmse_ds: xr.Dataset, sel_var: str, out_prefix: str, label: str) -> None:
    """Plot single ACC and RMSE curves."""
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
    ax.set_ylabel('RMSE (C)')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0., 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend()
    plt.savefig(f'{out_prefix}_rmse.png', dpi=300)
    plt.close()


def plot_skill_curves_dual(acc_train: xr.Dataset, rmse_train: xr.Dataset, 
                           acc_test: xr.Dataset, rmse_test: xr.Dataset,
                           sel_var: str, out_prefix: str, label: str) -> None:
    """Plot ACC and RMSE with both in-sample and out-of-sample periods."""
    
    # ACC plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    acc_train[sel_var].plot(ax=ax, label=f'{label} (in-sample)', c='green', lw=2.5, marker='o', markersize=4)
    acc_test[sel_var].plot(ax=ax, label=f'{label} (out-of-sample)', c='orange', lw=2.5, marker='s', markersize=4, linestyle='--')
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0.2, 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{label}: ACC (In-Sample vs Out-of-Sample)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_acc_dual.png', dpi=300)
    plt.close()

    # RMSE plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rmse_train[sel_var].plot(ax=ax, label=f'{label} (in-sample)', c='green', lw=2.5, marker='o', markersize=4)
    rmse_test[sel_var].plot(ax=ax, label=f'{label} (out-of-sample)', c='orange', lw=2.5, marker='s', markersize=4, linestyle='--')
    ax.set_ylabel('RMSE (C)', fontsize=11)
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0., 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{label}: RMSE (In-Sample vs Out-of-Sample)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_rmse_dual.png', dpi=300)
    plt.close()


def generate_stochastic_ensemble(model, model_fcst, train_ds, obs_ds, var_order, 
                                 train_period, test_period, args, base_dir, 
                                 model_name, extra_tag='', fig_suffix='', device='cpu'):
    """
    Generate stochastic ensemble forecasts with optional Stage 2 or simulation-based noise.
    
    Supports 4 noise variants:
    1. Post-hoc (default)
    2. Stage 2 (--train_noise_stage2)
    3. Simulation-based (--use_sim_noise)
    4. Combined (--use_sim_noise --train_noise_stage2)
    
    Args:
        model: Trained deterministic model
        model_fcst: Deterministic forecast
        train_ds: Training dataset
        obs_ds: Full observation dataset
        var_order: Variable order
        train_period: Training period slice
        test_period: Test period slice
        args: Argparse namespace with flags
        base_dir: Output directory
        model_name: Model name for files (e.g., 'nxro_res')
        extra_tag: Extra tag for filenames
        fig_suffix: Figure suffix
        device: Device
    """
    from nxro.stochastic import (compute_residuals_series, fit_seasonal_ar1_from_residuals,
                                 SeasonalAR1Noise, nxro_reforecast_stochastic,
                                 train_noise_stage2, fit_noise_from_simulations)
    from utils.xro_utils import plot_forecast_plume, evaluate_stochastic_ensemble
    
    print("  Generating stochastic ensemble forecasts...")
    
    # Determine noise source
    if args.use_sim_noise:
        print("  Fitting noise from simulation-observation differences...")
        # Prefer CESM2-LENS data if available, otherwise fall back to old XRO_indices files
        from nxro.data import discover_cesm2_climate_mode_files
        cesm2_paths = discover_cesm2_climate_mode_files()
        if cesm2_paths:
            print(f"  Using CESM2-LENS data ({len(cesm2_paths)} ensemble members)")
            sim_paths = cesm2_paths
        else:
            print("  CESM2-LENS data not found, falling back to XRO_indices_*_preproc.nc")
            sim_paths = glob.glob('data/XRO_indices_*_preproc.nc')
        a1_np, sigma_np = fit_noise_from_simulations(args.nc_path, sim_paths, var_order, train_period)
        noise_suffix = '_sim_noise'
    else:
        resid, months = compute_residuals_series(model, train_ds, var_order, device=device)
        a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
        noise_suffix = ''
    
    # Stage 2 optimization
    stage2_suffix = '_stage2' if args.train_noise_stage2 else ''
    combined_suffix = noise_suffix + stage2_suffix
    
    if args.train_noise_stage2:
        print("  Running Stage 2 noise optimization...")
        a1_np, sigma_np = train_noise_stage2(model, train_ds, var_order, 
                                             a1_np, sigma_np, 
                                             n_epochs=100, lr=1e-3, 
                                             device=device, verbose=True)
    
    # Generate ensemble
    a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    noise = SeasonalAR1Noise(a1, sigma)
    
    fcst_m = nxro_reforecast_stochastic(model, init_ds=obs_ds, n_month=21, var_order=var_order,
                                        noise_model=noise, n_members=args.members, device=device)
    
    # Save artifacts
    np.savez(f'{base_dir}/{model_name}_stochastic{combined_suffix}_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
    torch.save({'state_dict': model.state_dict(), 'var_order': var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
              f'{base_dir}/{model_name}_stochastic{combined_suffix}{extra_tag}.pt')
    fcst_m.to_netcdf(f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}_forecasts{extra_tag}.nc')
    
    # Evaluate
    evaluate_stochastic_ensemble(fcst_m, obs_ds, var='Nino34', 
                                out_prefix=f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}_eval{extra_tag}')
    
    # Plume plots
    init_dates = pick_sample_inits(obs_ds, n=3)
    if len(init_dates) > 0:
        plot_forecast_plume(model_fcst, fcst_m, obs_ds, init_dates, 
                           fname_prefix=f'{base_dir}/{model_name.upper()}_plume', fig_suffix=fig_suffix)
    
    # Ensemble-mean skills
    fcst_m_mean = fcst_m.mean('member')
    
    if hasattr(args, 'eval_all_datasets') and args.eval_all_datasets and hasattr(args, 'all_eval_datasets'):
        all_results_stoc = evaluate_on_all_datasets_dual(fcst_m_mean, args.all_eval_datasets, train_period, test_period)
        acc_train = all_results_stoc['ORAS5']['acc_train']
        rmse_train = all_results_stoc['ORAS5']['rmse_train']
        acc_test = all_results_stoc['ORAS5']['acc_test']
        rmse_test = all_results_stoc['ORAS5']['rmse_test']
        plot_skill_curves_multi_dataset(all_results_stoc, 'Nino34', 
                                       f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}{fig_suffix}', 
                                       f'{model_name.upper()} (stochastic mean)')
    else:
        acc_train = calc_forecast_skill(fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                        by_month=False, verify_periods=train_period)
        rmse_train = calc_forecast_skill(fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                         by_month=False, verify_periods=train_period)
        acc_test = calc_forecast_skill(fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                       by_month=False, verify_periods=test_period)
        rmse_test = calc_forecast_skill(fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                        by_month=False, verify_periods=test_period)
    
    plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                          sel_var='Nino34', out_prefix=f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}{fig_suffix}', 
                          label=f'{model_name.upper()} (stochastic mean)')
