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


def plot_observed_nino34(obs_ds: xr.Dataset, out_path: str, train_end: str, test_start: str) -> None:
    """Plot observed Nino3.4 with train/test split marked."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    obs_ds['Nino34'].plot(ax=ax, c='black', label='Observed')
    
    # Add vertical lines to mark train/test split
    train_end_date = pd.to_datetime(train_end)
    test_start_date = pd.to_datetime(test_start)
    ax.axvline(train_end_date, color='red', linestyle='--', alpha=0.7, label=f'Train end ({train_end})')
    ax.axvline(test_start_date, color='blue', linestyle='--', alpha=0.7, label=f'Test start ({test_start})')
    
    ax.set_title('Observed Nino3.4 SSTA (Out-of-Sample Setup)')
    ax.legend()
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
    """Deterministic long-run simulation for seasonal stddev plots."""
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


def plot_seasonal_sync(train_ds: xr.Dataset, sim_ds: xr.Dataset, sel_var: str, out_path: str, model_label: str = 'NXRO') -> None:
    stddev_obs = train_ds.groupby('time.month').std('time')
    stddev_sim = sim_ds.groupby('time.month').std('time')

    plt.plot(stddev_obs.month, stddev_obs[sel_var], c='black', label='ORAS5 (train)')
    plt.plot(stddev_sim.month, stddev_sim[sel_var], c='green', label=model_label)
    plt.legend()
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel(f'{sel_var} seasonal standard deviation (℃)')
    plt.xlabel('Calendar Month')
    plt.title('Seasonal synchronization')
    plt.savefig(out_path, dpi=300)
    plt.close()


def load_all_eval_datasets():
    """Load all available evaluation datasets."""
    import glob
    datasets = {}
    
    # Primary dataset (always present)
    datasets['ORAS5'] = xr.open_dataset('data/XRO_indices_oras5.nc')
    
    # Find all preprocessed datasets
    all_nc_files = glob.glob('data/XRO_indices_*_preproc.nc')
    for nc_file in all_nc_files:
        basename = os.path.basename(nc_file)
        # Extract dataset name (e.g., 'era5' from 'XRO_indices_era5_preproc.nc')
        dataset_name = basename.replace('XRO_indices_', '').replace('_preproc.nc', '').upper()
        if dataset_name and dataset_name != 'ORAS5':
            try:
                datasets[dataset_name] = xr.open_dataset(nc_file)
                print(f"  Loaded {dataset_name}: {nc_file}")
            except Exception as e:
                print(f"  Warning: Could not load {nc_file}: {e}")
    
    return datasets


def evaluate_on_all_datasets(fcst, datasets, train_period, test_period, sel_var='Nino34'):
    """Evaluate forecast on all datasets, return aggregated metrics."""
    results = {}
    
    for ds_name, obs_ds in datasets.items():
        # In-sample (train period)
        try:
            acc_train = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
        except Exception as e:
            print(f"    Warning: Could not evaluate on {ds_name} train period: {e}")
            continue
        
        # Out-of-sample (test period)
        try:
            acc_test = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        except Exception as e:
            print(f"    Warning: Could not evaluate on {ds_name} test period: {e}")
            continue
        
        results[ds_name] = {
            'acc_train': acc_train,
            'rmse_train': rmse_train,
            'acc_test': acc_test,
            'rmse_test': rmse_test,
        }
        
        # Print summary
        mean_acc_train = float(np.nanmean(acc_train[sel_var].values))
        mean_rmse_train = float(np.nanmean(rmse_train[sel_var].values))
        mean_acc_test = float(np.nanmean(acc_test[sel_var].values))
        mean_rmse_test = float(np.nanmean(rmse_test[sel_var].values))
        print(f"    {ds_name}: Train ACC={mean_acc_train:.3f}, RMSE={mean_rmse_train:.3f} | "
              f"Test ACC={mean_acc_test:.3f}, RMSE={mean_rmse_test:.3f}")
    
    return results


def plot_skill_curves_dual(acc_train: xr.Dataset, rmse_train: xr.Dataset, 
                           acc_test: xr.Dataset, rmse_test: xr.Dataset,
                           sel_var: str, out_prefix: str, label: str) -> None:
    """Plot ACC and RMSE with both in-sample (train) and out-of-sample (test) periods."""
    
    # ACC plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    acc_train[sel_var].plot(ax=ax, label=f'{label} (in-sample: train period)', c='green', lw=2.5, marker='o', markersize=4)
    acc_test[sel_var].plot(ax=ax, label=f'{label} (out-of-sample: test period)', c='orange', lw=2.5, marker='s', markersize=4, linestyle='--')
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
    rmse_train[sel_var].plot(ax=ax, label=f'{label} (in-sample: train period)', c='green', lw=2.5, marker='o', markersize=4)
    rmse_test[sel_var].plot(ax=ax, label=f'{label} (out-of-sample: test period)', c='orange', lw=2.5, marker='s', markersize=4, linestyle='--')
    ax.set_ylabel('RMSE (℃)', fontsize=11)
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


def plot_skill_curves_multi_dataset(all_results, sel_var, out_prefix, label):
    """Plot skills across multiple datasets."""
    if len(all_results) <= 1:
        return
    
    # ACC plot - train vs test for each dataset
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Train period ACC
    for ds_name, results in all_results.items():
        results['acc_train'][sel_var].plot(ax=axes[0], label=f'{ds_name}', marker='o', markersize=3)
    axes[0].set_ylabel('Correlation', fontsize=11)
    axes[0].set_xlabel('Forecast lead (months)', fontsize=11)
    axes[0].set_title(f'{label}: ACC (Train Period)', fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([1., 21])
    axes[0].set_ylim([0.2, 1.])
    
    # Test period ACC
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
    
    # RMSE plot - train vs test for each dataset
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Train period RMSE
    for ds_name, results in all_results.items():
        results['rmse_train'][sel_var].plot(ax=axes[0], label=f'{ds_name}', marker='o', markersize=3)
    axes[0].set_ylabel('RMSE (℃)', fontsize=11)
    axes[0].set_xlabel('Forecast lead (months)', fontsize=11)
    axes[0].set_title(f'{label}: RMSE (Train Period)', fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([1., 21])
    axes[0].set_ylim([0., 1.])
    
    # Test period RMSE
    for ds_name, results in all_results.items():
        results['rmse_test'][sel_var].plot(ax=axes[1], label=f'{ds_name}', marker='s', markersize=3, linestyle='--')
    axes[1].set_ylabel('RMSE (℃)', fontsize=11)
    axes[1].set_xlabel('Forecast lead (months)', fontsize=11)
    axes[1].set_title(f'{label}: RMSE (Test Period)', fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([1., 21])
    axes[1].set_ylim([0., 1.])
    
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_rmse_multi_dataset.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train NXRO models (Out-of-Sample Experiment)')
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'ro', 'rodiag', 'res', 'res_fullxro', 'neural', 'neural_phys', 'resmix', 'bilinear', 'attentive', 'graph', 'graph_pyg', 'all'])
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--gat', action='store_true')
    parser.add_argument('--res_reg', type=float, default=1e-4)
    parser.add_argument('--nc_path', type=str, default='data/XRO_indices_oras5.nc')
    # Out-of-sample defaults: train on 1979-2001, test on 2002-2022
    parser.add_argument('--train_start', type=str, default='1979-01')
    parser.add_argument('--train_end', type=str, default='2001-12')
    parser.add_argument('--test_start', type=str, default='2002-01')
    parser.add_argument('--test_end', type=str, default='2022-12')
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
    parser.add_argument('--extra_train_nc', type=str, nargs='*', default=None,
                        help='Additional NetCDFs for training only (test stays ORAS5).')
    parser.add_argument('--eval_all_datasets', action='store_true',
                        help='Evaluate on all available datasets (ORAS5, ERA5, GODAS, etc.) separately')
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

    # Use out-of-sample results directory
    base_results_dir = 'results_out_of_sample'
    ensure_dir(base_results_dir)
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
            return ''
        
        suffix_parts = []
        
        if args.warm_start is not None:
            if args.freeze is None:
                suffix_parts.append('ws')
            else:
                freeze_components = [c.strip().lower() for c in args.freeze.split(',')]
                
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
                    suffix_parts.append('fix' + '_'.join(sorted(freeze_components)))
        
        return '_' + '_'.join(suffix_parts) if suffix_parts else ''
    
    variant_suffix = get_variant_suffix()

    # Auto-discover helper for *_preproc.nc
    def discover_preprocessed_all(nc_path_base):
        base_dir = os.path.dirname(nc_path_base) or 'data'
        try:
            import glob
            cands = sorted(glob.glob(os.path.join(base_dir, 'XRO_indices_*_preproc.nc')))
            return cands
        except Exception:
            return []

    def resolve_preprocessed_paths(paths):
        if paths is None:
            return None
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
    
    def build_extra_tag(extra_paths):
        if not extra_paths:
            return ''
        return '_extra_data'

    extra_tag = build_extra_tag(args.extra_train_nc)
    fig_suffix = extra_tag

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    obs_ds = xr.open_dataset(args.nc_path)
    train_ds = obs_ds.sel(time=slice(args.train_start, args.train_end))
    test_ds = obs_ds.sel(time=slice(args.test_start, args.test_end))
    
    # Load all eval datasets if requested
    all_eval_datasets = None
    if args.eval_all_datasets:
        print("\nLoading all available datasets for evaluation...")
        all_eval_datasets = load_all_eval_datasets()
        print(f"✓ Loaded {len(all_eval_datasets)} datasets: {list(all_eval_datasets.keys())}\n")

    print("="*80)
    print("OUT-OF-SAMPLE EXPERIMENT SETUP")
    print("="*80)
    print(f"Training period: {args.train_start} to {args.train_end}")
    print(f"Test period: {args.test_start} to {args.test_end}")
    print(f"Results directory: {base_results_dir}/")
    if args.eval_all_datasets:
        print(f"Multi-dataset evaluation: ENABLED ({len(all_eval_datasets)} datasets)")
    print("="*80)

    # Periods for evaluation
    train_period = slice(args.train_start, args.train_end)
    test_period = slice(args.test_start, args.test_end)

    # Observed plot with train/test split marked
    plot_observed_nino34(obs_ds, out_path=f'{base_results_dir}/NXRO_observed_Nino34_out_of_sample.png',
                        train_end=args.train_end, test_start=args.test_start)

    def run_linear():
        base_dir = f'{base_results_dir}/linear'
        ensure_dir(base_dir)
        
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
        ax.plot(history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Linear training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_linear_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        # Save weights
        lin_save = f'{base_dir}/nxro_linear{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': model.state_dict(), 'var_order': var_order}, lin_save)
        print(f"✓ Saved to: {lin_save}")

        # Reforecast for skills on BOTH train and test periods
        NXRO_fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device=device)
        
        # Evaluate on single or multiple datasets
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_fcst, all_eval_datasets, train_period, test_period)
            # Use ORAS5 for primary plots
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            # Generate multi-dataset comparison plots
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_linear{fig_suffix}', 'NXRO-Linear')
        else:
            # In-sample (train period)
            acc_train = calc_forecast_skill(NXRO_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            
            # Out-of-sample (test period)
            acc_test = calc_forecast_skill(NXRO_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        # Plot dual (in-sample vs out-of-sample) for primary dataset
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_linear{fig_suffix}', 
                              label='NXRO-Linear')
        
        # Stochastic ensemble forecasts (optional)
        if args.stochastic:
            print("  Generating stochastic ensemble forecasts...")
            resid, months = compute_residuals_series(model, train_ds, var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            
            NXRO_fcst_m = nxro_reforecast_stochastic(model, init_ds=obs_ds, n_month=21, var_order=var_order,
                                                     noise_model=noise, n_members=args.members, device=device)
            
            # Save stochastic artifacts
            np.savez(f'{base_dir}/nxro_linear_stochastic_noise{extra_tag}.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': model.state_dict(), 'var_order': var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                      f'{base_dir}/nxro_linear_stochastic{extra_tag}.pt')
            NXRO_fcst_m.to_netcdf(f'{base_dir}/NXRO_linear_stochastic_forecasts{extra_tag}.nc')
            
            # Ensemble evaluation
            evaluate_stochastic_ensemble(NXRO_fcst_m, obs_ds, var='Nino34', 
                                        out_prefix=f'{base_dir}/NXRO_linear_stochastic_eval{extra_tag}')
            
            # Plume plots
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_fcst, NXRO_fcst_m, obs_ds, init_dates, 
                                   fname_prefix=f'{base_dir}/NXRO_linear_plume', fig_suffix=fig_suffix)
            
            # Skills on ensemble mean (both train and test)
            NXRO_fcst_m_mean = NXRO_fcst_m.mean('member')
            
            if args.eval_all_datasets and all_eval_datasets:
                all_results_stoc = evaluate_on_all_datasets(NXRO_fcst_m_mean, all_eval_datasets, train_period, test_period)
                acc_train_stoc = all_results_stoc['ORAS5']['acc_train']
                rmse_train_stoc = all_results_stoc['ORAS5']['rmse_train']
                acc_test_stoc = all_results_stoc['ORAS5']['acc_test']
                rmse_test_stoc = all_results_stoc['ORAS5']['rmse_test']
                plot_skill_curves_multi_dataset(all_results_stoc, 'Nino34', 
                                               f'{base_dir}/NXRO_linear_stochastic{fig_suffix}', 
                                               'NXRO-Linear (stochastic mean)')
            else:
                acc_train_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                                     by_month=False, verify_periods=train_period)
                rmse_train_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                                      by_month=False, verify_periods=train_period)
                acc_test_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                                    by_month=False, verify_periods=test_period)
                rmse_test_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                                     by_month=False, verify_periods=test_period)
            
            plot_skill_curves_dual(acc_train_stoc, rmse_train_stoc, acc_test_stoc, rmse_test_stoc,
                                  sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_linear_stochastic{fig_suffix}', 
                                  label='NXRO-Linear (stochastic mean)')
        
        # Seasonal synchronization
        sim_ds = simulate_nxro_longrun(model, X0_ds=train_ds, var_order=var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_linear_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-Linear')
        
        print(f"✓ NXRO-Linear complete (out-of-sample)")

    # Similar functions for other models (ro, rodiag, res, etc.)
    # For brevity, I'll show the pattern for one more model type

    def run_ro():
        base_dir = f'{base_results_dir}/ro'
        ensure_dir(base_dir)
        
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=True, include_diag=False) if args.warm_start else None
        
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
        
        # Training curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(ro_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(ro_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-RO training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_ro_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        # Save
        ro_save = f'{base_dir}/nxro_ro{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': ro_model.state_dict(), 'var_order': ro_var_order}, ro_save)
        print(f"✓ Saved to: {ro_save}")
        
        # Forecast and evaluate on both periods
        NXRO_ro_fcst = nxro_reforecast(ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device=device)
        
        # Evaluate on single or multiple datasets
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_ro_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_ro{fig_suffix}', 'NXRO-RO')
        else:
            acc_train = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            
            acc_test = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_ro{fig_suffix}', 
                              label='NXRO-RO')
        
        ro_sim_ds = simulate_nxro_longrun(ro_model, X0_ds=train_ds, var_order=ro_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, ro_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_ro_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-RO')
        
        print(f"✓ NXRO-RO complete (out-of-sample)")

    def run_rodiag():
        base_dir = f'{base_results_dir}/rodiag'
        ensure_dir(base_dir)
        
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
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rd_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rd_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-RO+Diag training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_rodiag_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        rd_save = f'{base_dir}/nxro_rodiag{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': rd_model.state_dict(), 'var_order': rd_var_order}, rd_save)
        print(f"✓ Saved to: {rd_save}")
        
        NXRO_rd_fcst = nxro_reforecast(rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device=device)
        
        # Evaluate on single or multiple datasets
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_rd_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_rodiag{fig_suffix}', 'NXRO-RO+Diag')
        else:
            acc_train = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            
            acc_test = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_rodiag{fig_suffix}', 
                              label='NXRO-RO+Diag')
        
        rd_sim_ds = simulate_nxro_longrun(rd_model, X0_ds=train_ds, var_order=rd_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rd_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_rodiag_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-RO+Diag')
        
        print(f"✓ NXRO-RO+Diag complete (out-of-sample)")

    def run_res():
        base_dir = f'{base_results_dir}/res'
        ensure_dir(base_dir)
        
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=False, include_diag=False) if args.warm_start else None
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
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rs_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rs_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Res training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_res_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        rs_save = f'{base_dir}/nxro_res{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': rs_model.state_dict(), 'var_order': rs_var_order}, rs_save)
        print(f"✓ Saved to: {rs_save}")
        
        NXRO_rs_fcst = nxro_reforecast(rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_rs_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_res{fig_suffix}', 'NXRO-Res')
        else:
            acc_train = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_res{fig_suffix}', 
                              label='NXRO-Res')
        
        rs_sim_ds = simulate_nxro_longrun(rs_model, X0_ds=train_ds, var_order=rs_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rs_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_res_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-Res')
        
        print(f"✓ NXRO-Res complete (out-of-sample)")

    def run_res_fullxro():
        assert args.warm_start is not None, "Variant res_fullxro requires --warm_start argument!"
        
        base_dir = f'{base_results_dir}/res_fullxro'
        ensure_dir(base_dir)
        
        xro_init = load_xro_init(args.warm_start, k_max=args.k_max, include_ro=True, include_diag=True)
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
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rs_fullxro_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rs_fullxro_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Res-FullXRO training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_res_fullxro_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        rs_fullxro_save = f'{base_dir}/nxro_res_fullxro_best{extra_tag}.pt'
        torch.save({'state_dict': rs_fullxro_model.state_dict(), 'var_order': rs_fullxro_var_order}, rs_fullxro_save)
        print(f"✓ Saved to: {rs_fullxro_save}")
        
        NXRO_rs_fullxro_fcst = nxro_reforecast(rs_fullxro_model, init_ds=obs_ds, n_month=21, var_order=rs_fullxro_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_rs_fullxro_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_res_fullxro{fig_suffix}', 'NXRO-Res-FullXRO')
        else:
            acc_train = calc_forecast_skill(NXRO_rs_fullxro_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_rs_fullxro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_rs_fullxro_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_rs_fullxro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_res_fullxro{fig_suffix}', 
                              label='NXRO-Res-FullXRO')
        
        rs_fullxro_sim_ds = simulate_nxro_longrun(rs_fullxro_model, X0_ds=train_ds, var_order=rs_fullxro_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rs_fullxro_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_res_fullxro_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-Res-FullXRO')
        
        print(f"✓ NXRO-Res-FullXRO complete (out-of-sample)")

    def run_neural():
        base_dir = f'{base_results_dir}/neural'
        ensure_dir(base_dir)
        
        nn_model, nn_var_order, nn_best_rmse, nn_history = train_nxro_neural(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only', device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(nn_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(nn_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-NeuralODE training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_neural_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        nn_save = f'{base_dir}/nxro_neural_best{extra_tag}.pt'
        torch.save({'state_dict': nn_model.state_dict(), 'var_order': nn_var_order}, nn_save)
        print(f"✓ Saved to: {nn_save}")
        
        NXRO_nn_fcst = nxro_reforecast(nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_nn_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_neural{fig_suffix}', 'NXRO-NeuralODE')
        else:
            acc_train = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_neural{fig_suffix}', 
                              label='NXRO-NeuralODE')
        
        nn_sim_ds = simulate_nxro_longrun(nn_model, X0_ds=train_ds, var_order=nn_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, nn_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_neural_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-NeuralODE')
        
        print(f"✓ NXRO-NeuralODE complete (out-of-sample)")

    def run_neural_phys():
        base_dir = f'{base_results_dir}/neural_phys'
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
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(np_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(np_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-NeuralODE (PhysReg) training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_neural_phys_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        np_save = f'{base_dir}/nxro_neural_phys_best{extra_tag}.pt'
        torch.save({'state_dict': np_model.state_dict(), 'var_order': np_var_order}, np_save)
        print(f"✓ Saved to: {np_save}")
        
        NXRO_np_fcst = nxro_reforecast(np_model, init_ds=obs_ds, n_month=21, var_order=np_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_np_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_neural_phys{fig_suffix}', 'NXRO-PhysReg')
        else:
            acc_train = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_neural_phys{fig_suffix}', 
                              label='NXRO-PhysReg')
        
        np_sim_ds = simulate_nxro_longrun(np_model, X0_ds=train_ds, var_order=np_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, np_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_neural_phys_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-PhysReg')
        
        print(f"✓ NXRO-PhysReg complete (out-of-sample)")

    def run_resmix():
        base_dir = f'{base_results_dir}/resmix'
        ensure_dir(base_dir)
        
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
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rx_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rx_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-ResidualMix training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_resmix_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        rx_save = f'{base_dir}/nxro_resmix{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': rx_model.state_dict(), 'var_order': rx_var_order}, rx_save)
        print(f"✓ Saved to: {rx_save}")
        
        NXRO_rx_fcst = nxro_reforecast(rx_model, init_ds=obs_ds, n_month=21, var_order=rx_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_rx_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_resmix{fig_suffix}', 'NXRO-ResidualMix')
        else:
            acc_train = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_resmix{fig_suffix}', 
                              label='NXRO-ResidualMix')
        
        rx_sim_ds = simulate_nxro_longrun(rx_model, X0_ds=train_ds, var_order=rx_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rx_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_resmix_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-ResidualMix')
        
        print(f"✓ NXRO-ResidualMix complete (out-of-sample)")

    def run_bilinear():
        base_dir = f'{base_results_dir}/bilinear'
        ensure_dir(base_dir)
        
        bl_model, bl_var_order, bl_best_rmse, bl_history = train_nxro_bilinear(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            n_channels=2, rank=2, device=device, rollout_k=args.rollout_k,
            extra_train_nc_paths=args.extra_train_nc
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(bl_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(bl_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Bilinear training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_bilinear_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        bl_save = f'{base_dir}/nxro_bilinear_best{extra_tag}.pt'
        torch.save({'state_dict': bl_model.state_dict(), 'var_order': bl_var_order}, bl_save)
        print(f"✓ Saved to: {bl_save}")
        
        NXRO_bl_fcst = nxro_reforecast(bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_bl_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_bilinear{fig_suffix}', 'NXRO-Bilinear')
        else:
            acc_train = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_bilinear{fig_suffix}', 
                              label='NXRO-Bilinear')
        
        bl_sim_ds = simulate_nxro_longrun(bl_model, X0_ds=train_ds, var_order=bl_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, bl_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_bilinear_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-Bilinear')
        
        print(f"✓ NXRO-Bilinear complete (out-of-sample)")

    def run_attentive():
        base_dir = f'{base_results_dir}/attentive'
        ensure_dir(base_dir)
        
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=False, include_diag=False) if args.warm_start else None
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
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(at_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(at_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Attentive training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_attentive_training_curves{fig_suffix}.png', dpi=300)
        plt.close()
        
        at_save = f'{base_dir}/nxro_attentive{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': at_model.state_dict(), 'var_order': at_var_order}, at_save)
        print(f"✓ Saved to: {at_save}")
        
        NXRO_at_fcst = nxro_reforecast(at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_at_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_attentive{fig_suffix}', 'NXRO-Attentive')
        else:
            acc_train = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_attentive{fig_suffix}', 
                              label='NXRO-Attentive')
        
        at_sim_ds = simulate_nxro_longrun(at_model, X0_ds=train_ds, var_order=at_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, at_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_attentive_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-Attentive')
        
        print(f"✓ NXRO-Attentive complete (out-of-sample)")

    def run_graph():
        warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                         include_ro=False, include_diag=False) if args.warm_start else None
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
        
        graph_kind = f"stat_{args.graph_stat_method}_k{args.graph_stat_topk}" if args.graph_stat_method else "xro"
        graph_mode = "learned" if args.graph_learned else "fixed"
        l1_tag = f"_l1{args.graph_l1}" if args.graph_learned and args.graph_l1 > 0 else ""
        graph_tag = f"_{graph_mode}_{graph_kind}{l1_tag}"
        base_dir = f"{base_results_dir}/graph/{graph_mode}_{graph_kind}{l1_tag}"
        ensure_dir(base_dir)
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(gr_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(gr_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'NXRO-Graph ({graph_mode}, {graph_kind}) training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_graph{graph_tag}{fig_suffix}_training_curves.png', dpi=300)
        plt.close()
        
        gr_save = f'{base_dir}/nxro_graph{graph_tag}{variant_suffix}_best{extra_tag}.pt'
        torch.save({'state_dict': gr_model.state_dict(), 'var_order': gr_var_order}, gr_save)
        print(f"✓ Saved to: {gr_save}")
        
        NXRO_gr_fcst = nxro_reforecast(gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_gr_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_graph{graph_tag}{fig_suffix}', 'NXRO-Graph')
        else:
            acc_train = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_graph{graph_tag}{fig_suffix}', 
                              label='NXRO-Graph')
        
        gr_sim_ds = simulate_nxro_longrun(gr_model, X0_ds=train_ds, var_order=gr_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, gr_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_graph{graph_tag}_seasonal_synchronization{fig_suffix}.png',
                          model_label='NXRO-Graph')
        
        print(f"✓ NXRO-Graph complete (out-of-sample)")

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
        base_dir = f'{base_results_dir}/graphpyg/{tag2}_{ktag}'
        ensure_dir(base_dir)
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(gp_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(gp_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'NXRO-GraphPyG ({tag2}, {ktag}) training (Out-of-Sample)')
        ax.legend()
        plt.savefig(f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_training_curves.png', dpi=300)
        plt.close()
        
        gp_save = f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}_best{extra_tag}.pt'
        torch.save({'state_dict': gp_model.state_dict(), 'var_order': gp_var_order}, gp_save)
        print(f"✓ Saved to: {gp_save}")
        
        NXRO_gp_fcst = nxro_reforecast(gp_model, init_ds=obs_ds, n_month=21, var_order=gp_var_order, device=device)
        
        if args.eval_all_datasets and all_eval_datasets:
            print("  Evaluating on all datasets...")
            all_results = evaluate_on_all_datasets(NXRO_gp_fcst, all_eval_datasets, train_period, test_period)
            acc_train = all_results['ORAS5']['acc_train']
            rmse_train = all_results['ORAS5']['rmse_train']
            acc_test = all_results['ORAS5']['acc_test']
            rmse_test = all_results['ORAS5']['rmse_test']
            plot_skill_curves_multi_dataset(all_results, 'Nino34', f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}', f'NXRO-GraphPyG ({tag2.upper()})')
        else:
            acc_train = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            acc_test = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
        
        plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                              sel_var='Nino34', out_prefix=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}', 
                              label=f'NXRO-GraphPyG ({tag2.upper()})')
        
        gp_sim_ds = simulate_nxro_longrun(gp_model, X0_ds=train_ds, var_order=gp_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, gp_sim_ds, sel_var='Nino34', 
                          out_path=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_seasonal_synchronization.png',
                          model_label=f'NXRO-GraphPyG ({tag2.upper()})')
        
        print(f"✓ NXRO-GraphPyG complete (out-of-sample)")

    # Run selected models
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
    
    print("\n" + "="*80)
    print("OUT-OF-SAMPLE EXPERIMENT COMPLETE")
    print("="*80)
    print(f"All results saved to: {base_results_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()

