import os
import warnings
warnings.filterwarnings("ignore")
import argparse

import torch
import xarray as xr

from nxro.nxro_utils import (
    plot_observed_nino34,
    load_all_eval_datasets,
)
from run_utils import (
    ensure_dir,
    run_linear,
    run_ro,
    run_rodiag,
    run_res,
    run_res_fullxro,
    run_neural,
    run_neural_phys,
    run_resmix,
    run_bilinear,
    run_attentive,
    run_graph,
    run_graph_pyg,
)
from run_utils_twostage import (
    run_linear_twostage,
    run_ro_twostage,
    run_rodiag_twostage,
    run_res_twostage,
    run_res_fullxro_twostage,
    run_neural_twostage,
    run_neural_phys_twostage,
    run_resmix_twostage,
    run_bilinear_twostage,
    run_attentive_twostage,
    run_graph_twostage,
    run_graph_pyg_twostage,
)


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
    parser.add_argument('--ar_p', type=int, default=1, help='AR lag order for stochastic noise (default: 1)')
    parser.add_argument('--train_noise_stage2', action='store_true', 
                        help='Use Stage 2 training (likelihood) instead of post-hoc for noise parameters')
    parser.add_argument('--use_sim_noise', action='store_true',
                        help='Use simulation-observation differences for noise instead of model residuals')
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
    parser.add_argument('--two_stage', action='store_true',
                        help='Enable two-stage training: 1. Synthetic Pre-training, 2. ORAS5 Fine-tuning.')
    args = parser.parse_args()

    # Use different results directory based on eval_all_datasets flag
    if args.eval_all_datasets:
        base_results_dir = 'results_all_outsample'
    else:
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

    # Run selected models
    if args.model in ('linear', 'all'):
        if args.two_stage:
            run_linear_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, 
                       load_xro_init, variant_suffix, extra_tag, fig_suffix)
        else:
            run_linear(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, 
                       load_xro_init, variant_suffix, extra_tag, fig_suffix)
    
    if args.model in ('ro', 'all'):
        if args.two_stage:
            run_ro_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                   base_results_dir, all_eval_datasets, device, 
                   load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
        else:
            run_ro(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                   base_results_dir, all_eval_datasets, device, 
                   load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
    
    if args.model in ('rodiag', 'all'):
        if args.two_stage:
            run_rodiag_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, 
                       load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
        else:
            run_rodiag(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, 
                       load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
    
    if args.model in ('res', 'all'):
        if args.two_stage:
            run_res_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                    base_results_dir, all_eval_datasets, device, 
                    load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
        else:
            run_res(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                    base_results_dir, all_eval_datasets, device, 
                    load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
    
    if args.model == 'res_fullxro':
        if args.two_stage:
            run_res_fullxro_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                            base_results_dir, all_eval_datasets, device, 
                            load_xro_init, extra_tag, fig_suffix)
        else:
            run_res_fullxro(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                            base_results_dir, all_eval_datasets, device, 
                            load_xro_init, extra_tag, fig_suffix)
    
    if args.model in ('neural', 'all'):
        if args.two_stage:
            run_neural_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)
        else:
            run_neural(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)
    
    if args.model in ('neural_phys', 'all'):
        if args.two_stage:
            run_neural_phys_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                            base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)
        else:
            run_neural_phys(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                            base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)
    
    if args.model in ('resmix', 'all'):
        if args.two_stage:
            run_resmix_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, 
                       load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
        else:
            run_resmix(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                       base_results_dir, all_eval_datasets, device, 
                       load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
    
    if args.model in ('bilinear', 'all'):
        if args.two_stage:
            run_bilinear_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                         base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)
        else:
            run_bilinear(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                         base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)
    
    if args.model in ('attentive', 'all'):
        if args.two_stage:
            run_attentive_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                          base_results_dir, all_eval_datasets, device, 
                          load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
        else:
            run_attentive(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                          base_results_dir, all_eval_datasets, device, 
                          load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
    
    if args.model in ('graph', 'all'):
        if args.two_stage:
            run_graph_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                      base_results_dir, all_eval_datasets, device, 
                      load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
        else:
            run_graph(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                      base_results_dir, all_eval_datasets, device, 
                      load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix)
    
    if args.model in ('graph_pyg', 'all'):
        if args.two_stage:
            run_graph_pyg_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                          base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)
        else:
            run_graph_pyg(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                          base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix)

    print("\n" + "="*80)
    print("OUT-OF-SAMPLE EXPERIMENT COMPLETE")
    print("="*80)
    print(f"All results saved to: {base_results_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
