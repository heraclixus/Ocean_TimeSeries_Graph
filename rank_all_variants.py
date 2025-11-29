#!/usr/bin/env python
"""
Rank ALL available NXRO variant checkpoints and display top N performers.

This script:
1. Discovers all checkpoint files across all model types and variants
2. Loads each model and generates forecasts
3. Calculates ACC and RMSE skill metrics
4. Ranks all variants together (not just one per model type)
5. Shows top N performers and generates comparison plots

Usage:
    python rank_all_variants.py --top_n 10 --metric rmse --test --eval_period train
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import glob
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from XRO.core import XRO
from nxro.models import (
    NXROLinearModel,
    NXROROModel,
    NXRORODiagModel,
    NXROResModel,
    NXROResFullXROModel,
    NXROResidualMixModel,
    NXROAttentiveModel,
    NXROGraphModel,
    NXROGraphPyGModel,
    NXRONeuralODEModel,
    NXROBilinearModel,
    NXROTransformerModel,
)
from utils.xro_utils import calc_forecast_skill, nxro_reforecast
from graph_construction import get_or_build_xro_graph, get_or_build_stat_knn_graph


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def discover_all_checkpoints(search_dirs=None, ckpt_suffix=''):
    """Discover all NXRO checkpoint files."""
    if search_dirs is None:
        search_dirs = [
            'results/linear',
            'results/ro',
            'results/rodiag',
            'results/res',
            'results/res_fullxro',
            'results/neural',
            'results/neural_phys',
            'results/attentive',
            'results/bilinear',
            'results/resmix',
            'results/graph',
            'results/graphpyg',
            'results/transformer',
        ]
    
    all_checkpoints = []
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        # Find all .pt files matching pattern
        pattern = f'{base_dir}/**/nxro_*_best{ckpt_suffix}*.pt'
        matches = glob.glob(pattern, recursive=True)
        all_checkpoints.extend(matches)
    
    # Also check root results directory
    root_pattern = f'results/nxro_*_best{ckpt_suffix}*.pt'
    all_checkpoints.extend(glob.glob(root_pattern))
    
    # Deduplicate
    all_checkpoints = list(set(all_checkpoints))
    
    return sorted(all_checkpoints)


def infer_model_class_and_kwargs(ckpt_path):
    """Infer model class and kwargs from checkpoint path and state_dict."""
    basename = os.path.basename(ckpt_path).lower()
    
    # Load checkpoint to check state_dict keys
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        return None, None
    
    var_order = ckpt.get('var_order', None)
    if var_order is None:
        return None, None
    
    n_vars = len(var_order)
    sd_keys = list(ckpt['state_dict'].keys())
    
    # Infer model type from filename and state_dict
    # Order matters: check specific patterns before general ones!
    
    if 'graph' in basename and 'pyg' in basename:
        # PyG models - need to reconstruct edge_index
        # Parse graph type from filename
        toks = basename.replace('.pt', '').split('_')
        use_gat = 'gat' in toks
        
        # Get top_k from filename (e.g., 'k3' → top_k=3)
        k_tok = next((t for t in toks if t.startswith('k') and t[1:].isdigit()), 'k3')
        top_k = int(k_tok[1:])
        
        # Determine graph prior
        if 'stat' in toks:
            i = toks.index('stat')
            prior = toks[i+1] if i + 1 < len(toks) else 'pearson'
        else:
            prior = 'xro'
        
        try:
            # Build adjacency matrix
            if prior == 'xro':
                A, _ = get_or_build_xro_graph(nc_path='data/XRO_indices_oras5.nc', 
                                             train_start='1979-01', train_end='2022-12', 
                                             var_order=var_order)
            else:
                A, _ = get_or_build_stat_knn_graph(data_path='data/XRO_indices_oras5_train.csv',
                                                   train_start='1979-01', train_end='2022-12',
                                                   var_order=var_order, method=prior, top_k=top_k)
            
            # Build edge_index from adjacency
            V = A.shape[0]
            A2 = A.clone()
            A2.fill_diagonal_(0.0)
            edges = []
            for i in range(V):
                vals, idx = torch.topk(A2[i], k=min(top_k, V - 1))
                for j in idx.tolist():
                    if i != j and A2[i, j] > 0:
                        edges.append([i, j])
                        edges.append([j, i])
            edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty(2, 0, dtype=torch.long)
            
            return NXROGraphPyGModel, {'n_vars': n_vars, 'k_max': 2, 'edge_index': edge_index,
                                       'hidden': 16, 'dropout': 0.1, 'use_gat': use_gat}
        except Exception:
            # If graph construction fails, skip
            return None, None
    
    elif 'resmix' in basename or 'residualmix' in basename:
        # Check resmix before res (more specific)
        alpha_learnable = any('alpha_param' in k for k in sd_keys)
        return NXROResidualMixModel, {'n_vars': n_vars, 'k_max': 2, 'hidden': 64, 
                                      'alpha_init': 0.1, 'alpha_learnable': alpha_learnable}
    
    elif 'rodiag' in basename:
        # Check rodiag before ro (more specific)
        return NXRORODiagModel, {'n_vars': n_vars, 'k_max': 2}
    
    elif 'neural' in basename:
        # Check neural before checking for 'ro' substring (neural contains 'ro')
        return NXRONeuralODEModel, {'n_vars': n_vars, 'k_max': 2, 'hidden': 64, 
                                    'depth': 2, 'dropout': 0.1}
    
    elif 'attentive' in basename or 'attention' in basename:
        return NXROAttentiveModel, {'n_vars': n_vars, 'k_max': 2, 'd': 32, 
                                    'dropout': 0.1, 'mask_mode': 'th_only'}
    
    elif 'bilinear' in basename:
        return NXROBilinearModel, {'n_vars': n_vars, 'k_max': 2, 'n_channels': 2, 'rank': 2}
    
    elif 'transformer' in basename:
        return NXROTransformerModel, {'n_vars': n_vars, 'k_max': 2, 'd_model': 64, 
                                      'nhead': 4, 'num_layers': 2, 
                                      'dim_feedforward': 256, 'dropout': 0.1}
    
    elif 'fullxro' in basename:
        # Cannot recreate without XRO components
        return None, None
    
    elif 'res' in basename:
        # Check res after resmix and fullxro
        return NXROResModel, {'n_vars': n_vars, 'k_max': 2, 'hidden': 64}
    
    elif 'graph' in basename:
        # Check graph after graphpyg
        use_fixed = 'fixed' in basename or 'learned' not in basename
        return NXROGraphModel, {'n_vars': n_vars, 'k_max': 2, 'use_fixed_graph': use_fixed}
    
    elif 'linear' in basename and not any(k.startswith('W_T') for k in sd_keys):
        # Linear model (check it doesn't have RO components)
        return NXROLinearModel, {'n_vars': n_vars, 'k_max': 2}
    
    elif 'ro' in basename:
        # Check ro last (very general, matches many words)
        # Verify it has RO components in state_dict
        if any(k.startswith('W_T') for k in sd_keys) and not any(k.startswith('B_diag') for k in sd_keys):
            return NXROROModel, {'n_vars': n_vars, 'k_max': 2}
        else:
            return None, None
    
    return None, None


def get_variant_label(ckpt_path):
    """Extract a human-readable label from checkpoint path."""
    basename = os.path.basename(ckpt_path)
    # Remove extension and common prefixes
    label = basename.replace('.pt', '').replace('nxro_', '').replace('_best_test', '').replace('_best', '')
    
    # Clean up extra tags
    for tag in ['_extra_data', '_sim100', '_sim50']:
        label = label.split(tag)[0]
    
    # Prettify
    label = label.replace('_', ' ').title()
    label = label.replace('Ws', 'WS').replace('Fixl', 'FixL').replace('Fixro', 'FixRO')
    label = label.replace('Fixdiag', 'FixDiag').replace('Fixnl', 'FixNL')
    label = label.replace('Fixphysics', 'FixPhysics')
    label = label.replace('Rodiag', 'RO+Diag').replace('Resmix', 'ResidualMix')
    
    return label


def load_and_evaluate(ckpt_path, obs_ds, eval_period):
    """Load checkpoint, forecast, and calculate skill."""
    model_class, model_kwargs = infer_model_class_and_kwargs(ckpt_path)
    
    if model_class is None:
        return None
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        var_order = ckpt['var_order']
        
        model = model_class(**model_kwargs)
        model.load_state_dict(ckpt['state_dict'])
        
        # Generate forecast
        fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device='cpu')
        
        # Calculate skill
        acc_ds = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=eval_period)
        rmse_ds = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=eval_period)
        
        # Mean skill across leads
        mean_acc = float(np.nanmean(acc_ds['Nino34'].values))
        mean_rmse = float(np.nanmean(rmse_ds['Nino34'].values))
        
        return {
            'label': get_variant_label(ckpt_path),
            'path': ckpt_path,
            'mean_acc': mean_acc,
            'mean_rmse': mean_rmse,
            'acc_ds': acc_ds,
            'rmse_ds': rmse_ds,
            'fcst': fcst,
            'model': model,
            'var_order': var_order,
        }
    
    except Exception as e:
        print(f"  ✗ Failed to load {os.path.basename(ckpt_path)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Rank all NXRO variants and show top performers')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top models to display/plot')
    parser.add_argument('--metric', type=str, choices=['acc', 'rmse', 'combined'], default='rmse',
                       help='Ranking metric: acc (higher better), rmse (lower better), or combined')
    parser.add_argument('--test', action='store_true', help='Use test-period checkpoints (*_best_test.pt)')
    parser.add_argument('--eval_period', type=str, choices=['train', 'test'], default='train',
                       help='Evaluation period: train (1979-2022) or test (2023-onwards)')
    parser.add_argument('--output_dir', type=str, default='results/rankings',
                       help='Output directory for ranking results')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if CSV exists')
    args = parser.parse_args()
    
    # Set evaluation period
    if args.eval_period == 'test':
        eval_start, eval_end = '2023-01', None
    else:
        eval_start, eval_end = '1979-01', '2022-12'
    
    eval_period = slice(eval_start, eval_end)
    ckpt_suffix = '_test' if args.test else ''
    
    ensure_dir(args.output_dir)
    
    print("="*80)
    print("RANKING ALL MODELS (XRO Baselines + NXRO Variants)")
    print("="*80)
    print(f"Checkpoint suffix: {ckpt_suffix or '(none)'}")
    print(f"Evaluation period: {eval_start} to {eval_end or 'end'}")
    print(f"Ranking metric: {args.metric}")
    print(f"Top N to display: {args.top_n}")
    print()
    print("Note: XRO baselines (XRO, XRO_ac0, Linear XRO) are always included")
    print("      NXRO variants are loaded from checkpoint files")
    print("="*80)
    print()
    
    # Check if cached results exist
    output_csv = f'{args.output_dir}/all_variants_ranked_{args.metric}_eval_{args.eval_period}.csv'
    
    # Check if cached CSV exists
    use_cache = os.path.exists(output_csv) and not args.force
    
    if use_cache:
        print(f"✓ Found existing ranking: {output_csv}")
        print("  Loading cached results (use --force to recompute)")
        print()
        df = pd.read_csv(output_csv)
        all_results = None  # Skip detailed loading when using cache
        
    else:
        if args.force and os.path.exists(output_csv):
            print(f"  Force recomputation requested (--force)")
        else:
            print(f"  No cached results found, computing from scratch...")
        print()
        
        # Load data
        obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
        train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))
        
        # Fit XRO baselines
        print("Fitting XRO baselines...")
        xro_ac2 = XRO(ncycle=12, ac_order=2)
        xro_ac0 = XRO(ncycle=12, ac_order=0)
        
        xro_ac2_fit = xro_ac2.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
        xro_ac0_fit = xro_ac0.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
        xro_lin_fit = xro_ac2.fit_matrix(train_ds, maskb=[], maskNT=[])
        
        xro_results = {}
        for label, xro_model, xro_fit in [
            ('XRO', xro_ac2, xro_ac2_fit),
            ('XRO_ac0', xro_ac0, xro_ac0_fit),
            ('Linear XRO', xro_ac2, xro_lin_fit),
        ]:
            fcst = xro_model.reforecast(fit_ds=xro_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
            acc_ds = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                         by_month=False, verify_periods=eval_period)
            rmse_ds = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                          by_month=False, verify_periods=eval_period)
            xro_results[label] = {
                'label': label,
                'path': 'N/A (XRO baseline)',
                'mean_acc': float(np.nanmean(acc_ds['Nino34'].values)),
                'mean_rmse': float(np.nanmean(rmse_ds['Nino34'].values)),
                'acc_ds': acc_ds,
                'rmse_ds': rmse_ds,
            }
        
        print(f"  ✓ XRO baselines ready: {len(xro_results)} variants (XRO, XRO_ac0, Linear XRO)")
        print()
        
        # Discover all NXRO checkpoints
        print("Discovering NXRO variant checkpoints...")
        all_checkpoints = discover_all_checkpoints(ckpt_suffix=ckpt_suffix)
        print(f"  Found {len(all_checkpoints)} checkpoint files")
        print()
        
        # Load and evaluate each
        print("Loading and evaluating variants...")
        nxro_results = {}
        for i, ckpt_path in enumerate(all_checkpoints, 1):
            basename = os.path.basename(ckpt_path)
            print(f"  [{i}/{len(all_checkpoints)}] Loading {basename}...", end=' ')
            
            result = load_and_evaluate(ckpt_path, obs_ds, eval_period)
            if result is not None:
                nxro_results[result['label']] = result
                print(f"✓ ACC={result['mean_acc']:.3f}, RMSE={result['mean_rmse']:.3f}")
            else:
                print("✗ Failed")
        
        print()
        print(f"✓ Successfully loaded {len(nxro_results)} NXRO variants")
        print()
        
        # Combine XRO and NXRO results
        all_results = {**xro_results, **nxro_results}
        
        # Rank all models
        print("="*80)
        print("RANKING ALL MODELS")
        print("="*80)
        
        # Create ranking dataframe
        ranking_data = []
        for label, result in all_results.items():
            ranking_data.append({
                'Model': label,
                'Mean_ACC': result['mean_acc'],
                'Mean_RMSE': result['mean_rmse'],
                'Path': result.get('path', 'N/A'),
            })
        
        df = pd.DataFrame(ranking_data)
        
        # Rank by chosen metric
        if args.metric == 'acc':
            df = df.sort_values('Mean_ACC', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
        elif args.metric == 'rmse':
            df = df.sort_values('Mean_RMSE', ascending=True)
            df['Rank'] = range(1, len(df) + 1)
        else:  # combined
            # Normalize both metrics to [0, 1] and combine
            acc_norm = (df['Mean_ACC'] - df['Mean_ACC'].min()) / (df['Mean_ACC'].max() - df['Mean_ACC'].min())
            rmse_norm = (df['Mean_RMSE'] - df['Mean_RMSE'].min()) / (df['Mean_RMSE'].max() - df['Mean_RMSE'].min())
            df['Combined_Score'] = acc_norm - rmse_norm  # Higher ACC good, lower RMSE good
            df = df.sort_values('Combined_Score', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
        
        # Save full ranking
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved full ranking to: {output_csv}")
        print()
    
    # Display top N
    print(f"TOP {args.top_n} MODELS (Ranked by {args.metric.upper()}):")
    print("-"*80)
    top_df = df.head(args.top_n)
    print(top_df[['Rank', 'Model', 'Mean_ACC', 'Mean_RMSE']].to_string(index=False))
    print()
    
    # Plot average rank bar chart (always generate, only needs df)
    print("Generating average rank bar plot...")
    
    # Sort by rank for display
    df_sorted = df.sort_values('Rank')
    
    # Plot bar chart of ranks
    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.5 * len(df)), 5))
    
    # Color XRO baselines differently
    colors = []
    for model in df_sorted['Model'].values:
        if model in ['XRO', 'XRO_ac0', 'Linear XRO']:
            colors.append('tab:red')
        else:
            colors.append('tab:blue')
    
    ax.bar(range(len(df_sorted)), df_sorted['Rank'].values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['Model'].values, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Rank (lower is better)', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_title(f'Overall Ranking (by {args.metric.upper()})', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tab:red', alpha=0.7, label='XRO Baselines'),
                       Patch(facecolor='tab:blue', alpha=0.7, label='NXRO Variants')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    bar_plot_path = f'{args.output_dir}/all_variants_rank_bar_{args.metric}_eval_{args.eval_period}.png'
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    
    print(f"  ✓ Saved rank bar plot: {bar_plot_path}")
    print()
    
    # Plot top N skill curves (only if we have full results, not from cache)
    if not use_cache and all_results is not None:
        print(f"Generating skill curve plots for top {args.top_n} + XRO baselines...")
        
        # Ensure XRO baselines are included (always plot them regardless of rank)
        xro_baseline_labels = ['XRO', 'XRO_ac0', 'Linear XRO']
        models_to_plot = set(top_df['Model'].values)
        for xro_label in xro_baseline_labels:
            if xro_label in all_results:
                models_to_plot.add(xro_label)
        
        # Get rows for models to plot
        plot_rows = df[df['Model'].isin(models_to_plot)].sort_values('Rank')
        
        # Plot ACC
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for idx, row in plot_rows.iterrows():
            label = row['Model']
            if label in all_results:
                result = all_results[label]
                acc_vals = result['acc_ds']['Nino34'].values
                lead_vals = result['acc_ds']['Nino34'].lead.values
                
                # Style XRO baselines differently
                if label in xro_baseline_labels:
                    ax.plot(lead_vals, acc_vals, label=f"{row['Rank']}. {label}", 
                           lw=2.5, linestyle='--', alpha=0.8)
                else:
                    ax.plot(lead_vals, acc_vals, label=f"{row['Rank']}. {label}", 
                           lw=2, marker='o', markersize=3)
        
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'Top {args.top_n} Models + XRO Baselines: ACC (Ranked by {args.metric.upper()})', fontsize=12)
        ax.set_xlim([0, 21])
        ax.set_ylim([0.2, 1.0])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/top{args.top_n}_acc_eval_{args.eval_period}.png', dpi=300)
        plt.close()
        
        # Plot RMSE
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for idx, row in plot_rows.iterrows():
            label = row['Model']
            if label in all_results:
                result = all_results[label]
                rmse_vals = result['rmse_ds']['Nino34'].values
                lead_vals = result['rmse_ds']['Nino34'].lead.values
                
                # Style XRO baselines differently
                if label in xro_baseline_labels:
                    ax.plot(lead_vals, rmse_vals, label=f"{row['Rank']}. {label}", 
                           lw=2.5, linestyle='--', alpha=0.8)
                else:
                    ax.plot(lead_vals, rmse_vals, label=f"{row['Rank']}. {label}", 
                           lw=2, marker='o', markersize=3)
        
        ax.set_ylabel('RMSE (°C)', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'Top {args.top_n} Models + XRO Baselines: RMSE (Ranked by {args.metric.upper()})', fontsize=12)
        ax.set_xlim([0, 21])
        ax.set_ylim([0.0, 1.0])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/top{args.top_n}_rmse_eval_{args.eval_period}.png', dpi=300)
        plt.close()
        
        print(f"  ✓ Saved ACC plot: {args.output_dir}/top{args.top_n}_acc_eval_{args.eval_period}.png")
        print(f"  ✓ Saved RMSE plot: {args.output_dir}/top{args.top_n}_rmse_eval_{args.eval_period}.png")
        print()
    elif use_cache:
        print("  ⊘ Skipping detailed skill curve plots (using cached data)")
        print("    Run with --force to regenerate with full forecast data")
        print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if not use_cache and all_results is not None:
        print(f"Total models evaluated: {len(all_results)}")
        print(f"  - XRO baselines: {len(xro_results)} (XRO, XRO_ac0, Linear XRO)")
        print(f"  - NXRO variants: {len(nxro_results)}")
    else:
        print(f"Total models in ranking: {len(df)}")
        # Count XRO baselines from df
        xro_count = df['Model'].isin(['XRO', 'XRO_ac0', 'Linear XRO']).sum()
        print(f"  - XRO baselines: {xro_count}")
        print(f"  - NXRO variants: {len(df) - xro_count}")
    print()
    
    # Show ranking of XRO baselines
    xro_ranks = df[df['Model'].isin(['XRO', 'XRO_ac0', 'Linear XRO'])][['Rank', 'Model', 'Mean_ACC', 'Mean_RMSE']]
    if not xro_ranks.empty:
        print("XRO baseline rankings:")
        print(xro_ranks.to_string(index=False))
        print()
    
    print(f"Best model by ACC: {df.iloc[0]['Model']} (ACC={df.iloc[0]['Mean_ACC']:.4f})")
    print(f"Best model by RMSE: {df.sort_values('Mean_RMSE').iloc[0]['Model']} (RMSE={df.sort_values('Mean_RMSE').iloc[0]['Mean_RMSE']:.4f})")
    print()
    print("="*80)
    print("✓ RANKING COMPLETE!")
    print("="*80)
    print()
    print("Output files:")
    print(f"  - {output_csv}")
    print(f"  - {bar_plot_path}")
    if not use_cache:
        print(f"  - {args.output_dir}/top{args.top_n}_acc_eval_{args.eval_period}.png")
        print(f"  - {args.output_dir}/top{args.top_n}_rmse_eval_{args.eval_period}.png")
    print()


if __name__ == '__main__':
    main()

