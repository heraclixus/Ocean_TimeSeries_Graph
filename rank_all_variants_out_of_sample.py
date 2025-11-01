#!/usr/bin/env python
"""
Rank ALL available NXRO variant checkpoints for OUT-OF-SAMPLE experiment.

This script ranks models trained on 1979-2001 and evaluates on both:
- In-sample (train): 1979-2001
- Out-of-sample (test): 2002-2022

Usage:
    python rank_all_variants_out_of_sample.py --top_n 10 --metric rmse
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
)
from utils.xro_utils import calc_forecast_skill, nxro_reforecast
from graph_construction import get_or_build_xro_graph, get_or_build_stat_knn_graph


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def discover_all_checkpoints(search_dirs=None, ckpt_suffix=''):
    """Discover all NXRO checkpoint files from out-of-sample results."""
    if search_dirs is None:
        search_dirs = [
            'results_out_of_sample/linear',
            'results_out_of_sample/ro',
            'results_out_of_sample/rodiag',
            'results_out_of_sample/res',
            'results_out_of_sample/res_fullxro',
            'results_out_of_sample/neural',
            'results_out_of_sample/neural_phys',
            'results_out_of_sample/attentive',
            'results_out_of_sample/bilinear',
            'results_out_of_sample/resmix',
            'results_out_of_sample/graph',
            'results_out_of_sample/graphpyg',
        ]
    
    all_checkpoints = []
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        pattern = f'{base_dir}/**/nxro_*_best{ckpt_suffix}*.pt'
        matches = glob.glob(pattern, recursive=True)
        all_checkpoints.extend(matches)
    
    root_pattern = f'results_out_of_sample/nxro_*_best{ckpt_suffix}*.pt'
    all_checkpoints.extend(glob.glob(root_pattern))
    
    all_checkpoints = list(set(all_checkpoints))
    
    return sorted(all_checkpoints)


def infer_model_class_and_kwargs(ckpt_path):
    """Infer model class and kwargs from checkpoint path and state_dict."""
    basename = os.path.basename(ckpt_path).lower()
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        return None, None
    
    var_order = ckpt.get('var_order', None)
    if var_order is None:
        return None, None
    
    n_vars = len(var_order)
    sd_keys = list(ckpt['state_dict'].keys())
    
    # Infer model type (same logic as original rank_all_variants.py)
    if 'graph' in basename and 'pyg' in basename:
        toks = basename.replace('.pt', '').split('_')
        use_gat = 'gat' in toks
        
        k_tok = next((t for t in toks if t.startswith('k') and t[1:].isdigit()), 'k3')
        top_k = int(k_tok[1:])
        
        if 'stat' in toks:
            i = toks.index('stat')
            prior = toks[i+1] if i + 1 < len(toks) else 'pearson'
        else:
            prior = 'xro'
        
        try:
            if prior == 'xro':
                A, _ = get_or_build_xro_graph(nc_path='data/XRO_indices_oras5.nc', 
                                             train_start='1979-01', train_end='2001-12', 
                                             var_order=var_order)
            else:
                A, _ = get_or_build_stat_knn_graph(data_path='data/XRO_indices_oras5_train.csv',
                                                   train_start='1979-01', train_end='2001-12',
                                                   var_order=var_order, method=prior, top_k=top_k)
            
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
            return None, None
    
    elif 'resmix' in basename or 'residualmix' in basename:
        alpha_learnable = any('alpha_param' in k for k in sd_keys)
        return NXROResidualMixModel, {'n_vars': n_vars, 'k_max': 2, 'hidden': 64, 
                                      'alpha_init': 0.1, 'alpha_learnable': alpha_learnable}
    
    elif 'rodiag' in basename:
        return NXRORODiagModel, {'n_vars': n_vars, 'k_max': 2}
    
    elif 'neural' in basename:
        return NXRONeuralODEModel, {'n_vars': n_vars, 'k_max': 2, 'hidden': 64, 
                                    'depth': 2, 'dropout': 0.1}
    
    elif 'attentive' in basename or 'attention' in basename:
        return NXROAttentiveModel, {'n_vars': n_vars, 'k_max': 2, 'd': 32, 
                                    'dropout': 0.1, 'mask_mode': 'th_only'}
    
    elif 'bilinear' in basename:
        return NXROBilinearModel, {'n_vars': n_vars, 'k_max': 2, 'n_channels': 2, 'rank': 2}
    
    elif 'fullxro' in basename:
        return None, None
    
    elif 'res' in basename:
        return NXROResModel, {'n_vars': n_vars, 'k_max': 2, 'hidden': 64}
    
    elif 'graph' in basename:
        use_fixed = 'fixed' in basename or 'learned' not in basename
        return NXROGraphModel, {'n_vars': n_vars, 'k_max': 2, 'use_fixed_graph': use_fixed}
    
    elif 'linear' in basename and not any(k.startswith('W_T') for k in sd_keys):
        return NXROLinearModel, {'n_vars': n_vars, 'k_max': 2}
    
    elif 'ro' in basename:
        if any(k.startswith('W_T') for k in sd_keys) and not any(k.startswith('B_diag') for k in sd_keys):
            return NXROROModel, {'n_vars': n_vars, 'k_max': 2}
        else:
            return None, None
    
    return None, None


def get_variant_label(ckpt_path):
    """Extract a human-readable label from checkpoint path."""
    basename = os.path.basename(ckpt_path)
    label = basename.replace('.pt', '').replace('nxro_', '').replace('_best_test', '').replace('_best', '')
    
    for tag in ['_extra_data', '_sim100', '_sim50']:
        label = label.split(tag)[0]
    
    label = label.replace('_', ' ').title()
    label = label.replace('Ws', 'WS').replace('Fixl', 'FixL').replace('Fixro', 'FixRO')
    label = label.replace('Fixdiag', 'FixDiag').replace('Fixnl', 'FixNL')
    label = label.replace('Fixphysics', 'FixPhysics')
    label = label.replace('Rodiag', 'RO+Diag').replace('Resmix', 'ResidualMix')
    
    return label


def load_and_evaluate_dual(ckpt_path, obs_ds, train_period, test_period):
    """Load checkpoint, forecast, and calculate skill for both train and test periods."""
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
        
        # Calculate skill for train period (in-sample)
        acc_train = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                        by_month=False, verify_periods=train_period)
        rmse_train = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                         by_month=False, verify_periods=train_period)
        
        # Calculate skill for test period (out-of-sample)
        acc_test = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                       by_month=False, verify_periods=test_period)
        rmse_test = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                        by_month=False, verify_periods=test_period)
        
        mean_acc_train = float(np.nanmean(acc_train['Nino34'].values))
        mean_rmse_train = float(np.nanmean(rmse_train['Nino34'].values))
        mean_acc_test = float(np.nanmean(acc_test['Nino34'].values))
        mean_rmse_test = float(np.nanmean(rmse_test['Nino34'].values))
        
        return {
            'label': get_variant_label(ckpt_path),
            'path': ckpt_path,
            'mean_acc_train': mean_acc_train,
            'mean_rmse_train': mean_rmse_train,
            'mean_acc_test': mean_acc_test,
            'mean_rmse_test': mean_rmse_test,
            'acc_train': acc_train,
            'rmse_train': rmse_train,
            'acc_test': acc_test,
            'rmse_test': rmse_test,
            'fcst': fcst,
            'model': model,
            'var_order': var_order,
        }
    
    except Exception as e:
        print(f"  ✗ Failed to load {os.path.basename(ckpt_path)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Rank NXRO variants (Out-of-Sample Experiment)')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top models to display/plot')
    parser.add_argument('--metric', type=str, choices=['acc', 'rmse', 'combined'], default='combined',
                       help='Ranking metric (evaluated on out-of-sample period)')
    parser.add_argument('--output_dir', type=str, default='results_out_of_sample/rankings',
                       help='Output directory for ranking results')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if CSV exists')
    args = parser.parse_args()
    
    # Out-of-sample setup
    train_start, train_end = '1979-01', '2001-12'
    test_start, test_end = '2002-01', '2022-12'
    
    train_period = slice(train_start, train_end)
    test_period = slice(test_start, test_end)
    
    ensure_dir(args.output_dir)
    
    print("="*80)
    print("RANKING ALL MODELS (OUT-OF-SAMPLE EXPERIMENT)")
    print("="*80)
    print(f"Train period (in-sample): {train_start} to {train_end}")
    print(f"Test period (out-of-sample): {test_start} to {test_end}")
    print(f"Ranking metric: {args.metric} (evaluated on out-of-sample)")
    print(f"Top N to display: {args.top_n}")
    print("="*80)
    print()
    
    # Check if cached results exist
    output_csv = f'{args.output_dir}/all_variants_ranked_{args.metric}_out_of_sample.csv'
    
    use_cache = os.path.exists(output_csv) and not args.force
    
    if use_cache:
        print(f"✓ Found existing ranking: {output_csv}")
        print("  Loading cached results (use --force to recompute)")
        print()
        df = pd.read_csv(output_csv)
        all_results = None
        
    else:
        if args.force and os.path.exists(output_csv):
            print(f"  Force recomputation requested (--force)")
        else:
            print(f"  No cached results found, computing from scratch...")
        print()
        
        # Load data
        obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
        train_ds = obs_ds.sel(time=train_period)
        
        # Fit XRO baselines on train period only
        print("Fitting XRO baselines on train period (1979-2001)...")
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
            
            # Train period
            acc_train = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            
            # Test period
            acc_test = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
            rmse_test = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
            
            xro_results[label] = {
                'label': label,
                'path': 'N/A (XRO baseline)',
                'mean_acc_train': float(np.nanmean(acc_train['Nino34'].values)),
                'mean_rmse_train': float(np.nanmean(rmse_train['Nino34'].values)),
                'mean_acc_test': float(np.nanmean(acc_test['Nino34'].values)),
                'mean_rmse_test': float(np.nanmean(rmse_test['Nino34'].values)),
                'acc_train': acc_train,
                'rmse_train': rmse_train,
                'acc_test': acc_test,
                'rmse_test': rmse_test,
            }
        
        print(f"  ✓ XRO baselines ready: {len(xro_results)} variants")
        print()
        
        # Discover NXRO checkpoints
        print("Discovering NXRO variant checkpoints...")
        all_checkpoints = discover_all_checkpoints(ckpt_suffix='')
        print(f"  Found {len(all_checkpoints)} checkpoint files")
        print()
        
        # Load and evaluate each
        print("Loading and evaluating variants...")
        nxro_results = {}
        for i, ckpt_path in enumerate(all_checkpoints, 1):
            basename = os.path.basename(ckpt_path)
            print(f"  [{i}/{len(all_checkpoints)}] Loading {basename}...", end=' ')
            
            result = load_and_evaluate_dual(ckpt_path, obs_ds, train_period, test_period)
            if result is not None:
                nxro_results[result['label']] = result
                print(f"✓ Train ACC={result['mean_acc_train']:.3f}, Test ACC={result['mean_acc_test']:.3f}")
            else:
                print("✗ Failed")
        
        print()
        print(f"✓ Successfully loaded {len(nxro_results)} NXRO variants")
        print()
        
        # Combine results
        all_results = {**xro_results, **nxro_results}
        
        # Create ranking dataframe
        ranking_data = []
        for label, result in all_results.items():
            ranking_data.append({
                'Model': label,
                'Mean_ACC_Train': result['mean_acc_train'],
                'Mean_RMSE_Train': result['mean_rmse_train'],
                'Mean_ACC_Test': result['mean_acc_test'],
                'Mean_RMSE_Test': result['mean_rmse_test'],
                'Path': result.get('path', 'N/A'),
            })
        
        df = pd.DataFrame(ranking_data)
        
        # Rank based on OUT-OF-SAMPLE (test) performance
        if args.metric == 'acc':
            df = df.sort_values('Mean_ACC_Test', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
        elif args.metric == 'rmse':
            df = df.sort_values('Mean_RMSE_Test', ascending=True)
            df['Rank'] = range(1, len(df) + 1)
        else:  # combined
            acc_norm = (df['Mean_ACC_Test'] - df['Mean_ACC_Test'].min()) / (df['Mean_ACC_Test'].max() - df['Mean_ACC_Test'].min())
            rmse_norm = (df['Mean_RMSE_Test'] - df['Mean_RMSE_Test'].min()) / (df['Mean_RMSE_Test'].max() - df['Mean_RMSE_Test'].min())
            df['Combined_Score'] = acc_norm - rmse_norm
            df = df.sort_values('Combined_Score', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
        
        # Save full ranking
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved full ranking to: {output_csv}")
        print()
    
    # Display top N
    print(f"TOP {args.top_n} MODELS (Ranked by Out-of-Sample {args.metric.upper()}):")
    print("-"*80)
    top_df = df.head(args.top_n)
    print(top_df[['Rank', 'Model', 'Mean_ACC_Train', 'Mean_RMSE_Train', 'Mean_ACC_Test', 'Mean_RMSE_Test']].to_string(index=False))
    print()
    
    # Plot comparative bar charts (in-sample vs out-of-sample)
    print("Generating comparative bar plots...")
    
    df_sorted = df.sort_values('Rank')
    
    # Dual bar plot: in-sample vs out-of-sample
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.4 * len(df)), 8))
    
    x = np.arange(len(df_sorted))
    width = 0.35
    
    # ACC comparison
    axes[0].bar(x - width/2, df_sorted['Mean_ACC_Train'].values, width, 
               label='In-sample (train)', color='tab:blue', alpha=0.7)
    axes[0].bar(x + width/2, df_sorted['Mean_ACC_Test'].values, width,
               label='Out-of-sample (test)', color='tab:orange', alpha=0.7)
    axes[0].set_ylabel('Mean ACC', fontsize=11)
    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].set_title('ACC: In-Sample vs Out-of-Sample', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_sorted['Model'].values, rotation=45, ha='right', fontsize=7)
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # RMSE comparison
    axes[1].bar(x - width/2, df_sorted['Mean_RMSE_Train'].values, width,
               label='In-sample (train)', color='tab:blue', alpha=0.7)
    axes[1].bar(x + width/2, df_sorted['Mean_RMSE_Test'].values, width,
               label='Out-of-sample (test)', color='tab:orange', alpha=0.7)
    axes[1].set_ylabel('Mean RMSE (°C)', fontsize=11)
    axes[1].set_xlabel('Model', fontsize=11)
    axes[1].set_title('RMSE: In-Sample vs Out-of-Sample', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_sorted['Model'].values, rotation=45, ha='right', fontsize=7)
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    bar_plot_path = f'{args.output_dir}/all_variants_comparison_{args.metric}_out_of_sample.png'
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    
    print(f"  ✓ Saved comparison bar plot: {bar_plot_path}")
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
        
        # Plot ACC - TRAIN period (in-sample)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for idx, row in plot_rows.iterrows():
            label = row['Model']
            if label in all_results:
                result = all_results[label]
                acc_vals = result['acc_train']['Nino34'].values
                lead_vals = result['acc_train']['Nino34'].lead.values
                
                # Style XRO baselines differently
                if label in xro_baseline_labels:
                    ax.plot(lead_vals, acc_vals, label=f"{row['Rank']}. {label}", 
                           lw=2.5, linestyle='--', alpha=0.8)
                else:
                    ax.plot(lead_vals, acc_vals, label=f"{row['Rank']}. {label}", 
                           lw=2, marker='o', markersize=3)
        
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'Top {args.top_n} + XRO Baselines: ACC (In-Sample)', fontsize=12)
        ax.set_xlim([0, 21])
        ax.set_ylim([0.2, 1.0])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/top{args.top_n}_acc_train_out_of_sample.png', dpi=300)
        plt.close()
        
        # Plot ACC - TEST period (out-of-sample)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for idx, row in plot_rows.iterrows():
            label = row['Model']
            if label in all_results:
                result = all_results[label]
                acc_vals = result['acc_test']['Nino34'].values
                lead_vals = result['acc_test']['Nino34'].lead.values
                
                if label in xro_baseline_labels:
                    ax.plot(lead_vals, acc_vals, label=f"{row['Rank']}. {label}", 
                           lw=2.5, linestyle='--', alpha=0.8)
                else:
                    ax.plot(lead_vals, acc_vals, label=f"{row['Rank']}. {label}", 
                           lw=2, marker='o', markersize=3)
        
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'Top {args.top_n} + XRO Baselines: ACC (Out-of-Sample)', fontsize=12)
        ax.set_xlim([0, 21])
        ax.set_ylim([0.2, 1.0])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/top{args.top_n}_acc_test_out_of_sample.png', dpi=300)
        plt.close()
        
        # Plot RMSE - TRAIN period (in-sample)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for idx, row in plot_rows.iterrows():
            label = row['Model']
            if label in all_results:
                result = all_results[label]
                rmse_vals = result['rmse_train']['Nino34'].values
                lead_vals = result['rmse_train']['Nino34'].lead.values
                
                if label in xro_baseline_labels:
                    ax.plot(lead_vals, rmse_vals, label=f"{row['Rank']}. {label}", 
                           lw=2.5, linestyle='--', alpha=0.8)
                else:
                    ax.plot(lead_vals, rmse_vals, label=f"{row['Rank']}. {label}", 
                           lw=2, marker='o', markersize=3)
        
        ax.set_ylabel('RMSE (°C)', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'Top {args.top_n} + XRO Baselines: RMSE (In-Sample)', fontsize=12)
        ax.set_xlim([0, 21])
        ax.set_ylim([0.0, 1.0])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/top{args.top_n}_rmse_train_out_of_sample.png', dpi=300)
        plt.close()
        
        # Plot RMSE - TEST period (out-of-sample)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for idx, row in plot_rows.iterrows():
            label = row['Model']
            if label in all_results:
                result = all_results[label]
                rmse_vals = result['rmse_test']['Nino34'].values
                lead_vals = result['rmse_test']['Nino34'].lead.values
                
                if label in xro_baseline_labels:
                    ax.plot(lead_vals, rmse_vals, label=f"{row['Rank']}. {label}", 
                           lw=2.5, linestyle='--', alpha=0.8)
                else:
                    ax.plot(lead_vals, rmse_vals, label=f"{row['Rank']}. {label}", 
                           lw=2, marker='o', markersize=3)
        
        ax.set_ylabel('RMSE (°C)', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title(f'Top {args.top_n} + XRO Baselines: RMSE (Out-of-Sample)', fontsize=12)
        ax.set_xlim([0, 21])
        ax.set_ylim([0.0, 1.0])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/top{args.top_n}_rmse_test_out_of_sample.png', dpi=300)
        plt.close()
        
        print(f"  ✓ Saved ACC train plot: {args.output_dir}/top{args.top_n}_acc_train_out_of_sample.png")
        print(f"  ✓ Saved ACC test plot: {args.output_dir}/top{args.top_n}_acc_test_out_of_sample.png")
        print(f"  ✓ Saved RMSE train plot: {args.output_dir}/top{args.top_n}_rmse_train_out_of_sample.png")
        print(f"  ✓ Saved RMSE test plot: {args.output_dir}/top{args.top_n}_rmse_test_out_of_sample.png")
        print()
    elif use_cache:
        print("  ⊘ Skipping detailed skill curve plots (using cached data)")
        print("    Run with --force to regenerate with full forecast data")
        print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS (OUT-OF-SAMPLE EXPERIMENT)")
    print("="*80)
    print(f"Total models evaluated: {len(df)}")
    xro_count = df['Model'].isin(['XRO', 'XRO_ac0', 'Linear XRO']).sum()
    print(f"  - XRO baselines: {xro_count}")
    print(f"  - NXRO variants: {len(df) - xro_count}")
    print()
    
    # Show XRO baseline rankings
    xro_ranks = df[df['Model'].isin(['XRO', 'XRO_ac0', 'Linear XRO'])][['Rank', 'Model', 'Mean_ACC_Test', 'Mean_RMSE_Test']]
    if not xro_ranks.empty:
        print("XRO baseline rankings (out-of-sample):")
        print(xro_ranks.to_string(index=False))
        print()
    
    print(f"Best model by test ACC: {df.iloc[0]['Model']} (ACC={df.iloc[0]['Mean_ACC_Test']:.4f})")
    print(f"Best model by test RMSE: {df.sort_values('Mean_RMSE_Test').iloc[0]['Model']} (RMSE={df.sort_values('Mean_RMSE_Test').iloc[0]['Mean_RMSE_Test']:.4f})")
    print()
    print("="*80)
    print("✓ RANKING COMPLETE!")
    print("="*80)
    print()
    print("Output files:")
    print(f"  - {output_csv}")
    print(f"  - {bar_plot_path}")
    if not use_cache:
        print(f"  - {args.output_dir}/top{args.top_n}_acc_train_out_of_sample.png")
        print(f"  - {args.output_dir}/top{args.top_n}_acc_test_out_of_sample.png")
        print(f"  - {args.output_dir}/top{args.top_n}_rmse_train_out_of_sample.png")
        print(f"  - {args.output_dir}/top{args.top_n}_rmse_test_out_of_sample.png")
    print()


if __name__ == '__main__':
    main()

