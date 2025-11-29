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
    NXROTransformerModel,
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
            'results_out_of_sample/transformer',
        ]
    
    all_checkpoints = []
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        # If ckpt_suffix is provided (e.g. "real_finetuned"), search for that specifically
        # otherwise use the standard pattern but potentially exclude "stage1" or "finetuned" if we are doing normal ranking
        
        if ckpt_suffix:
            pattern = f'{base_dir}/**/nxro_*_{ckpt_suffix}*.pt'
        else:
            pattern = f'{base_dir}/**/nxro_*_best*.pt'
            
        matches = glob.glob(pattern, recursive=True)
        
        # Filter out two-stage models if we are doing normal ranking (suffix='')
        if not ckpt_suffix:
            matches = [m for m in matches if 'real_finetuned' not in m and 'synthetic_pretrained' not in m]
            
        all_checkpoints.extend(matches)
    
    if ckpt_suffix:
        root_pattern = f'results_out_of_sample/nxro_*_{ckpt_suffix}*.pt'
    else:
        root_pattern = f'results_out_of_sample/nxro_*_best*.pt'
        
    matches = glob.glob(root_pattern)
    if not ckpt_suffix:
        matches = [m for m in matches if 'real_finetuned' not in m and 'synthetic_pretrained' not in m]
        
    all_checkpoints.extend(matches)
    
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
    
    elif 'transformer' in basename:
        return NXROTransformerModel, {'n_vars': n_vars, 'k_max': 2, 'd_model': 64, 
                                      'nhead': 4, 'num_layers': 2, 
                                      'dim_feedforward': 256, 'dropout': 0.1}
    
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
    label = label.replace('_real_finetuned', ' (Two-Stage)').replace('_synthetic_pretrained', ' (Stage 1)')
    
    for tag in ['_extra_data', '_sim100', '_sim50']:
        label = label.split(tag)[0]
    
    label = label.replace('_', ' ').title()
    label = label.replace('Ws', 'WS').replace('Fixl', 'FixL').replace('Fixro', 'FixRO')
    label = label.replace('Fixdiag', 'FixDiag').replace('Fixnl', 'FixNL')
    label = label.replace('Fixphysics', 'FixPhysics')
    label = label.replace('Rodiag', 'RO+Diag').replace('Resmix', 'ResidualMix')
    
    return label


def compute_usefulness_metrics(rmse_test, xro_rmse_test):
    """
    Compute usefulness metrics that evaluate operational forecast skill.
    
    A "useful" model must consistently outperform XRO at short-to-medium range (1-12 months).
    
    Args:
        rmse_test: xarray DataArray with RMSE values indexed by lead time
        xro_rmse_test: xarray DataArray with XRO's RMSE values indexed by lead time
    
    Returns:
        dict with:
            - consistency_score: Weighted fraction of leads 1-12 where model beats XRO [0-1]
            - rmse_improvement_1_12: Weighted average RMSE improvement over XRO for leads 1-12 (°C)
            - wins_1_7: Number of wins in months 1-7
            - wins_8_12: Number of wins in months 8-12
    """
    # Extract lead times 1-12 (indices 0-11 since lead starts at 1)
    leads_short = rmse_test['Nino34'].lead.values[:12]
    rmse_vals = rmse_test['Nino34'].values[:12]
    xro_vals = xro_rmse_test['Nino34'].values[:12]
    
    # Define weights: 1x for months 1-7, 2x for months 8-12
    weights = np.ones(12)
    weights[7:12] = 2.0  # Months 8-12 get 2x weight (indices 7-11)
    
    # Compute wins (model RMSE < XRO RMSE)
    wins = (rmse_vals < xro_vals).astype(float)
    wins_1_7 = int(np.sum(wins[:7]))
    wins_8_12 = int(np.sum(wins[7:12]))
    
    # Weighted consistency score
    weighted_wins = wins * weights
    consistency_score = float(np.sum(weighted_wins) / np.sum(weights))
    
    # Weighted average RMSE improvement (positive = better than XRO)
    rmse_diff = xro_vals - rmse_vals  # Positive when model is better
    weighted_improvement = np.sum(rmse_diff * weights) / np.sum(weights)
    rmse_improvement_1_12 = float(weighted_improvement)
    
    return {
        'consistency_score': consistency_score,
        'rmse_improvement_1_12': rmse_improvement_1_12,
        'wins_1_7': wins_1_7,
        'wins_8_12': wins_8_12,
    }


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
    parser.add_argument('--top_n', type=int, default=5, help='Number of top models to display/plot')
    parser.add_argument('--metric', type=str, choices=['acc', 'rmse', 'combined', 'usefulness'], default='rmse',
                       help='Ranking metric (evaluated on out-of-sample period)')
    parser.add_argument('--output_dir', type=str, default='results_out_of_sample/rankings',
                       help='Output directory for ranking results')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if CSV exists')
    parser.add_argument('--two_stage', action='store_true', help='Rank two-stage models (trained on synthetic then fine-tuned)')
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
    two_stage_suffix = '_two_stage' if args.two_stage else ''
    output_csv = f'{args.output_dir}/all_variants_ranked_{args.metric}_out_of_sample{two_stage_suffix}.csv'
    
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
        if args.two_stage:
            # For two-stage, look for "real_finetuned" specifically
            all_checkpoints = discover_all_checkpoints(ckpt_suffix='real_finetuned')
        else:
            # For standard, look for "best" and filter out two-stage ones
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
        
        # Get XRO baseline RMSE for usefulness metrics
        xro_rmse_test = xro_results['XRO']['rmse_test']
        
        # Create ranking dataframe with usefulness metrics
        ranking_data = []
        for label, result in all_results.items():
            row = {
                'Model': label,
                'Mean_ACC_Train': result['mean_acc_train'],
                'Mean_RMSE_Train': result['mean_rmse_train'],
                'Mean_ACC_Test': result['mean_acc_test'],
                'Mean_RMSE_Test': result['mean_rmse_test'],
                'Path': result.get('path', 'N/A'),
            }
            
            # Compute usefulness metrics (compare against XRO)
            usefulness = compute_usefulness_metrics(result['rmse_test'], xro_rmse_test)
            row['Consistency_Score'] = usefulness['consistency_score']
            row['RMSE_Improvement_1_12'] = usefulness['rmse_improvement_1_12']
            row['Wins_1_7'] = usefulness['wins_1_7']
            row['Wins_8_12'] = usefulness['wins_8_12']
            
            ranking_data.append(row)
        
        df = pd.DataFrame(ranking_data)
        
        # Rank based on OUT-OF-SAMPLE (test) performance
        if args.metric == 'acc':
            df = df.sort_values('Mean_ACC_Test', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
        elif args.metric == 'rmse':
            df = df.sort_values('Mean_RMSE_Test', ascending=True)
            df['Rank'] = range(1, len(df) + 1)
        elif args.metric == 'usefulness':
            # Rank by consistency score (higher is better), then by RMSE improvement
            df = df.sort_values(['Consistency_Score', 'RMSE_Improvement_1_12'], ascending=[False, False])
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
    
    # Display columns based on metric
    if args.metric == 'usefulness':
        display_cols = ['Rank', 'Model', 'Consistency_Score', 'RMSE_Improvement_1_12', 
                       'Wins_1_7', 'Wins_8_12', 'Mean_RMSE_Test']
    else:
        display_cols = ['Rank', 'Model', 'Mean_ACC_Train', 'Mean_RMSE_Train', 
                       'Mean_ACC_Test', 'Mean_RMSE_Test']
    
    print(top_df[display_cols].to_string(index=False))
    print()
    
    # Show usefulness summary for all models
    if not use_cache:
        print("="*80)
        print("USEFULNESS METRICS (Operational Forecast Skill vs XRO at Leads 1-12)")
        print("="*80)
        print("Consistency Score: Weighted fraction of leads where model beats XRO")
        print("                   (1x weight for months 1-7, 2x for months 8-12)")
        print("RMSE Improvement: Weighted average RMSE advantage over XRO (°C)")
        print("-"*80)
        
        # Show top 10 by usefulness
        df_useful = df.sort_values('Consistency_Score', ascending=False).head(10)
        print("\nTop 10 Most Useful Models (by Consistency Score):")
        print(df_useful[['Model', 'Consistency_Score', 'RMSE_Improvement_1_12', 'Wins_1_7', 'Wins_8_12', 'Mean_RMSE_Test']].to_string(index=False))
        print()
        
        # Identify truly useful models (consistency > 0.5 = win more than lose)
        useful_models = df[df['Consistency_Score'] > 0.5]
        print(f"\nModels with Consistency Score > 0.5 (beat XRO more often than not): {len(useful_models)}")
        if len(useful_models) > 0:
            print(useful_models[['Model', 'Consistency_Score', 'RMSE_Improvement_1_12', 'Mean_RMSE_Test']].to_string(index=False))
        print()
    
    print("="*80)
    
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
    two_stage_suffix = '_two_stage' if args.two_stage else ''
    bar_plot_path = f'{args.output_dir}/all_variants_comparison_{args.metric}_out_of_sample{two_stage_suffix}.png'
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    
    print(f"  ✓ Saved comparison bar plot: {bar_plot_path}")
    print()
    
    # Plot top N skill curves (only if we have full results, not from cache)
    if not use_cache and all_results is not None:
        print(f"Generating skill curve plots for top {args.top_n} vs XRO baseline...")
        
        # Only include XRO baseline (not XRO_ac0 or Linear XRO)
        xro_baseline_label = 'XRO'
        top_models = top_df['Model'].values
        
        # Get XRO data
        if xro_baseline_label not in all_results:
            print(f"  Warning: XRO baseline not found, skipping plots")
        else:
            xro_result = all_results[xro_baseline_label]
            
            # PLOT 1: Top N models + XRO (ACC, out-of-sample)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot XRO baseline
            xro_acc_vals = xro_result['acc_test']['Nino34'].values
            xro_lead_vals = xro_result['acc_test']['Nino34'].lead.values
            xro_rank = df[df['Model'] == xro_baseline_label]['Rank'].values[0]
            ax.plot(xro_lead_vals, xro_acc_vals, label=f"XRO (Rank {xro_rank})", 
                   lw=3, linestyle='--', color='black', alpha=0.8)
            
            # Plot top N models
            for idx, model_name in enumerate(top_models):
                if model_name in all_results and model_name != xro_baseline_label:
                    result = all_results[model_name]
                    acc_vals = result['acc_test']['Nino34'].values
                    lead_vals = result['acc_test']['Nino34'].lead.values
                    rank = df[df['Model'] == model_name]['Rank'].values[0]
                    ax.plot(lead_vals, acc_vals, label=f"{rank}. {model_name}", 
                           lw=2, marker='o', markersize=3)
            
            ax.set_ylabel('Correlation', fontsize=12)
            ax.set_xlabel('Forecast Lead (months)', fontsize=12)
            ax.set_title(f'Top {args.top_n} Models vs XRO: ACC (Out-of-Sample)', fontsize=13, fontweight='bold')
            ax.set_xlim([0, 21])
            ax.set_ylim([0.2, 1.0])
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{args.output_dir}/top{args.top_n}_vs_xro_acc_test{two_stage_suffix}.png', dpi=300)
            plt.close()
            
            # PLOT 2: Top N models + XRO (RMSE, out-of-sample)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot XRO baseline
            xro_rmse_vals = xro_result['rmse_test']['Nino34'].values
            xro_lead_vals = xro_result['rmse_test']['Nino34'].lead.values
            ax.plot(xro_lead_vals, xro_rmse_vals, label=f"XRO (Rank {xro_rank})", 
                   lw=3, linestyle='--', color='black', alpha=0.8)
            
            # Plot top N models
            for idx, model_name in enumerate(top_models):
                if model_name in all_results and model_name != xro_baseline_label:
                    result = all_results[model_name]
                    rmse_vals = result['rmse_test']['Nino34'].values
                    lead_vals = result['rmse_test']['Nino34'].lead.values
                    rank = df[df['Model'] == model_name]['Rank'].values[0]
                    ax.plot(lead_vals, rmse_vals, label=f"{rank}. {model_name}", 
                           lw=2, marker='o', markersize=3)
            
            ax.set_ylabel('RMSE (°C)', fontsize=12)
            ax.set_xlabel('Forecast Lead (months)', fontsize=12)
            ax.set_title(f'Top {args.top_n} Models vs XRO: RMSE (Out-of-Sample)', fontsize=13, fontweight='bold')
            ax.set_xlim([0, 21])
            ax.set_ylim([0.0, 1.0])
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{args.output_dir}/top{args.top_n}_vs_xro_rmse_test{two_stage_suffix}.png', dpi=300)
            plt.close()
            
            print(f"  ✓ Saved combined ACC plot: {args.output_dir}/top{args.top_n}_vs_xro_acc_test{two_stage_suffix}.png")
            print(f"  ✓ Saved combined RMSE plot: {args.output_dir}/top{args.top_n}_vs_xro_rmse_test{two_stage_suffix}.png")
            
            # PLOTS 3-12: Individual comparison plots (each top model vs XRO)
            print(f"\n  Generating individual comparison plots...")
            for idx, model_name in enumerate(top_models, 1):
                if model_name in all_results and model_name != xro_baseline_label:
                    result = all_results[model_name]
                    rank = df[df['Model'] == model_name]['Rank'].values[0]
                    
                    # Individual ACC plot
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    
                    # XRO
                    ax.plot(xro_lead_vals, xro_acc_vals, label=f"XRO (Rank {xro_rank})", 
                           lw=3, linestyle='--', color='gray', alpha=0.7)
                    
                    # Model
                    acc_vals = result['acc_test']['Nino34'].values
                    lead_vals = result['acc_test']['Nino34'].lead.values
                    ax.plot(lead_vals, acc_vals, label=f"{model_name} (Rank {rank})", 
                           lw=2.5, marker='o', markersize=4, color='C0')
                    
                    ax.set_ylabel('Correlation', fontsize=12)
                    ax.set_xlabel('Forecast Lead (months)', fontsize=12)
                    ax.set_title(f'Rank {rank}: {model_name} vs XRO - ACC (Out-of-Sample)', 
                               fontsize=13, fontweight='bold')
                    ax.set_xlim([0, 21])
                    ax.set_ylim([0.2, 1.0])
                    ax.legend(fontsize=10, loc='best')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    safe_name = model_name.replace(' ', '_').replace('+', 'plus').lower().replace('(', '').replace(')', '')
                    plt.savefig(f'{args.output_dir}/rank{rank}_{safe_name}_vs_xro_acc_test{two_stage_suffix}.png', dpi=300)
                    plt.close()
                    
                    # Individual RMSE plot
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    
                    # XRO
                    ax.plot(xro_lead_vals, xro_rmse_vals, label=f"XRO (Rank {xro_rank})", 
                           lw=3, linestyle='--', color='gray', alpha=0.7)
                    
                    # Model
                    rmse_vals = result['rmse_test']['Nino34'].values
                    lead_vals = result['rmse_test']['Nino34'].lead.values
                    ax.plot(lead_vals, rmse_vals, label=f"{model_name} (Rank {rank})", 
                           lw=2.5, marker='o', markersize=4, color='C1')
                    
                    ax.set_ylabel('RMSE (°C)', fontsize=12)
                    ax.set_xlabel('Forecast Lead (months)', fontsize=12)
                    ax.set_title(f'Rank {rank}: {model_name} vs XRO - RMSE (Out-of-Sample)', 
                               fontsize=13, fontweight='bold')
                    ax.set_xlim([0, 21])
                    ax.set_ylim([0.0, 1.0])
                    ax.legend(fontsize=10, loc='best')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    plt.savefig(f'{args.output_dir}/rank{rank}_{safe_name}_vs_xro_rmse_test{two_stage_suffix}.png', dpi=300)
                    plt.close()
                    
                    print(f"    ✓ Saved plots for Rank {rank}: {model_name}")
            
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
        print(f"  - {args.output_dir}/top{args.top_n}_vs_xro_acc_test{two_stage_suffix}.png (combined)")
        print(f"  - {args.output_dir}/top{args.top_n}_vs_xro_rmse_test{two_stage_suffix}.png (combined)")
        print(f"  - {args.output_dir}/rank*_vs_xro_*.png (individual comparisons)")
    print()


if __name__ == '__main__':
    main()

