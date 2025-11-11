#!/usr/bin/env python
"""
Rank ALL available NXRO variant checkpoints (OUT-OF-SAMPLE, ALL DATASETS).

This script ranks models from results_all_outsample/ directory.

Usage:
    python rank_all_variants_outsample.py --top_n 10 --metric rmse
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import glob

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
    """Discover all NXRO checkpoint files from out-of-sample all-datasets results."""
    if search_dirs is None:
        search_dirs = [
            'results_all_outsample/linear',
            'results_all_outsample/ro',
            'results_all_outsample/rodiag',
            'results_all_outsample/res',
            'results_all_outsample/res_fullxro',
            'results_all_outsample/neural',
            'results_all_outsample/neural_phys',
            'results_all_outsample/attentive',
            'results_all_outsample/bilinear',
            'results_all_outsample/resmix',
            'results_all_outsample/graph',
            'results_all_outsample/graphpyg',
        ]
    
    all_checkpoints = []
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        pattern = f'{base_dir}/**/nxro_*_best{ckpt_suffix}*.pt'
        matches = glob.glob(pattern, recursive=True)
        all_checkpoints.extend(matches)
    
    root_pattern = f'results_all_outsample/nxro_*_best{ckpt_suffix}*.pt'
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
        
        fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device='cpu')
        
        acc_train = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                        by_month=False, verify_periods=train_period)
        rmse_train = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                         by_month=False, verify_periods=train_period)
        
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
        print(f"  [X] Failed to load {os.path.basename(ckpt_path)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Rank NXRO variants (Out-of-Sample, All Datasets)')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top models to display/plot')
    parser.add_argument('--metric', type=str, choices=['acc', 'rmse', 'combined'], default='rmse',
                       help='Ranking metric (evaluated on out-of-sample period)')
    parser.add_argument('--output_dir', type=str, default='results_all_outsample/rankings',
                       help='Output directory for ranking results')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if CSV exists')
    args = parser.parse_args()
    
    train_start, train_end = '1979-01', '2001-12'
    test_start, test_end = '2002-01', '2022-12'
    
    train_period = slice(train_start, train_end)
    test_period = slice(test_start, test_end)
    
    ensure_dir(args.output_dir)
    
    print("="*80)
    print("RANKING ALL MODELS (OUT-OF-SAMPLE, ALL DATASETS)")
    print("="*80)
    print(f"Train period (in-sample): {train_start} to {train_end}")
    print(f"Test period (out-of-sample): {test_start} to {test_end}")
    print(f"Ranking metric: {args.metric} (evaluated on out-of-sample)")
    print(f"Top N to display: {args.top_n}")
    print("="*80)
    print()
    
    output_csv = f'{args.output_dir}/all_variants_ranked_{args.metric}_all_outsample.csv'
    
    use_cache = os.path.exists(output_csv) and not args.force
    
    if use_cache:
        print(f"[OK] Found existing ranking: {output_csv}")
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
        
        obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
        train_ds = obs_ds.sel(time=train_period)
        
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
            
            acc_train = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                           by_month=False, verify_periods=train_period)
            rmse_train = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                            by_month=False, verify_periods=train_period)
            
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
        
        print(f"  [OK] XRO baselines ready: {len(xro_results)} variants")
        print()
        
        print("Discovering NXRO variant checkpoints...")
        all_checkpoints = discover_all_checkpoints(ckpt_suffix='')
        print(f"  Found {len(all_checkpoints)} checkpoint files")
        print()
        
        print("Loading and evaluating variants...")
        nxro_results = {}
        for i, ckpt_path in enumerate(all_checkpoints, 1):
            basename = os.path.basename(ckpt_path)
            print(f"  [{i}/{len(all_checkpoints)}] Loading {basename}...", end=' ')
            
            result = load_and_evaluate_dual(ckpt_path, obs_ds, train_period, test_period)
            if result is not None:
                nxro_results[result['label']] = result
                print(f"[OK] Train ACC={result['mean_acc_train']:.3f}, Test ACC={result['mean_acc_test']:.3f}")
            else:
                print("[X] Failed")
        
        print()
        print(f"[OK] Successfully loaded {len(nxro_results)} NXRO variants")
        print()
        
        all_results = {**xro_results, **nxro_results}
        
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
        
        if args.metric == 'acc':
            df = df.sort_values('Mean_ACC_Test', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
        elif args.metric == 'rmse':
            df = df.sort_values('Mean_RMSE_Test', ascending=True)
            df['Rank'] = range(1, len(df) + 1)
        else:
            acc_norm = (df['Mean_ACC_Test'] - df['Mean_ACC_Test'].min()) / (df['Mean_ACC_Test'].max() - df['Mean_ACC_Test'].min())
            rmse_norm = (df['Mean_RMSE_Test'] - df['Mean_RMSE_Test'].min()) / (df['Mean_RMSE_Test'].max() - df['Mean_RMSE_Test'].min())
            df['Combined_Score'] = acc_norm - rmse_norm
            df = df.sort_values('Combined_Score', ascending=False)
            df['Rank'] = range(1, len(df) + 1)
        
        df.to_csv(output_csv, index=False)
        print(f"[OK] Saved full ranking to: {output_csv}")
        print()
    
    print(f"TOP {args.top_n} MODELS (Ranked by Out-of-Sample {args.metric.upper()}):")
    print("-"*80)
    top_df = df.head(args.top_n)
    print(top_df[['Rank', 'Model', 'Mean_ACC_Train', 'Mean_RMSE_Train', 'Mean_ACC_Test', 'Mean_RMSE_Test']].to_string(index=False))
    print()
    
    print("="*80)
    print("[OK] RANKING COMPLETE!")
    print("="*80)
    print()
    print("Output files:")
    print(f"  - {output_csv}")
    print()


if __name__ == '__main__':
    main()

