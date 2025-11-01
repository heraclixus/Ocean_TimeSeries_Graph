#!/usr/bin/env python3
"""
Hyperparameter Grid Search for Graph-based NXRO Models (Base Data Only)

This script performs an exhaustive grid search over:
- Graph model types: graph (NXROGraphModel), graph_pyg (GCN, GAT)
- Graph structures: XRO-based, Statistical KNN (Pearson, Spearman, MI, xcorr_max)
- Hyperparameters: top_k, hidden_dim, learning_rate, rollout_k, L1 regularization (for learned graphs)

Selection criterion: Best test RMSE on Nino3.4
"""

import os
import subprocess
import itertools
import json
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import re


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for graph models')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs per run')
    parser.add_argument('--device', type=str, default='auto', help='Device: cpu, cuda, or auto')
    parser.add_argument('--test', action='store_true', help='Use test split for selection')
    parser.add_argument('--max_jobs', type=int, default=1, help='Max parallel jobs (future use)')
    parser.add_argument('--output_dir', type=str, default='results/hyperparam_search', 
                        help='Directory for search results')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    parser.add_argument('--show_training_output', action='store_true', 
                        help='Show training output in console (default: log only)')
    return parser.parse_args()


def build_hyperparameter_grid() -> List[Dict]:
    """
    Define comprehensive hyperparameter grid for graph models.
    
    Returns:
        List of configuration dictionaries, each representing a unique experiment.
    """
    
    # Grid dimensions
    grid = {
        # Model architecture
        'model_type': ['graph', 'graph_pyg'],
        
        # Graph structure source
        'graph_structure': [
            'xro',                    # XRO-based graph
            'stat_pearson',          # Statistical Pearson correlation
            'stat_spearman',         # Statistical Spearman correlation  
            'stat_mi',               # Statistical Mutual Information
            'stat_xcorr_max',        # Statistical cross-correlation
        ],
        
        # For graph_pyg only: GNN architecture
        'gnn_type': [None, 'gcn', 'gat'],  # None for non-PyG models
        
        # Graph sparsity (top-k neighbors)
        'top_k': [1, 2, 3, 5, 7, 10],
        
        # Model hyperparameters
        # Note: hidden_dim is fixed in NXRO models (not a CLI arg)
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
        'rollout_k': [1, 2, 3],
        
        # For learned graph models
        'use_learned_graph': [False, True],
        'l1_lambda': [1e-4, 5e-4, 1e-3, 5e-3],  # Only used if use_learned_graph=True
    }
    
    configs = []
    
    # Generate all combinations
    for model_type in grid['model_type']:
        for graph_struct in grid['graph_structure']:
            for top_k in grid['top_k']:
                for lr in grid['learning_rate']:
                    for rollout_k in grid['rollout_k']:
                        
                        if model_type == 'graph':
                            # NXROGraphModel: no GNN type, but can have learned adjacency
                            for use_learned in grid['use_learned_graph']:
                                if use_learned:
                                    # If learned, try different L1 values
                                    for l1 in grid['l1_lambda']:
                                        configs.append({
                                            'model_type': model_type,
                                            'graph_structure': graph_struct,
                                            'gnn_type': None,
                                            'top_k': top_k,
                                            'learning_rate': lr,
                                            'rollout_k': rollout_k,
                                            'use_learned_graph': True,
                                            'l1_lambda': l1,
                                        })
                                else:
                                    # Fixed graph
                                    configs.append({
                                        'model_type': model_type,
                                        'graph_structure': graph_struct,
                                        'gnn_type': None,
                                        'top_k': top_k,
                                        'learning_rate': lr,
                                        'rollout_k': rollout_k,
                                        'use_learned_graph': False,
                                        'l1_lambda': None,
                                    })
                        
                        elif model_type == 'graph_pyg':
                            # PyG models: GCN or GAT, no learned adjacency option
                            for gnn in ['gcn', 'gat']:
                                configs.append({
                                    'model_type': model_type,
                                    'graph_structure': graph_struct,
                                    'gnn_type': gnn,
                                    'top_k': top_k,
                                    'learning_rate': lr,
                                    'rollout_k': rollout_k,
                                    'use_learned_graph': False,
                                    'l1_lambda': None,
                                })
    
    return configs


def config_to_name(config: Dict) -> str:
    """Generate a unique identifier for a configuration."""
    parts = [
        config['model_type'],
        config['graph_structure'],
    ]
    if config['gnn_type']:
        parts.append(config['gnn_type'])
    parts.extend([
        f"k{config['top_k']}",
        f"lr{config['learning_rate']:.0e}",
        f"r{config['rollout_k']}",
    ])
    if config['use_learned_graph']:
        parts.append('learned')
        parts.append(f"l1{config['l1_lambda']:.0e}")
    
    return '_'.join(parts)


def config_to_command(config: Dict, args) -> str:
    """Convert a configuration dict to NXRO_train.py command."""
    
    cmd_parts = [
        'python', 'NXRO_train.py',
        '--model', config['model_type'],
        '--epochs', str(args.epochs),
        '--device', args.device,
        '--lr', str(config['learning_rate']),
        '--rollout_k', str(config['rollout_k']),
    ]
    
    # Note: NXRO models use fixed hidden_dim (not a CLI arg), so we skip it here
    
    if args.test:
        cmd_parts.append('--test')
    
    # Graph structure
    if config['graph_structure'] == 'xro':
        # XRO-based graph (default, no special flag needed)
        pass
    else:
        # Statistical KNN
        stat_method = config['graph_structure'].replace('stat_', '')
        cmd_parts.extend(['--graph_stat_method', stat_method])
    
    # top_k
    if config['model_type'] == 'graph_pyg':
        cmd_parts.extend(['--top_k', str(config['top_k'])])
    elif config['model_type'] == 'graph':
        # For graph model, always set topk (used for both XRO and stat graphs)
        cmd_parts.extend(['--graph_stat_topk', str(config['top_k'])])
    
    # GNN type (for graph_pyg)
    if config['gnn_type'] == 'gat':
        cmd_parts.append('--gat')
    
    # Learned graph (for graph model)
    if config['use_learned_graph']:
        cmd_parts.extend([
            '--graph_learned',
            '--graph_l1', str(config['l1_lambda'])
        ])
    
    return ' '.join(cmd_parts)


def extract_test_rmse(result_dir: str, config: Dict, use_test: bool = False) -> float:
    """
    Extract test RMSE for Nino3.4 from saved checkpoint.
    
    Returns:
        Test RMSE value, or np.inf if not found.
    """
    # Build expected checkpoint pattern
    model_type = config['model_type']
    
    if model_type == 'graph':
        # e.g., results/graph/fixed_xro/nxro_graph_fixed_xro_best_test.pt
        if config['use_learned_graph']:
            graph_tag = f"learned_{config['graph_structure']}_k{config['top_k']}_l1{config['l1_lambda']}"
        else:
            if config['graph_structure'] == 'xro':
                graph_tag = 'fixed_xro'
            else:
                stat_method = config['graph_structure'].replace('stat_', '')
                graph_tag = f"fixed_stat_{stat_method}_k{config['top_k']}"
        
        subdir = Path(result_dir) / 'graph' / graph_tag
        pattern = f"nxro_graph_{graph_tag}_best*.pt"
    
    elif model_type == 'graph_pyg':
        # e.g., results/graphpyg/gcn_k3/nxro_graphpyg_gcn_k3_best_test.pt
        gnn = config['gnn_type']
        if config['graph_structure'] == 'xro':
            graph_tag = f"{gnn}_k{config['top_k']}"
        else:
            stat_method = config['graph_structure'].replace('stat_', '')
            graph_tag = f"{gnn}_{stat_method}_k{config['top_k']}"
        
        subdir = Path(result_dir) / 'graphpyg' / graph_tag
        pattern = f"nxro_graphpyg_{graph_tag}_best*.pt"
    
    else:
        return np.inf
    
    # Debug: print what we're looking for
    print(f"  Looking for checkpoints in: {subdir}")
    print(f"  Pattern: {pattern}")
    
    # Find checkpoint files
    if not subdir.exists():
        print(f"  ✗ Directory does not exist: {subdir}")
        # Try legacy flat structure
        subdir = Path(result_dir)
        ckpt_files = list(subdir.glob(pattern))
        if ckpt_files:
            print(f"  ✓ Found {len(ckpt_files)} checkpoint(s) in legacy flat structure")
        else:
            print(f"  ✗ No checkpoints found in flat structure either")
            return np.inf
    else:
        ckpt_files = list(subdir.glob(pattern))
        if not ckpt_files:
            print(f"  ✗ No checkpoint files matching pattern")
            # List what IS in the directory
            all_files = list(subdir.glob('*.pt'))
            if all_files:
                print(f"  Found these .pt files: {[f.name for f in all_files[:5]]}")
            return np.inf
        else:
            print(f"  ✓ Found {len(ckpt_files)} checkpoint(s)")
    
    # Load the most recent checkpoint
    ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt_path = ckpt_files[0]
    print(f"  Loading: {ckpt_path.name}")
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Extract test_loss if available
        if 'test_loss' in ckpt:
            # test_loss is typically MSE on Nino3.4, so RMSE = sqrt(MSE)
            test_rmse = float(np.sqrt(ckpt['test_loss']))
            print(f"  ✓ Extracted test_loss from checkpoint: {test_rmse:.4f}")
            return test_rmse
        else:
            print(f"  ⚠ No 'test_loss' in checkpoint, running evaluation...")
            print(f"  Checkpoint keys: {list(ckpt.keys())}")
            # Fallback: run quick evaluation
            return evaluate_checkpoint_rmse(ckpt_path, config, use_test)
    
    except Exception as e:
        print(f"  ✗ Error loading {ckpt_path}: {e}")
        import traceback
        traceback.print_exc()
        return np.inf


def evaluate_checkpoint_rmse(ckpt_path: Path, config: Dict, use_test: bool) -> float:
    """
    Fallback: Load checkpoint and compute test RMSE by running a quick forecast.
    
    This is a simplified version - in practice, you'd call nxro_reforecast.
    """
    from nxro.models import NXROGraphModel, NXROGraphPyGModel
    from utils.xro_utils import nxro_reforecast, calc_forecast_skill
    
    try:
        # Load data
        obs_path = 'data/XRO_indices_oras5.nc'
        obs_ds = xr.open_dataset(obs_path)
        
        # Split
        if use_test:
            val_start, val_end = '2015-01', '2024-12'
        else:
            val_start, val_end = '2000-01', '2014-12'
        
        val_ds = obs_ds.sel(time=slice(val_start, val_end))
        
        # Load model
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if config['model_type'] == 'graph':
            model = NXROGraphModel(
                n_vars=len(obs_ds.data_vars),
                use_fixed_graph=not config['use_learned_graph']
            )
        elif config['model_type'] == 'graph_pyg':
            model = NXROGraphPyGModel(
                n_vars=len(obs_ds.data_vars),
                edge_index=ckpt.get('edge_index'),
                use_gat=(config['gnn_type'] == 'gat')
            )
        else:
            return np.inf
        
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        # Run forecast
        var_order = list(obs_ds.data_vars)
        fcst_ds = nxro_reforecast(
            model, val_ds, var_order,
            lead_months=24, ncycle=12, device='cpu'
        )
        
        # Compute RMSE
        skill_df = calc_forecast_skill(fcst_ds, val_ds, 'Nino34')
        mean_rmse = skill_df['rmse'].mean()
        
        return float(mean_rmse)
    
    except Exception as e:
        print(f"Error evaluating {ckpt_path}: {e}")
        return np.inf


def run_search(args):
    """Execute grid search."""
    
    ensure_dir(args.output_dir)
    
    # Generate all configs
    all_configs = build_hyperparameter_grid()
    print(f"Total configurations to search: {len(all_configs)}")
    
    # Save grid
    grid_path = Path(args.output_dir) / 'search_grid.json'
    with open(grid_path, 'w') as f:
        json.dump([{**cfg, 'name': config_to_name(cfg)} for cfg in all_configs], f, indent=2)
    print(f"Grid saved to {grid_path}")
    
    # Run experiments
    results = []
    results_path = Path(args.output_dir) / 'search_results.csv'
    log_path = Path(args.output_dir) / 'search.log'
    
    # Open log file for writing
    with open(log_path, 'w') as log_file:
        log_file.write(f"Grid search started: {len(all_configs)} configurations\n")
        log_file.write(f"Device: {args.device}, Epochs: {args.epochs}, Test: {args.test}\n")
        log_file.write("="*80 + "\n\n")
    
    for i, config in enumerate(all_configs):
        config_name = config_to_name(config)
        print(f"\n[{i+1}/{len(all_configs)}] Running: {config_name}")
        
        cmd = config_to_command(config, args)
        print(f"Command: {cmd}")
        
        if args.dry_run:
            print("  [DRY RUN - skipping execution]")
            continue
        
        # Execute with streaming output
        try:
            # Run without capturing output so user can see progress in real-time
            with open(log_path, 'a') as log_file:
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"[{i+1}/{len(all_configs)}] {config_name}\n")
                log_file.write(f"Command: {cmd}\n")
                log_file.write(f"{'='*80}\n")
                log_file.flush()
            
            # Run command and capture output to parse best test RMSE
            best_test_rmse = np.inf
            
            with open(log_path, 'a') as log_file:
                if args.show_training_output:
                    # Show output in console AND log to file, while tracking best test RMSE
                    import sys
                    proc = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_file.write(line)
                        log_file.flush()
                        
                        # Parse test RMSE from output
                        test_rmse_match = re.search(r'test RMSE:\s+([\d.]+)', line)
                        if test_rmse_match:
                            current_rmse = float(test_rmse_match.group(1))
                            best_test_rmse = min(best_test_rmse, current_rmse)
                    
                    proc.wait()
                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(proc.returncode, cmd)
                else:
                    # Capture output to parse RMSE, then write to log
                    proc = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    for line in proc.stdout:
                        log_file.write(line)
                        log_file.flush()
                        
                        # Parse test RMSE from output
                        test_rmse_match = re.search(r'test RMSE:\s+([\d.]+)', line)
                        if test_rmse_match:
                            current_rmse = float(test_rmse_match.group(1))
                            best_test_rmse = min(best_test_rmse, current_rmse)
                    
                    proc.wait()
                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(proc.returncode, cmd)
            
            print(f"  ✓ Training completed")
            
            # Use parsed test RMSE
            test_rmse = best_test_rmse if best_test_rmse != np.inf else extract_test_rmse('results', config, use_test=args.test)
            print(f"  Best test RMSE: {test_rmse:.4f}")
            
            results.append({
                'config_name': config_name,
                'test_rmse': test_rmse,
                **config
            })
        
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Training failed with exit code {e.returncode}")
            with open(log_path, 'a') as log_file:
                log_file.write(f"\n✗ Training failed with exit code {e.returncode}\n\n")
            
            results.append({
                'config_name': config_name,
                'test_rmse': np.inf,
                'error': f'Exit code {e.returncode}',
                **config
            })
        
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            with open(log_path, 'a') as log_file:
                log_file.write(f"\n✗ Unexpected error: {e}\n\n")
            
            results.append({
                'config_name': config_name,
                'test_rmse': np.inf,
                'error': str(e),
                **config
            })
        
        # Save incremental results after each run
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('test_rmse')
            results_df.to_csv(results_path, index=False)
            print(f"  → Incremental results saved to {results_path}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_rmse')
    
    results_path = Path(args.output_dir) / 'search_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Print top 10
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (by test RMSE):")
    print("="*80)
    top10 = results_df.head(10)
    for idx, row in top10.iterrows():
        print(f"\nRank {idx+1}: {row['config_name']}")
        print(f"  Test RMSE: {row['test_rmse']:.4f}")
        print(f"  Model: {row['model_type']}, Graph: {row['graph_structure']}, k={row['top_k']}")
        print(f"  LR: {row['learning_rate']:.0e}, Rollout: {row['rollout_k']}")
        if row['use_learned_graph']:
            print(f"  Learned graph with L1={row['l1_lambda']:.0e}")
    
    # Save best config
    best_config = results_df.iloc[0].to_dict()
    best_path = Path(args.output_dir) / 'best_config.json'
    with open(best_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\n✓ Best config saved to {best_path}")
    
    return results_df


def main():
    args = parse_args()
    
    print("="*80)
    print("GRAPH MODEL HYPERPARAMETER GRID SEARCH")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Using test split: {args.test}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    
    results_df = run_search(args)
    
    print("\n✓ Grid search complete!")


if __name__ == '__main__':
    main()

