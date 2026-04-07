"""Aggregate rebuttal experiment results across seeds.

Reads *_summary.json files and forecast NetCDFs from results_rebuttal_*/
and produces tables with mean ± std for RMSE, ACC, and CRPS.

Usage:
    conda run -n signature python scripts/aggregate_rebuttal_results.py --experiment multiseed
    conda run -n signature python scripts/aggregate_rebuttal_results.py --experiment gate_ablation
    conda run -n signature python scripts/aggregate_rebuttal_results.py --experiment stochastic_ablation
    conda run -n signature python scripts/aggregate_rebuttal_results.py --experiment data_scarcity
    conda run -n signature python scripts/aggregate_rebuttal_results.py --experiment all
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from utils.xro_utils import calc_forecast_skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_summary_files(results_dir):
    """Find all *_summary.json files under a results directory."""
    return sorted(glob.glob(os.path.join(results_dir, '**', '*_summary.json'), recursive=True))


def parse_model_seed_from_path(path, results_dir):
    """Extract (model_name, seed, extra_tag) from a summary file path."""
    rel = os.path.relpath(path, results_dir)
    parts = rel.split(os.sep)
    model_name = parts[0] if parts else 'unknown'

    # Extract seed from filename (pattern: *_seed<N>_summary.json)
    basename = os.path.basename(path)
    seed_match = re.search(r'seed(\d+)', basename)
    seed = int(seed_match.group(1)) if seed_match else None

    # Extract any extra tag (gate, noise, size)
    tag_match = re.search(r'_(with_gate|no_gate|posthoc|stage2|\d+yr)', basename)
    tag = tag_match.group(1) if tag_match else ''

    return model_name, seed, tag


def load_forecast_skill(results_dir, model_subdir, nc_pattern, obs_path,
                        test_period, sel_var='Nino34'):
    """Load a forecast NetCDF and compute RMSE/ACC by lead."""
    nc_files = glob.glob(os.path.join(results_dir, model_subdir, nc_pattern))
    if not nc_files:
        return None, None

    obs_ds = xr.open_dataset(obs_path)

    all_rmse, all_acc = [], []
    for nc_file in nc_files:
        try:
            fcst = xr.open_dataset(nc_file)
            rmse = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=test_period)
            acc = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=test_period)
            all_rmse.append(rmse[sel_var].values)
            all_acc.append(acc[sel_var].values)
        except Exception as e:
            print(f"  Warning: Could not process {nc_file}: {e}")
            continue

    if not all_rmse:
        return None, None
    return np.array(all_rmse), np.array(all_acc)


# ---------------------------------------------------------------------------
# Experiment-specific aggregators
# ---------------------------------------------------------------------------

def aggregate_multiseed(results_dir='results_rebuttal_multiseed',
                        obs_path='data/XRO_indices_oras5.nc'):
    """Aggregate multi-seed experiment: mean ± std of test RMSE per model."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Multi-Seed Core Models (Train/Val/Test Split)")
    print("="*80)

    summaries = find_summary_files(results_dir)
    if not summaries:
        print(f"  No results found in {results_dir}/")
        return

    # Group by model
    model_data = {}
    for sf in summaries:
        model, seed, _ = parse_model_seed_from_path(sf, results_dir)
        with open(sf) as f:
            data = json.load(f)
        if model not in model_data:
            model_data[model] = []
        model_data[model].append({
            'seed': seed,
            'best_test_rmse': data.get('best_test_rmse'),
            'best_epoch': data.get('best_epoch'),
            'selection_on': data.get('history', {}).get('selection_on', 'unknown'),
        })

    # Summary table
    rows = []
    for model in sorted(model_data.keys()):
        rmses = [d['best_test_rmse'] for d in model_data[model] if d['best_test_rmse'] is not None]
        epochs = [d['best_epoch'] for d in model_data[model] if d['best_epoch'] is not None]
        if rmses:
            rows.append({
                'Model': model,
                'N_seeds': len(rmses),
                'RMSE_mean': np.mean(rmses),
                'RMSE_std': np.std(rmses),
                'RMSE_min': np.min(rmses),
                'RMSE_max': np.max(rmses),
                'Epoch_mean': np.mean(epochs) if epochs else None,
                'Selection': model_data[model][0].get('selection_on', ''),
            })

    df = pd.DataFrame(rows)
    print("\nBest Test RMSE (1-step, across seeds):")
    print(df.to_string(index=False, float_format='%.4f'))

    out_path = os.path.join(results_dir, 'multiseed_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also compute per-lead RMSE if forecast files exist
    test_period = slice('2002-01', '2022-12')
    print("\nPer-lead RMSE (mean ± std across seeds):")
    for model in sorted(model_data.keys()):
        nc_pattern = f'*seed*_forecasts*.nc'
        rmse_arr, acc_arr = load_forecast_skill(
            results_dir, model, nc_pattern, obs_path, test_period)
        if rmse_arr is not None and len(rmse_arr) > 1:
            mean = rmse_arr.mean(axis=0)
            std = rmse_arr.std(axis=0)
            avg_mean = mean.mean()
            avg_std = std.mean()
            print(f"  {model:20s}: avg RMSE = {avg_mean:.4f} ± {avg_std:.4f} (n={len(rmse_arr)})")


def aggregate_gate_ablation(results_dir='results_rebuttal_gate_ablation'):
    """Aggregate seasonal gate ablation results."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Seasonal Gate Ablation (SPB Analysis)")
    print("="*80)

    summaries = find_summary_files(results_dir)
    if not summaries:
        print(f"  No results found in {results_dir}/")
        return

    # Group by (model, gate_condition)
    groups = {}
    for sf in summaries:
        model, seed, tag = parse_model_seed_from_path(sf, results_dir)
        key = (model, tag or 'with_gate')
        with open(sf) as f:
            data = json.load(f)
        if key not in groups:
            groups[key] = []
        groups[key].append(data.get('best_test_rmse'))

    rows = []
    for (model, condition), rmses in sorted(groups.items()):
        rmses = [r for r in rmses if r is not None]
        if rmses:
            rows.append({
                'Model': model,
                'Condition': condition,
                'N': len(rmses),
                'RMSE_mean': np.mean(rmses),
                'RMSE_std': np.std(rmses),
            })

    df = pd.DataFrame(rows)
    print("\nGate Ablation Results:")
    print(df.to_string(index=False, float_format='%.4f'))

    out_path = os.path.join(results_dir, 'gate_ablation_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


def aggregate_stochastic_ablation(results_dir='results_rebuttal_stochastic_ablation'):
    """Aggregate stochastic ablation: post-hoc vs stage2 noise."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Stochastic Ablation (Drift vs Noise)")
    print("="*80)

    summaries = find_summary_files(results_dir)
    if not summaries:
        print(f"  No results found in {results_dir}/")
        return

    # Also look for stochastic eval files
    eval_files = sorted(glob.glob(os.path.join(results_dir, '**', '*_eval*.json'), recursive=True))

    groups = {}
    for sf in summaries:
        model, seed, tag = parse_model_seed_from_path(sf, results_dir)
        key = (model, tag or 'posthoc')
        with open(sf) as f:
            data = json.load(f)
        if key not in groups:
            groups[key] = []
        groups[key].append({
            'seed': seed,
            'det_rmse': data.get('best_test_rmse'),
        })

    rows = []
    for (model, noise_mode), runs in sorted(groups.items()):
        det_rmses = [r['det_rmse'] for r in runs if r['det_rmse'] is not None]
        if det_rmses:
            rows.append({
                'Model': model,
                'Noise_Mode': noise_mode,
                'N': len(det_rmses),
                'Det_RMSE_mean': np.mean(det_rmses),
                'Det_RMSE_std': np.std(det_rmses),
            })

    df = pd.DataFrame(rows)
    print("\nStochastic Ablation Results (Deterministic RMSE):")
    print(df.to_string(index=False, float_format='%.4f'))

    out_path = os.path.join(results_dir, 'stochastic_ablation_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


def aggregate_data_scarcity(results_dir='results_rebuttal_data_scarcity'):
    """Aggregate data scarcity experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Data Scarcity Curve")
    print("="*80)

    summaries = find_summary_files(results_dir)
    if not summaries:
        print(f"  No results found in {results_dir}/")
        return

    groups = {}
    for sf in summaries:
        model, seed, tag = parse_model_seed_from_path(sf, results_dir)
        size_tag = tag if tag else 'unknown'
        key = (model, size_tag)
        with open(sf) as f:
            data = json.load(f)
        if key not in groups:
            groups[key] = []
        groups[key].append(data.get('best_test_rmse'))

    rows = []
    for (model, size), rmses in sorted(groups.items()):
        rmses = [r for r in rmses if r is not None]
        if rmses:
            rows.append({
                'Model': model,
                'Train_Size': size,
                'N': len(rmses),
                'RMSE_mean': np.mean(rmses),
                'RMSE_std': np.std(rmses),
            })

    df = pd.DataFrame(rows)

    # Pivot for easier reading
    print("\nData Scarcity Results:")
    if not df.empty:
        pivot = df.pivot_table(index='Model', columns='Train_Size',
                               values='RMSE_mean', aggfunc='first')
        # Reorder columns
        size_order = ['10yr', '13yr', '16yr', '19yr', '23yr']
        cols = [c for c in size_order if c in pivot.columns]
        if cols:
            pivot = pivot[cols]
        print(pivot.to_string(float_format='%.4f'))

    out_path = os.path.join(results_dir, 'data_scarcity_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Aggregate rebuttal experiment results')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['multiseed', 'gate_ablation', 'stochastic_ablation',
                                 'data_scarcity', 'all'])
    parser.add_argument('--obs_path', type=str, default='data/XRO_indices_oras5.nc')
    args = parser.parse_args()

    if args.experiment in ('multiseed', 'all'):
        aggregate_multiseed(obs_path=args.obs_path)
    if args.experiment in ('gate_ablation', 'all'):
        aggregate_gate_ablation()
    if args.experiment in ('stochastic_ablation', 'all'):
        aggregate_stochastic_ablation()
    if args.experiment in ('data_scarcity', 'all'):
        aggregate_data_scarcity()

    print("\n" + "="*80)
    print("Aggregation complete.")
    print("="*80)


if __name__ == '__main__':
    main()
