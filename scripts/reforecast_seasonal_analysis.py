"""Load checkpoints from gate ablation and multiseed, reforecast, compute per-season RMSE.

Outputs a CSV with columns: model, condition, seed, season, lead, rmse

Usage:
    python scripts/reforecast_seasonal_analysis.py --experiment gate_ablation
    python scripts/reforecast_seasonal_analysis.py --experiment multiseed
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import xarray as xr

from utils.xro_utils import nxro_reforecast


def load_model_from_checkpoint(ckpt_path, device='cpu'):
    """Load a model from checkpoint, inferring model class from the state dict."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
    var_order = ckpt['var_order']
    n_vars = len(var_order)

    # Infer model type from state dict keys
    keys = set(state_dict.keys())

    if 'conv1.lin.weight' in keys or 'conv1.weight' in keys:
        # GraphPyG model
        from nxro.models import NXROGraphPyGModel, build_edge_index_from_corr
        from nxro.data import get_dataloaders as _gdl
        # Need edge_index — build from training data
        obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
        train_ds = obs_ds.sel(time=slice('1979-01', '1995-12'))
        arrs = np.stack([train_ds[v].values for v in var_order], axis=-1)
        corr = np.corrcoef(arrs.T)
        edge_index = build_edge_index_from_corr(torch.tensor(corr, dtype=torch.float32), top_k=3)
        has_gate = 'alpha_w' in keys
        model = NXROGraphPyGModel(n_vars=n_vars, k_max=2, edge_index=edge_index,
                                   hidden=16, disable_seasonal_gate=not has_gate)
        model.load_state_dict(state_dict)
    elif 'Wq.weight' in keys:
        # Attentive model
        from nxro.models import NXROAttentiveModel
        has_gate = 'alpha_w' in keys
        model = NXROAttentiveModel(n_vars=n_vars, k_max=2, d=16,
                                    disable_seasonal_gate=not has_gate)
        model.load_state_dict(state_dict)
    elif any('res_net' in k for k in keys):
        from nxro.models import NXROResModel
        model = NXROResModel(n_vars=n_vars, k_max=2, hidden=32)
        model.load_state_dict(state_dict)
    else:
        from nxro.models import NXROLinearModel
        model = NXROLinearModel(n_vars=n_vars, k_max=2)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model, var_order


def compute_seasonal_rmse(fcst_ds, obs_ds, var='Nino34', test_period=('2002-01', '2022-12')):
    """Compute RMSE by init season and lead."""
    obs = obs_ds[var]
    init_times = pd.to_datetime(fcst_ds.init.values)
    leads = fcst_ds.lead.values

    mask = (init_times >= pd.Timestamp(test_period[0])) & (init_times <= pd.Timestamp(test_period[1]))
    fcst_test = fcst_ds.isel(init=mask)
    init_test = init_times[mask]

    seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    rows = []

    for season_name, months in seasons.items():
        season_mask = np.isin(init_test.month, months)
        if season_mask.sum() == 0:
            continue
        fcst_season = fcst_test.isel(init=season_mask)
        init_season = init_test[season_mask]

        for iL, L in enumerate(leads):
            target_times = init_season + pd.DateOffset(months=int(L))
            fcst_vals = fcst_season[var].isel(lead=iL).values
            obs_vals = np.array([float(obs.sel(time=tt).values) if tt in obs.time.values else np.nan
                                 for tt in target_times])
            valid = np.isfinite(fcst_vals) & np.isfinite(obs_vals)
            if valid.sum() > 0:
                rmse = np.sqrt(np.mean((fcst_vals[valid] - obs_vals[valid])**2))
            else:
                rmse = np.nan
            rows.append({'season': season_name, 'lead': int(L), 'rmse': rmse, 'n_inits': int(valid.sum())})

    return pd.DataFrame(rows)


def process_experiment(experiment, device='cpu'):
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')

    if experiment == 'gate_ablation':
        results_dir = 'results_rebuttal_gate_ablation'
    else:
        results_dir = 'results_rebuttal_multiseed'

    # Find all checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(results_dir, '**', '*_best_*seed*.pt'), recursive=True))
    # Exclude stochastic/noise checkpoints
    ckpt_files = [f for f in ckpt_files if 'stochastic' not in f and 'noise' not in f and 'checkpoint' not in f]

    if not ckpt_files:
        print(f"No checkpoint files found in {results_dir}")
        return

    print(f"Found {len(ckpt_files)} checkpoints in {results_dir}")

    all_rows = []
    for ckpt_path in ckpt_files:
        basename = os.path.basename(ckpt_path)
        seed_match = re.search(r'seed(\d+)', basename)
        seed = int(seed_match.group(1)) if seed_match else 0

        # Determine model and condition from path/filename
        if 'attentive' in ckpt_path:
            model_name = 'attentive'
        elif 'graphpyg' in ckpt_path or 'graph_pyg' in ckpt_path:
            model_name = 'graph_pyg'
        elif 'res' in ckpt_path:
            model_name = 'res'
        elif 'linear' in ckpt_path:
            model_name = 'linear'
        elif 'pure_neural_ode' in ckpt_path:
            model_name = 'pure_neural_ode'
        elif 'pure_transformer' in ckpt_path:
            model_name = 'pure_transformer'
        else:
            model_name = 'unknown'

        condition = 'default'
        if 'no_gate' in basename:
            condition = 'no_gate'
        elif 'with_gate' in basename:
            condition = 'with_gate'

        print(f"  Processing {model_name} {condition} seed={seed}...")
        try:
            model, var_order = load_model_from_checkpoint(ckpt_path, device=device)
            fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device=device)
            df = compute_seasonal_rmse(fcst, obs_ds)
            df['model'] = model_name
            df['condition'] = condition
            df['seed'] = seed
            all_rows.append(df)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
        out_path = os.path.join(results_dir, 'seasonal_rmse_analysis.csv')
        result.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("PER-SEASON AVERAGE RMSE (averaged over leads and seeds)")
        print("=" * 80)
        summary = result.groupby(['model', 'condition', 'season'])['rmse'].agg(['mean', 'std', 'count'])
        print(summary.to_string())

        # Spring barrier focus: MAM at lead 3-9
        print("\n" + "=" * 80)
        print("SPRING BARRIER FOCUS: MAM inits, leads 3-9 months")
        print("=" * 80)
        spring = result[(result['season'] == 'MAM') & (result['lead'].between(3, 9))]
        spring_summary = spring.groupby(['model', 'condition'])['rmse'].agg(['mean', 'std'])
        print(spring_summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='gate_ablation',
                        choices=['gate_ablation', 'multiseed'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    process_experiment(args.experiment, device=args.device)
