"""Analyze seasonal gate effect on forecast skill by initialization month.

Reads forecast NetCDFs from the gate ablation and multiseed experiments,
computes RMSE grouped by initialization season (MAM=spring barrier period).

Usage:
    conda run -n signature python scripts/analyze_seasonal_gate_effect.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr


def compute_rmse_by_init_season(fcst_ds, obs_ds, var='Nino34', test_period=('2002-01', '2022-12')):
    """Compute RMSE by initialization season and lead time."""
    obs = obs_ds[var].sel(time=slice(*test_period))

    # Get init times and leads from forecast
    init_times = pd.to_datetime(fcst_ds.init.values)
    leads = fcst_ds.lead.values

    # Filter to test period
    mask = (init_times >= pd.Timestamp(test_period[0])) & (init_times <= pd.Timestamp(test_period[1]))
    fcst_test = fcst_ds.isel(init=mask)
    init_test = init_times[mask]

    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11],
    }

    results = {}
    for season_name, months in seasons.items():
        season_mask = np.isin(init_test.month, months)
        if season_mask.sum() == 0:
            continue

        fcst_season = fcst_test.isel(init=season_mask)
        init_season = init_test[season_mask]

        rmse_by_lead = []
        for iL, L in enumerate(leads):
            # Target times for this lead
            target_times = init_season + pd.DateOffset(months=int(L))

            # Get forecast and obs values
            fcst_vals = fcst_season[var].isel(lead=iL).values
            obs_vals = []
            valid_mask = []
            for tt in target_times:
                if tt in obs.time.values:
                    obs_vals.append(float(obs.sel(time=tt).values))
                    valid_mask.append(True)
                else:
                    obs_vals.append(np.nan)
                    valid_mask.append(False)

            obs_arr = np.array(obs_vals)
            valid = np.array(valid_mask) & np.isfinite(fcst_vals) & np.isfinite(obs_arr)

            if valid.sum() > 0:
                rmse = np.sqrt(np.mean((fcst_vals[valid] - obs_arr[valid])**2))
            else:
                rmse = np.nan
            rmse_by_lead.append(rmse)

        results[season_name] = np.array(rmse_by_lead)

    return results, leads


def main():
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    test_period = ('2002-01', '2022-12')

    # =========================================================================
    # 1. Gate ablation: compare with_gate vs no_gate by season
    # =========================================================================
    print("=" * 80)
    print("SEASONAL GATE EFFECT ON FORECAST SKILL")
    print("=" * 80)

    for model in ['attentive', 'graph_pyg']:
        model_dir = 'graphpyg' if model == 'graph_pyg' else model

        print(f"\n--- {model.upper()} ---")

        for condition in ['with_gate', 'no_gate']:
            # Find forecast NetCDFs
            base = f'results_rebuttal_gate_ablation/{model_dir}'
            if model == 'graph_pyg':
                base = f'results_rebuttal_gate_ablation/graphpyg/gcn_k3'
            pattern = os.path.join(base, f'*{condition}_seed*_forecasts*.nc')
            # Try various patterns
            nc_files = glob.glob(pattern)
            if not nc_files:
                pattern = os.path.join(f'results_rebuttal_gate_ablation/{model}', f'*{condition}_seed*_forecasts*.nc')
                nc_files = glob.glob(pattern)
            if not nc_files:
                # Try to find any forecast NC in the model dir
                for root, dirs, files in os.walk(f'results_rebuttal_gate_ablation'):
                    for fn in files:
                        if condition in fn and 'seed' in fn and fn.endswith('_forecasts.nc') and 'stochastic' not in fn:
                            nc_files.append(os.path.join(root, fn))

            if not nc_files:
                print(f"  {condition}: no forecast files found")
                continue

            all_season_rmse = {}
            for nc_file in nc_files:
                try:
                    fcst = xr.open_dataset(nc_file)
                    season_rmse, leads = compute_rmse_by_init_season(fcst, obs_ds, test_period=test_period)
                    for s, vals in season_rmse.items():
                        if s not in all_season_rmse:
                            all_season_rmse[s] = []
                        all_season_rmse[s].append(vals)
                except Exception as e:
                    print(f"  Warning: {nc_file}: {e}")
                    continue

            if all_season_rmse:
                print(f"\n  {condition} (n={len(nc_files)} seeds):")
                print(f"  {'Season':>8s}", end='')
                for L in [3, 6, 9, 12, 15, 18, 21]:
                    print(f"  lead={L:2d}", end='')
                print(f"  {'avg':>8s}")

                for season in ['DJF', 'MAM', 'JJA', 'SON']:
                    if season in all_season_rmse:
                        arr = np.array(all_season_rmse[season])  # [n_seeds, n_leads]
                        mean = arr.mean(axis=0)
                        print(f"  {season:>8s}", end='')
                        for L_idx in [2, 5, 8, 11, 14, 17, 20]:
                            if L_idx < len(mean):
                                print(f"  {mean[L_idx]:8.3f}", end='')
                        print(f"  {np.nanmean(mean):8.3f}")

    # =========================================================================
    # 2. Multiseed: compute per-season RMSE for attentive and graph_pyg
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-SEASON RMSE FROM MULTISEED (with gate, default)")
    print("=" * 80)

    for model in ['attentive', 'graph_pyg', 'linear']:
        model_dir = model
        if model == 'graph_pyg':
            model_dir = 'graphpyg/gcn_k3'

        base = f'results_rebuttal_multiseed/{model_dir}'
        nc_files = []
        for root, dirs, files in os.walk(f'results_rebuttal_multiseed'):
            for fn in files:
                if 'seed' in fn and fn.endswith('_forecasts.nc') and 'stochastic' not in fn:
                    full = os.path.join(root, fn)
                    if model_dir.split('/')[0] in full:
                        nc_files.append(full)

        if not nc_files:
            print(f"\n  {model}: no forecast files found")
            continue

        all_season_rmse = {}
        for nc_file in nc_files[:5]:  # Use up to 5 seeds
            try:
                fcst = xr.open_dataset(nc_file)
                season_rmse, leads = compute_rmse_by_init_season(fcst, obs_ds, test_period=test_period)
                for s, vals in season_rmse.items():
                    if s not in all_season_rmse:
                        all_season_rmse[s] = []
                    all_season_rmse[s].append(vals)
            except Exception:
                continue

        if all_season_rmse:
            print(f"\n  {model} (n={len(all_season_rmse.get('MAM', []))} seeds):")
            print(f"  {'Season':>8s}", end='')
            for L in [3, 6, 9, 12]:
                print(f"  lead={L:2d}", end='')
            print(f"  {'avg':>8s}")

            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                if season in all_season_rmse:
                    arr = np.array(all_season_rmse[season])
                    mean = arr.mean(axis=0)
                    print(f"  {season:>8s}", end='')
                    for L_idx in [2, 5, 8, 11]:
                        if L_idx < len(mean):
                            print(f"  {mean[L_idx]:8.3f}", end='')
                    print(f"  {np.nanmean(mean):8.3f}")


if __name__ == '__main__':
    main()
