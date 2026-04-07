"""Run classical baselines (ARIMA, VAR, Persistence, Climatology) on ORAS5 data.

Uses the same train/test split and evaluation protocol as NXRO experiments.
Outputs per-lead RMSE and ACC for direct comparison.

Usage:
    python scripts/run_classical_baselines.py
"""
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')


def persistence_forecast(obs_ds, var='Nino34', test_period=('2002-01', '2022-12'), max_lead=21):
    """Persistence baseline: forecast = last observed value."""
    obs = obs_ds[var]
    init_times = pd.to_datetime(obs.sel(time=slice(*test_period)).time.values)
    leads = np.arange(1, max_lead + 1)

    fcst_vals = np.full((len(init_times), len(leads)), np.nan)
    obs_vals = np.full((len(init_times), len(leads)), np.nan)

    for i, init_t in enumerate(init_times):
        init_val = float(obs.sel(time=init_t).values)
        for j, L in enumerate(leads):
            target_t = init_t + pd.DateOffset(months=int(L))
            fcst_vals[i, j] = init_val
            if target_t in obs.time.values:
                obs_vals[i, j] = float(obs.sel(time=target_t).values)

    rmse = np.array([np.sqrt(np.nanmean((fcst_vals[:, j] - obs_vals[:, j])**2)) for j in range(len(leads))])
    # ACC
    acc = np.array([np.corrcoef(fcst_vals[np.isfinite(obs_vals[:, j]), j],
                                 obs_vals[np.isfinite(obs_vals[:, j]), j])[0, 1]
                    if np.isfinite(obs_vals[:, j]).sum() > 2 else np.nan
                    for j in range(len(leads))])
    return leads, rmse, acc


def climatology_forecast(obs_ds, var='Nino34', train_period=('1979-01', '1995-12'),
                         test_period=('2002-01', '2022-12'), max_lead=21):
    """Climatology baseline: forecast = monthly mean from training period."""
    obs = obs_ds[var]
    train = obs.sel(time=slice(*train_period))
    monthly_mean = train.groupby('time.month').mean()

    init_times = pd.to_datetime(obs.sel(time=slice(*test_period)).time.values)
    leads = np.arange(1, max_lead + 1)

    fcst_vals = np.full((len(init_times), len(leads)), np.nan)
    obs_vals = np.full((len(init_times), len(leads)), np.nan)

    for i, init_t in enumerate(init_times):
        for j, L in enumerate(leads):
            target_t = init_t + pd.DateOffset(months=int(L))
            target_month = target_t.month
            fcst_vals[i, j] = float(monthly_mean.sel(month=target_month).values)
            if target_t in obs.time.values:
                obs_vals[i, j] = float(obs.sel(time=target_t).values)

    rmse = np.array([np.sqrt(np.nanmean((fcst_vals[:, j] - obs_vals[:, j])**2)) for j in range(len(leads))])
    acc = np.array([np.corrcoef(fcst_vals[np.isfinite(obs_vals[:, j]), j],
                                 obs_vals[np.isfinite(obs_vals[:, j]), j])[0, 1]
                    if np.isfinite(obs_vals[:, j]).sum() > 2 else np.nan
                    for j in range(len(leads))])
    return leads, rmse, acc


def arima_forecast(obs_ds, var='Nino34', train_period=('1979-01', '1995-12'),
                   test_period=('2002-01', '2022-12'), max_lead=21, order=(2, 0, 1)):
    """Univariate ARIMA on Nino3.4."""
    obs = obs_ds[var]
    train = obs.sel(time=slice(*train_period)).values
    test_times = pd.to_datetime(obs.sel(time=slice(*test_period)).time.values)
    all_data = obs.values
    all_times = pd.to_datetime(obs.time.values)
    leads = np.arange(1, max_lead + 1)

    fcst_vals = np.full((len(test_times), len(leads)), np.nan)
    obs_vals = np.full((len(test_times), len(leads)), np.nan)

    for i, init_t in enumerate(test_times):
        # Use all data up to init_t for fitting
        init_idx = np.where(all_times == init_t)[0]
        if len(init_idx) == 0:
            continue
        init_idx = init_idx[0]
        history = all_data[:init_idx + 1]

        try:
            model = ARIMA(history, order=order)
            fitted = model.fit()
            pred = fitted.forecast(steps=max_lead)
            for j, L in enumerate(leads):
                fcst_vals[i, j] = pred[j]
                target_t = init_t + pd.DateOffset(months=int(L))
                if target_t in obs.time.values:
                    obs_vals[i, j] = float(obs.sel(time=target_t).values)
        except Exception:
            continue

    rmse = np.array([np.sqrt(np.nanmean((fcst_vals[:, j] - obs_vals[:, j])**2)) for j in range(len(leads))])
    acc = np.array([np.corrcoef(fcst_vals[np.isfinite(obs_vals[:, j]), j],
                                 obs_vals[np.isfinite(obs_vals[:, j]), j])[0, 1]
                    if np.isfinite(obs_vals[:, j]).sum() > 2 else np.nan
                    for j in range(len(leads))])
    return leads, rmse, acc


def var_forecast(obs_ds, train_period=('1979-01', '1995-12'),
                 test_period=('2002-01', '2022-12'), max_lead=21, var_lags=3):
    """Multivariate VAR on all climate indices."""
    var_names = list(obs_ds.data_vars)
    sel_var = 'Nino34'
    sel_idx = var_names.index(sel_var)

    all_data = np.stack([obs_ds[v].values for v in var_names], axis=-1)
    all_times = pd.to_datetime(obs_ds.time.values)
    test_times = pd.to_datetime(obs_ds.sel(time=slice(*test_period)).time.values)
    leads = np.arange(1, max_lead + 1)

    fcst_vals = np.full((len(test_times), len(leads)), np.nan)
    obs_vals = np.full((len(test_times), len(leads)), np.nan)

    for i, init_t in enumerate(test_times):
        init_idx = np.where(all_times == init_t)[0]
        if len(init_idx) == 0:
            continue
        init_idx = init_idx[0]
        history = all_data[:init_idx + 1]

        try:
            model = VAR(history)
            fitted = model.fit(maxlags=var_lags, ic=None)
            pred = fitted.forecast(history[-var_lags:], steps=max_lead)
            for j, L in enumerate(leads):
                fcst_vals[i, j] = pred[j, sel_idx]
                target_t = init_t + pd.DateOffset(months=int(L))
                if target_t in obs_ds.time.values:
                    obs_vals[i, j] = float(obs_ds[sel_var].sel(time=target_t).values)
        except Exception as e:
            if i == 0:
                print(f'    VAR error at init {init_t}: {e}')
            continue

    rmse = np.array([np.sqrt(np.nanmean((fcst_vals[:, j] - obs_vals[:, j])**2)) for j in range(len(leads))])
    acc = np.array([np.corrcoef(fcst_vals[np.isfinite(obs_vals[:, j]), j],
                                 obs_vals[np.isfinite(obs_vals[:, j]), j])[0, 1]
                    if np.isfinite(obs_vals[:, j]).sum() > 2 else np.nan
                    for j in range(len(leads))])
    return leads, rmse, acc


def main():
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    train_period = ('1979-01', '1995-12')
    test_period = ('2002-01', '2022-12')

    results = {}

    print('Running persistence baseline...')
    leads, rmse, acc = persistence_forecast(obs_ds, test_period=test_period)
    results['Persistence'] = {'leads': leads, 'rmse': rmse, 'acc': acc}
    print(f'  Avg RMSE: {np.nanmean(rmse):.4f}')

    print('Running climatology baseline...')
    leads, rmse, acc = climatology_forecast(obs_ds, train_period=train_period, test_period=test_period)
    results['Climatology'] = {'leads': leads, 'rmse': rmse, 'acc': acc}
    print(f'  Avg RMSE: {np.nanmean(rmse):.4f}')

    print('Running ARIMA(2,0,1) baseline...')
    leads, rmse, acc = arima_forecast(obs_ds, train_period=train_period, test_period=test_period, order=(2, 0, 1))
    results['ARIMA(2,0,1)'] = {'leads': leads, 'rmse': rmse, 'acc': acc}
    print(f'  Avg RMSE: {np.nanmean(rmse):.4f}')

    print('Running VAR(3) baseline...')
    leads, rmse, acc = var_forecast(obs_ds, train_period=train_period, test_period=test_period, var_lags=3)
    results['VAR(3)'] = {'leads': leads, 'rmse': rmse, 'acc': acc}
    print(f'  Avg RMSE: {np.nanmean(rmse):.4f}')

    # Save results
    rows = []
    for name, data in results.items():
        for j, L in enumerate(data['leads']):
            rows.append({'model': name, 'lead': int(L), 'rmse': data['rmse'][j], 'acc': data['acc'][j]})
    df = pd.DataFrame(rows)
    df.to_csv('results_rebuttal_classical_baselines.csv', index=False)
    print(f'\nSaved results_rebuttal_classical_baselines.csv')

    # Print summary
    print(f"\n{'Model':20s} {'Avg RMSE':>10s}  Selected leads RMSE (3, 6, 9, 12, 15, 18, 21)")
    print('-' * 80)
    for name, data in results.items():
        avg = np.nanmean(data['rmse'])
        sel = ' '.join([f"{data['rmse'][l-1]:.3f}" for l in [3, 6, 9, 12, 15, 18, 21] if l - 1 < len(data['rmse'])])
        print(f"{name:20s} {avg:10.4f}  {sel}")


if __name__ == '__main__':
    main()
