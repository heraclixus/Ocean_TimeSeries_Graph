import datetime
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from dateutil.relativedelta import relativedelta


def calc_forecast_skill(fcst_ds: xr.Dataset, ref_ds: xr.Dataset, metric: str = 'acc', is_mv3: bool = True,
                        comparison: str = "e2o", by_month: bool = False,
                        verify_periods: slice = slice('1979-01', '2022-12')) -> xr.Dataset:
    """Compute hindcast skill via climpred if available, else manual correlation/RMSE across init.

    Returns xr.Dataset with data_vars per variable and coord lead (months).
    """
    try:
        from climpred import HindcastEnsemble
    except Exception:
        HindcastEnsemble = None

    try:
        fcst_ds = fcst_ds.squeeze().drop('member')
    except Exception:
        pass

    if is_mv3:
        fcst_ds = fcst_ds.rolling(init=3, center=True, min_periods=1).mean('init')
        ref_mv3 = ref_ds.rolling(time=3, center=True, min_periods=1).mean().dropna(dim='time')
    else:
        ref_mv3 = ref_ds

    if HindcastEnsemble is not None:
        hc = HindcastEnsemble(fcst_ds.sel(init=verify_periods))
        try:
            hc = hc.add_observations(ref_mv3)
            skill = hc.verify(metric=metric, comparison=comparison, alignment="maximize",
                              dim=["init"], skipna=True, groupby='month' if by_month else None)
        except Exception:
            skill = hc.verify(observations=ref_mv3, metric=metric, comparison=comparison, alignment="maximize",
                              dim=["init"], skipna=True, groupby='month' if by_month else None)
        try:
            del skill.attrs['skipna']
            skill = skill.drop('skill')
        except Exception:
            pass
        for var in skill.data_vars:
            if var != 'model':
                skill[var].encoding['dtype'] = 'float32'
                skill[var].encoding['_FillValue'] = 1e20
        return skill

    # Manual fallback: correlate across init for each lead
    fcst_sel = fcst_ds.sel(init=verify_periods)
    leads = fcst_sel['lead'].values
    init_times = pd.DatetimeIndex(fcst_sel['init'].values)
    var_names = [v for v in fcst_sel.data_vars if v != 'model']
    results = {}
    for v in var_names:
        stats_per_lead = []
        for L in leads:
            fc = fcst_sel[v].sel(lead=L)
            target_times = init_times + pd.DateOffset(months=int(L))
            obs = ref_mv3[v].sel(time=xr.DataArray(target_times, dims='init'))
            fc, obs = xr.align(fc, obs, join='inner')
            if metric == 'acc':
                val = xr.corr(fc, obs, dim='init', skipna=True)
            elif metric == 'rmse':
                diff = fc - obs
                val = np.sqrt((diff ** 2).mean(dim='init', skipna=True))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            stats_per_lead.append(val)
        results[v] = xr.concat(stats_per_lead, dim='lead').assign_coords(lead=leads)
    return xr.Dataset(results)


def plot_forecast_plume(fcst_det: xr.Dataset, fcst_stoc: xr.Dataset, obs_ds: xr.Dataset,
                        date_arrs: List[str], fname_prefix: str = 'XRO_forecast_plume') -> None:
    for idx, sel_date in enumerate(date_arrs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        # Select deterministic and stochastic forecasts (Nino34)
        sel_fcst_d = fcst_det['Nino34'].sel(init=sel_date).squeeze()
        sel_fcst_m = fcst_stoc['Nino34'].sel(init=sel_date).mean('member').squeeze()
        sel_fcst_e = fcst_stoc['Nino34'].sel(init=sel_date).std('member').squeeze()
        # Align lead lengths across deterministic (may include t0) and stochastic
        nlead_d = int(getattr(sel_fcst_d, 'sizes', {}).get('lead', sel_fcst_d.size))
        nlead_m = int(getattr(sel_fcst_m, 'sizes', {}).get('lead', sel_fcst_m.size))
        nlead = min(nlead_d, nlead_m)
        if 'lead' in sel_fcst_d.dims:
            sel_fcst_d = sel_fcst_d.isel(lead=slice(0, nlead))
        else:
            sel_fcst_d = sel_fcst_d[..., :nlead]
        if 'lead' in sel_fcst_m.dims:
            sel_fcst_m = sel_fcst_m.isel(lead=slice(0, nlead))
            sel_fcst_e = sel_fcst_e.isel(lead=slice(0, nlead))
        else:
            sel_fcst_m = sel_fcst_m[..., :nlead]
            sel_fcst_e = sel_fcst_e[..., :nlead]
        nlead = len(sel_fcst_m.lead)
        xdate_init = datetime.datetime.strptime(sel_date + '-01', "%Y-%m-%d").date()
        xdate_strt = xdate_init + relativedelta(months=-2)
        xdate_last = xdate_init + relativedelta(months=nlead - 1)
        xtime_fcst = [xdate_init + relativedelta(months=i) for i in range(nlead)]
        sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
        xtime_obs = sel_obs.time.values

        ax.plot(xtime_fcst, sel_fcst_m, c='orangered', marker='.', lw=3, label='100-members stochastic mean')
        ax.fill_between(xtime_fcst, sel_fcst_m - sel_fcst_e, sel_fcst_m + sel_fcst_e, fc='red', alpha=0.2)
        ax.plot(xtime_fcst, sel_fcst_d, c='blue', marker='.', lw=1, label='Deterministic forecast')
        ax.plot(xtime_obs, sel_obs, c='black', marker='.', lw=3, label='Observation', alpha=0.5)

        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator((1, 4, 7, 10), bymonthday=2))
        ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=1, bymonthday=1))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", which="minor", length=2)
        ax.tick_params(axis="y", which="major", length=2)
        ax.tick_params(axis="x", which="major", length=4, color=(0., 0., 0., 0))
        plt.setp(ax.get_xticklabels(minor=False), rotation=0, ha="center")
        plt.setp(ax.get_xticklabels(minor=True), rotation=0, ha="center")
        ax.set_xlim([xdate_strt, xdate_last])
        ax.set_title(f"Initialized from {sel_date}")
        ax.set_ylim([-4., 4.])
        ax.set_ylabel('Nino3.4 SST anomaly (℃)')
        ax.axhline(0.5, c='red', ls='--', dashes=[3, 3], alpha=0.3)
        ax.axhline(-0.5, c='blue', ls='--', dashes=[3, 3], alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.savefig(f'{fname_prefix}_{sel_date}_{idx}.png', dpi=300)
        plt.close()


@torch.no_grad()
def nxro_reforecast(model, init_ds: xr.Dataset, n_month: int = 21, var_order: Optional[list] = None,
                    device: str = 'cpu') -> xr.Dataset:
    """Build deterministic reforecast for NXRO by monthly Euler stepping.

    Returns xr.Dataset with dims [init, lead] and data_vars per variable.
    """
    if var_order is None:
        var_order = list(init_ds.data_vars)
    # time in decimal years
    time = init_ds.time.to_index()
    years = time.year + (time.month - 1) / 12.0
    ncycle = 12
    dt = 1.0 / ncycle

    X0_all = np.stack([init_ds[v].values for v in var_order], axis=-1)  # [T, n_vars]
    T = X0_all.shape[0]
    n_vars = X0_all.shape[1]
    lead = np.arange(0, n_month + 1, dtype=np.int32)
    out = np.zeros((n_vars, len(lead), T), dtype=np.float32)

    model.eval()
    for i in range(T):
        x = torch.from_numpy(X0_all[i:i+1]).to(device)  # [1, n_vars]
        t0 = float(years[i])
        out[:, 0, i] = x.squeeze(0).cpu().numpy()
        for j in range(1, len(lead)):
            t_year = torch.tensor([t0 + (j - 1) * dt], dtype=torch.float32, device=device)
            dxdt = model(x, t_year)
            x = x + dxdt * dt
            out[:, j, i] = x.squeeze(0).cpu().numpy()

    coords = {
        'ranky': np.arange(1, n_vars + 1, dtype=np.int32),
        'lead': lead,
        'init': ('init', init_ds.time.values)
    }
    da = xr.DataArray(out, dims=['ranky', 'lead', 'init'], coords=coords)
    # map to variables
    for k, var in enumerate(var_order):
        tmp = da.sel(ranky=k + 1).drop('ranky')
        if k == 0:
            fcst = xr.Dataset({var: tmp})
        else:
            fcst[var] = tmp
    # Ensure climpred-compatible lead units
    xr_lead = xr.DataArray(lead, dims={'lead': lead}, coords={'lead': lead},
                           attrs={'units': 'months', 'long_name': 'Lead'})
    fcst = fcst.assign_coords(lead=xr_lead)
    return fcst



def _align_obs_for_lead(obs_var: xr.DataArray, init_times: np.ndarray, lead_val: int) -> xr.DataArray:
    """Align observations to forecast inits shifted by lead months.

    Uses reindex to tolerate missing targets (beyond obs range). Returns a DataArray
    over dimension 'init' with NaNs where observations are unavailable.
    """
    target_times = pd.to_datetime(init_times)
    target_times = pd.DatetimeIndex(target_times) + pd.DateOffset(months=int(lead_val))
    target_da = xr.DataArray(target_times, dims=['time'])
    obs_re = obs_var.reindex(time=target_da)
    # Rename time->init to match forecast dim
    obs_re = obs_re.rename({'time': 'init'})
    return obs_re


def evaluate_stochastic_ensemble(
    fcst_m: xr.Dataset,
    obs_ds: xr.Dataset,
    var: str = 'Nino34',
    out_prefix: str = 'results/NXRO_eval',
    intervals: list = [0.5, 0.8, 0.9],
    threshold: float = 0.5,
) -> xr.Dataset:
    """Evaluate ensemble forecasts: coverage, width, spread, RMSE(mean), CRPS, Brier, reliability plot.

    Saves plots and CSVs with the given out_prefix. Returns an xr.Dataset of lead-wise metrics.
    """
    leads = fcst_m['lead'].values
    init_vals = fcst_m['init'].values
    fc = fcst_m[var]  # dims: lead, init, member
    obs_var = obs_ds[var]

    # Pre-allocate
    cov_intervals = np.zeros((len(leads), len(intervals)), dtype=np.float32)
    wid_intervals = np.zeros((len(leads), len(intervals)), dtype=np.float32)
    spread = np.zeros(len(leads), dtype=np.float32)
    rmse_mean = np.zeros(len(leads), dtype=np.float32)
    crps = np.zeros(len(leads), dtype=np.float32)
    brier = np.zeros(len(leads), dtype=np.float32)

    # Reliability bins
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rel_counts = np.zeros((len(leads), len(bin_centers)), dtype=np.float32)
    rel_obsfreq = np.zeros((len(leads), len(bin_centers)), dtype=np.float32)

    for iL, L in enumerate(leads):
        fc_L = fc.sel(lead=L)  # [init, member]
        obs_L = _align_obs_for_lead(obs_var, init_vals, int(L))  # [init]

        # Mask to available observations
        valid_mask = xr.ufuncs.isfinite(obs_L)
        fc_L_valid = fc_L.where(valid_mask, drop=True)
        obs_L_valid = obs_L.where(valid_mask, drop=True)
        # Ensemble mean and spread
        mean_L = fc_L_valid.mean('member')
        std_L = fc_L_valid.std('member')
        spread[iL] = float(std_L.mean('init')) if std_L.size > 0 else np.nan
        # RMSE of ensemble mean
        diff = (mean_L - obs_L_valid)
        rmse_mean[iL] = float(np.sqrt((diff ** 2).mean('init'))) if diff.size > 0 else np.nan

        # Coverage and width per interval
        ens_vals = fc_L_valid.values  # [I, M]
        obs_vals = obs_L_valid.values  # [I]
        if ens_vals.size == 0 or obs_vals.size == 0:
            continue
        for j, c in enumerate(intervals):
            alpha = (1.0 - c) / 2.0
            lo = np.quantile(ens_vals, alpha, axis=1)
            hi = np.quantile(ens_vals, 1.0 - alpha, axis=1)
            covered = (obs_vals >= lo) & (obs_vals <= hi)
            cov_intervals[iL, j] = float(covered.mean())
            wid_intervals[iL, j] = float((hi - lo).mean())

        # CRPS (ensemble) using fair estimator
        # (1/M) sum |x_m - y| - 0.5/M^2 sum |x_m - x_n|
        M = ens_vals.shape[1]
        term1 = np.mean(np.abs(ens_vals - obs_vals[:, None]), axis=1)
        # pairwise abs differences averaged over all pairs
        diff_pairs = np.abs(ens_vals[:, :, None] - ens_vals[:, None, :])
        term2 = np.mean(diff_pairs, axis=(1, 2))
        crps[iL] = float(np.mean(term1 - 0.5 * term2))

        # Brier for threshold event
        p_hat = (ens_vals > threshold).mean(axis=1)
        y_evt = (obs_vals > threshold).astype(np.float32)
        brier[iL] = float(np.mean((p_hat - y_evt) ** 2))

        # Reliability components
        bin_idx = np.digitize(p_hat, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)
        for b in range(len(bin_centers)):
            sel = bin_idx == b
            if np.any(sel):
                rel_counts[iL, b] = np.sum(sel)
                rel_obsfreq[iL, b] = float(y_evt[sel].mean())

    # Build xr outputs
    coords = {'lead': leads}
    ds_out = xr.Dataset({
        'coverage': (['lead', 'interval'], cov_intervals),
        'interval_width': (['lead', 'interval'], wid_intervals),
        'spread': (['lead'], spread),
        'rmse_mean': (['lead'], rmse_mean),
        'crps': (['lead'], crps),
        'brier': (['lead'], brier),
    }, coords={**coords, 'interval': intervals})

    # Save CSV summaries
    pd.DataFrame({'lead': leads, 'spread': spread, 'rmse_mean': rmse_mean, 'crps': crps, 'brier': brier}).to_csv(
        f'{out_prefix}_lead_metrics.csv', index=False)
    cov_df = pd.DataFrame(cov_intervals, columns=[f'cov_{int(c*100)}%' for c in intervals])
    cov_df.insert(0, 'lead', leads)
    cov_df.to_csv(f'{out_prefix}_coverage.csv', index=False)
    wid_df = pd.DataFrame(wid_intervals, columns=[f'width_{int(c*100)}%' for c in intervals])
    wid_df.insert(0, 'lead', leads)
    wid_df.to_csv(f'{out_prefix}_interval_width.csv', index=False)

    # Plots
    # Coverage heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 3 + 0.3 * len(intervals)))
    im = ax.imshow(cov_intervals.T, aspect='auto', origin='lower', vmin=0.0, vmax=1.0, cmap='viridis')
    ax.set_yticks(np.arange(len(intervals)))
    ax.set_yticklabels([f'{int(c*100)}%' for c in intervals])
    ax.set_xticks(np.arange(len(leads)))
    ax.set_xticklabels(leads)
    ax.set_xlabel('Lead (months)')
    ax.set_ylabel('Central interval')
    ax.set_title(f'Coverage rates ({var})')
    plt.colorbar(im, ax=ax, label='Coverage')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_coverage_heatmap.png', dpi=300)
    plt.close()

    # Spread vs RMSE
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(leads, spread, label='Ensemble spread', marker='o')
    ax.plot(leads, rmse_mean, label='RMSE (ensemble mean)', marker='s')
    ax.set_xlabel('Lead (months)')
    ax.set_ylabel('℃')
    ax.set_title(f'Spread vs RMSE ({var})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_spread_rmse.png', dpi=300)
    plt.close()

    # CRPS per lead
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(leads, crps, label='CRPS', c='tab:purple')
    ax.set_xlabel('Lead (months)')
    ax.set_ylabel('CRPS')
    ax.set_title(f'CRPS by lead ({var})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_crps.png', dpi=300)
    plt.close()

    # Reliability diagram for threshold
    # Aggregate across leads for a compact plot
    total_counts = rel_counts.sum(axis=0)
    with np.errstate(invalid='ignore'):
        avg_obsfreq = np.where(total_counts > 0, (rel_obsfreq * rel_counts).sum(axis=0) / total_counts, np.nan)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.plot(bin_centers, avg_obsfreq, marker='o', label=f'Threshold {threshold:.2f}℃')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Forecast probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(f'Reliability diagram ({var})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_reliability.png', dpi=300)
    plt.close()

    return ds_out

