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


