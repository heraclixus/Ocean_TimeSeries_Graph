import importlib.util
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

# Function to check and install a package if not found
def install_if_missing(package):
    if importlib.util.find_spec(package) is None:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")

# List of required packages
required_packages = [
    "matplotlib", "numpy", "xarray", "climpred", "dateutil", "nc-time-axis",
]

# # Install missing packages
# for pkg in required_packages:
#     install_if_missing(pkg)

# Importing libraries after ensuring they are installed
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import datetime
from dateutil.relativedelta import *

import numpy as np
import xarray as xr

from climpred import HindcastEnsemble

print("All required libraries are installed and imported successfully!")


from XRO.core import XRO
from utils.xro_utils import calc_forecast_skill, plot_forecast_plume

# XRO model with annual mean, annual cycle, and semi-annual cycle
XROac2 = XRO(ncycle=12, ac_order=2)

# XRO model without annual cycles
XROac0 = XRO(ncycle=12, ac_order=0)


# load observed state vectors of XRO: which include ENSO, WWV, and other modes SST indices
# the order of variables is important, with first two must be ENSO SST and WWV;
obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
print(obs_ds)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
obs_ds['Nino34'].plot(ax=ax, c='black', )
ax.set_title('Observed Nino3.4 SSTA')
plt.savefig('XRO_observed_Nino34.png', dpi=300)
plt.close()

# select 1979-01 to 2022-12 as training data
train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))

# XRO model used as control experiment in the paper
XROac2_fit = XROac2.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
print('XRO control model parameters')
print(XROac2_fit)

# XRO ac=0 model
XROac0_fit = XROac0.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
print('XRO(ac=0) model parameters')
print(XROac0_fit)

XROac2Lin_fit = XROac2.fit_matrix(train_ds, maskb=[], maskNT=[])
print('XRO linear model parameters')
print(XROac2Lin_fit)

seed = 2000
XROac2_sim = XROac2.simulate(fit_ds=XROac2_fit, X0_ds=train_ds.isel(time=0), nyear=100, ncopy=100, is_xi_stdac=False, seed=seed)  
print('XRO control stochastic simulation')
print(XROac2_sim)

XROac0_sim = XROac0.simulate(fit_ds=XROac0_fit, X0_ds=train_ds.isel(time=0), nyear=100, ncopy=100, is_xi_stdac=False, seed=seed)  #set seed=1000 to get the exact same result
print('XRO(ac=0) model stochastic simulation')
print(XROac0_sim)

XROac2Lin_sim = XROac0.simulate(fit_ds=XROac2Lin_fit, X0_ds=train_ds.isel(time=0), nyear=100, ncopy=100, is_xi_stdac=False, seed=seed)  #set seed=1000 to get the exact same result
print('XRO(ac=2) linear model stochastic simulation')
print(XROac2Lin_sim)


nmember=5
fig, axes = plt.subplots(nmember, 1, figsize=(8, nmember*2), layout='compressed')

for i, ax in enumerate(axes.flat):
    XROac2_sim.isel(member=i+1)['Nino34'].plot(ax=ax, c='r', lw=1.5, label='XRO')
    XROac2Lin_sim.isel(member=i+1)['Nino34'].plot(ax=ax, c='cyan', lw=1., label='Linear XRO')
    ax.set_xlabel('')
    ax.legend()

plt.savefig('XRO_simulation.png', dpi=300)
plt.close()




# as exmaple shown the 
stddev_obs = train_ds.groupby('time.month').std('time')

stddev_XROac2 = XROac2_sim.groupby('time.month').std('time')
stddev_XROac2_m = stddev_XROac2.mean('member')
stddev_XROac2_e = stddev_XROac2.std('member')

stddev_XROac0 = XROac0_sim.groupby('time.month').std('time')
stddev_XROac0_m = stddev_XROac0.mean('member')
stddev_XROac0_e = stddev_XROac0.std('member')

stddev_XROac2Lin = XROac2Lin_sim.groupby('time.month').std('time')
stddev_XROac2Lin_m = stddev_XROac2Lin.mean('member')
stddev_XROac2Lin_e = stddev_XROac2Lin.std('member')

sel_var = 'Nino34'
plt.plot(stddev_obs.month, stddev_obs[sel_var], c='black', label='ORAS5')
plt.plot(stddev_XROac2_m.month, stddev_XROac2_m[sel_var], c='orangered', label='XRO')
plt.fill_between(stddev_XROac2_m.month, (stddev_XROac2_m-stddev_XROac2_e)[sel_var], (stddev_XROac2_m+stddev_XROac2_e)[sel_var], fc='orangered', alpha=0.15)
plt.plot(stddev_XROac0_m.month, stddev_XROac0_m[sel_var], c='deepskyblue', label='XRO$_{ac=0}$')
plt.fill_between(stddev_XROac0_m.month, (stddev_XROac0_m-stddev_XROac0_e)[sel_var], (stddev_XROac0_m+stddev_XROac0_e)[sel_var], fc='deepskyblue', alpha=0.15)

plt.plot(stddev_XROac2_m.month, stddev_XROac2_m[sel_var], c='cyan', label='Linear XRO', marker='.', ls='None', ms=8)
# plt.fill_between(stddev_XROac2_m.month, (stddev_XROac2_m-stddev_XROac2_e)[sel_var], (stddev_XROac2_m+stddev_XROac2_e)[sel_var], fc='orange', alpha=0.1)

plt.legend()
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.ylabel('Nino34 seasonal standard deviation (℃)')
plt.xlabel('Calendar Month')
plt.title('XRO accurately simulates ENSO seasonal synchronization')
plt.savefig('XRO_seasonal_synchronization.png', dpi=300)
plt.close() 



XROac0_fcst = XROac0.reforecast(fit_ds=XROac0_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
XROac0_fcst

XROac2_fcst = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
print(XROac2_fcst)

XROac2Lin_fcst = XROac2.reforecast(fit_ds=XROac2Lin_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
# print(XROac2_fcst)

def _deprecated_local_calc_forecast_skill(*args, **kwargs):
    raise RuntimeError("Use utils.xro_utils.calc_forecast_skill instead.")



acc_XROac0 = calc_forecast_skill(XROac0_fcst, obs_ds, metric='acc', is_mv3=True, by_month=False, verify_periods=slice('1979-01', '2022-12'))
# print(acc_XROac0)

acc_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='acc', is_mv3=True, by_month=False, verify_periods=slice('1979-01', '2022-12'))
# print(acc_XROac2)

acc_XROac2Lin = calc_forecast_skill(XROac2Lin_fcst, obs_ds, metric='acc', is_mv3=True, by_month=False, verify_periods=slice('1979-01', '2022-12'))
print(acc_XROac2Lin)


sel_var = 'Nino34'
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
acc_XROac2[sel_var].plot(ax=ax, label='XRO', c='orangered', lw=2)
acc_XROac0[sel_var].plot(ax=ax, label='XRO$_{ac=0}$', c='deepskyblue', lw=2)
acc_XROac2Lin[sel_var].plot(ax=ax, label='Linear XRO', c='cyan', ls='None', marker='.', ms=8)

ax.set_ylabel('{0} skill'.format('Correlation') )

ax.set_yticks(np.arange(0, 2.01, step=0.1))
ax.set_xticks(np.arange(1, 24, step=2))
ax.set_ylim([0.2, 1.])
ax.set_xlim([1., 21])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Forecast lead (months)')
ax.axhline(0.5, ls='--', c='black', lw=1.)
ax.set_title('In-sample Niño3.4 forecast correlation skill (1979-2024)')
ax.legend()

plt.savefig('XRO_forecast_skill.png', dpi=300)
plt.close()



rmse_XROac0 = calc_forecast_skill(XROac0_fcst, obs_ds, metric='rmse', is_mv3=True, by_month=False, verify_periods=slice('1979-01', '2022-12'))
rmse_XROac0

rmse_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='rmse', is_mv3=True, by_month=False, verify_periods=slice('1979-01', '2022-12'))
rmse_XROac2

rmse_XROac2Lin = calc_forecast_skill(XROac2Lin_fcst, obs_ds, metric='rmse', is_mv3=True, by_month=False, verify_periods=slice('1979-01', '2022-12'))
rmse_XROac2Lin


sel_var = 'Nino34'
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
rmse_XROac2[sel_var].plot(ax=ax, label='XRO', c='orangered', lw=2)
rmse_XROac0[sel_var].plot(ax=ax, label='XRO$_{ac=0}$', c='deepskyblue', lw=2)

rmse_XROac2Lin[sel_var].plot(ax=ax, label='Linear XRO', c='cyan', ls='None', marker='.', ms=8)

ax.set_ylabel('{0} (℃)'.format('RMSE') )

ax.set_yticks(np.arange(0, 2.01, step=0.1))
ax.set_xticks(np.arange(1, 24, step=2))
ax.set_ylim([0., 1.])
ax.set_xlim([1., 21])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Forecast lead (months)')
ax.set_title('In-sample Niño3.4 forecast RMSE skill (1979-2024)')
ax.legend()

plt.savefig('XRO_forecast_skill_rmse.png', dpi=300)
plt.close()


XROac2_fcst_stoc = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=100, noise_type='red')
print(XROac2_fcst_stoc)

XROac2Lin_fcst_stoc = XROac2.reforecast(fit_ds=XROac2Lin_fit, init_ds=obs_ds, n_month=21, ncopy=100, noise_type='red')
print(XROac2_fcst_stoc)


def plot_forecast_plume(date_arrs, idx=0):
    n_arr = len(date_arrs)
    fig, axes = plt.subplots(n_arr, 1, figsize=(6, 4 * n_arr), sharex=False, sharey=False)
    
    for i, sel_date in enumerate(date_arrs):
        ax = axes.flat[i] if n_arr > 1 else axes  # Handle single subplot case
    
        # Select deterministic and stochastic forecasts
        sel_fcst_d = XROac2_fcst['Nino34'].sel(init=sel_date).squeeze()
        
        sel_fcst_m = XROac2_fcst_stoc['Nino34'].sel(init=sel_date).mean('member').squeeze()
        sel_fcst_e = XROac2_fcst_stoc['Nino34'].sel(init=sel_date).std('member').squeeze()
    
        sel_Linfcst_m = XROac2Lin_fcst_stoc['Nino34'].sel(init=sel_date).mean('member').squeeze()
        sel_Linfcst_e = XROac2Lin_fcst_stoc['Nino34'].sel(init=sel_date).std('member').squeeze()
    
        nlead = len(sel_fcst_m.lead)  # Number of lead months
    
        # Corrected x-axis time handling
        xdate_init = datetime.datetime.strptime(sel_date + '-01', "%Y-%m-%d").date()
        xdate_strt = xdate_init + relativedelta(months= -2)
        xdate_last = xdate_init + relativedelta(months=nlead - 1)
    
        # Forecast time axis based on `lead` months
        xtime_fcst = [xdate_init + relativedelta(months=i) for i in range(nlead)]
    
        # Select observations in the matching time range
        sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
        xtime_obs = sel_obs.time.values  # Ensure NumPy array format for compatibility
    
        # Ensure forecast arrays are sliced correctly
        sel_fcst_m = sel_fcst_m.isel(lead=slice(0, nlead))
        sel_fcst_e = sel_fcst_e.isel(lead=slice(0, nlead))
        sel_Linfcst_m = sel_Linfcst_m.isel(lead=slice(0, nlead))
        sel_fcst_d = sel_fcst_d.isel(lead=slice(0, nlead))
    
        # Plot stochastic forecast with uncertainty
        ax.plot(xtime_fcst, sel_fcst_m, c='orangered', marker='.', lw=3, label='100-members XRO stochastic forecasts')
        ax.fill_between(xtime_fcst, sel_fcst_m - sel_fcst_e, sel_fcst_m + sel_fcst_e, fc='red', alpha=0.2)
    
        # Plot linear stochastic forecast
        ax.plot(xtime_fcst, sel_Linfcst_m, c='cyan', marker='.', ls='None', ms=6, label='100-members Linear XRO stochastic forecasts')
    
        # Plot deterministic forecast
        ax.plot(xtime_fcst, sel_fcst_d, c='blue', marker='.', lw=1, label='Deterministic XRO forecast')
    
        # Plot observations
        ax.plot(xtime_obs, sel_obs, c='black', marker='.', lw=3, label='Observation', alpha=0.5)
    
        # Formatting
        # ax.axhline(y=0., c='black', ls='-', lw=0.5)
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
    
        ax.axhline(0.5, c='red', ls='--', dashes=[3,3], alpha=0.3)
        ax.axhline(-0.5, c='blue', ls='--', dashes=[3,3], alpha=0.3)
    
        ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'XRO_forecast_plume_{sel_date}_{idx}.png', dpi=300)
    plt.close()

date_arrs = ['1997-04', '1997-12', '2022-09']
plot_forecast_plume(date_arrs, idx=0)


date_arrs = ['2022-12', '2024-12']
plot_forecast_plume(date_arrs, idx=1)