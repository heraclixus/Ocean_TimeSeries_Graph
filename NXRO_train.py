import os
import warnings
warnings.filterwarnings("ignore")
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from nxro.train import (
    train_nxro_linear,
    train_nxro_ro,
    train_nxro_rodiag,
    train_nxro_res,
    train_nxro_neural,
    train_nxro_bilinear,
    train_nxro_attentive,
    train_nxro_graph,
    train_nxro_graph_pyg,
    train_nxro_neural_phys,
    train_nxro_resmix,
)
from utils.xro_utils import calc_forecast_skill, nxro_reforecast, plot_forecast_plume
from nxro.stochastic import compute_residuals_series, fit_seasonal_ar1_from_residuals, SeasonalAR1Noise, nxro_reforecast_stochastic


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_observed_nino34(obs_ds: xr.Dataset, out_path: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    obs_ds['Nino34'].plot(ax=ax, c='black')
    ax.set_title('Observed Nino3.4 SSTA')
    plt.savefig(out_path, dpi=300)
    plt.close()


def pick_sample_inits(ds: xr.Dataset, n: int = 3) -> list:
    time_index = pd.to_datetime(ds.time.values)
    T = len(time_index)
    if T == 0:
        return []
    candidates = [min(12, T - 1), T // 2, max(T - 12, 0)]
    uniq = sorted(set(candidates))
    out = []
    for i in uniq[:n]:
        out.append(f"{time_index[i].year:04d}-{time_index[i].month:02d}")
    return out


def simulate_nxro_longrun(model, X0_ds: xr.Dataset, var_order: list, nyear: int = 100, device: str = 'cpu') -> xr.Dataset:
    """Deterministic long-run simulation for seasonal stddev plots.

    Starts from X0_ds first time step; monthly Euler stepping without noise.
    """
    ncycle = 12
    dt = 1.0 / ncycle
    start_time = pd.to_datetime(str(X0_ds.time.values[0]))
    n_months = nyear * ncycle
    time_index = pd.date_range(start=start_time, periods=n_months, freq='MS')
    years = time_index.year + (time_index.month - 1) / 12.0

    X0 = np.stack([X0_ds[v].isel(time=0).item() for v in var_order], axis=-1).astype(np.float32)
    x = torch.from_numpy(X0[None, :]).to(device)  # [1, n_vars]
    n_vars = x.shape[1]
    out = np.zeros((n_months, n_vars), dtype=np.float32)
    out[0] = x.squeeze(0).cpu().numpy()

    model.eval()
    with torch.no_grad():
        for t in range(1, n_months):
            t_year = torch.tensor([float(years[t-1])], dtype=torch.float32, device=device)
            dxdt = model(x, t_year)
            x = x + dxdt * dt
            out[t] = x.squeeze(0).detach().cpu().numpy()

    ds = xr.Dataset({var: (['time'], out[:, i]) for i, var in enumerate(var_order)}, coords={'time': time_index})
    return ds


def plot_seasonal_sync(train_ds: xr.Dataset, sim_ds: xr.Dataset, sel_var: str, out_path: str) -> None:
    stddev_obs = train_ds.groupby('time.month').std('time')
    stddev_sim = sim_ds.groupby('time.month').std('time')

    plt.plot(stddev_obs.month, stddev_obs[sel_var], c='black', label='ORAS5')
    plt.plot(stddev_sim.month, stddev_sim[sel_var], c='green', label='NXRO-Linear')
    plt.legend()
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel(f'{sel_var} seasonal standard deviation (℃)')
    plt.xlabel('Calendar Month')
    plt.title('Seasonal synchronization')
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_skill_curves(acc_ds: xr.Dataset, rmse_ds: xr.Dataset, sel_var: str, out_prefix: str, label: str) -> None:
    # ACC
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    acc_ds[sel_var].plot(ax=ax, label=label, c='green', lw=2)
    ax.set_ylabel('Correlation')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0.2, 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend()
    plt.savefig(f'{out_prefix}_acc.png', dpi=300)
    plt.close()

    # RMSE
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    rmse_ds[sel_var].plot(ax=ax, label=label, c='green', lw=2)
    ax.set_ylabel('RMSE (℃)')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0., 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend()
    plt.savefig(f'{out_prefix}_rmse.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train NXRO models')
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'ro', 'rodiag', 'res', 'neural', 'neural_phys', 'resmix', 'bilinear', 'attentive', 'graph', 'graph_pyg', 'all'])
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--gat', action='store_true')
    parser.add_argument('--res_reg', type=float, default=1e-4)
    parser.add_argument('--nc_path', type=str, default='data/XRO_indices_oras5.nc')
    parser.add_argument('--train_start', type=str, default='1979-01')
    parser.add_argument('--train_end', type=str, default='2022-12')
    parser.add_argument('--test_start', type=str, default='2023-01')
    parser.add_argument('--test_end', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--k_max', type=int, default=2)
    parser.add_argument('--jac_reg', type=float, default=1e-4)
    parser.add_argument('--div_reg', type=float, default=0.0)
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--alpha_init', type=float, default=0.1)
    parser.add_argument('--alpha_learnable', action='store_true')
    parser.add_argument('--alpha_max', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--stochastic', action='store_true', help='Enable stochastic reforecast outputs')
    parser.add_argument('--members', type=int, default=100, help='Number of ensemble members if stochastic')
    parser.add_argument('--rollout_k', type=int, default=1, help='K-step rollout loss if >1')
    args = parser.parse_args()

    ensure_dir('results')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    obs_ds = xr.open_dataset(args.nc_path)
    train_ds = obs_ds.sel(time=slice(args.train_start, args.train_end))

    # Observed plot
    plot_observed_nino34(obs_ds, out_path='results/NXRO_observed_Nino34.png')

    def run_linear():
        model, var_order, best_rmse, history = train_nxro_linear(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, device=device, rollout_k=args.rollout_k
        )
        # Plot training curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Linear training')
        ax.legend()
        plt.savefig('results/NXRO_linear_training_curves.png', dpi=300)
        plt.close()
        # Save weights
        torch.save({'state_dict': model.state_dict(), 'var_order': var_order}, 'results/nxro_linear_best.pt')

        # Reforecast for skills (deterministic; stochastic optional)
        NXRO_fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device=device)
        acc_NXRO = calc_forecast_skill(NXRO_fcst, obs_ds, metric='acc', is_mv3=True,
                                       by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO = calc_forecast_skill(NXRO_fcst, obs_ds, metric='rmse', is_mv3=True,
                                        by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO, rmse_NXRO, sel_var='Nino34', out_prefix='results/NXRO_linear', label='NXRO-Linear')
        if args.stochastic:
            resid, months = compute_residuals_series(model, train_ds, var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_fcst_m = nxro_reforecast_stochastic(model, init_ds=obs_ds, n_month=21, var_order=var_order,
                                                     noise_model=noise, n_members=args.members, device=device)
            # Save noise params and forecasts
            np.savez('results/nxro_linear_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': model.state_dict(), 'var_order': var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_linear_stochastic.pt')
            NXRO_fcst_m.to_netcdf('results/NXRO_linear_stochastic_forecasts.nc')
            # Skills on ensemble mean
            NXRO_fcst_m_mean = NXRO_fcst_m.mean('member')
            acc_lin_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_lin_stoc = calc_forecast_skill(NXRO_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                                by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_lin_stoc, rmse_lin_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_linear_stochastic', label='NXRO-Linear (stochastic mean)')
            # Plume plots
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_fcst, NXRO_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_linear_plume')

        # Seasonal synchronization via long-run deterministic simulation
        sim_ds = simulate_nxro_longrun(model, X0_ds=train_ds, var_order=var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, sim_ds, sel_var='Nino34', out_path='results/NXRO_linear_seasonal_synchronization.png')

    def run_ro():
        ro_model, ro_var_order, ro_best_rmse, ro_history = train_nxro_ro(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(ro_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(ro_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-RO training')
        ax.legend()
        plt.savefig('results/NXRO_ro_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': ro_model.state_dict(), 'var_order': ro_var_order}, 'results/nxro_ro_best.pt')
        # Skills & seasonal sync
        NXRO_ro_fcst = nxro_reforecast(ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device=device)
        acc_NXRO_ro = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_ro = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_ro, rmse_NXRO_ro, sel_var='Nino34', out_prefix='results/NXRO_ro', label='NXRO-RO')
        if args.stochastic:
            resid, months = compute_residuals_series(ro_model, train_ds, ro_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_ro_fcst_m = nxro_reforecast_stochastic(ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_ro_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': ro_model.state_dict(), 'var_order': ro_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_ro_stochastic.pt')
            NXRO_ro_fcst_m.to_netcdf('results/NXRO_ro_stochastic_forecasts.nc')
            NXRO_ro_fcst_m_mean = NXRO_ro_fcst_m.mean('member')
            acc_ro_stoc = calc_forecast_skill(NXRO_ro_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_ro_stoc = calc_forecast_skill(NXRO_ro_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_ro_stoc, rmse_ro_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_ro_stochastic', label='NXRO-RO (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_ro_fcst, NXRO_ro_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_ro_plume')
        ro_sim_ds = simulate_nxro_longrun(ro_model, X0_ds=train_ds, var_order=ro_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, ro_sim_ds, sel_var='Nino34', out_path='results/NXRO_ro_seasonal_synchronization.png')

    def run_rodiag():
        rd_model, rd_var_order, rd_best_rmse, rd_history = train_nxro_rodiag(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rd_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rd_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-RO+Diag training')
        ax.legend()
        plt.savefig('results/NXRO_rodiag_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': rd_model.state_dict(), 'var_order': rd_var_order}, 'results/nxro_rodiag_best.pt')
        # Skills & seasonal sync
        NXRO_rd_fcst = nxro_reforecast(rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device=device)
        acc_NXRO_rd = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_rd = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_rd, rmse_NXRO_rd, sel_var='Nino34', out_prefix='results/NXRO_rodiag', label='NXRO-RO+Diag')
        if args.stochastic:
            resid, months = compute_residuals_series(rd_model, train_ds, rd_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_rd_fcst_m = nxro_reforecast_stochastic(rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_rodiag_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': rd_model.state_dict(), 'var_order': rd_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_rodiag_stochastic.pt')
            NXRO_rd_fcst_m.to_netcdf('results/NXRO_rodiag_stochastic_forecasts.nc')
            NXRO_rd_fcst_m_mean = NXRO_rd_fcst_m.mean('member')
            acc_rd_stoc = calc_forecast_skill(NXRO_rd_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_rd_stoc = calc_forecast_skill(NXRO_rd_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_rd_stoc, rmse_rd_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_rodiag_stochastic', label='NXRO-RO+Diag (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_rd_fcst, NXRO_rd_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_rodiag_plume')
        rd_sim_ds = simulate_nxro_longrun(rd_model, X0_ds=train_ds, var_order=rd_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rd_sim_ds, sel_var='Nino34', out_path='results/NXRO_rodiag_seasonal_synchronization.png')

    def run_res():
        rs_model, rs_var_order, rs_best_rmse, rs_history = train_nxro_res(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            res_reg=args.res_reg, device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rs_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rs_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Res training')
        ax.legend()
        plt.savefig('results/NXRO_res_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': rs_model.state_dict(), 'var_order': rs_var_order}, 'results/nxro_res_best.pt')
        # Skills & seasonal sync
        NXRO_rs_fcst = nxro_reforecast(rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device=device)
        acc_NXRO_rs = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_rs = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_rs, rmse_NXRO_rs, sel_var='Nino34', out_prefix='results/NXRO_res', label='NXRO-Res')
        if args.stochastic:
            resid, months = compute_residuals_series(rs_model, train_ds, rs_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_rs_fcst_m = nxro_reforecast_stochastic(rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_res_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': rs_model.state_dict(), 'var_order': rs_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_res_stochastic.pt')
            NXRO_rs_fcst_m.to_netcdf('results/NXRO_res_stochastic_forecasts.nc')
            NXRO_rs_fcst_m_mean = NXRO_rs_fcst_m.mean('member')
            acc_res_stoc = calc_forecast_skill(NXRO_rs_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_res_stoc = calc_forecast_skill(NXRO_rs_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                                by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_res_stoc, rmse_res_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_res_stochastic', label='NXRO-Res (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_rs_fcst, NXRO_rs_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_res_plume')
        rs_sim_ds = simulate_nxro_longrun(rs_model, X0_ds=train_ds, var_order=rs_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rs_sim_ds, sel_var='Nino34', out_path='results/NXRO_res_seasonal_synchronization.png')

    def run_neural():
        nn_model, nn_var_order, nn_best_rmse, nn_history = train_nxro_neural(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only', device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(nn_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(nn_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-NeuralODE training')
        ax.legend()
        plt.savefig('results/NXRO_neural_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': nn_model.state_dict(), 'var_order': nn_var_order}, 'results/nxro_neural_best.pt')
        # Skills & seasonal sync
        NXRO_nn_fcst = nxro_reforecast(nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device=device)
        acc_NXRO_nn = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_nn = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_nn, rmse_NXRO_nn, sel_var='Nino34', out_prefix='results/NXRO_neural', label='NXRO-NeuralODE')
        if args.stochastic:
            resid, months = compute_residuals_series(nn_model, train_ds, nn_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_nn_fcst_m = nxro_reforecast_stochastic(nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_neural_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': nn_model.state_dict(), 'var_order': nn_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_neural_stochastic.pt')
            NXRO_nn_fcst_m.to_netcdf('results/NXRO_neural_stochastic_forecasts.nc')
            NXRO_nn_fcst_m_mean = NXRO_nn_fcst_m.mean('member')
            acc_nn_stoc = calc_forecast_skill(NXRO_nn_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_nn_stoc = calc_forecast_skill(NXRO_nn_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_nn_stoc, rmse_nn_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_neural_stochastic', label='NXRO-NeuralODE (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_nn_fcst, NXRO_nn_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_neural_plume')
        nn_sim_ds = simulate_nxro_longrun(nn_model, X0_ds=train_ds, var_order=nn_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, nn_sim_ds, sel_var='Nino34', out_path='results/NXRO_neural_seasonal_synchronization.png')

    def run_neural_phys():
        np_model, np_var_order, np_best_rmse, np_history = train_nxro_neural_phys(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only',
            jac_reg=args.jac_reg, div_reg=args.div_reg, noise_std=args.noise_std, device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(np_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(np_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-NeuralODE (PhysReg) training')
        ax.legend()
        plt.savefig('results/NXRO_neural_phys_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': np_model.state_dict(), 'var_order': np_var_order}, 'results/nxro_neural_phys_best.pt')
        # Skills & seasonal sync
        NXRO_np_fcst = nxro_reforecast(np_model, init_ds=obs_ds, n_month=21, var_order=np_var_order, device=device)
        acc_NXRO_np = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_np = calc_forecast_skill(NXRO_np_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_np, rmse_NXRO_np, sel_var='Nino34', out_prefix='results/NXRO_neural_phys', label='NXRO-NeuralODE (PhysReg)')
        np_sim_ds = simulate_nxro_longrun(np_model, X0_ds=train_ds, var_order=np_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, np_sim_ds, sel_var='Nino34', out_path='results/NXRO_neural_phys_seasonal_synchronization.png')
        if args.stochastic:
            resid, months = compute_residuals_series(np_model, train_ds, np_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_np_fcst_m = nxro_reforecast_stochastic(np_model, init_ds=obs_ds, n_month=21, var_order=np_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_neural_phys_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': np_model.state_dict(), 'var_order': np_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_neural_phys_stochastic.pt')
            NXRO_np_fcst_m.to_netcdf('results/NXRO_neural_phys_stochastic_forecasts.nc')
            NXRO_np_fcst_m_mean = NXRO_np_fcst_m.mean('member')
            acc_np_stoc = calc_forecast_skill(NXRO_np_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_np_stoc = calc_forecast_skill(NXRO_np_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_np_stoc, rmse_np_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_neural_phys_stochastic', label='NXRO-NeuralODE (PhysReg, stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_np_fcst, NXRO_np_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_neural_phys_plume')

    def run_resmix():
        rx_model, rx_var_order, rx_best_rmse, rx_history = train_nxro_resmix(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            hidden=64, alpha_init=args.alpha_init, alpha_learnable=args.alpha_learnable,
            alpha_max=args.alpha_max, res_reg=args.res_reg, device=device
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(rx_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(rx_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-ResidualMix training')
        ax.legend()
        plt.savefig('results/NXRO_resmix_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': rx_model.state_dict(), 'var_order': rx_var_order}, 'results/nxro_resmix_best.pt')
        # Skills & seasonal sync
        NXRO_rx_fcst = nxro_reforecast(rx_model, init_ds=obs_ds, n_month=21, var_order=rx_var_order, device=device)
        acc_NXRO_rx = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_rx = calc_forecast_skill(NXRO_rx_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_rx, rmse_NXRO_rx, sel_var='Nino34', out_prefix='results/NXRO_resmix', label='NXRO-ResidualMix')
        rx_sim_ds = simulate_nxro_longrun(rx_model, X0_ds=train_ds, var_order=rx_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, rx_sim_ds, sel_var='Nino34', out_path='results/NXRO_resmix_seasonal_synchronization.png')
        if args.stochastic:
            resid, months = compute_residuals_series(rx_model, train_ds, rx_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_rx_fcst_m = nxro_reforecast_stochastic(rx_model, init_ds=obs_ds, n_month=21, var_order=rx_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_resmix_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': rx_model.state_dict(), 'var_order': rx_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_resmix_stochastic.pt')
            NXRO_rx_fcst_m.to_netcdf('results/NXRO_resmix_stochastic_forecasts.nc')
            NXRO_rx_fcst_m_mean = NXRO_rx_fcst_m.mean('member')
            acc_rx_stoc = calc_forecast_skill(NXRO_rx_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_rx_stoc = calc_forecast_skill(NXRO_rx_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_rx_stoc, rmse_rx_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_resmix_stochastic', label='NXRO-ResidualMix (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_rx_fcst, NXRO_rx_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_resmix_plume')

    def run_bilinear():
        bl_model, bl_var_order, bl_best_rmse, bl_history = train_nxro_bilinear(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            n_channels=2, rank=2, device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(bl_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(bl_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Bilinear training')
        ax.legend()
        plt.savefig('results/NXRO_bilinear_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': bl_model.state_dict(), 'var_order': bl_var_order}, 'results/nxro_bilinear_best.pt')
        # Skills & seasonal sync
        NXRO_bl_fcst = nxro_reforecast(bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device=device)
        acc_NXRO_bl = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_bl = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_bl, rmse_NXRO_bl, sel_var='Nino34', out_prefix='results/NXRO_bilinear', label='NXRO-Bilinear')
        if args.stochastic:
            resid, months = compute_residuals_series(bl_model, train_ds, bl_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_bl_fcst_m = nxro_reforecast_stochastic(bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_bilinear_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': bl_model.state_dict(), 'var_order': bl_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_bilinear_stochastic.pt')
            NXRO_bl_fcst_m.to_netcdf('results/NXRO_bilinear_stochastic_forecasts.nc')
            NXRO_bl_fcst_m_mean = NXRO_bl_fcst_m.mean('member')
            acc_bl_stoc = calc_forecast_skill(NXRO_bl_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_bl_stoc = calc_forecast_skill(NXRO_bl_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_bl_stoc, rmse_bl_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_bilinear_stochastic', label='NXRO-Bilinear (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_bl_fcst, NXRO_bl_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_bilinear_plume')
        bl_sim_ds = simulate_nxro_longrun(bl_model, X0_ds=train_ds, var_order=bl_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, bl_sim_ds, sel_var='Nino34', out_path='results/NXRO_bilinear_seasonal_synchronization.png')

    def run_attentive():
        at_model, at_var_order, at_best_rmse, at_history = train_nxro_attentive(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            d=32, dropout=0.1, mask_mode='th_only', device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(at_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(at_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-AttentiveCoupling training')
        ax.legend()
        plt.savefig('results/NXRO_attentive_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': at_model.state_dict(), 'var_order': at_var_order}, 'results/nxro_attentive_best.pt')
        # Skills & seasonal sync
        NXRO_at_fcst = nxro_reforecast(at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device=device)
        acc_NXRO_at = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_at = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_at, rmse_NXRO_at, sel_var='Nino34', out_prefix='results/NXRO_attentive', label='NXRO-AttentiveCoupling')
        if args.stochastic:
            resid, months = compute_residuals_series(at_model, train_ds, at_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_at_fcst_m = nxro_reforecast_stochastic(at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_attentive_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': at_model.state_dict(), 'var_order': at_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_attentive_stochastic.pt')
            NXRO_at_fcst_m.to_netcdf('results/NXRO_attentive_stochastic_forecasts.nc')
            NXRO_at_fcst_m_mean = NXRO_at_fcst_m.mean('member')
            acc_at_stoc = calc_forecast_skill(NXRO_at_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_at_stoc = calc_forecast_skill(NXRO_at_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_at_stoc, rmse_at_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_attentive_stochastic', label='NXRO-AttentiveCoupling (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_at_fcst, NXRO_at_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_attentive_plume')
        at_sim_ds = simulate_nxro_longrun(at_model, X0_ds=train_ds, var_order=at_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, at_sim_ds, sel_var='Nino34', out_path='results/NXRO_attentive_seasonal_synchronization.png')

    def run_graph():
        gr_model, gr_var_order, gr_best_rmse, gr_history = train_nxro_graph(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            use_fixed_graph=True, device=device, rollout_k=args.rollout_k
        )
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(gr_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(gr_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('NXRO-Graph training')
        ax.legend()
        plt.savefig('results/NXRO_graph_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': gr_model.state_dict(), 'var_order': gr_var_order}, 'results/nxro_graph_best.pt')
        # Skills & seasonal sync
        NXRO_gr_fcst = nxro_reforecast(gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device=device)
        acc_NXRO_gr = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_gr = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_gr, rmse_NXRO_gr, sel_var='Nino34', out_prefix='results/NXRO_graph', label='NXRO-Graph')
        if args.stochastic:
            resid, months = compute_residuals_series(gr_model, train_ds, gr_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_gr_fcst_m = nxro_reforecast_stochastic(gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez('results/nxro_graph_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': gr_model.state_dict(), 'var_order': gr_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       'results/nxro_graph_stochastic.pt')
            NXRO_gr_fcst_m.to_netcdf('results/NXRO_graph_stochastic_forecasts.nc')
            NXRO_gr_fcst_m_mean = NXRO_gr_fcst_m.mean('member')
            acc_gr_stoc = calc_forecast_skill(NXRO_gr_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_gr_stoc = calc_forecast_skill(NXRO_gr_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_gr_stoc, rmse_gr_stoc, sel_var='Nino34',
                              out_prefix='results/NXRO_graph_stochastic', label='NXRO-Graph (stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_gr_fcst, NXRO_gr_fcst_m, obs_ds, init_dates, fname_prefix='results/NXRO_graph_plume')
        gr_sim_ds = simulate_nxro_longrun(gr_model, X0_ds=train_ds, var_order=gr_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, gr_sim_ds, sel_var='Nino34', out_path='results/NXRO_graph_seasonal_synchronization.png')

    def run_graph_pyg():
        gp_model, gp_var_order, gp_best_rmse, gp_history = train_nxro_graph_pyg(
            nc_path=args.nc_path,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
            top_k=args.top_k, hidden=16, dropout=0.1, use_gat=args.gat, device=device, rollout_k=args.rollout_k
        )
        tag2 = 'gat' if args.gat else 'gcn'
        # Curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(gp_history['train_rmse'], label='train RMSE', c='tab:blue')
        ax.plot(gp_history['test_rmse'], label='test RMSE', c='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'NXRO-GraphPyG ({tag2}) training')
        ax.legend()
        plt.savefig(f'results/NXRO_graphpyg_{tag2}_training_curves.png', dpi=300)
        plt.close()
        # Save
        torch.save({'state_dict': gp_model.state_dict(), 'var_order': gp_var_order}, f'results/nxro_graphpyg_{tag2}_best.pt')
        # Skills & seasonal sync
        NXRO_gp_fcst = nxro_reforecast(gp_model, init_ds=obs_ds, n_month=21, var_order=gp_var_order, device=device)
        acc_NXRO_gp = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=slice(args.train_start, args.train_end))
        rmse_NXRO_gp = calc_forecast_skill(NXRO_gp_fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=slice(args.train_start, args.train_end))
        plot_skill_curves(acc_NXRO_gp, rmse_NXRO_gp, sel_var='Nino34', out_prefix=f'results/NXRO_graphpyg_{tag2}', label=f'NXRO-GraphPyG ({tag2.upper()})')
        if args.stochastic:
            resid, months = compute_residuals_series(gp_model, train_ds, gp_var_order, device=device)
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
            sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
            noise = SeasonalAR1Noise(a1, sigma)
            NXRO_gp_fcst_m = nxro_reforecast_stochastic(gp_model, init_ds=obs_ds, n_month=21, var_order=gp_var_order,
                                                        noise_model=noise, n_members=args.members, device=device)
            np.savez(f'results/nxro_graphpyg_{tag2}_stochastic_noise.npz', a1=a1_np, sigma=sigma_np)
            torch.save({'state_dict': gp_model.state_dict(), 'var_order': gp_var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
                       f'results/nxro_graphpyg_{tag2}_stochastic.pt')
            NXRO_gp_fcst_m.to_netcdf(f'results/NXRO_graphpyg_{tag2}_stochastic_forecasts.nc')
            NXRO_gp_fcst_m_mean = NXRO_gp_fcst_m.mean('member')
            acc_gp_stoc = calc_forecast_skill(NXRO_gp_fcst_m_mean, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(args.train_start, args.train_end))
            rmse_gp_stoc = calc_forecast_skill(NXRO_gp_fcst_m_mean, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(args.train_start, args.train_end))
            plot_skill_curves(acc_gp_stoc, rmse_gp_stoc, sel_var='Nino34',
                              out_prefix=f'results/NXRO_graphpyg_{tag2}_stochastic', label=f'NXRO-GraphPyG ({tag2.upper()} stochastic mean)')
            init_dates = pick_sample_inits(obs_ds, n=3)
            if len(init_dates) > 0:
                plot_forecast_plume(NXRO_gp_fcst, NXRO_gp_fcst_m, obs_ds, init_dates, fname_prefix=f'results/NXRO_graphpyg_{tag2}_plume')
        gp_sim_ds = simulate_nxro_longrun(gp_model, X0_ds=train_ds, var_order=gp_var_order, nyear=100, device=device)
        plot_seasonal_sync(train_ds, gp_sim_ds, sel_var='Nino34', out_path=f'results/NXRO_graphpyg_{tag2}_seasonal_synchronization.png')

    if args.model in ('linear', 'all'):
        run_linear()
    if args.model in ('ro', 'all'):
        run_ro()
    if args.model in ('rodiag', 'all'):
        run_rodiag()
    if args.model in ('res', 'all'):
        run_res()
    if args.model in ('neural', 'all'):
        run_neural()
    if args.model in ('neural_phys', 'all'):
        run_neural_phys()
    if args.model in ('resmix', 'all'):
        run_resmix()
    if args.model in ('bilinear', 'all'):
        run_bilinear()
    if args.model in ('attentive', 'all'):
        run_attentive()
    if args.model in ('graph', 'all'):
        run_graph()
    if args.model in ('graph_pyg', 'all'):
        run_graph_pyg()


if __name__ == '__main__':
    main()


