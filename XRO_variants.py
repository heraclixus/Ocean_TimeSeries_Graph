import warnings
warnings.filterwarnings("ignore")
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import torch
import argparse
import pandas as pd
from XRO.core import XRO
from nxro.train import train_nxro_linear, train_nxro_ro, train_nxro_rodiag, train_nxro_res, train_nxro_neural, train_nxro_bilinear, train_nxro_attentive, train_nxro_graph
from utils.xro_utils import calc_forecast_skill, plot_forecast_plume, nxro_reforecast


def main():
    os.makedirs('results', exist_ok=True)
    # Data
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))

    # ------- Baseline XRO (control, ac=2; linear; and ac=0) -------
    XROac2 = XRO(ncycle=12, ac_order=2)
    XROac0 = XRO(ncycle=12, ac_order=0)

    XROac2_fit = XROac2.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    XROac0_fit = XROac0.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    XROac2Lin_fit = XROac2.fit_matrix(train_ds, maskb=[], maskNT=[])

    # Deterministic reforecasts
    XROac2_fcst = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
    XROac0_fcst = XROac0.reforecast(fit_ds=XROac0_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
    XROac2Lin_fcst = XROac2.reforecast(fit_ds=XROac2Lin_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')

    # ------- NXRO-Linear (load if exists, else train) -------
    linear_path = 'results/nxro_linear_best.pt'
    if os.path.exists(linear_path):
        ckpt = torch.load(linear_path, map_location='cpu')
        from nxro.models import NXROLinearModel
        var_order = ckpt['var_order']
        nxro_model = NXROLinearModel(n_vars=len(var_order), k_max=2)
        nxro_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_model, var_order, best_rmse, _ = train_nxro_linear(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_model.state_dict(), 'var_order': var_order}, linear_path)
    NXRO_fcst = nxro_reforecast(nxro_model, init_ds=obs_ds, n_month=21, var_order=var_order, device='cpu')

    # ------- NXRO-RO (load if exists, else train) -------
    ro_path = 'results/nxro_ro_best.pt'
    if os.path.exists(ro_path):
        from nxro.models import NXROROModel
        ckpt = torch.load(ro_path, map_location='cpu')
        ro_var_order = ckpt['var_order']
        nxro_ro_model = NXROROModel(n_vars=len(ro_var_order), k_max=2)
        nxro_ro_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_ro_model, ro_var_order, _, _ = train_nxro_ro(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_ro_model.state_dict(), 'var_order': ro_var_order}, ro_path)
    NXRO_ro_fcst = nxro_reforecast(nxro_ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device='cpu')

    # ------- NXRO-RO+Diag (load if exists, else train) -------
    rodiag_path = 'results/nxro_rodiag_best.pt'
    if os.path.exists(rodiag_path):
        from nxro.models import NXRORODiagModel
        ckpt = torch.load(rodiag_path, map_location='cpu')
        rd_var_order = ckpt['var_order']
        nxro_rd_model = NXRORODiagModel(n_vars=len(rd_var_order), k_max=2)
        nxro_rd_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_rd_model, rd_var_order, _, _ = train_nxro_rodiag(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_rd_model.state_dict(), 'var_order': rd_var_order}, rodiag_path)
    NXRO_rd_fcst = nxro_reforecast(nxro_rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device='cpu')

    # ------- NXRO-Res (load if exists, else train) -------
    res_path = 'results/nxro_res_best.pt'
    if os.path.exists(res_path):
        from nxro.models import NXROResModel
        ckpt = torch.load(res_path, map_location='cpu')
        rs_var_order = ckpt['var_order']
        nxro_rs_model = NXROResModel(n_vars=len(rs_var_order), k_max=2)
        nxro_rs_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_rs_model, rs_var_order, _, _ = train_nxro_res(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_rs_model.state_dict(), 'var_order': rs_var_order}, res_path)
    NXRO_rs_fcst = nxro_reforecast(nxro_rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device='cpu')

    # ------- NXRO-NeuralODE (load if exists, else train) -------
    neural_path = 'results/nxro_neural_best.pt'
    if os.path.exists(neural_path):
        from nxro.models import NXRONeuralODEModel
        ckpt = torch.load(neural_path, map_location='cpu')
        nn_var_order = ckpt['var_order']
        nxro_nn_model = NXRONeuralODEModel(n_vars=len(nn_var_order), k_max=2, hidden=64, depth=2, dropout=0.1, allow_cross=False)
        nxro_nn_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_nn_model, nn_var_order, _, _ = train_nxro_neural(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, hidden=64, depth=2, dropout=0.1,
            allow_cross=False, mask_mode='th_only', device='cpu'
        )
        torch.save({'state_dict': nxro_nn_model.state_dict(), 'var_order': nn_var_order}, neural_path)
    NXRO_nn_fcst = nxro_reforecast(nxro_nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device='cpu')

    # ------- NXRO-Bilinear (load if exists, else train) -------
    bl_path = 'results/nxro_bilinear_best.pt'
    if os.path.exists(bl_path):
        from nxro.models import NXROBilinearModel
        ckpt = torch.load(bl_path, map_location='cpu')
        bl_var_order = ckpt['var_order']
        nxro_bl_model = NXROBilinearModel(n_vars=len(bl_var_order), k_max=2, n_channels=2, rank=2)
        nxro_bl_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_bl_model, bl_var_order, _, _ = train_nxro_bilinear(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, n_channels=2, rank=2, device='cpu'
        )
        torch.save({'state_dict': nxro_bl_model.state_dict(), 'var_order': bl_var_order}, bl_path)
    NXRO_bl_fcst = nxro_reforecast(nxro_bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device='cpu')

    # ------- NXRO-Attentive (load if exists, else train) -------
    at_path = 'results/nxro_attentive_best.pt'
    if os.path.exists(at_path):
        from nxro.models import NXROAttentiveModel
        ckpt = torch.load(at_path, map_location='cpu')
        at_var_order = ckpt['var_order']
        nxro_at_model = NXROAttentiveModel(n_vars=len(at_var_order), k_max=2, d=32, dropout=0.1, mask_mode='th_only')
        nxro_at_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_at_model, at_var_order, _, _ = train_nxro_attentive(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, d=32, dropout=0.1, mask_mode='th_only', device='cpu'
        )
        torch.save({'state_dict': nxro_at_model.state_dict(), 'var_order': at_var_order}, at_path)
    NXRO_at_fcst = nxro_reforecast(nxro_at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device='cpu')

    # ------- NXRO-Graph (load if exists, else train) -------
    gr_path = 'results/nxro_graph_best.pt'
    if os.path.exists(gr_path):
        from nxro.models import NXROGraphModel
        ckpt = torch.load(gr_path, map_location='cpu')
        gr_var_order = ckpt['var_order']
        nxro_gr_model = NXROGraphModel(n_vars=len(gr_var_order), k_max=2, use_fixed_graph=True)
        nxro_gr_model.load_state_dict(ckpt['state_dict'])
    else:
        nxro_gr_model, gr_var_order, _, _ = train_nxro_graph(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, use_fixed_graph=True, device='cpu'
        )
        torch.save({'state_dict': nxro_gr_model.state_dict(), 'var_order': gr_var_order}, gr_path)
    NXRO_gr_fcst = nxro_reforecast(nxro_gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device='cpu')

    # ------- Skills (ACC and RMSE) for Nino34 -------
    acc_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_XROac0 = calc_forecast_skill(XROac0_fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_XROac2Lin = calc_forecast_skill(XROac2Lin_fcst, obs_ds, metric='acc', is_mv3=True,
                                        by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO = calc_forecast_skill(NXRO_fcst, obs_ds, metric='acc', is_mv3=True,
                                   by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO_ro = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO_rd = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO_rs = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO_nn = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO_bl = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO_at = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_NXRO_gr = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))

    sel_var = 'Nino34'
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    acc_XROac2[sel_var].plot(ax=ax, label='XRO', c='orangered', lw=2)
    acc_XROac0[sel_var].plot(ax=ax, label='XRO$_{ac=0}$', c='deepskyblue', lw=2)
    acc_XROac2Lin[sel_var].plot(ax=ax, label='Linear XRO', c='cyan', ls='None', marker='.', ms=8)
    acc_NXRO[sel_var].plot(ax=ax, label='NXRO-Linear', c='green', lw=2)
    acc_NXRO_ro[sel_var].plot(ax=ax, label='NXRO-RO', c='purple', lw=2)
    acc_NXRO_rd[sel_var].plot(ax=ax, label='NXRO-RO+Diag', c='brown', lw=2)
    acc_NXRO_rs[sel_var].plot(ax=ax, label='NXRO-Res', c='darkgreen', lw=2)
    acc_NXRO_nn[sel_var].plot(ax=ax, label='NXRO-NeuralODE', c='black', lw=2)
    acc_NXRO_bl[sel_var].plot(ax=ax, label='NXRO-Bilinear', c='orange', lw=2)
    acc_NXRO_at[sel_var].plot(ax=ax, label='NXRO-Attentive', c='pink', lw=2)
    acc_NXRO_gr[sel_var].plot(ax=ax, label='NXRO-Graph', c='gray', lw=2)
    ax.set_ylabel('Correlation')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0.2, 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend()
    plt.savefig('results/variants_acc_skill.png', dpi=300)
    plt.close()

    rmse_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_XROac0 = calc_forecast_skill(XROac0_fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_XROac2Lin = calc_forecast_skill(XROac2Lin_fcst, obs_ds, metric='rmse', is_mv3=True,
                                         by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO = calc_forecast_skill(NXRO_fcst, obs_ds, metric='rmse', is_mv3=True,
                                    by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO_ro = calc_forecast_skill(NXRO_ro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO_rd = calc_forecast_skill(NXRO_rd_fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO_rs = calc_forecast_skill(NXRO_rs_fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO_nn = calc_forecast_skill(NXRO_nn_fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO_bl = calc_forecast_skill(NXRO_bl_fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO_at = calc_forecast_skill(NXRO_at_fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_NXRO_gr = calc_forecast_skill(NXRO_gr_fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=slice('1979-01', '2022-12'))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    rmse_XROac2[sel_var].plot(ax=ax, label='XRO', c='orangered', lw=2)
    rmse_XROac0[sel_var].plot(ax=ax, label='XRO$_{ac=0}$', c='deepskyblue', lw=2)
    rmse_XROac2Lin[sel_var].plot(ax=ax, label='Linear XRO', c='cyan', ls='None', marker='.', ms=8)
    rmse_NXRO[sel_var].plot(ax=ax, label='NXRO-Linear', c='green', lw=2)
    rmse_NXRO_ro[sel_var].plot(ax=ax, label='NXRO-RO', c='purple', lw=2)
    rmse_NXRO_rd[sel_var].plot(ax=ax, label='NXRO-RO+Diag', c='brown', lw=2)
    rmse_NXRO_rs[sel_var].plot(ax=ax, label='NXRO-Res', c='darkgreen', lw=2)
    rmse_NXRO_nn[sel_var].plot(ax=ax, label='NXRO-NeuralODE', c='black', lw=2)
    rmse_NXRO_bl[sel_var].plot(ax=ax, label='NXRO-Bilinear', c='orange', lw=2)
    rmse_NXRO_at[sel_var].plot(ax=ax, label='NXRO-Attentive', c='pink', lw=2)
    rmse_NXRO_gr[sel_var].plot(ax=ax, label='NXRO-Graph', c='gray', lw=2)
    ax.set_ylabel('RMSE (℃)')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0., 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend()
    plt.savefig('results/variants_rmse_skill.png', dpi=300)
    plt.close()

    # ------- Rank tables and heatmaps (per-lead ranking across models) -------
    def compute_rank_df(metric_map, higher_is_better=True):
        model_names = list(metric_map.keys())
        # Common leads across all models
        lead_sets = [set(metric_map[m]['lead'].values.tolist()) for m in model_names]
        common_leads = sorted(list(set.intersection(*lead_sets)))
        rows = []
        for L in common_leads:
            vals = []
            for m in model_names:
                v = metric_map[m].sel(lead=L).values
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
            vals = np.array(vals, dtype=float)
            # Skip lead if any NaN present
            if np.isnan(vals).any():
                continue
            order = np.argsort(-vals) if higher_is_better else np.argsort(vals)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            rows.append((L, ranks))
        leads_final = [L for (L, _) in rows]
        rank_mat = np.stack([r for (_, r) in rows], axis=0) if rows else np.zeros((0, len(model_names)), dtype=int)
        df = pd.DataFrame(rank_mat, index=leads_final, columns=model_names)
        df.index.name = 'lead'
        return df

    # Build metric maps for selected variable
    acc_map = {
        'XRO': acc_XROac2[sel_var],
        'XRO_ac0': acc_XROac0[sel_var],
        'Linear XRO': acc_XROac2Lin[sel_var],
        'NXRO-Linear': acc_NXRO[sel_var],
        'NXRO-RO': acc_NXRO_ro[sel_var],
        'NXRO-RO+Diag': acc_NXRO_rd[sel_var],
        'NXRO-Res': acc_NXRO_rs[sel_var],
        'NXRO-NeuralODE': acc_NXRO_nn[sel_var],
        'NXRO-Bilinear': acc_NXRO_bl[sel_var],
        'NXRO-Attentive': acc_NXRO_at[sel_var],
        'NXRO-Graph': acc_NXRO_gr[sel_var],
    }
    rmse_map = {
        'XRO': rmse_XROac2[sel_var],
        'XRO_ac0': rmse_XROac0[sel_var],
        'Linear XRO': rmse_XROac2Lin[sel_var],
        'NXRO-Linear': rmse_NXRO[sel_var],
        'NXRO-RO': rmse_NXRO_ro[sel_var],
        'NXRO-RO+Diag': rmse_NXRO_rd[sel_var],
        'NXRO-Res': rmse_NXRO_rs[sel_var],
        'NXRO-NeuralODE': rmse_NXRO_nn[sel_var],
        'NXRO-Bilinear': rmse_NXRO_bl[sel_var],
        'NXRO-Attentive': rmse_NXRO_at[sel_var],
        'NXRO-Graph': rmse_NXRO_gr[sel_var],
    }

    acc_rank_df = compute_rank_df(acc_map, higher_is_better=True)
    rmse_rank_df = compute_rank_df(rmse_map, higher_is_better=False)
    acc_rank_df.to_csv('results/variants_rank_acc.csv')
    rmse_rank_df.to_csv('results/variants_rank_rmse.csv')

    def plot_rank_heatmap(df: pd.DataFrame, title: str, out_path: str):
        if df.empty:
            return
        fig_w = max(8, 1.0 * len(df.columns))
        fig_h = max(4, 0.4 * len(df.index))
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        im = ax.imshow(df.values, aspect='auto', origin='lower', cmap='viridis_r')
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_xlabel('Model')
        ax.set_ylabel('Lead (months)')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rank (1=best)')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    plot_rank_heatmap(acc_rank_df, f'Model ranks by ACC ({sel_var})', 'results/variants_rank_acc_heatmap.png')
    plot_rank_heatmap(rmse_rank_df, f'Model ranks by RMSE ({sel_var})', 'results/variants_rank_rmse_heatmap.png')

    # Overall rank (average across leads)
    def compute_overall_rank(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.mean(axis=0).sort_values()

    def plot_rank_bar(series: pd.Series, title: str, out_path: str):
        if series.empty:
            return
        fig_w = max(8, 0.6 * len(series.index))
        fig_h = 4
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        ax.bar(np.arange(len(series.index)), series.values, color='tab:blue', alpha=0.8)
        ax.set_xticks(np.arange(len(series.index)))
        ax.set_xticklabels(series.index, rotation=45, ha='right')
        ax.set_ylabel('Average rank (lower is better)')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    acc_overall = compute_overall_rank(acc_rank_df)
    rmse_overall = compute_overall_rank(rmse_rank_df)
    acc_overall.to_frame('avg_rank').to_csv('results/variants_overall_rank_acc.csv')
    rmse_overall.to_frame('avg_rank').to_csv('results/variants_overall_rank_rmse.csv')
    plot_rank_bar(acc_overall, f'Overall average rank by ACC ({sel_var})', 'results/variants_overall_rank_acc_bar.png')
    plot_rank_bar(rmse_overall, f'Overall average rank by RMSE ({sel_var})', 'results/variants_overall_rank_rmse_bar.png')

    # Combined overall rank (mean of ACC and RMSE average ranks)
    if not acc_overall.empty and not rmse_overall.empty:
        common_models = acc_overall.index.intersection(rmse_overall.index)
        combined_overall = ((acc_overall[common_models] + rmse_overall[common_models]) / 2.0).sort_values()
        combined_overall.to_frame('avg_rank').to_csv('results/variants_overall_rank_combined.csv')
        plot_rank_bar(combined_overall, f'Combined overall average rank (ACC + RMSE) ({sel_var})', 'results/variants_overall_rank_combined_bar.png')

    # Plumes
    XROac2_fcst_stoc = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=100, noise_type='red')
    dates = ['1997-04', '1997-12', '2022-09', '2022-12', '2024-12']
    plot_forecast_plume(XROac2_fcst, XROac2_fcst_stoc, obs_ds, dates, fname_prefix='results/variants_plume')


if __name__ == '__main__':
    main()


