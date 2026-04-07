"""Generate per-lead RMSE/ACC skill curves from multiseed checkpoints.

Loads best checkpoints, runs reforecasts, computes skill, and plots
RMSE and ACC vs lead time with uncertainty bands — matching the paper's
Figure 2 (1f_core_models_combined.png) format.

Usage:
    python scripts/generate_rebuttal_skill_curves.py [--device cuda]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.xro_utils import calc_forecast_skill, nxro_reforecast
from nxro.models import (
    NXROResModel, NXROAttentiveModel, NXROGraphPyGModel,
    PureNeuralODEModel, PureTransformerModel,
    build_edge_index_from_corr,
)

OUT_DIR = 'tex/rebuttal/figures'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

COLORS = {
    'XRO': '#888888',
    'attentive': '#E91E63',
    'graph_pyg': '#4CAF50',
    'res': '#FF9800',
    'pure_neural_ode': '#9C27B0',
    'pure_transformer': '#795548',
}
LABELS = {
    'XRO': 'XRO',
    'attentive': 'NXRO-Attention',
    'graph_pyg': 'NXRO-GNN',
    'res': 'NXRO-MLP',
    'pure_neural_ode': 'NeuralODE',
    'pure_transformer': 'Transformer',
}
LINESTYLES = {
    'XRO': '--',
    'attentive': '-',
    'graph_pyg': '-',
    'res': '-',
    'pure_neural_ode': ':',
    'pure_transformer': ':',
}


def load_model(ckpt_path, device='cpu'):
    """Load model from checkpoint, inferring architecture from state dict keys."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
    var_order = ckpt['var_order']
    n_vars = len(var_order)
    keys = set(state_dict.keys())

    if 'conv1.lin.weight' in keys or 'conv1.weight' in keys:
        obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
        train_ds = obs_ds.sel(time=slice('1979-01', '1995-12'))
        arrs = np.stack([train_ds[v].values for v in var_order], axis=-1)
        corr = np.corrcoef(arrs.T)
        edge_index = build_edge_index_from_corr(torch.tensor(corr, dtype=torch.float32), top_k=3)
        model = NXROGraphPyGModel(n_vars=n_vars, k_max=2, edge_index=edge_index, hidden=16)
    elif 'Wq.weight' in keys:
        model = NXROAttentiveModel(n_vars=n_vars, k_max=2, d=16)
    elif any('res_net' in k or 'residual.0.weight' in k for k in keys):
        # NXROResModel — infer hidden from weight shape
        hidden = 32
        for k in keys:
            if 'res_net.0.weight' in k or 'residual.0.weight' in k:
                hidden = state_dict[k].shape[0]
                break
        model = NXROResModel(n_vars=n_vars, k_max=2, hidden=hidden)
    elif 'input_proj.weight' in keys:
        # PureTransformerModel — no k_max
        d_model = state_dict['input_proj.weight'].shape[0]
        model = PureTransformerModel(n_vars=n_vars, d_model=d_model)
    elif 'drift_net.0.weight' in keys or 'drift.0.weight' in keys:
        # PureNeuralODEModel — no k_max
        weight_key = 'drift_net.0.weight' if 'drift_net.0.weight' in keys else 'drift.0.weight'
        hidden = state_dict[weight_key].shape[0]
        # Detect dropout from layer indices: if drift.3 is Dropout (no weight), indices skip
        has_dropout = any('drift.3' in k for k in keys) == False and any('drift.5' in k for k in keys)
        model = PureNeuralODEModel(n_vars=n_vars, hidden=hidden,
                                    dropout=0.1 if has_dropout else 0.0)
    else:
        return None, None

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, var_order


def compute_xro_skill(obs_ds, test_period, sel_var='Nino34'):
    """Compute XRO baseline skill from saved XRO forecast."""
    # Try to load pre-computed XRO forecast
    xro_paths = glob.glob('results_out_of_sample/xro/*.nc') + \
                glob.glob('data/XRO_*forecast*.nc')
    # Fall back to computing from XRO model
    from XRO.core import XRO
    xro = XRO()
    train_ds = obs_ds.sel(time=slice('1979-01', '2001-12'))
    data = np.stack([train_ds[v].values for v in list(train_ds.data_vars)], axis=-1)
    xro.fit(data)
    # Use nxro_reforecast-style computation but with XRO
    # For simplicity, return None and we'll use the known XRO RMSE = 0.605
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--results_dir', default='results_rebuttal_multiseed')
    parser.add_argument('--mlp_dir', default='results_rebuttal_mlp_best')
    args = parser.parse_args()
    device = args.device

    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    test_period = slice('2002-01', '2022-12')
    sel_var = 'Nino34'  # Match paper's evaluation metric

    # Collect per-model per-seed skill curves
    model_skills = {}  # model_name -> list of (rmse_by_lead, acc_by_lead)

    # 1. Main multiseed models (attentive, graph_pyg, pure_neural_ode, pure_transformer)
    for model_name in ['attentive', 'graph_pyg', 'pure_neural_ode', 'pure_transformer']:
        ckpt_pattern = os.path.join(args.results_dir, '**', f'*{model_name}*best*seed*.pt')
        if model_name == 'graph_pyg':
            ckpt_pattern = os.path.join(args.results_dir, 'graphpyg', '**', '*best*seed*.pt')
        ckpts = glob.glob(ckpt_pattern, recursive=True)
        ckpts = [c for c in ckpts if 'stochastic' not in c and 'noise' not in c and 'checkpoint' not in os.path.basename(c)]
        if not ckpts:
            print(f'  {model_name}: no checkpoints found')
            continue

        print(f'  {model_name}: {len(ckpts)} checkpoints')
        skills = []
        for ckpt_path in ckpts[:10]:  # max 10 seeds
            try:
                model, var_order = load_model(ckpt_path, device=device)
                if model is None:
                    continue
                fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21,
                                       var_order=var_order, device=device)
                rmse = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
                acc = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
                skills.append({
                    'rmse': rmse[sel_var].values,
                    'acc': acc[sel_var].values,
                    'leads': rmse.lead.values,
                })
            except Exception as e:
                print(f'    ERROR {os.path.basename(ckpt_path)}: {e}')
                continue

        if skills:
            model_skills[model_name] = skills

    # 2. MLP from mlp_best directory
    mlp_ckpts = glob.glob(os.path.join(args.mlp_dir, '**', '*best*seed*.pt'), recursive=True)
    mlp_ckpts = [c for c in mlp_ckpts if 'stochastic' not in c and 'noise' not in c]
    if mlp_ckpts:
        print(f'  res (MLP h=4): {len(mlp_ckpts)} checkpoints')
        skills = []
        for ckpt_path in mlp_ckpts[:10]:
            try:
                model, var_order = load_model(ckpt_path, device=device)
                if model is None:
                    continue
                fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21,
                                       var_order=var_order, device=device)
                rmse = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=test_period)
                acc = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=test_period)
                skills.append({
                    'rmse': rmse[sel_var].values,
                    'acc': acc[sel_var].values,
                    'leads': rmse.lead.values,
                })
            except Exception as e:
                print(f'    ERROR {os.path.basename(ckpt_path)}: {e}')
                continue
        if skills:
            model_skills['res'] = skills

    if not model_skills:
        print('No skill data computed. Exiting.')
        return

    # Get leads from first result
    leads = model_skills[list(model_skills.keys())[0]][0]['leads']

    # =========================================================================
    # Print per-model average RMSE for verification
    # =========================================================================
    print(f'\n  Per-model average Nino34 RMSE (leads 1-21, across seeds):')
    for model_name in ['attentive', 'graph_pyg', 'res', 'pure_transformer', 'pure_neural_ode']:
        if model_name not in model_skills:
            continue
        skills = model_skills[model_name]
        arr = np.array([s['rmse'] for s in skills])  # [n_seeds, n_leads]
        avg = arr[:, 1:].mean()
        print(f'    {model_name:20s}: {avg:.4f} (n_seeds={len(skills)})')

    # =========================================================================
    # Load XRO per-lead skill from saved evaluation
    # =========================================================================
    xro_skill = None
    xro_eval_path = 'results_out_of_sample/xro_baseline/xro_stochastic_eval_lead_metrics.csv'
    if os.path.exists(xro_eval_path):
        xro_df = pd.read_csv(xro_eval_path)
        xro_skill = {'leads': xro_df['lead'].values, 'rmse': xro_df['rmse_mean'].values}
        print(f'    XRO (from file)     : {xro_df[xro_df["lead"]>0]["rmse_mean"].mean():.4f}')

    # Load classical baselines if available
    classical = {}
    classical_path = 'results_rebuttal_classical_baselines.csv'
    if os.path.exists(classical_path):
        cl_df = pd.read_csv(classical_path)
        for name in cl_df['model'].unique():
            sub = cl_df[(cl_df['model'] == name) & (cl_df['lead'] > 0)]
            classical[name] = {'leads': sub['lead'].values, 'rmse': sub['rmse'].values}
            print(f'    {name:20s}: {sub["rmse"].mean():.4f} (classical)')

    # =========================================================================
    # Plot: RMSE and ACC vs lead time (matching paper Figure 2)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Focus on NXRO variants (the paper's core models) + baselines
    plot_order = ['attentive', 'graph_pyg', 'res']

    CLASSICAL_COLORS = {
        'Persistence': '#AAAAAA',
        'Climatology': '#BBBBBB',
        'ARIMA(2,0,1)': '#666666',
        'VAR(3)': '#444444',
    }

    for metric_idx, (metric, ax, ylabel) in enumerate([
        ('rmse', axes[0], 'Nino3.4 RMSE (°C)'),
        ('acc', axes[1], 'Nino3.4 ACC'),
    ]):
        ax.set_title(f'({"ab"[metric_idx]}) {metric.upper()} vs Forecast Lead')

        # Plot XRO reference
        if metric == 'rmse' and xro_skill is not None:
            xro_leads = xro_skill['leads']
            xro_rmse = xro_skill['rmse']
            mask = xro_leads > 0
            ax.plot(xro_leads[mask], xro_rmse[mask], '--', color=COLORS['XRO'],
                    linewidth=2.5, label='XRO', zorder=4)

        # Plot classical baselines (RMSE only)
        if metric == 'rmse' and classical:
            for cl_name in ['VAR(3)', 'ARIMA(2,0,1)']:
                if cl_name in classical:
                    cl = classical[cl_name]
                    ax.plot(cl['leads'], cl['rmse'], ':', color=CLASSICAL_COLORS[cl_name],
                            linewidth=1.5, label=cl_name, zorder=1, alpha=0.7)

        # Plot NXRO and pure neural models
        for model_name in plot_order:
            if model_name not in model_skills:
                continue
            skills = model_skills[model_name]
            arr = np.array([s[metric] for s in skills])  # [n_seeds, n_leads]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)

            lbl = LABELS[model_name]
            if model_name == 'res':
                lbl = 'NXRO-MLP (h=4)'

            ax.plot(leads, mean, LINESTYLES[model_name],
                    color=COLORS[model_name], linewidth=2,
                    label=lbl, zorder=3)
            ax.fill_between(leads, mean - std, mean + std,
                            color=COLORS[model_name], alpha=0.10, zorder=2)

        ax.set_xlabel('Forecast Lead (months)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left' if metric == 'rmse' else 'lower left', fontsize=7.5)
        ax.grid(alpha=0.3, zorder=0)
        ax.set_xlim(leads[0], leads[-1])
        if metric == 'rmse':
            ax.set_ylim(0, 1.4)

    fig.suptitle('Out-of-Sample Forecast Skill (Val-Selected, Multi-Seed)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_skill_curves_combined.png')
    plt.savefig(f'{OUT_DIR}/fig7_skill_curves_combined.pdf')
    plt.close()
    print(f'\nSaved fig7_skill_curves_combined')


if __name__ == '__main__':
    main()
