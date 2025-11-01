#!/usr/bin/env python
"""
Generate systematic comparison plots for NXRO variants.

Two types of comparisons:
1. Within-category: Compare all sub-variants within each category (e.g., 1, 1a or 2, 2a, 2a-FixL, etc.)
2. Between-category: Compare best variant from each main category

Outputs saved to results/summary/ with descriptive names.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import glob
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from XRO.core import XRO
from nxro.models import (
    NXROLinearModel,
    NXROROModel,
    NXRORODiagModel,
    NXROResModel,
    NXROResFullXROModel,
    NXROResidualMixModel,
    NXROAttentiveModel,
    NXROGraphModel,
    NXRONeuralODEModel,
)
from utils.xro_utils import calc_forecast_skill, nxro_reforecast


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_checkpoint_and_reforecast(ckpt_path, model_class, obs_ds, model_kwargs=None):
    """Load checkpoint and generate reforecast."""
    if not os.path.exists(ckpt_path):
        return None, None
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    var_order = ckpt['var_order']
    n_vars = len(var_order)
    
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs['n_vars'] = n_vars
    model_kwargs.setdefault('k_max', 2)
    
    model = model_class(**model_kwargs)
    model.load_state_dict(ckpt['state_dict'])
    
    fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device='cpu')
    return fcst, var_order


def plot_within_category_comparison(category_fcsts: Dict[str, xr.Dataset], 
                                    obs_ds: xr.Dataset,
                                    category_name: str,
                                    out_dir: str,
                                    eval_period: slice,
                                    out_suffix: str = ''):
    """Generate ACC and RMSE comparison plots for variants within a category."""
    
    if not category_fcsts:
        print(f"  Skipping {category_name}: no models found")
        return
    
    sel_var = 'Nino34'
    
    # Compute ACC and RMSE for all variants
    acc_dict = {}
    rmse_dict = {}
    
    for label, fcst in category_fcsts.items():
        acc_dict[label] = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                               by_month=False, verify_periods=eval_period)
        rmse_dict[label] = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                                by_month=False, verify_periods=eval_period)
    
    # Plot ACC
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for label, acc_ds in acc_dict.items():
        # Explicitly use lead as x-axis
        ax.plot(acc_ds[sel_var].lead.values, acc_ds[sel_var].values, label=label, lw=2, marker='o', markersize=3)
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Forecast lead (months)')
    ax.set_title(f'{category_name}: ACC Comparison')
    ax.set_xlim([0, 21])
    ax.set_ylim([0.2, 1.0])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{category_name}_within_category_acc{out_suffix}.png', dpi=300)
    plt.close()
    
    # Plot RMSE
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for label, rmse_ds in rmse_dict.items():
        # Explicitly use lead as x-axis
        ax.plot(rmse_ds[sel_var].lead.values, rmse_ds[sel_var].values, label=label, lw=2, marker='o', markersize=3)
    ax.set_ylabel('RMSE (°C)')
    ax.set_xlabel('Forecast lead (months)')
    ax.set_title(f'{category_name}: RMSE Comparison')
    ax.set_xlim([0, 21])
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{category_name}_within_category_rmse{out_suffix}.png', dpi=300)
    plt.close()
    
    # Save CSV with skill values
    acc_df = pd.DataFrame({label: acc_dict[label][sel_var].values for label in acc_dict.keys()},
                          index=acc_dict[list(acc_dict.keys())[0]]['lead'].values)
    acc_df.index.name = 'lead'
    acc_df.to_csv(f'{out_dir}/{category_name}_within_category_acc{out_suffix}.csv')
    
    rmse_df = pd.DataFrame({label: rmse_dict[label][sel_var].values for label in rmse_dict.keys()},
                           index=rmse_dict[list(rmse_dict.keys())[0]]['lead'].values)
    rmse_df.index.name = 'lead'
    rmse_df.to_csv(f'{out_dir}/{category_name}_within_category_rmse{out_suffix}.csv')
    
    print(f"  ✓ Generated plots for {category_name} ({len(category_fcsts)} variants)")


def plot_between_category_comparison(best_per_category: Dict[str, Tuple[str, xr.Dataset]],
                                     xro_fcsts: Dict[str, xr.Dataset],
                                     obs_ds: xr.Dataset,
                                     out_dir: str,
                                     eval_period: slice,
                                     out_suffix: str = ''):
    """Generate comparison plots across categories using best variant from each."""
    
    sel_var = 'Nino34'
    
    # Compute ACC and RMSE
    acc_dict = {}
    rmse_dict = {}
    
    # Add XRO baselines
    for xro_label, xro_fcst in xro_fcsts.items():
        acc_dict[xro_label] = calc_forecast_skill(xro_fcst, obs_ds, metric='acc', is_mv3=True,
                                                   by_month=False, verify_periods=eval_period)
        rmse_dict[xro_label] = calc_forecast_skill(xro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                                    by_month=False, verify_periods=eval_period)
    
    # Add best from each category
    for category, (variant_label, fcst) in best_per_category.items():
        full_label = f"{category}: {variant_label}"
        acc_dict[full_label] = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                                    by_month=False, verify_periods=eval_period)
        rmse_dict[full_label] = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                                     by_month=False, verify_periods=eval_period)
    
    # Plot ACC
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Color scheme
    color_map = {
        'XRO': '#FF1744',
        'XRO_ac0': '#2196F3',
        'Linear XRO': '#00BCD4',
        'Cat1': '#4CAF50',
        'Cat2': '#9C27B0',
        'Cat3': '#FF6F00',
        'Cat4': '#00897B',
        'Cat5': '#1A237E',
    }
    
    # Plot XRO baselines
    for xro_label, acc_ds in acc_dict.items():
        if xro_label in ['XRO', 'XRO_ac0', 'Linear XRO']:
            # Use matplotlib plot directly with explicit x and y
            ax.plot(acc_ds[sel_var].lead.values, acc_ds[sel_var].values, 
                   label=xro_label, c=color_map.get(xro_label, None), lw=2.5)
    
    # Plot category representatives
    cat_idx = 1
    for category, (variant_label, _) in sorted(best_per_category.items()):
        full_label = f"{category}: {variant_label}"
        if full_label in acc_dict:
            color = color_map.get(f'Cat{cat_idx}', None)
            ax.plot(acc_dict[full_label][sel_var].lead.values, acc_dict[full_label][sel_var].values,
                   label=full_label, c=color, lw=2, marker='o', markersize=3)
            cat_idx += 1
    
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_xlabel('Forecast lead (months)', fontsize=11)
    ax.set_title('Between-Category Comparison: ACC (Best from Each Category)', fontsize=12)
    ax.set_xlim([0, 21])
    ax.set_ylim([0.2, 1.0])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/between_category_best_acc{out_suffix}.png', dpi=300)
    plt.close()
    
    # Plot RMSE
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot XRO baselines
    for xro_label, rmse_ds in rmse_dict.items():
        if xro_label in ['XRO', 'XRO_ac0', 'Linear XRO']:
            # Use matplotlib plot directly with explicit x and y
            ax.plot(rmse_ds[sel_var].lead.values, rmse_ds[sel_var].values,
                   label=xro_label, c=color_map.get(xro_label, None), lw=2.5)
    
    # Plot category representatives
    cat_idx = 1
    for category, (variant_label, _) in sorted(best_per_category.items()):
        full_label = f"{category}: {variant_label}"
        if full_label in rmse_dict:
            color = color_map.get(f'Cat{cat_idx}', None)
            ax.plot(rmse_dict[full_label][sel_var].lead.values, rmse_dict[full_label][sel_var].values,
                   label=full_label, c=color, lw=2, marker='o', markersize=3)
            cat_idx += 1
    
    ax.set_ylabel('RMSE (°C)', fontsize=11)
    ax.set_xlabel('Forecast lead (months)', fontsize=11)
    ax.set_title('Between-Category Comparison: RMSE (Best from Each Category)', fontsize=12)
    ax.set_xlim([0, 21])
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/between_category_best_rmse{out_suffix}.png', dpi=300)
    plt.close()
    
    print(f"  ✓ Generated between-category comparison plots")


def main():
    parser = argparse.ArgumentParser(description='Generate systematic NXRO variant comparison plots')
    parser.add_argument('--test', action='store_true', help='Use test-period checkpoints (*_best_test.pt)')
    parser.add_argument('--eval_start', type=str, default='1979-01', help='Evaluation period start')
    parser.add_argument('--eval_end', type=str, default='2022-12', help='Evaluation period end')
    parser.add_argument('--eval_period', type=str, choices=['train', 'test'], default='train',
                       help='Evaluation period: train (1979-2022) or test (2023-onwards)')
    args = parser.parse_args()
    
    # Set eval period based on choice
    if args.eval_period == 'test':
        args.eval_start = '2023-01'
        args.eval_end = None
    # else use defaults (1979-01 to 2022-12)
    
    eval_period = slice(args.eval_start, args.eval_end)
    out_dir = 'results/summary'
    ensure_dir(out_dir)
    
    # Suffix for checkpoint files (--test flag determines which checkpoints to load)
    ckpt_suffix = '_test' if args.test else ''
    # Suffix for output files (based on evaluation period)
    eval_suffix = f'_eval_{args.eval_period}'
    
    print("="*80)
    print("XRO Variant Systematic Comparison")
    print("="*80)
    print(f"Checkpoint suffix: {ckpt_suffix or '(none)'} - Loading *{ckpt_suffix}.pt files")
    print(f"Evaluation period: {args.eval_start} to {args.eval_end or 'end'}")
    print(f"Output directory: {out_dir}")
    print(f"Output suffix: {eval_suffix}")
    print("="*80)
    print()
    
    # Load data
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))
    
    # Fit XRO baselines
    print("Fitting XRO baselines...")
    xro_ac2 = XRO(ncycle=12, ac_order=2)
    xro_ac0 = XRO(ncycle=12, ac_order=0)
    
    xro_ac2_fit = xro_ac2.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    xro_ac0_fit = xro_ac0.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    xro_lin_fit = xro_ac2.fit_matrix(train_ds, maskb=[], maskNT=[])
    
    xro_fcsts = {
        'XRO': xro_ac2.reforecast(fit_ds=xro_ac2_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero'),
        'XRO_ac0': xro_ac0.reforecast(fit_ds=xro_ac0_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero'),
        'Linear XRO': xro_ac2.reforecast(fit_ds=xro_lin_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero'),
    }
    print("  ✓ XRO baselines ready\n")
    
    # ============================================================================
    # CATEGORY 1: NXRO-Linear (Variants 1, 1a)
    # ============================================================================
    print("Category 1: NXRO-Linear")
    print("-" * 40)
    cat1_fcsts = {}
    
    # Variant 1: Random
    ckpt_1 = f'results/linear/nxro_linear_best{ckpt_suffix}.pt'
    fcst_1, _ = load_checkpoint_and_reforecast(ckpt_1, NXROLinearModel, obs_ds)
    if fcst_1 is not None:
        cat1_fcsts['V1: Random'] = fcst_1
        print("  ✓ Loaded Variant 1 (Random)")
    
    # Variant 1a: Warm-start
    ckpt_1a = f'results/linear/nxro_linear_ws_best{ckpt_suffix}.pt'
    fcst_1a, _ = load_checkpoint_and_reforecast(ckpt_1a, NXROLinearModel, obs_ds)
    if fcst_1a is not None:
        cat1_fcsts['V1a: Warm-start'] = fcst_1a
        print("  ✓ Loaded Variant 1a (Warm-start)")
    
    plot_within_category_comparison(cat1_fcsts, obs_ds, 'Cat1_Linear', out_dir, eval_period, eval_suffix)
    print()
    
    # ============================================================================
    # CATEGORY 2: NXRO-RO (Variants 2, 2a, 2a-FixL, 2a-FixRO)
    # ============================================================================
    print("Category 2: NXRO-RO")
    print("-" * 40)
    cat2_fcsts = {}
    
    variants_2 = [
        ('V2: Random', f'results/ro/nxro_ro_best{ckpt_suffix}.pt'),
        ('V2a: Warm-start', f'results/ro/nxro_ro_ws_best{ckpt_suffix}.pt'),
        ('V2a-FixL: Freeze L', f'results/ro/nxro_ro_fixL_best{ckpt_suffix}.pt'),
        ('V2a-FixRO: Freeze RO', f'results/ro/nxro_ro_fixRO_best{ckpt_suffix}.pt'),
    ]
    
    for label, ckpt_path in variants_2:
        fcst, _ = load_checkpoint_and_reforecast(ckpt_path, NXROROModel, obs_ds)
        if fcst is not None:
            cat2_fcsts[label] = fcst
            print(f"  ✓ Loaded {label}")
    
    plot_within_category_comparison(cat2_fcsts, obs_ds, 'Cat2_RO', out_dir, eval_period, eval_suffix)
    print()
    
    # ============================================================================
    # CATEGORY 3: NXRO-RO+Diag (Variants 3, 3a, 3a-Fix*)
    # ============================================================================
    print("Category 3: NXRO-RO+Diag")
    print("-" * 40)
    cat3_fcsts = {}
    
    variants_3 = [
        ('V3: Random', f'results/rodiag/nxro_rodiag_best{ckpt_suffix}.pt'),
        ('V3a: Warm-start', f'results/rodiag/nxro_rodiag_ws_best{ckpt_suffix}.pt'),
        ('V3a-FixL: Freeze L', f'results/rodiag/nxro_rodiag_fixL_best{ckpt_suffix}.pt'),
        ('V3a-FixRO: Freeze RO', f'results/rodiag/nxro_rodiag_fixRO_best{ckpt_suffix}.pt'),
        ('V3a-FixDiag: Freeze Diag', f'results/rodiag/nxro_rodiag_fixDiag_best{ckpt_suffix}.pt'),
        ('V3a-FixNL: Freeze RO+Diag', f'results/rodiag/nxro_rodiag_fixNL_best{ckpt_suffix}.pt'),
    ]
    
    for label, ckpt_path in variants_3:
        fcst, _ = load_checkpoint_and_reforecast(ckpt_path, NXRORODiagModel, obs_ds)
        if fcst is not None:
            cat3_fcsts[label] = fcst
            print(f"  ✓ Loaded {label}")
    
    plot_within_category_comparison(cat3_fcsts, obs_ds, 'Cat3_RODiag', out_dir, eval_period, eval_suffix)
    print()
    
    # ============================================================================
    # CATEGORY 4: NXRO-Res (Variants 4, 4a, 4b)
    # ============================================================================
    print("Category 4: NXRO-Res")
    print("-" * 40)
    cat4_fcsts = {}
    
    # Variant 4: Random
    ckpt_4 = f'results/res/nxro_res_best{ckpt_suffix}.pt'
    fcst_4, _ = load_checkpoint_and_reforecast(ckpt_4, NXROResModel, obs_ds)
    if fcst_4 is not None:
        cat4_fcsts['V4: Random'] = fcst_4
        print("  ✓ Loaded Variant 4 (Random)")
    
    # Variant 4a: Frozen linear
    ckpt_4a = f'results/res/nxro_res_fixL_best{ckpt_suffix}.pt'
    fcst_4a, _ = load_checkpoint_and_reforecast(ckpt_4a, NXROResModel, obs_ds)
    if fcst_4a is not None:
        cat4_fcsts['V4a: Frozen L'] = fcst_4a
        print("  ✓ Loaded Variant 4a (FixL)")
    
    # Variant 4b: Frozen full XRO
    ckpt_4b = f'results/res_fullxro/nxro_res_fullxro_best{ckpt_suffix}.pt'
    if os.path.exists(ckpt_4b):
        ckpt = torch.load(ckpt_4b, map_location='cpu')
        # Cannot recreate 4b without XRO components; skip for now
        print("  ⊘ Variant 4b requires XRO components, skipping")
    
    plot_within_category_comparison(cat4_fcsts, obs_ds, 'Cat4_Res', out_dir, eval_period, eval_suffix)
    print()
    
    # ============================================================================
    # CATEGORY 5: NXRO-ResidualMix (Variants 5d, 5d-WS, 5d-Fix*)
    # ============================================================================
    print("Category 5d: NXRO-ResidualMix")
    print("-" * 40)
    cat5d_fcsts = {}
    
    variants_5d = [
        ('V5d: Random', f'results/resmix/nxro_resmix_best{ckpt_suffix}.pt'),
        ('V5d-WS: Warm-start', f'results/resmix/nxro_resmix_ws_best{ckpt_suffix}.pt'),
        ('V5d-FixL: Freeze L', f'results/resmix/nxro_resmix_fixL_best{ckpt_suffix}.pt'),
        ('V5d-FixRO: Freeze RO', f'results/resmix/nxro_resmix_fixRO_best{ckpt_suffix}.pt'),
        ('V5d-FixDiag: Freeze Diag', f'results/resmix/nxro_resmix_fixDiag_best{ckpt_suffix}.pt'),
        ('V5d-FixNL: Freeze RO+Diag', f'results/resmix/nxro_resmix_fixNL_best{ckpt_suffix}.pt'),
        ('V5d-FixPhysics: Freeze All Physics', f'results/resmix/nxro_resmix_fixPhysics_best{ckpt_suffix}.pt'),
    ]
    
    for label, ckpt_path in variants_5d:
        fcst, _ = load_checkpoint_and_reforecast(ckpt_path, NXROResidualMixModel, obs_ds,
                                                  model_kwargs={'hidden': 64, 'alpha_init': 0.1})
        if fcst is not None:
            cat5d_fcsts[label] = fcst
            print(f"  ✓ Loaded {label}")
    
    plot_within_category_comparison(cat5d_fcsts, obs_ds, 'Cat5d_ResidualMix', out_dir, eval_period, eval_suffix)
    print()
    
    # ============================================================================
    # CATEGORY 5a: NXRO-Attentive (Variants 5a, 5a-WS, 5a-FixL)
    # ============================================================================
    print("Category 5a: NXRO-Attentive")
    print("-" * 40)
    cat5a_fcsts = {}
    
    variants_5a = [
        ('V5a: Random', f'results/attentive/nxro_attentive_best{ckpt_suffix}.pt'),
        ('V5a-WS: Warm-start', f'results/attentive/nxro_attentive_ws_best{ckpt_suffix}.pt'),
        ('V5a-FixL: Freeze L', f'results/attentive/nxro_attentive_fixL_best{ckpt_suffix}.pt'),
    ]
    
    for label, ckpt_path in variants_5a:
        fcst, _ = load_checkpoint_and_reforecast(ckpt_path, NXROAttentiveModel, obs_ds,
                                                  model_kwargs={'d': 32, 'dropout': 0.1, 'mask_mode': 'th_only'})
        if fcst is not None:
            cat5a_fcsts[label] = fcst
            print(f"  ✓ Loaded {label}")
    
    plot_within_category_comparison(cat5a_fcsts, obs_ds, 'Cat5a_Attentive', out_dir, eval_period, eval_suffix)
    print()
    
    # ============================================================================
    # Between-Category Comparison: Select Best from Each
    # ============================================================================
    print("Between-Category Comparison")
    print("-" * 40)
    print("Selecting best variant from each category...")
    
    def select_best_in_category(fcst_dict, obs_ds, eval_period):
        """Select best variant by mean RMSE."""
        if not fcst_dict:
            return None, None
        
        best_label = None
        best_rmse = float('inf')
        
        for label, fcst in fcst_dict.items():
            rmse_ds = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                          by_month=False, verify_periods=eval_period)
            mean_rmse = float(np.nanmean(rmse_ds['Nino34'].values))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_label = label
        
        return best_label, fcst_dict.get(best_label)
    
    best_per_category = {}
    
    if cat1_fcsts:
        best_label, best_fcst = select_best_in_category(cat1_fcsts, obs_ds, eval_period)
        if best_fcst is not None:
            best_per_category['Cat1_Linear'] = (best_label, best_fcst)
            print(f"  Cat1: {best_label}")
    
    if cat2_fcsts:
        best_label, best_fcst = select_best_in_category(cat2_fcsts, obs_ds, eval_period)
        if best_fcst is not None:
            best_per_category['Cat2_RO'] = (best_label, best_fcst)
            print(f"  Cat2: {best_label}")
    
    if cat3_fcsts:
        best_label, best_fcst = select_best_in_category(cat3_fcsts, obs_ds, eval_period)
        if best_fcst is not None:
            best_per_category['Cat3_RODiag'] = (best_label, best_fcst)
            print(f"  Cat3: {best_label}")
    
    if cat4_fcsts:
        best_label, best_fcst = select_best_in_category(cat4_fcsts, obs_ds, eval_period)
        if best_fcst is not None:
            best_per_category['Cat4_Res'] = (best_label, best_fcst)
            print(f"  Cat4: {best_label}")
    
    if cat5d_fcsts:
        best_label, best_fcst = select_best_in_category(cat5d_fcsts, obs_ds, eval_period)
        if best_fcst is not None:
            best_per_category['Cat5d_ResidualMix'] = (best_label, best_fcst)
            print(f"  Cat5d: {best_label}")
    
    if cat5a_fcsts:
        best_label, best_fcst = select_best_in_category(cat5a_fcsts, obs_ds, eval_period)
        if best_fcst is not None:
            best_per_category['Cat5a_Attentive'] = (best_label, best_fcst)
            print(f"  Cat5a: {best_label}")
    
    print()
    plot_between_category_comparison(best_per_category, xro_fcsts, obs_ds, out_dir, eval_period, eval_suffix)
    
    # ============================================================================
    # Summary Statistics
    # ============================================================================
    print()
    print("="*80)
    print("Summary Statistics")
    print("="*80)
    
    all_categories = {
        'Cat1_Linear': cat1_fcsts,
        'Cat2_RO': cat2_fcsts,
        'Cat3_RODiag': cat3_fcsts,
        'Cat4_Res': cat4_fcsts,
        'Cat5d_ResidualMix': cat5d_fcsts,
        'Cat5a_Attentive': cat5a_fcsts,
    }
    
    summary_rows = []
    for cat_name, fcst_dict in all_categories.items():
        if not fcst_dict:
            continue
        for variant_label, fcst in fcst_dict.items():
            acc_ds = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                          by_month=False, verify_periods=eval_period)
            rmse_ds = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                           by_month=False, verify_periods=eval_period)
            mean_acc = float(np.nanmean(acc_ds['Nino34'].values))
            mean_rmse = float(np.nanmean(rmse_ds['Nino34'].values))
            summary_rows.append({
                'Category': cat_name,
                'Variant': variant_label,
                'Mean_ACC': mean_acc,
                'Mean_RMSE': mean_rmse,
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{out_dir}/variant_summary_statistics{eval_suffix}.csv', index=False)
    print(f"✓ Saved summary statistics to {out_dir}/variant_summary_statistics{eval_suffix}.csv")
    print()
    print(summary_df.to_string(index=False))
    print()
    
    print("="*80)
    print("✓ ALL COMPARISONS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {out_dir}/:")
    print("  Within-category comparisons:")
    print(f"    - Cat*_within_category_acc{eval_suffix}.png")
    print(f"    - Cat*_within_category_rmse{eval_suffix}.png")
    print(f"    - Cat*_within_category_acc{eval_suffix}.csv")
    print(f"    - Cat*_within_category_rmse{eval_suffix}.csv")
    print("  Between-category comparison:")
    print(f"    - between_category_best_acc{eval_suffix}.png")
    print(f"    - between_category_best_rmse{eval_suffix}.png")
    print("  Summary:")
    print(f"    - variant_summary_statistics{eval_suffix}.csv")
    print()


if __name__ == '__main__':
    main()

