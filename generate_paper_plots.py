#!/usr/bin/env python
"""
Generate Paper Plots

This script generates all plots required for the paper, organized according to plots.md.
The output is organized in a `plots/` directory with subdirectories reflecting the hierarchy.

Directory Structure:
    plots/
    ├── 1_deterministic_oras5/          # ORAS5-only models
    │   ├── 1_all_models_rmse_ranking.png
    │   ├── 1a_better_than_xro_barplot.png
    │   ├── 1b_better_than_xro_skill_curves.png
    │   └── 1c_core_models_ranking.png
    ├── 2_deterministic_two_stage/      # Two-stage models
    │   ├── 2_all_models_rmse_ranking.png
    │   ├── 2a_better_than_xro_barplot.png
    │   └── 2b_better_than_xro_skill_curves.png
    ├── 3_deterministic_combined/       # All models combined
    │   ├── 3_all_models_rmse_ranking.png
    │   ├── 3a_better_than_xro_barplot.png
    │   └── 3b_better_than_xro_skill_curves.png
    ├── 4_uncertainty_oras5/            # ORAS5-only CRPS rankings
    │   └── 4_crps_ranking.png
    ├── 5_uncertainty_two_stage/        # Two-stage CRPS rankings
    │   └── 5_crps_ranking.png
    ├── 6_generalization_gaps/          # Train/Test gap plots
    │   ├── 6a_oras5_train_test_gap.png
    │   ├── 6b_two_stage_train_test_gap.png
    │   └── 6c_single_vs_two_stage_gap.png
    ├── 7_ensemble_xro/                 # XRO ensemble forecasts
    │   └── (monthly subdirectories)
    ├── 8_ensemble_top5_oras5/          # Top 5 ORAS5 models ensemble
    │   └── (monthly subdirectories)
    └── 9_ensemble_top5_two_stage/      # Top 5 two-stage models ensemble
        └── (monthly subdirectories)

Usage:
    python generate_paper_plots.py
    python generate_paper_plots.py --sections 1,2,3    # Generate only sections 1-3
    python generate_paper_plots.py --sections 7        # Generate only section 7
"""

import os
import re
import argparse
import warnings
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = 'results_out_of_sample'
OUTPUT_DIR = 'plots'

# Time periods
TRAIN_START, TRAIN_END = '1979-01', '2001-12'
TEST_START, TEST_END = '2002-01', '2022-12'

# Colors for models (keyed by display name prefix)
MODEL_COLORS = {
    'XRO': '#FF1744',
    'XRO-NoSeasonal': '#FF5252',
    'XRO-LinearOnly': '#FF8A80',
    'NXRO-MLP': '#4CAF50',
    'NXRO-NeuralODE': '#9C27B0',
    'NeuralODE': '#7B1FA2',  # Pure Neural ODE (no physical priors)
    'Transformer': '#311B92',  # Pure Transformer (no physical priors)
    'NXRO-Linear': '#2196F3',
    'NXRO-Attention': '#EC407A',
    'NXRO-GCN': '#00BCD4',
    'NXRO-GNN': '#00BCD4',  # Same color as GCN
    'NXRO-GAT': '#009688',
    'NXRO-Graph': '#78909C',
    'NXRO-Graph-Learned': '#546E7A',
    'NXRO-RO': '#FF6F00',
    'NXRO-ROOnly': '#FF9800',
    'NXRO-Transformer': '#795548',
    'NXRO-Bilinear': '#607D8B',
    'NXRO-PhysReg': '#673AB7',
    'NXRO-HybridMLP': '#8BC34A',
    # Fallbacks for base name matching
    'MLP': '#4CAF50',
    'Linear': '#2196F3',
    'Attention': '#EC407A',
    'GCN': '#00BCD4',
    'GAT': '#009688',
}

# ============================================================================
# NAME MAPPING - Convert internal names to paper-friendly display names
# ============================================================================

def get_display_name(model_name):
    """
    Convert internal model names to clear, paper-friendly display names.
    
    Naming convention:
    - XRO variants: XRO, XRO-LinearOnly (no nonlinear terms)
    - NXRO models: NXRO-{ModelType} where ModelType describes the architecture
    
    Examples:
        'Res' -> 'NXRO-MLP'
        'Neural' -> 'NXRO-DeepMLP'
        'Linear' -> 'NXRO-Linear'
        'Graphpyg Gcn K3' -> 'NXRO-GCN'
        'Graphpyg Gat K5' -> 'NXRO-GAT'
        'Attentive' -> 'NXRO-Attention'
        'RO+Diag' -> 'NXRO-RO'
        'Linear XRO' -> 'XRO-LinearOnly'
    """
    original = model_name
    
    # Preserve two-stage suffix
    two_stage_suffix = ''
    if '(Two-Stage)' in model_name:
        two_stage_suffix = ' (Two-Stage)'
        model_name = model_name.replace(' (Two-Stage)', '')
    elif '(Stage 1)' in model_name:
        two_stage_suffix = ' (Stage 1)'
        model_name = model_name.replace(' (Stage 1)', '')
    
    # Preserve training variant suffixes
    variant_suffix = ''
    for suffix in [' WS', ' FixL', ' FixRO', ' FixDiag', ' FixNL', ' FixPhysics']:
        if suffix in model_name:
            variant_suffix = suffix
            model_name = model_name.replace(suffix, '')
    
    # Clean up the base name
    name_lower = model_name.lower().strip()
    
    # XRO baselines (keep as-is with minor cleanup)
    if name_lower == 'xro':
        return 'XRO' + variant_suffix + two_stage_suffix
    if name_lower == 'xro_ac0' or name_lower == 'xro ac0':
        return 'XRO-NoSeasonal' + variant_suffix + two_stage_suffix  # ac_order=0 means no seasonal variation in L
    if 'linear xro' in name_lower:
        return 'XRO-LinearOnly' + variant_suffix + two_stage_suffix  # No nonlinear RO terms
    
    # NXRO models - map to clearer names
    
    # Res -> NXRO-MLP
    if name_lower == 'res' or name_lower.startswith('res '):
        return 'NXRO-MLP' + variant_suffix + two_stage_suffix
    
    # Pure Neural ODE (no physical priors - baseline)
    if 'pure_neural_ode' in name_lower or 'pureneuralode' in name_lower or 'pure neural ode' in name_lower:
        return 'NeuralODE' + variant_suffix + two_stage_suffix
    
    # Pure Transformer (no physical priors - baseline)
    if 'pure_transformer' in name_lower or 'puretransformer' in name_lower or 'pure transformer' in name_lower:
        return 'Transformer' + variant_suffix + two_stage_suffix
    
    # Neural -> NXRO-NeuralODE
    if name_lower == 'neural' or name_lower.startswith('neural '):
        if 'phys' in name_lower:
            return 'NXRO-NeuralODE-Phys' + variant_suffix + two_stage_suffix
        return 'NXRO-NeuralODE' + variant_suffix + two_stage_suffix
    
    # Linear -> NXRO-Linear
    if name_lower == 'linear' or name_lower == 'nxro-linear':
        return 'NXRO-Linear' + variant_suffix + two_stage_suffix
    
    # Attentive -> NXRO-Attention
    if 'attentive' in name_lower or 'attention' in name_lower:
        return 'NXRO-Attention' + variant_suffix + two_stage_suffix
    
    # Graph PyG models -> NXRO-GCN-Kn or NXRO-GAT-Kn (preserve K value)
    if 'graphpyg' in name_lower or 'graph_pyg' in name_lower:
        # Extract K value if present (e.g., "Graphpyg Gcn K2" -> "K2")
        k_match = re.search(r'k(\d+)', name_lower)
        k_suffix = f'-K{k_match.group(1)}' if k_match else ''
        if 'gat' in name_lower:
            return 'NXRO-GAT' + k_suffix + variant_suffix + two_stage_suffix
        else:
            return 'NXRO-GCN' + k_suffix + variant_suffix + two_stage_suffix
    
    # Deep GCN (best model from hyperparameter search) -> NXRO-GCN
    if 'deep_gcn' in name_lower or 'deep gcn' in name_lower or 'deepgcn' in name_lower:
        return 'NXRO-GCN' + variant_suffix + two_stage_suffix
    
    # Plain Graph models -> NXRO-Graph
    if name_lower.startswith('graph') and 'pyg' not in name_lower:
        if 'fixed' in name_lower:
            return 'NXRO-Graph' + variant_suffix + two_stage_suffix
        elif 'learned' in name_lower:
            return 'NXRO-Graph-Learned' + variant_suffix + two_stage_suffix
        return 'NXRO-Graph' + variant_suffix + two_stage_suffix
    
    # RO+Diag -> NXRO-RO
    if 'rodiag' in name_lower or 'ro+diag' in name_lower or 'ro diag' in name_lower:
        return 'NXRO-RO' + variant_suffix + two_stage_suffix
    
    # ResidualMix -> NXRO-HybridMLP
    if 'residualmix' in name_lower or 'resmix' in name_lower:
        return 'NXRO-HybridMLP' + variant_suffix + two_stage_suffix
    
    # Transformer -> NXRO-Transformer
    if 'transformer' in name_lower:
        return 'NXRO-Transformer' + variant_suffix + two_stage_suffix
    
    # Bilinear -> NXRO-Bilinear
    if 'bilinear' in name_lower:
        return 'NXRO-Bilinear' + variant_suffix + two_stage_suffix
    
    # Neural Phys / PhysReg -> NXRO-PhysReg
    if 'neural phys' in name_lower or 'neural_phys' in name_lower or 'physreg' in name_lower:
        return 'NXRO-PhysReg' + variant_suffix + two_stage_suffix
    
    # RO (plain) -> NXRO-ROOnly
    if name_lower == 'ro' or name_lower.startswith('ro '):
        return 'NXRO-ROOnly' + variant_suffix + two_stage_suffix
    
    # Fallback: add NXRO- prefix if not already present
    if not model_name.startswith('NXRO-') and not model_name.startswith('XRO'):
        return 'NXRO-' + model_name.strip() + variant_suffix + two_stage_suffix
    
    return model_name + variant_suffix + two_stage_suffix


def apply_display_names(df):
    """Apply display name mapping to a DataFrame with 'Model' column."""
    df = df.copy()
    df['Model'] = df['Model'].apply(get_display_name)
    return df


def filter_pathological_models(df, rmse_threshold=10.0, crps_threshold=10.0):
    """
    Filter out models with inf, -inf, NaN, or pathologically large values in key metrics.
    Also filters out NXRO-Linear by default.
    
    Args:
        df: DataFrame with columns like 'Mean_RMSE_Test', 'Mean_ACC_Test', etc.
        rmse_threshold: Maximum reasonable RMSE value (default 10.0)
        crps_threshold: Maximum reasonable CRPS value (default 10.0)
    
    Returns:
        Filtered DataFrame with problematic models removed.
    """
    df = df.copy()
    
    # Models to exclude by default (regardless of metrics)
    excluded_models = ['NXRO-Linear', 'NXRO-Linear (Two-Stage)', 
                       'NXRO-NeuralODE', 'NXRO-NeuralODE (Two-Stage)']
    
    # Columns to check for inf/nan values
    metric_cols = [col for col in df.columns if col.startswith('Mean_') or col.endswith('_CRPS') or col == 'Avg_CRPS']
    
    if not metric_cols:
        # Still filter excluded models even if no metric columns
        if 'Model' in df.columns:
            excluded_mask = df['Model'].isin(excluded_models)
            if excluded_mask.any():
                print(f"  [!] Excluded NXRO-Linear from results")
            return df[~excluded_mask].copy()
        return df
    
    # Create mask for valid rows
    valid_mask = pd.Series(True, index=df.index)
    
    for col in metric_cols:
        if col in df.columns:
            # Check for inf/nan
            valid_mask &= np.isfinite(df[col])
            
            # Check for pathologically large values
            if 'RMSE' in col:
                valid_mask &= (df[col] < rmse_threshold)
            elif 'CRPS' in col:
                valid_mask &= (df[col] < crps_threshold)
    
    # Also exclude specific models by name
    if 'Model' in df.columns:
        valid_mask &= ~df['Model'].isin(excluded_models)
    
    # Report removed models
    removed = df[~valid_mask]
    if len(removed) > 0:
        pathological = removed[~removed['Model'].isin(excluded_models)] if 'Model' in removed.columns else removed
        excluded = removed[removed['Model'].isin(excluded_models)] if 'Model' in removed.columns else pd.DataFrame()
        
        if len(pathological) > 0:
            print(f"  [!] Filtered out {len(pathological)} model(s) with inf/NaN/pathological values:")
            for _, row in pathological.iterrows():
                print(f"      - {row['Model']}")
        if len(excluded) > 0:
            print(f"  [!] Excluded {len(excluded)} model(s) by default: {', '.join(excluded['Model'].tolist())}")
    
    return df[valid_mask].copy()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_base_model_name(model_name):
    """Extract base model name from full model name for color lookup."""
    # First apply display name conversion
    display_name = get_display_name(model_name)
    
    # Remove common suffixes
    name = display_name.replace(' (Two-Stage)', '').replace(' (Stage 1)', '')
    name = name.replace(' WS', '').replace(' FixL', '').replace(' FixRO', '')
    name = name.replace(' FixDiag', '').replace(' FixNL', '').replace(' FixPhysics', '')
    
    # Return the base NXRO-* or XRO* name
    return name.strip()


def get_model_color(model_name):
    """Get color for a model based on its display name."""
    display_name = get_display_name(model_name)
    base_name = get_base_model_name(model_name)
    
    # Try exact match first
    if base_name in MODEL_COLORS:
        return MODEL_COLORS[base_name]
    
    # Try prefix matching
    for key in MODEL_COLORS:
        if base_name.startswith(key) or key in base_name:
            return MODEL_COLORS[key]
    
    # Default color
    return '#666666'


def load_ranking_csv(csv_path):
    """Load ranking CSV and return dataframe."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ranking CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# ============================================================================
# SECTION 1-3: DETERMINISTIC RANKING PLOTS
# ============================================================================

def plot_rmse_ranking_barplot(df, out_path, title, highlight_xro=True):
    """
    Create horizontal bar plot ranking models by RMSE.
    
    Args:
        df: DataFrame with columns ['Model', 'Mean_RMSE_Test', 'Rank']
        out_path: Output path for plot
        title: Plot title
        highlight_xro: Whether to highlight XRO baseline
    """
    df_sorted = df.sort_values('Mean_RMSE_Test', ascending=True)
    n_models = len(df_sorted)
    
    # Publication mode for small number of models (like 2c with ~7 models)
    if n_models <= 10:
        # Publication quality: larger fonts, bold, appropriate bar sizing
        height_per_model = 0.6
        fig_height = max(4, n_models * height_per_model)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        
        ytick_fontsize = 14
        xlabel_fontsize = 14
        title_fontsize = 16
        value_fontsize = 12
        legend_fontsize = 12
        bar_height = 0.6
        bold_ticks = True
    else:
        # Standard mode for many models
        height_per_model = 0.28 if n_models > 30 else 0.35
        fig_height = max(5, n_models * height_per_model)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        ytick_fontsize = 9 if n_models <= 30 else 8
        xlabel_fontsize = 11
        title_fontsize = 13
        value_fontsize = 7 if n_models > 30 else 8
        legend_fontsize = 9
        bar_height = 0.8
        bold_ticks = False
    
    # Determine colors
    colors = []
    for model in df_sorted['Model']:
        if 'XRO' in model and 'Linear XRO' not in model and 'NXRO' not in model:
            colors.append('#FF1744')  # Red for XRO baseline
        else:
            colors.append(get_model_color(model))
    
    y_pos = np.arange(n_models)
    bars = ax.barh(y_pos, df_sorted['Mean_RMSE_Test'], height=bar_height, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['Model'], fontsize=ytick_fontsize, 
                       fontweight='bold' if bold_ticks else 'normal')
    ax.set_xlabel('Test RMSE (°C)', fontsize=xlabel_fontsize, fontweight='bold' if bold_ticks else 'normal')
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.invert_yaxis()  # Best at top
    ax.set_ylim(n_models - 0.5, -0.5)  # Tight y-axis limits
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['Mean_RMSE_Test'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=value_fontsize, 
               fontweight='bold' if bold_ticks else 'normal')
    
    # Add XRO reference line if present
    if highlight_xro:
        xro_rows = df_sorted[df_sorted['Model'] == 'XRO']
        if len(xro_rows) > 0:
            xro_rmse = xro_rows['Mean_RMSE_Test'].values[0]
            ax.axvline(xro_rmse, color='#FF1744', linestyle='--', linewidth=2, alpha=0.7, label='XRO Baseline')
            # Place legend at bottom right, outside the plot area
            ax.legend(fontsize=legend_fontsize, loc='upper right', 
                     bbox_to_anchor=(1.0, -0.08), ncol=1, frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_better_than_xro_barplot(df, out_path, title):
    """
    Create bar plot showing only models better than XRO.
    
    Args:
        df: DataFrame with columns ['Model', 'Mean_RMSE_Test']
        out_path: Output path for plot
        title: Plot title
    """
    # Get XRO RMSE
    xro_rows = df[df['Model'] == 'XRO']
    if len(xro_rows) == 0:
        print(f"  [!] XRO baseline not found in data, skipping {out_path}")
        return
    
    xro_rmse = xro_rows['Mean_RMSE_Test'].values[0]
    
    # Filter models better than XRO
    better_df = df[df['Mean_RMSE_Test'] < xro_rmse].copy()
    
    if len(better_df) == 0:
        print(f"  [!] No models better than XRO, skipping {out_path}")
        return
    
    better_df = better_df.sort_values('Mean_RMSE_Test', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(better_df) * 0.4)))
    
    colors = [get_model_color(m) for m in better_df['Model']]
    y_pos = np.arange(len(better_df))
    
    bars = ax.barh(y_pos, better_df['Mean_RMSE_Test'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(better_df['Model'], fontsize=10)
    ax.set_xlabel('Test RMSE (°C)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add XRO reference line
    ax.axvline(xro_rmse, color='#FF1744', linestyle='--', linewidth=2, alpha=0.7, label=f'XRO ({xro_rmse:.3f})')
    ax.legend(fontsize=9, loc='lower right')
    
    # Add value labels with improvement
    for i, (bar, val) in enumerate(zip(bars, better_df['Mean_RMSE_Test'])):
        improvement = xro_rmse - val
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f} (−{improvement:.3f})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")
    
    return better_df['Model'].tolist()


def plot_skill_curves(models_to_plot, results_dir, obs_ds, out_path, title, test_period):
    """
    Plot RMSE vs lead time for selected models.
    
    Args:
        models_to_plot: List of model names to plot
        results_dir: Directory containing model results
        obs_ds: Observation dataset
        out_path: Output path for plot
        title: Plot title
        test_period: Time slice for test period
    """
    from utils.xro_utils import calc_forecast_skill, nxro_reforecast
    from XRO.core import XRO
    from rank_all_variants_out_of_sample import infer_model_class_and_kwargs
    import torch
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Fit and plot XRO baseline
    train_ds = obs_ds.sel(time=slice(TRAIN_START, TRAIN_END))
    xro = XRO(ncycle=12, ac_order=2)
    xro_fit = xro.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    xro_fcst = xro.reforecast(fit_ds=xro_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
    xro_rmse = calc_forecast_skill(xro_fcst, obs_ds, metric='rmse', is_mv3=True,
                                   by_month=False, verify_periods=test_period)
    
    leads = xro_rmse['Nino34'].lead.values
    ax.plot(leads, xro_rmse['Nino34'].values, label='XRO (Baseline)', 
           lw=3, linestyle='--', color='#FF1744', alpha=0.9)
    
    # Plot each NXRO model
    for model_name in models_to_plot:
        if model_name == 'XRO' or 'Linear XRO' in model_name:
            continue
            
        # Find checkpoint path
        ckpt_path = find_checkpoint_path(model_name, results_dir)
        if ckpt_path is None:
            print(f"    [!] Checkpoint not found for {model_name}")
            continue
        
        try:
            model_class, model_kwargs = infer_model_class_and_kwargs(ckpt_path)
            if model_class is None:
                continue
            
            ckpt = torch.load(ckpt_path, map_location='cpu')
            var_order = ckpt['var_order']
            
            model = model_class(**model_kwargs)
            model.load_state_dict(ckpt['state_dict'])
            
            fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device='cpu')
            rmse = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=test_period)
            
            color = get_model_color(model_name)
            ax.plot(leads, rmse['Nino34'].values, label=model_name, 
                   lw=2, marker='o', markersize=4, color=color)
            
        except Exception as e:
            print(f"    [!] Error loading {model_name}: {e}")
            continue
    
    ax.set_xlabel('Forecast Lead (months)', fontsize=12)
    ax.set_ylabel('RMSE (°C)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim([0, 21])
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def find_checkpoint_path(model_name, results_dir):
    """Find checkpoint path for a model name."""
    import glob
    
    # Clean model name for file search
    search_name = model_name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
    search_name = search_name.replace('two-stage', 'real_finetuned')
    
    # Map model names (both old and new display names) to directories
    # Format: search_key -> (subdir, filename_pattern)
    model_dir_map = {
        # New display names -> directories
        'nxro-mlp': ('res', 'nxro_res_*best*.pt'),
        'nxro-neuralode': ('neural', 'nxro_neural_*best*.pt'),
        'nxro-linear': ('linear', 'nxro_linear_*best*.pt'),
        'nxro-attention': ('attentive', 'nxro_attentive_*best*.pt'),
        'nxro-gcn': ('graphpyg', 'nxro_graphpyg_gcn_*.pt'),
        'nxro-gat': ('graphpyg', 'nxro_graphpyg_gat_*.pt'),
        'nxro-graph': ('graph', 'nxro_graph_*best*.pt'),
        'nxro-ro': ('rodiag', 'nxro_rodiag_*best*.pt'),
        'nxro-hybridmlp': ('resmix', 'nxro_resmix_*best*.pt'),
        'nxro-transformer': ('transformer', 'nxro_transformer_*best*.pt'),
        'nxro-physreg': ('neural_phys', 'nxro_neural_phys_*best*.pt'),
        'nxro-bilinear': ('bilinear', 'nxro_bilinear_*best*.pt'),
        'neuralode': ('pure_neural_ode', 'pure_neural_ode_*best*.pt'),
        'transformer': ('pure_transformer', 'pure_transformer_*best*.pt'),
        # Old names (for backwards compatibility)
        'res': ('res', 'nxro_res_*best*.pt'),
        'pure_neural_ode': ('pure_neural_ode', 'pure_neural_ode_*best*.pt'),
        'pure_transformer': ('pure_transformer', 'pure_transformer_*best*.pt'),
        'neural': ('neural', 'nxro_neural_*best*.pt'),
        'neural_phys': ('neural_phys', 'nxro_neural_phys_*best*.pt'),
        'neural phys': ('neural_phys', 'nxro_neural_phys_*best*.pt'),
        'linear': ('linear', 'nxro_linear_*best*.pt'),
        'attentive': ('attentive', 'nxro_attentive_*best*.pt'),
        'graph': ('graph', 'nxro_graph_*best*.pt'),
        'graphpyg': ('graphpyg', 'nxro_graphpyg_*.pt'),
        'rodiag': ('rodiag', 'nxro_rodiag_*best*.pt'),
        'ro+diag': ('rodiag', 'nxro_rodiag_*best*.pt'),
        'transformer': ('transformer', 'nxro_transformer_*best*.pt'),
        'resmix': ('resmix', 'nxro_resmix_*best*.pt'),
        'residualmix': ('resmix', 'nxro_resmix_*best*.pt'),
        'bilinear': ('bilinear', 'nxro_bilinear_*best*.pt'),
    }
    
    # Check for two-stage (finetuned) models
    is_two_stage = 'two-stage' in search_name or 'finetuned' in search_name
    
    # Try to find the checkpoint by checking each key
    for key, (subdir, file_pattern) in model_dir_map.items():
        if key in search_name:
            search_dir = f'{results_dir}/{subdir}'
            if not os.path.exists(search_dir):
                continue
                
            # For two-stage, look for finetuned checkpoints first
            if is_two_stage:
                pattern = f'{search_dir}/**/nxro_*finetuned*.pt'
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    return matches[0]
            
            # Otherwise look for best checkpoints
            pattern = f'{search_dir}/**/{file_pattern}'
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Prefer 'best' over 'finetuned' for single-stage
                best_matches = [m for m in matches if 'best' in m]
                if best_matches:
                    return best_matches[0]
                return matches[0]
    
    return None


def generate_deterministic_plots(df, out_dir, section_num, section_name, results_dir=None, obs_ds=None, strip_two_stage_suffix=False):
    """Generate all deterministic ranking plots for a section.
    
    Args:
        strip_two_stage_suffix: If True, remove ' (Two-Stage)' suffix from model names
    """
    ensure_dir(out_dir)
    
    # Apply display names to make plots clearer
    df = apply_display_names(df)
    
    # Optionally strip '(Two-Stage)' suffix for cleaner display
    if strip_two_stage_suffix:
        df = df.copy()
        df['Model'] = df['Model'].str.replace(' (Two-Stage)', '', regex=False)
    
    # Filter out models with inf/NaN values
    df = filter_pathological_models(df)
    
    # Plot 1: All models RMSE ranking
    plot_rmse_ranking_barplot(
        df, 
        f'{out_dir}/{section_num}_all_models_rmse_ranking.png',
        f'All Models RMSE Ranking ({section_name}, Out-of-Sample)'
    )
    
    # Plot 1a: Models better than XRO
    better_models = plot_better_than_xro_barplot(
        df,
        f'{out_dir}/{section_num}a_better_than_xro_barplot.png',
        f'Models Better Than XRO ({section_name})'
    )
    
    # Plot 1b: Skill curves for models better than XRO
    if better_models and results_dir and obs_ds is not None:
        test_period = slice(TEST_START, TEST_END)
        plot_skill_curves(
            better_models[:10],  # Top 10 at most
            results_dir,
            obs_ds,
            f'{out_dir}/{section_num}b_better_than_xro_skill_curves.png',
            f'Forecast Skill: Models Better Than XRO ({section_name})',
            test_period
        )


# ============================================================================
# SECTION 4-5: UNCERTAINTY RANKING PLOTS (CRPS)
# ============================================================================

def load_crps_results(results_dir, include_two_stage=False):
    """Load CRPS results from stochastic evaluation files."""
    import glob
    import re
    
    results = {}
    
    # XRO baseline
    xro_pattern = f'{results_dir}/xro_baseline/*_stochastic_eval_lead_metrics.csv'
    xro_matches = glob.glob(xro_pattern)
    if xro_matches:
        results['XRO'] = pd.read_csv(xro_matches[0])
    
    # NXRO models - map directory to model type
    model_dirs = {
        'res': 'NXRO-MLP',
        'linear': 'NXRO-Linear', 
        'attentive': 'NXRO-Attention',
        'neural': 'NXRO-NeuralODE',
        'pure_neural_ode': 'NeuralODE',
        'pure_transformer': 'Transformer',
        'rodiag': 'NXRO-RO',
        'graphpyg': 'NXRO-Graph',
        'graph': 'NXRO-Graph',
    }
    
    for model_dir, base_display_name in model_dirs.items():
        pattern = f'{results_dir}/{model_dir}/**/*stochastic*eval*metrics.csv'
        matches = glob.glob(pattern, recursive=True)
        
        for match in matches:
            basename = os.path.basename(match).lower()
            
            # Determine noise type from filename
            noise_type = None
            if 'sim_noise' in basename:
                noise_type = 'Sim Noise'
            elif 'arp2' in basename:
                noise_type = 'ARP2'
            elif 'arp3' in basename:
                noise_type = 'ARP3'
            elif 'arp4' in basename:
                noise_type = 'ARP4'
            elif 'arp5' in basename:
                noise_type = 'ARP5'
            
            # Determine if two-stage
            is_two_stage = ('stage2' in basename or 'extra_data' in basename or 'finetuned' in basename)
            
            # Skip if not including two-stage and this is two-stage
            if not include_two_stage and is_two_stage:
                continue
            
            # Skip noise variants for cleaner plots (only include base stochastic)
            # unless it's the only file type available
            if noise_type is not None:
                continue  # For paper plots, skip noise variants
            
            # Handle graphpyg variants (GCN vs GAT)
            display_name = base_display_name
            if model_dir == 'graphpyg':
                if 'gat' in basename:
                    display_name = 'NXRO-GAT'
                elif 'gcn' in basename:
                    display_name = 'NXRO-GCN'
            
            # Add two-stage suffix
            if is_two_stage:
                display_name += ' (Two-Stage)'
            
            # Only add if not already in results (avoid duplicates)
            if display_name not in results:
                try:
                    results[display_name] = pd.read_csv(match)
                except Exception as e:
                    print(f"    [!] Error loading {match}: {e}")
    
    return results


def plot_crps_ranking(results, out_path, title, top_n=15):
    """
    Create bar plot ranking models by average CRPS.
    
    Args:
        results: Dictionary of {model_name: DataFrame with 'crps' column}
        out_path: Output path for plot
        title: Plot title
        top_n: Number of top models to show
    """
    # Models to exclude by default
    excluded_models = ['NXRO-Linear', 'NXRO-Linear (Two-Stage)', 
                       'NXRO-NeuralODE', 'NXRO-NeuralODE (Two-Stage)']
    
    # Compute average CRPS
    summary = []
    for model_name, df in results.items():
        avg_crps = float(np.nanmean(df['crps'].values))
        summary.append({'Model': model_name, 'Avg_CRPS': avg_crps})
    
    summary_df = pd.DataFrame(summary)
    
    # Filter out excluded models by default
    excluded_mask = summary_df['Model'].isin(excluded_models)
    if excluded_mask.any():
        print(f"  [!] Excluded {excluded_mask.sum()} model(s) by default: {', '.join(summary_df[excluded_mask]['Model'].tolist())}")
    summary_df = summary_df[~excluded_mask]
    
    # Filter out models with inf/NaN or pathologically large CRPS values
    # CRPS should typically be in the range 0-2 for normalized data
    CRPS_THRESHOLD = 10.0
    valid_mask = np.isfinite(summary_df['Avg_CRPS']) & (summary_df['Avg_CRPS'] < CRPS_THRESHOLD)
    removed = summary_df[~valid_mask]
    if len(removed) > 0:
        print(f"  [!] Filtered out {len(removed)} model(s) with inf/NaN/pathological CRPS:")
        for _, row in removed.iterrows():
            print(f"      - {row['Model']} (CRPS={row['Avg_CRPS']:.2e})")
    summary_df = summary_df[valid_mask]
    
    summary_df = summary_df.sort_values('Avg_CRPS').head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(summary_df) * 0.4)))
    
    # Determine colors
    colors = []
    for model in summary_df['Model']:
        if model == 'XRO':
            colors.append('#FF1744')
        else:
            colors.append(get_model_color(model))
    
    y_pos = np.arange(len(summary_df))
    bars = ax.barh(y_pos, summary_df['Avg_CRPS'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df['Model'], fontsize=10)
    ax.set_xlabel('Average CRPS (lower is better)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, summary_df['Avg_CRPS'])):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, 
               f'{val:.4f}', va='center', fontsize=9)
    
    # Mark XRO reference
    xro_rows = summary_df[summary_df['Model'] == 'XRO']
    if len(xro_rows) > 0:
        xro_crps = xro_rows['Avg_CRPS'].values[0]
        ax.axvline(xro_crps, color='#FF1744', linestyle='--', linewidth=2, alpha=0.7, label='XRO Baseline')
        ax.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")
    
    return summary_df


# ============================================================================
# SECTION 6: GENERALIZATION GAP PLOTS
# ============================================================================

def plot_train_test_gap_stacked(df, out_path, title):
    """
    Create stacked bar plot showing train/test RMSE for each model.
    
    Args:
        df: DataFrame with columns ['Model', 'Mean_RMSE_Train', 'Mean_RMSE_Test']
        out_path: Output path for plot
        title: Plot title
    """
    # Apply display names and filter pathological models
    df = apply_display_names(df)
    df = filter_pathological_models(df)
    df_sorted = df.sort_values('Mean_RMSE_Test', ascending=True).head(15)  # Top 15
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df_sorted))
    width = 0.35
    
    train_bars = ax.bar(x - width/2, df_sorted['Mean_RMSE_Train'], width, 
                        label='Training RMSE', color='tab:blue', alpha=0.7)
    test_bars = ax.bar(x + width/2, df_sorted['Mean_RMSE_Test'], width,
                       label='Test RMSE', color='tab:orange', alpha=0.7)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('RMSE (°C)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Model'], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add gap annotation
    for i, (train, test) in enumerate(zip(df_sorted['Mean_RMSE_Train'], df_sorted['Mean_RMSE_Test'])):
        gap = test - train
        if gap > 0:
            ax.annotate(f'+{gap:.3f}', xy=(i, max(train, test) + 0.02), 
                       ha='center', fontsize=7, color='red')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_single_vs_two_stage_gap(df_single, df_two, out_path, title, filter_models=None, rename_map=None):
    """
    Create stacked bar plot comparing single-stage vs two-stage test performance.
    
    Args:
        df_single: DataFrame for single-stage models
        df_two: DataFrame for two-stage models
        out_path: Output path for plot
        title: Plot title
        filter_models: Optional list of model names to include (after display name conversion)
        rename_map: Optional dict to rename models for display (e.g., {'NXRO-GCN-K2': 'NXRO-GNN'})
    """
    # Apply display names and filter pathological models
    df_single = apply_display_names(df_single)
    df_single = filter_pathological_models(df_single)
    df_two = apply_display_names(df_two)
    df_two = filter_pathological_models(df_two)
    
    # Match models between single and two-stage
    df_two_clean = df_two.copy()
    df_two_clean['Match_Name'] = df_two_clean['Model'].str.replace(' (Two-Stage)', '', regex=False)
    
    merged = pd.merge(
        df_single[['Model', 'Mean_RMSE_Test']].rename(columns={'Mean_RMSE_Test': 'RMSE_Single'}),
        df_two_clean[['Match_Name', 'Mean_RMSE_Test']].rename(columns={'Match_Name': 'Model', 'Mean_RMSE_Test': 'RMSE_TwoStage'}),
        on='Model',
        how='inner'
    )
    
    if len(merged) == 0:
        print(f"  [!] No matching models found for single vs two-stage comparison")
        return
    
    # Remove XRO models (they don't have true two-stage variants, values are identical)
    xro_models = ['XRO', 'XRO-LinearOnly', 'XRO-NoSeasonal']
    merged = merged[~merged['Model'].isin(xro_models)]
    
    # Apply filter if specified
    if filter_models is not None:
        merged = merged[merged['Model'].isin(filter_models)]
        if len(merged) == 0:
            print(f"  [!] No models found after filtering to: {filter_models}")
            return
    
    if len(merged) == 0:
        print(f"  [!] No non-XRO models found for comparison")
        return
    
    # Apply rename map if specified
    if rename_map is not None:
        merged['Model'] = merged['Model'].replace(rename_map)
    
    merged = merged.sort_values('RMSE_Single')
    
    n_models = len(merged)
    
    # Adjust aesthetics based on number of models
    if n_models <= 8:
        # Publication mode: larger bold fonts, appropriate bar sizing, lighter colors
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.30  # Moderate bar width for clear visibility
        alpha = 0.65
        # Lighter colors for publication
        color_blue = '#6BAED6'  # Light blue
        color_green = '#74C476'  # Light green
        color_red = '#FC9272'  # Light red/coral
        legend_fontsize = 13
        xtick_fontsize = 14
        xlabel_fontsize = 14
        ylabel_fontsize = 14
        title_fontsize = 16
        bold_text = True
    else:
        # Many models: standard settings
        fig, ax = plt.subplots(figsize=(14, 7))
        width = 0.35
        alpha = 0.7
        color_blue = 'tab:blue'
        color_green = '#2ca02c'
        color_red = '#d62728'
        legend_fontsize = 10
        xtick_fontsize = 11
        xlabel_fontsize = 12
        ylabel_fontsize = 12
        title_fontsize = 14
        bold_text = False
    
    x = np.arange(n_models)
    
    # ORAS5-only bars
    single_bars = ax.bar(x - width/2, merged['RMSE_Single'], width,
                         label='ORAS5 only', color=color_blue, alpha=alpha)
    
    # CESM pretrained bars: green if improved (lower RMSE), red if worse (higher RMSE)
    two_colors = [color_green if two < single else color_red 
                  for single, two in zip(merged['RMSE_Single'], merged['RMSE_TwoStage'])]
    two_bars = ax.bar(x + width/2, merged['RMSE_TwoStage'], width, color=two_colors, alpha=alpha)
    
    # Create legend with both colors for CESM pretrained
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_blue, alpha=alpha, label='ORAS5 only'),
        Patch(facecolor=color_green, alpha=alpha, label='CESM pretrained (improved)'),
        Patch(facecolor=color_red, alpha=alpha, label='CESM pretrained (worse)'),
    ]
    
    ax.set_xlabel('Model', fontsize=xlabel_fontsize, fontweight='bold' if bold_text else 'normal')
    ax.set_ylabel('Test RMSE (°C)', fontsize=ylabel_fontsize, fontweight='bold' if bold_text else 'normal')
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=40)
    ax.set_xticks(x)
    ax.set_xticklabels(merged['Model'], rotation=45, ha='right', fontsize=xtick_fontsize,
                       fontweight='bold' if bold_text else 'normal')
    # Place legend in one row below title, above bars
    ax.legend(handles=legend_elements, fontsize=legend_fontsize, loc='upper center', 
              bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# SECTION 7-9: ENSEMBLE FORECAST PLUME PLOTS
# ============================================================================

def plot_ensemble_forecast_plume(fcst_ds, obs_ds, init_date, out_path, title, color='blue'):
    """
    Plot ensemble forecast plume for a single initialization date.
    
    Args:
        fcst_ds: xarray Dataset with ensemble forecasts (dims: init, lead, member)
        obs_ds: xarray Dataset with observations
        init_date: Initialization date string (e.g., '1979-01')
        out_path: Output path for plot
        title: Plot title
        color: Color for forecast
    """
    from dateutil.relativedelta import relativedelta
    import matplotlib.dates as mdates
    
    try:
        fcst_var = fcst_ds['Nino34'].sel(init=init_date)
    except KeyError:
        print(f"    [!] Init date {init_date} not in forecast")
        return False
    
    fcst_mean = fcst_var.mean('member').squeeze()
    fcst_std = fcst_var.std('member').squeeze()
    
    nlead = len(fcst_mean.lead)
    
    # Setup time axis
    xdate_init = datetime.datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
    xdate_strt = xdate_init + relativedelta(months=-2)
    xdate_last = xdate_init + relativedelta(months=nlead-1)
    xtime_fcst = [xdate_init + relativedelta(months=int(i)) for i in range(nlead)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot forecast
    ax.plot(xtime_fcst, fcst_mean.values, c=color, marker='.', lw=2.5, label='Ensemble Mean')
    ax.fill_between(xtime_fcst, 
                   fcst_mean.values - fcst_std.values, 
                   fcst_mean.values + fcst_std.values,
                   fc=color, alpha=0.3, label='±1 Std Dev')
    
    # Plot observations
    try:
        sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
        ax.plot(sel_obs.time.values, sel_obs.values, c='black', 
               marker='o', markersize=3, lw=2, label='Observed', alpha=0.8)
    except:
        pass
    
    # Formatting
    ax.axhline(0, c='gray', ls='-', lw=0.5, alpha=0.5)
    ax.axhline(0.5, c='red', ls='--', lw=1, alpha=0.3)
    ax.axhline(-0.5, c='blue', ls='--', lw=1, alpha=0.3)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim([xdate_strt, xdate_last])
    ax.set_ylim([-3, 3])
    ax.set_ylabel('Niño3.4 SSTA (°C)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return True


def generate_monthly_ensemble_plots(fcst_ds, obs_ds, out_dir, model_name, start_year=1979, end_year=2022):
    """
    Generate ensemble forecast plots for each month of each year.
    
    Args:
        fcst_ds: xarray Dataset with ensemble forecasts
        obs_ds: xarray Dataset with observations
        out_dir: Output directory
        model_name: Name of the model for titles
        start_year: Starting year
        end_year: Ending year
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    color = get_model_color(model_name)
    
    for month_idx, month_name in enumerate(months, 1):
        month_dir = ensure_dir(f'{out_dir}/{month_name}')
        
        for year in range(start_year, end_year + 1):
            init_date = f'{year}-{month_idx:02d}'
            out_path = f'{month_dir}/{model_name.replace(" ", "_")}_{year}_{month_name}.png'
            
            success = plot_ensemble_forecast_plume(
                fcst_ds, obs_ds, init_date, out_path,
                f'{model_name} Forecast - Init: {month_name} {year}',
                color=color
            )
            
            if success:
                print(f"    ✓ {month_name} {year}")


def plot_special_case_3x2(model_fcsts, obs_ds, init_date, out_path, title):
    """
    Create a 3x2 subplot showing ensemble forecasts for 6 models.
    
    Args:
        model_fcsts: Dict of {model_name: xarray.Dataset with forecasts}
        obs_ds: xarray Dataset with observations
        init_date: Initialization date string (e.g., '1997-04')
        out_path: Output path for plot
        title: Overall plot title
    """
    import datetime
    from dateutil.relativedelta import relativedelta
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    model_names = list(model_fcsts.keys())
    
    for idx, (model_name, fcst_ds) in enumerate(model_fcsts.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        color = get_model_color(model_name)
        
        try:
            fcst_var = fcst_ds['Nino34'].sel(init=init_date)
        except KeyError:
            ax.text(0.5, 0.5, f'No data for {init_date}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(model_name, fontsize=11, fontweight='bold')
            continue
        
        fcst_mean = fcst_var.mean('member').squeeze()
        fcst_std = fcst_var.std('member').squeeze()
        
        nlead = len(fcst_mean.lead)
        
        # Setup time axis
        xdate_init = datetime.datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
        xdate_strt = xdate_init + relativedelta(months=-2)
        xdate_last = xdate_init + relativedelta(months=nlead-1)
        xtime_fcst = [xdate_init + relativedelta(months=int(i)) for i in range(nlead)]
        
        # Plot forecast
        ax.plot(xtime_fcst, fcst_mean.values, c=color, marker='.', lw=2, label='Ensemble Mean')
        ax.fill_between(xtime_fcst, 
                       fcst_mean.values - fcst_std.values, 
                       fcst_mean.values + fcst_std.values,
                       fc=color, alpha=0.3, label='±1σ')
        
        # Plot observations
        try:
            sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
            ax.plot(sel_obs.time.values, sel_obs.values, c='black', 
                   marker='o', markersize=2, lw=1.5, label='Observed', alpha=0.8)
        except:
            pass
        
        # Formatting
        ax.axhline(0, c='gray', ls='-', lw=0.5, alpha=0.5)
        ax.axhline(0.5, c='red', ls='--', lw=0.8, alpha=0.3)
        ax.axhline(-0.5, c='blue', ls='--', lw=0.8, alpha=0.3)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_xlim([xdate_strt, xdate_last])
        ax.set_ylim([-4, 5])  # Fixed y-axis range for all subplots (expanded for extreme events)
        ax.set_ylabel('Niño3.4 (°C)', fontsize=9)
        ax.set_title(model_name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return True


def plot_special_case_1x2(model_fcsts, obs_ds, init_date, out_path, model_order=None):
    """
    Create a 1x2 comparison plot for specific models.
    
    Args:
        model_fcsts: dict of model_name -> xarray Dataset with forecasts
        obs_ds: xarray Dataset with observations
        init_date: initialization date string (YYYY-MM)
        out_path: output path for the figure
        model_order: list of model names to plot (in order), defaults to dict keys
    """
    from dateutil.relativedelta import relativedelta
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if model_order is None:
        model_order = list(model_fcsts.keys())[:2]
    
    for idx, model_name in enumerate(model_order):
        if idx >= 2:
            break
        if model_name not in model_fcsts:
            print(f"  [!] Model {model_name} not found in forecasts")
            continue
            
        ax = axes[idx]
        fcst_ds = model_fcsts[model_name]
        color = get_model_color(model_name)
        
        try:
            fcst_var = fcst_ds['Nino34'].sel(init=init_date)
        except KeyError:
            ax.text(0.5, 0.5, f'No data for {init_date}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            continue
        
        fcst_mean = fcst_var.mean('member').squeeze()
        fcst_std = fcst_var.std('member').squeeze()
        
        nlead = len(fcst_mean.lead)
        
        # Setup time axis
        xdate_init = datetime.datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
        xdate_strt = xdate_init + relativedelta(months=-2)
        xdate_last = xdate_init + relativedelta(months=nlead-1)
        xtime_fcst = [xdate_init + relativedelta(months=int(i)) for i in range(nlead)]
        
        # Plot forecast
        ax.plot(xtime_fcst, fcst_mean.values, c=color, marker='.', lw=2, label='Ensemble Mean')
        ax.fill_between(xtime_fcst, 
                       fcst_mean.values - fcst_std.values, 
                       fcst_mean.values + fcst_std.values,
                       fc=color, alpha=0.3, label='±1σ')
        
        # Plot observations
        try:
            sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
            ax.plot(sel_obs.time.values, sel_obs.values, c='black', 
                   marker='o', markersize=3, lw=1.5, label='Observed', alpha=0.8)
        except:
            pass
        
        # Formatting
        ax.axhline(0, c='gray', ls='-', lw=0.5, alpha=0.5)
        ax.axhline(0.5, c='red', ls='--', lw=0.8, alpha=0.3)
        ax.axhline(-0.5, c='blue', ls='--', lw=0.8, alpha=0.3)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_xlim([xdate_strt, xdate_last])
        ax.set_ylim([-4, 5])
        ax.set_ylabel('Niño3.4 (°C)', fontsize=11)
        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return True


def generate_special_case_plots(results_dir, output_dir, obs_ds):
    """
    Generate special case 3x2 plots for key ENSO events and spring barrier.
    
    Uses top 5 CRPS models + XRO baseline.
    """
    # Define special cases
    special_cases = {
        # Spring Predictability Barrier
        'spring_barrier_1997': ('1997-04', 'Spring Barrier: April 1997 (Pre-El Niño)'),
        'spring_barrier_2015': ('2015-04', 'Spring Barrier: April 2015 (Pre-El Niño)'),
        'spring_barrier_1998': ('1998-04', 'Spring Barrier: April 1998 (El Niño → La Niña)'),
        
        # Major El Niño 1997-98
        'elnino_1997_development': ('1997-07', '1997-98 El Niño: Development Phase (July 1997)'),
        'elnino_1997_peak': ('1997-12', '1997-98 El Niño: Near Peak (Dec 1997)'),
        'elnino_1998_transition': ('1998-01', '1997-98 El Niño: Peak & Transition (Jan 1998)'),
        
        # Major El Niño 2015-16
        'elnino_2015_spring': ('2015-05', '2015-16 El Niño: Spring Development (May 2015)'),
        'elnino_2015_summer': ('2015-07', '2015-16 El Niño: Summer Intensification (July 2015)'),
        'elnino_2016_peak': ('2016-01', '2015-16 El Niño: Near Peak (Jan 2016)'),
        
        # La Niña events
        'lanina_2010_dev': ('2010-07', '2010-11 La Niña: Development (July 2010)'),
        'lanina_2011_peak': ('2011-01', '2010-11 La Niña: Strong Phase (Jan 2011)'),
        
        # Transition cases
        'transition_1998_summer': ('1998-07', 'Post-El Niño Transition (July 1998)'),
    }
    
    # Find stochastic forecast files for top models
    # Use specific file patterns that match the actual naming convention
    model_files = {
        'XRO': f'{results_dir}/xro_baseline/xro_stochastic_fcst.nc',
        'NXRO-Linear': f'{results_dir}/linear/NXRO_LINEAR_stochastic_forecasts.nc',
        'NXRO-Attention': f'{results_dir}/attentive/NXRO_ATTENTIVE_stochastic_forecasts.nc',
        'NXRO-MLP': f'{results_dir}/res/NXRO_RES_stochastic_forecasts.nc',
        'NXRO-NeuralODE': f'{results_dir}/neural/NXRO_NEURAL_stochastic_forecasts.nc',
        'NXRO-GCN': f'{results_dir}/graphpyg/gcn_k3/NXRO_GRAPHPYG_GCN_K3_stochastic_forecasts.nc',
    }
    
    # Load available forecasts
    model_fcsts = {}
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                fcst_ds = xr.open_dataset(filepath)
                model_fcsts[model_name] = fcst_ds
                print(f"  ✓ Loaded {model_name}: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"  [!] Failed to load {model_name}: {e}")
        else:
            print(f"  [!] File not found for {model_name}: {filepath}")
    
    if len(model_fcsts) == 0:
        print("  [!] No stochastic forecasts found. Run stochastic evaluation first.")
        return
    
    print(f"\n  Loaded {len(model_fcsts)} models for special case plots\n")
    
    # Generate plots for each special case
    for case_name, (init_date, title) in special_cases.items():
        out_path = f'{output_dir}/{case_name}.png'
        
        success = plot_special_case_3x2(model_fcsts, obs_ds, init_date, out_path, title)
        
        if success:
            print(f"  ✓ {case_name}: {init_date}")
        else:
            print(f"  [!] Failed: {case_name}")
    
    # Generate 1x2 comparison plots for selected cases
    print("\n  Generating 1x2 comparison plots...")
    
    # XRO vs NXRO-MLP for 2015-16 El Niño Spring
    comparison_1x2 = {
        'elnino_2015_spring_xro_vs_mlp': ('2015-05', ['XRO', 'NXRO-MLP']),
    }
    
    for case_name, (init_date, model_list) in comparison_1x2.items():
        out_path = f'{output_dir}/{case_name}.png'
        success = plot_special_case_1x2(model_fcsts, obs_ds, init_date, out_path, model_order=model_list)
        if success:
            print(f"  ✓ {case_name}: {init_date} ({' vs '.join(model_list)})")
        else:
            print(f"  [!] Failed: {case_name}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Paper Plots')
    parser.add_argument('--sections', type=str, default='all',
                       help='Comma-separated section numbers (e.g., "1,2,3") or "all"')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                       help='Results directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Parse sections
    if args.sections == 'all':
        sections = list(range(1, 10))  # Sections 1-9
    else:
        sections = [int(s.strip()) for s in args.sections.split(',')]
    
    print("="*80)
    print("GENERATING PAPER PLOTS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sections to generate: {sections}")
    print("="*80)
    print()
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load observation data
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    
    # Load ranking CSVs
    # Use rmse CSV as primary source (most complete with all models including pure baselines)
    single_stage_csv = f'{args.results_dir}/rankings/all_variants_ranked_rmse_out_of_sample.csv'
    two_stage_csv = f'{args.results_dir}/rankings/all_variants_ranked_rmse_out_of_sample_two_stage.csv'
    
    df_single = None
    df_two = None
    
    if os.path.exists(single_stage_csv):
        df_single = load_ranking_csv(single_stage_csv)
        print(f"✓ Loaded single-stage rankings: {len(df_single)} models")
    else:
        print(f"[!] Single-stage rankings not found: {single_stage_csv}")
    
    if os.path.exists(two_stage_csv):
        df_two = load_ranking_csv(two_stage_csv)
        print(f"✓ Loaded two-stage rankings: {len(df_two)} models")
    else:
        # Try alternative path
        alt_path = 'results_out_of_sample_no_wwv/rankings/all_variants_ranked_combined_out_of_sample_two_stage_no_wwv.csv'
        if os.path.exists(alt_path):
            df_two = load_ranking_csv(alt_path)
            print(f"✓ Loaded two-stage rankings (no WWV): {len(df_two)} models")
        else:
            print(f"[!] Two-stage rankings not found")
    
    print()
    
    # ========================================================================
    # SECTION 1: DETERMINISTIC RANKING - ORAS5 ONLY
    # ========================================================================
    if 1 in sections and df_single is not None:
        print("="*60)
        print("SECTION 1: Deterministic Ranking (ORAS5-only)")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/1_deterministic_oras5')
        generate_deterministic_plots(df_single, out_dir, '1', 'ORAS5-only', 
                                    args.results_dir, obs_ds)
        
        # Additional plot: Core models only (with XRO baseline)
        df_single_display = apply_display_names(df_single.copy())
        df_single_display = filter_pathological_models(df_single_display)
        
        # Filter to core models only (including baselines)
        # Note: NXRO-Linear and NXRO-NeuralODE excluded by default
        # Use NXRO-GCN-K2 (best GNN variant) instead of NXRO-Graph
        # Core models for plot 1c - use NXRO-GCN (from deep_gcn) or NXRO-GCN-K2 as fallback
        core_models = ['NXRO-MLP', 'NXRO-GCN', 'NXRO-GCN-K2', 'NXRO-Attention', 
                       'NeuralODE', 'Transformer', 'XRO']
        df_core = df_single_display[df_single_display['Model'].isin(core_models)].copy()
        
        # Prefer NXRO-GCN (deep_gcn) over NXRO-GCN-K2; rename K2 to GCN if no deep_gcn
        if 'NXRO-GCN' not in df_core['Model'].values and 'NXRO-GCN-K2' in df_core['Model'].values:
            df_core['Model'] = df_core['Model'].replace('NXRO-GCN-K2', 'NXRO-GCN')
        else:
            # Remove NXRO-GCN-K2 if NXRO-GCN exists (avoid duplicates)
            df_core = df_core[df_core['Model'] != 'NXRO-GCN-K2']
        
        if len(df_core) > 0:
            plot_rmse_ranking_barplot(
                df_core,
                f'{out_dir}/1c_core_models_ranking.png',
                'Core Neural Models (ORAS5 Only)',
                highlight_xro=True
            )
        print()
    
    # ========================================================================
    # SECTION 2: DETERMINISTIC RANKING - CESM PRETRAINED
    # ========================================================================
    if 2 in sections and df_two is not None:
        print("="*60)
        print("SECTION 2: Deterministic Ranking (CESM Pretrained)")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/2_deterministic_two_stage')
        generate_deterministic_plots(df_two, out_dir, '2', 'CESM Pretrained', 
                                    args.results_dir, obs_ds, strip_two_stage_suffix=True)
        
        # Additional plot: Core models only (with XRO baseline)
        df_two_display = apply_display_names(df_two.copy())
        df_two_display = filter_pathological_models(df_two_display)
        df_two_display['Model'] = df_two_display['Model'].str.replace(' (Two-Stage)', '', regex=False)
        
        # Filter to core models only (including baselines)
        # Use NXRO-GCN-K2 (best GNN variant) instead of NXRO-Graph, same as 1c
        # Note: NXRO-Linear excluded by default
        # Core models for plot 2c - use NXRO-GCN (from deep_gcn) or NXRO-GCN-K2 as fallback
        core_models = ['NXRO-MLP', 'NXRO-GCN', 'NXRO-GCN-K2', 'NXRO-Attention', 
                       'NeuralODE', 'Transformer', 'XRO']
        df_core = df_two_display[df_two_display['Model'].isin(core_models)].copy()
        
        # Prefer NXRO-GCN (deep_gcn) over NXRO-GCN-K2; rename K2 to GCN if no deep_gcn
        if 'NXRO-GCN' not in df_core['Model'].values and 'NXRO-GCN-K2' in df_core['Model'].values:
            df_core['Model'] = df_core['Model'].replace('NXRO-GCN-K2', 'NXRO-GCN')
        else:
            # Remove NXRO-GCN-K2 if NXRO-GCN exists (avoid duplicates)
            df_core = df_core[df_core['Model'] != 'NXRO-GCN-K2']
        
        if len(df_core) > 0:
            plot_rmse_ranking_barplot(
                df_core,
                f'{out_dir}/2c_core_models_ranking.png',
                'Core Neural Models (CESM Pretrained)',
                highlight_xro=True
            )
        print()
    
    # ========================================================================
    # SECTION 3: DETERMINISTIC RANKING - COMBINED
    # ========================================================================
    if 3 in sections and df_single is not None:
        print("="*60)
        print("SECTION 3: Deterministic Ranking (Combined)")
        print("="*60)
        
        # Combine single and two-stage
        if df_two is not None:
            df_combined = pd.concat([df_single, df_two], ignore_index=True)
            df_combined = df_combined.sort_values('Mean_RMSE_Test').reset_index(drop=True)
            df_combined['Rank'] = range(1, len(df_combined) + 1)
        else:
            df_combined = df_single
        
        out_dir = ensure_dir(f'{args.output_dir}/3_deterministic_combined')
        generate_deterministic_plots(df_combined, out_dir, '3', 'All Models',
                                    args.results_dir, obs_ds)
        print()
    
    # ========================================================================
    # SECTION 4: UNCERTAINTY RANKING - ORAS5 (CRPS)
    # ========================================================================
    if 4 in sections:
        print("="*60)
        print("SECTION 4: Uncertainty Ranking (ORAS5, CRPS)")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/4_uncertainty_oras5')
        
        try:
            crps_results = load_crps_results(args.results_dir, include_two_stage=False)
            if crps_results:
                plot_crps_ranking(crps_results, f'{out_dir}/4_crps_ranking.png',
                                'CRPS Ranking (ORAS5-trained Models)')
            else:
                print("  [!] No CRPS results found")
        except Exception as e:
            print(f"  [!] Error loading CRPS results: {e}")
        print()
    
    # ========================================================================
    # SECTION 5: UNCERTAINTY RANKING - TWO-STAGE (CRPS)
    # ========================================================================
    if 5 in sections:
        print("="*60)
        print("SECTION 5: Uncertainty Ranking (Two-Stage, CRPS)")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/5_uncertainty_two_stage')
        
        try:
            crps_results = load_crps_results(args.results_dir, include_two_stage=True)
            # Filter to two-stage only
            crps_two_stage = {k: v for k, v in crps_results.items() 
                            if 'Two-Stage' in k or k == 'XRO'}
            if crps_two_stage:
                plot_crps_ranking(crps_two_stage, f'{out_dir}/5_crps_ranking.png',
                                'CRPS Ranking (Two-Stage Models)')
            else:
                print("  [!] No two-stage CRPS results found")
        except Exception as e:
            print(f"  [!] Error loading CRPS results: {e}")
        print()
    
    # ========================================================================
    # SECTION 6: GENERALIZATION GAPS
    # ========================================================================
    if 6 in sections:
        print("="*60)
        print("SECTION 6: Generalization Gaps")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/6_generalization_gaps')
        
        # 6a: ORAS5-only train/test gap
        if df_single is not None:
            plot_train_test_gap_stacked(df_single, f'{out_dir}/6a_oras5_train_test_gap.png',
                                       'Training vs Test RMSE (ORAS5-only)')
        
        # 6b: Two-stage train/test gap
        if df_two is not None:
            plot_train_test_gap_stacked(df_two, f'{out_dir}/6b_two_stage_train_test_gap.png',
                                       'Training vs Test RMSE (Two-Stage)')
        
        # 6c: ORAS5-only vs CESM pretrained comparison
        if df_single is not None and df_two is not None:
            plot_single_vs_two_stage_gap(df_single, df_two, f'{out_dir}/6c_single_vs_two_stage_gap.png',
                                        'ORAS5 Only vs CESM Pretrained: Test Performance')
        
        # 6d: Core models only comparison (subset)
        # Use same models as 1c and 2c: NXRO-GCN-K2 renamed to NXRO-GNN
        # Note: NXRO-Linear and NXRO-NeuralODE excluded by default
        if df_single is not None and df_two is not None:
            # Core models for plot 6d - prefer NXRO-GCN (deep_gcn) over K2
            core_models_6d = ['NXRO-MLP', 'NXRO-Attention', 
                             'NXRO-GCN', 'NXRO-GCN-K2', 'NeuralODE', 'Transformer']
            # Check which GCN variant exists
            gcn_exists = 'NXRO-GCN' in df_single['Model'].values or 'NXRO-GCN' in df_two['Model'].values
            rename_map_6d = {} if gcn_exists else {'NXRO-GCN-K2': 'NXRO-GCN'}
            plot_single_vs_two_stage_gap(df_single, df_two, f'{out_dir}/6d_core_models_single_vs_two_stage.png',
                                        'Core Models: ORAS5 Only vs CESM Pretrained',
                                        filter_models=core_models_6d,
                                        rename_map=rename_map_6d)
        print()
    
    # ========================================================================
    # SECTION 7: ENSEMBLE FORECASTS - XRO BASELINE
    # ========================================================================
    if 7 in sections:
        print("="*60)
        print("SECTION 7: Ensemble Forecasts (XRO Baseline)")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/7_ensemble_xro')
        
        xro_fcst_path = f'{args.results_dir}/xro_baseline/xro_stochastic_fcst.nc'
        if os.path.exists(xro_fcst_path):
            xro_fcst = xr.open_dataset(xro_fcst_path)
            print(f"  Generating monthly ensemble plots for XRO...")
            generate_monthly_ensemble_plots(xro_fcst, obs_ds, out_dir, 'XRO', 
                                           start_year=1979, end_year=2022)
            xro_fcst.close()
        else:
            print(f"  [!] XRO forecast file not found: {xro_fcst_path}")
        print()
    
    # ========================================================================
    # SECTION 8 & 9: ENSEMBLE FORECASTS - TOP 5 MODELS
    # ========================================================================
    for section_num, include_two_stage in [(8, False), (9, True)]:
        if section_num in sections:
            print("="*60)
            stage_name = 'Two-Stage' if include_two_stage else 'ORAS5'
            print(f"SECTION {section_num}: Ensemble Forecasts (Top 5 {stage_name})")
            print("="*60)
            
            out_dir = ensure_dir(f'{args.output_dir}/{section_num}_ensemble_top5_{"two_stage" if include_two_stage else "oras5"}')
            
            # Get top 5 models by CRPS
            try:
                crps_results = load_crps_results(args.results_dir, include_two_stage=include_two_stage)
                
                if include_two_stage:
                    crps_results = {k: v for k, v in crps_results.items() 
                                  if 'Two-Stage' in k}
                else:
                    crps_results = {k: v for k, v in crps_results.items() 
                                  if 'Two-Stage' not in k and k != 'XRO'}
                
                if crps_results:
                    # Sort by average CRPS
                    sorted_models = sorted(crps_results.items(), 
                                          key=lambda x: np.nanmean(x[1]['crps'].values))[:5]
                    
                    print(f"  Top 5 models by CRPS:")
                    for model_name, _ in sorted_models:
                        print(f"    - {model_name}")
                    
                    # TODO: Load and plot forecasts for top 5 models
                    # This requires finding and loading the stochastic forecast files
                    print(f"  [!] Ensemble plot generation for top 5 models requires additional implementation")
                else:
                    print(f"  [!] No CRPS results found for {stage_name} models")
            except Exception as e:
                print(f"  [!] Error: {e}")
            print()
    
    # ========================================================================
    # SECTION 10: SPECIAL CASE ENSEMBLE PLOTS (3x2)
    # ========================================================================
    if 10 in sections:
        print("="*60)
        print("SECTION 10: Special Case Ensemble Plots (3x2)")
        print("="*60)
        print("  Key ENSO events: 1997-98, 2015-16 El Niño, 2010-11 La Niña")
        print("  Spring Predictability Barrier cases")
        print()
        
        out_dir = ensure_dir(f'{args.output_dir}/10_special_cases')
        generate_special_case_plots(args.results_dir, out_dir, obs_ds)
        print()
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    print("="*80)
    print("PLOT GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}/")
    print("\nGenerated sections:")
    for s in sections:
        print(f"  - Section {s}")
    print()
    
    obs_ds.close()


if __name__ == '__main__':
    main()
