#!/usr/bin/env python
"""
Generate Ensemble Forecast Plume Plots

This script generates ensemble forecast plume plots for:
- Section 7: XRO baseline (all months, all years 1979-2022)
- Section 8: Top 5 ORAS5-trained models by CRPS
- Section 9: Top 5 Two-stage trained models by CRPS

Each plot shows the ensemble mean forecast with uncertainty bands (±1 std dev)
along with observations for verification.

Usage:
    python generate_ensemble_plumes.py
    python generate_ensemble_plumes.py --section 7       # XRO only
    python generate_ensemble_plumes.py --section 8       # Top 5 ORAS5
    python generate_ensemble_plumes.py --section 9       # Top 5 Two-stage
    python generate_ensemble_plumes.py --year 1997       # Specific year only
    python generate_ensemble_plumes.py --month Jan       # Specific month only
"""

import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = 'results_out_of_sample'
OUTPUT_DIR = 'plots'

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Colors for models (keyed by display name prefix)
MODEL_COLORS = {
    'XRO': '#FF1744',
    'XRO-NoSeasonal': '#FF5252',
    'XRO-LinearOnly': '#FF8A80',
    'NXRO-MLP': '#4CAF50',
    'NXRO-NeuralODE': '#9C27B0',
    'NXRO-Linear': '#2196F3',
    'NXRO-Attention': '#EC407A',
    'NXRO-GCN': '#00BCD4',
    'NXRO-GAT': '#009688',
    'NXRO-Graph': '#78909C',
    'NXRO-Graph-Learned': '#546E7A',
    'NXRO-RO': '#FF6F00',
    'NXRO-ROOnly': '#FF9800',
    'NXRO-Transformer': '#795548',
    'NXRO-Bilinear': '#607D8B',
    'NXRO-PhysReg': '#673AB7',
    'NXRO-HybridMLP': '#8BC34A',
}


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_display_name(model_name):
    """
    Convert internal model names to clear, paper-friendly display names.
    (Same mapping as in generate_paper_plots.py)
    """
    original = model_name
    
    # Preserve two-stage suffix
    two_stage_suffix = ''
    if '(Two-Stage)' in model_name:
        two_stage_suffix = ' (Two-Stage)'
        model_name = model_name.replace(' (Two-Stage)', '')
    
    name_lower = model_name.lower().strip()
    
    # XRO baselines
    if name_lower == 'xro':
        return 'XRO' + two_stage_suffix
    if name_lower == 'xro_ac0' or name_lower == 'xro ac0':
        return 'XRO-NoSeasonal' + two_stage_suffix  # ac_order=0 means no seasonal variation in L
    if 'linear xro' in name_lower:
        return 'XRO-LinearOnly' + two_stage_suffix  # No nonlinear RO terms
    
    # NXRO models
    if name_lower == 'res' or name_lower.startswith('res '):
        return 'NXRO-MLP' + two_stage_suffix
    # Neural Phys / PhysReg -> NXRO-PhysReg (check before plain Neural)
    if 'neural phys' in name_lower or 'neural_phys' in name_lower or 'physreg' in name_lower:
        return 'NXRO-PhysReg' + two_stage_suffix
    if name_lower == 'neural' or name_lower.startswith('neural '):
        return 'NXRO-NeuralODE' + two_stage_suffix
    if name_lower == 'linear':
        return 'NXRO-Linear' + two_stage_suffix
    if 'attentive' in name_lower or 'attention' in name_lower:
        return 'NXRO-Attention' + two_stage_suffix
    if 'graphpyg' in name_lower:
        # Extract K value if present (e.g., "Graphpyg Gcn K2" -> "K2")
        k_match = re.search(r'k(\d+)', name_lower)
        k_suffix = f'-K{k_match.group(1)}' if k_match else ''
        if 'gat' in name_lower:
            return 'NXRO-GAT' + k_suffix + two_stage_suffix
        return 'NXRO-GCN' + k_suffix + two_stage_suffix
    if name_lower.startswith('graph'):
        return 'NXRO-Graph' + two_stage_suffix
    if 'rodiag' in name_lower or 'ro+diag' in name_lower:
        return 'NXRO-RO' + two_stage_suffix
    if 'residualmix' in name_lower or 'resmix' in name_lower:
        return 'NXRO-HybridMLP' + two_stage_suffix
    if 'transformer' in name_lower:
        return 'NXRO-Transformer' + two_stage_suffix
    
    # Fallback
    if not model_name.startswith('NXRO-') and not model_name.startswith('XRO'):
        return 'NXRO-' + model_name.strip() + two_stage_suffix
    
    return model_name + two_stage_suffix


def get_model_color(model_name):
    """Get color for a model based on its display name."""
    display_name = get_display_name(model_name)
    
    # Try exact match first
    for key in MODEL_COLORS:
        if display_name.startswith(key):
            return MODEL_COLORS[key]
    
    # Try partial match
    for key, color in MODEL_COLORS.items():
        if key in display_name:
            return color
    
    return '#666666'


# ============================================================================
# CRPS RANKING UTILITIES
# ============================================================================

def load_crps_results(results_dir, include_two_stage=False):
    """Load CRPS results from stochastic evaluation files."""
    results = {}
    
    # XRO baseline
    xro_pattern = f'{results_dir}/xro_baseline/*_stochastic_eval_lead_metrics.csv'
    xro_matches = glob.glob(xro_pattern)
    if xro_matches:
        df = pd.read_csv(xro_matches[0])
        results['XRO'] = {
            'df': df,
            'avg_crps': float(np.nanmean(df['crps'].values)),
            'fcst_path': f'{results_dir}/xro_baseline/xro_stochastic_fcst.nc'
        }
    
    # NXRO models - map directory names to display names
    model_dirs = {
        'res': 'NXRO-MLP',
        'linear': 'NXRO-Linear', 
        'attentive': 'NXRO-Attention',
        'neural': 'NXRO-NeuralODE',
        'neural_phys': 'NXRO-PhysReg',
        'rodiag': 'NXRO-RO',
        'graphpyg': 'NXRO-GCN',  # Default to GCN, will check for GAT below
        'graph': 'NXRO-Graph',
        'resmix': 'NXRO-HybridMLP',
        'transformer': 'NXRO-Transformer',
        'bilinear': 'NXRO-Bilinear',
    }
    
    for dir_name, base_display_name in model_dirs.items():
        pattern = f'{results_dir}/{dir_name}/**/*stochastic*eval_lead_metrics.csv'
        matches = glob.glob(pattern, recursive=True)
        
        for match in matches:
            basename = os.path.basename(match)
            
            # Skip non-standard files
            is_two_stage = '_extra_data' in match or 'finetuned' in basename
            
            if not include_two_stage and is_two_stage:
                continue
            
            # Find corresponding forecast file
            fcst_pattern = match.replace('_eval_lead_metrics.csv', '_forecasts.nc')
            if not os.path.exists(fcst_pattern):
                # Try alternative patterns
                dir_path = os.path.dirname(match)
                alt_patterns = [
                    f'{dir_path}/*stochastic*forecasts.nc',
                    f'{dir_path}/*_stochastic_fcst.nc',
                ]
                fcst_path = None
                for p in alt_patterns:
                    alt_matches = glob.glob(p)
                    if alt_matches:
                        fcst_path = alt_matches[0]
                        break
            else:
                fcst_path = fcst_pattern
            
            if fcst_path is None:
                continue
            
            # Create model name - check for GAT vs GCN in graphpyg
            display_name = base_display_name
            if dir_name == 'graphpyg' and 'gat' in basename.lower():
                display_name = 'NXRO-GAT'
            if dir_name == 'neural_phys' or (dir_name == 'neural' and 'phys' in basename.lower()):
                display_name = 'NXRO-PhysReg'
            
            if is_two_stage:
                display_name += ' (Two-Stage)'
            
            df = pd.read_csv(match)
            results[display_name] = {
                'df': df,
                'avg_crps': float(np.nanmean(df['crps'].values)),
                'fcst_path': fcst_path
            }
    
    return results


def get_top_n_models(crps_results, n=5, include_xro=False, two_stage_only=False, exclude_models=None):
    """Get top N models by average CRPS.
    
    Args:
        crps_results: Dictionary of model results
        n: Number of top models to return
        include_xro: Whether to include XRO baseline
        two_stage_only: Only include two-stage models
        exclude_models: List of model name patterns to exclude (e.g., ['NXRO-RO'])
    """
    if exclude_models is None:
        exclude_models = ['NXRO-RO']  # Default: exclude pathological models
    
    filtered = {}
    for name, data in crps_results.items():
        if name == 'XRO' and not include_xro:
            continue
        # Exclude pathological models
        if any(excl in name for excl in exclude_models):
            continue
        if two_stage_only and 'Two-Stage' not in name:
            continue
        if not two_stage_only and 'Two-Stage' in name:
            continue
        filtered[name] = data
    
    sorted_models = sorted(filtered.items(), key=lambda x: x[1]['avg_crps'])
    return dict(sorted_models[:n])


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_single_plume(fcst_ds, obs_ds, init_date, out_path, title, color='blue'):
    """
    Plot ensemble forecast plume for a single initialization date.
    
    Args:
        fcst_ds: xarray Dataset with ensemble forecasts
        obs_ds: xarray Dataset with observations
        init_date: Initialization date string (e.g., '1979-01')
        out_path: Output path for plot
        title: Plot title
        color: Color for forecast
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        fcst_var = fcst_ds['Nino34'].sel(init=init_date)
    except KeyError:
        return False
    
    fcst_mean = fcst_var.mean('member').squeeze()
    fcst_std = fcst_var.std('member').squeeze()
    
    nlead = len(fcst_mean.lead)
    
    # Setup time axis
    xdate_init = datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
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
    
    # El Niño / La Niña thresholds
    ax.axhline(0, c='gray', ls='-', lw=0.5, alpha=0.5)
    ax.axhline(0.5, c='red', ls='--', lw=1, alpha=0.3, label='El Niño threshold')
    ax.axhline(-0.5, c='blue', ls='--', lw=1, alpha=0.3, label='La Niña threshold')
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim([xdate_strt, xdate_last])
    ax.set_ylim([-3.5, 3.5])
    ax.set_ylabel('Niño3.4 SSTA (°C)', fontsize=11)
    ax.set_xlabel('')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def generate_plumes_for_model(model_name, fcst_path, obs_ds, out_dir, 
                              start_year=1979, end_year=2022,
                              specific_month=None, specific_year=None):
    """
    Generate ensemble plume plots for a single model.
    
    Args:
        model_name: Name of the model
        fcst_path: Path to forecast NetCDF file
        obs_ds: xarray Dataset with observations
        out_dir: Output directory
        start_year: Starting year
        end_year: Ending year
        specific_month: If specified, only generate for this month
        specific_year: If specified, only generate for this year
    """
    if not os.path.exists(fcst_path):
        print(f"  [!] Forecast file not found: {fcst_path}")
        return
    
    fcst_ds = xr.open_dataset(fcst_path)
    color = get_model_color(model_name)
    
    safe_name = model_name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')
    
    months_to_process = MONTHS if specific_month is None else [specific_month]
    years_to_process = range(start_year, end_year + 1) if specific_year is None else [specific_year]
    
    total_plots = 0
    success_plots = 0
    
    for month_idx, month_name in enumerate(MONTHS, 1):
        if month_name not in months_to_process:
            continue
            
        month_dir = ensure_dir(f'{out_dir}/{month_name}')
        
        for year in years_to_process:
            init_date = f'{year}-{month_idx:02d}'
            out_path = f'{month_dir}/{safe_name}_{year}_{month_name}.png'
            
            total_plots += 1
            success = plot_single_plume(
                fcst_ds, obs_ds, init_date, out_path,
                f'{model_name} Forecast - Init: {month_name} {year}',
                color=color
            )
            
            if success:
                success_plots += 1
    
    fcst_ds.close()
    print(f"  ✓ {model_name}: {success_plots}/{total_plots} plots generated")


def generate_combined_plumes(models, obs_ds, out_dir, start_year=1979, end_year=2022,
                            specific_month=None, specific_year=None):
    """
    Generate combined ensemble plume plots with multiple models per plot.
    
    Args:
        models: Dictionary of {model_name: {'fcst_path': path}}
        obs_ds: xarray Dataset with observations
        out_dir: Output directory
        start_year: Starting year
        end_year: Ending year
        specific_month: If specified, only generate for this month
        specific_year: If specified, only generate for this year
    """
    # Load all forecast datasets
    fcst_datasets = {}
    for model_name, model_data in models.items():
        fcst_path = model_data['fcst_path']
        if os.path.exists(fcst_path):
            fcst_datasets[model_name] = xr.open_dataset(fcst_path)
    
    if not fcst_datasets:
        print("  [!] No forecast datasets found")
        return
    
    months_to_process = MONTHS if specific_month is None else [specific_month]
    years_to_process = range(start_year, end_year + 1) if specific_year is None else [specific_year]
    
    total_plots = 0
    success_plots = 0
    
    for month_idx, month_name in enumerate(MONTHS, 1):
        if month_name not in months_to_process:
            continue
            
        month_dir = ensure_dir(f'{out_dir}/combined/{month_name}')
        
        for year in years_to_process:
            init_date = f'{year}-{month_idx:02d}'
            out_path = f'{month_dir}/combined_{year}_{month_name}.png'
            
            total_plots += 1
            
            # Create combined plot
            fig, axes = plt.subplots(len(fcst_datasets), 1, 
                                    figsize=(12, 4 * len(fcst_datasets)))
            if len(fcst_datasets) == 1:
                axes = [axes]
            
            success = True
            for ax_idx, (model_name, fcst_ds) in enumerate(fcst_datasets.items()):
                try:
                    fcst_var = fcst_ds['Nino34'].sel(init=init_date)
                    fcst_mean = fcst_var.mean('member').squeeze()
                    fcst_std = fcst_var.std('member').squeeze()
                    
                    nlead = len(fcst_mean.lead)
                    xdate_init = datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
                    xdate_strt = xdate_init + relativedelta(months=-2)
                    xdate_last = xdate_init + relativedelta(months=nlead-1)
                    xtime_fcst = [xdate_init + relativedelta(months=int(i)) for i in range(nlead)]
                    
                    ax = axes[ax_idx]
                    color = get_model_color(model_name)
                    
                    ax.plot(xtime_fcst, fcst_mean.values, c=color, marker='.', lw=2, label='Ensemble Mean')
                    ax.fill_between(xtime_fcst,
                                   fcst_mean.values - fcst_std.values,
                                   fcst_mean.values + fcst_std.values,
                                   fc=color, alpha=0.3, label='±1 Std Dev')
                    
                    # Observations
                    try:
                        sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
                        ax.plot(sel_obs.time.values, sel_obs.values, c='black',
                               marker='o', markersize=2, lw=1.5, label='Observed', alpha=0.8)
                    except:
                        pass
                    
                    ax.axhline(0, c='gray', ls='-', lw=0.5, alpha=0.5)
                    ax.axhline(0.5, c='red', ls='--', lw=1, alpha=0.2)
                    ax.axhline(-0.5, c='blue', ls='--', lw=1, alpha=0.2)
                    
                    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
                    ax.set_xlim([xdate_strt, xdate_last])
                    ax.set_ylim([-3.5, 3.5])
                    ax.set_ylabel('Niño3.4 (°C)', fontsize=9)
                    ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
                    ax.legend(fontsize=7, loc='upper left', ncol=3)
                    ax.grid(True, alpha=0.3)
                    
                except KeyError:
                    success = False
                    ax = axes[ax_idx]
                    ax.text(0.5, 0.5, f'Init date {init_date} not available',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
            
            plt.suptitle(f'Ensemble Forecasts - Init: {month_name} {year}', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if success:
                success_plots += 1
    
    # Close datasets
    for ds in fcst_datasets.values():
        ds.close()
    
    print(f"  ✓ Combined plots: {success_plots}/{total_plots} generated")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Ensemble Forecast Plume Plots')
    parser.add_argument('--section', type=int, choices=[7, 8, 9], default=None,
                       help='Section to generate (7=XRO, 8=Top5 ORAS5, 9=Top5 Two-Stage)')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                       help='Results directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--year', type=int, default=None,
                       help='Generate only for specific year')
    parser.add_argument('--month', type=str, default=None, choices=MONTHS,
                       help='Generate only for specific month')
    parser.add_argument('--start_year', type=int, default=1979,
                       help='Start year')
    parser.add_argument('--end_year', type=int, default=2022,
                       help='End year')
    parser.add_argument('--combined', action='store_true',
                       help='Generate combined plots with all models')
    args = parser.parse_args()
    
    sections = [7, 8, 9] if args.section is None else [args.section]
    
    print("="*80)
    print("GENERATING ENSEMBLE FORECAST PLUME PLOTS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sections: {sections}")
    if args.year:
        print(f"Specific year: {args.year}")
    if args.month:
        print(f"Specific month: {args.month}")
    print("="*80)
    print()
    
    # Load observations
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    
    # Load CRPS results for ranking
    crps_results = load_crps_results(args.results_dir, include_two_stage=True)
    print(f"✓ Loaded CRPS results for {len(crps_results)} models")
    print()
    
    # ========================================================================
    # SECTION 7: XRO BASELINE
    # ========================================================================
    if 7 in sections:
        print("="*60)
        print("SECTION 7: XRO Baseline Ensemble Forecasts")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/7_ensemble_xro')
        
        if 'XRO' in crps_results:
            xro_data = crps_results['XRO']
            generate_plumes_for_model(
                'XRO', xro_data['fcst_path'], obs_ds, out_dir,
                start_year=args.start_year, end_year=args.end_year,
                specific_month=args.month, specific_year=args.year
            )
        else:
            print("  [!] XRO results not found")
        print()
    
    # ========================================================================
    # SECTION 8: TOP 5 ORAS5 MODELS
    # ========================================================================
    if 8 in sections:
        print("="*60)
        print("SECTION 8: Top 5 ORAS5-trained Models Ensemble Forecasts")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/8_ensemble_top5_oras5')
        
        top5_oras5 = get_top_n_models(crps_results, n=5, include_xro=False, two_stage_only=False)
        
        if top5_oras5:
            print(f"  Top 5 models by CRPS:")
            for name, data in top5_oras5.items():
                print(f"    - {name}: CRPS={data['avg_crps']:.4f}")
            print()
            
            for model_name, model_data in top5_oras5.items():
                generate_plumes_for_model(
                    model_name, model_data['fcst_path'], obs_ds, out_dir,
                    start_year=args.start_year, end_year=args.end_year,
                    specific_month=args.month, specific_year=args.year
                )
            
            if args.combined:
                print("\n  Generating combined plots...")
                generate_combined_plumes(
                    top5_oras5, obs_ds, out_dir,
                    start_year=args.start_year, end_year=args.end_year,
                    specific_month=args.month, specific_year=args.year
                )
        else:
            print("  [!] No ORAS5-trained models found")
        print()
    
    # ========================================================================
    # SECTION 9: TOP 5 TWO-STAGE MODELS
    # ========================================================================
    if 9 in sections:
        print("="*60)
        print("SECTION 9: Top 5 Two-Stage Models Ensemble Forecasts")
        print("="*60)
        
        out_dir = ensure_dir(f'{args.output_dir}/9_ensemble_top5_two_stage')
        
        top5_two_stage = get_top_n_models(crps_results, n=5, include_xro=False, two_stage_only=True)
        
        if top5_two_stage:
            print(f"  Top 5 two-stage models by CRPS:")
            for name, data in top5_two_stage.items():
                print(f"    - {name}: CRPS={data['avg_crps']:.4f}")
            print()
            
            for model_name, model_data in top5_two_stage.items():
                generate_plumes_for_model(
                    model_name, model_data['fcst_path'], obs_ds, out_dir,
                    start_year=args.start_year, end_year=args.end_year,
                    specific_month=args.month, specific_year=args.year
                )
            
            if args.combined:
                print("\n  Generating combined plots...")
                generate_combined_plumes(
                    top5_two_stage, obs_ds, out_dir,
                    start_year=args.start_year, end_year=args.end_year,
                    specific_month=args.month, specific_year=args.year
                )
        else:
            print("  [!] No two-stage models found")
        print()
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    print("="*80)
    print("ENSEMBLE PLUME GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}/")
    print()
    
    obs_ds.close()


if __name__ == '__main__':
    main()
