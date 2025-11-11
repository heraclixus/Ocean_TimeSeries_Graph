#!/usr/bin/env python
"""
Visualize and Compare Stochastic Ensemble Performance

This script compares stochastic ensemble forecasts from top 5 NXRO models
against XRO baseline, generating comprehensive probabilistic skill visualizations.

Usage:
    python visualize_stochastic_comparison.py
    python visualize_stochastic_comparison.py --results_dir results_out_of_sample
"""

import os
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

def load_stochastic_results(base_dir, compare_stages=False):
    """Load stochastic evaluation results for all models.
    
    Args:
        base_dir: Base results directory
        compare_stages: If True, load all stages (post-hoc, stage2, sim_noise, etc.)
                       If False, only load most recent for each model
    """
    results = {}
    
    # XRO baseline
    xro_pattern = f'{base_dir}/xro_baseline/*_stochastic_eval_lead_metrics.csv'
    xro_matches = glob.glob(xro_pattern)
    if xro_matches:
        results['XRO'] = pd.read_csv(xro_matches[0])
        print(f"  Loaded XRO baseline: {xro_matches[0]}")
    
    # NXRO models
    model_dirs = {
        'Res': f'{base_dir}/res',
        'Graph': f'{base_dir}/graphpyg',
        'Attentive': f'{base_dir}/attentive',
        'RO+Diag': f'{base_dir}/rodiag',
        'Linear': f'{base_dir}/linear',
    }
    
    for model_name, model_dir in model_dirs.items():
        # Find stochastic eval CSV (look for _lead_metrics.csv specifically)
        pattern = f'{model_dir}/**/*_stochastic_*_eval_lead_metrics.csv'
        matches = glob.glob(pattern, recursive=True)
        
        if not matches:
            print(f"  [!] Not found: {model_name} (searched: {pattern})")
            continue
        
        if compare_stages:
            # Load ALL stages for this model
            for match in matches:
                # Determine stage from filename
                basename = os.path.basename(match)
                if '_sim_noise_stage2_' in basename:
                    stage_label = f'{model_name} (S2+S3)'
                elif '_stage2_' in basename:
                    stage_label = f'{model_name} (S2)'
                elif '_sim_noise_' in basename:
                    stage_label = f'{model_name} (S3)'
                else:
                    stage_label = f'{model_name} (Post-hoc)'
                
                results[stage_label] = pd.read_csv(match)
                print(f"  Loaded {stage_label}: {match}")
        else:
            # Use most recent only
            eval_file = max(matches, key=os.path.getmtime)
            results[f'NXRO-{model_name}'] = pd.read_csv(eval_file)
            print(f"  Loaded NXRO-{model_name}: {eval_file}")
    
    return results


def plot_crps_comparison(results, out_dir):
    """Plot CRPS by lead time for all models."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Base colors for models
    base_colors = {
        'XRO': '#FF1744',
        'Res': '#4CAF50',
        'Graph': '#78909C',
        'Attentive': '#EC407A',
        'RO+Diag': '#FF6F00',
        'Linear': '#2196F3',
    }
    
    # Linestyles for stages
    stage_styles = {
        'Post-hoc': '-',
        'S2': '--',
        'S3': '-.',
        'S2+S3': ':',
    }
    
    for model_name, df in results.items():
        # Extract base model name and stage
        base_model = model_name.split()[0].replace('NXRO-', '')
        
        # Determine color and linestyle
        color = base_colors.get(base_model, None)
        if '(S2+S3)' in model_name:
            linestyle = stage_styles['S2+S3']
        elif '(S2)' in model_name:
            linestyle = stage_styles['S2']
        elif '(S3)' in model_name:
            linestyle = stage_styles['S3']
        elif '(Post-hoc)' in model_name:
            linestyle = stage_styles['Post-hoc']
        else:
            linestyle = '-'
        
        marker = 'o' if model_name == 'XRO' else 's'
        lw = 2.5 if model_name == 'XRO' else 2
        
        ax.plot(df['lead'], df['crps'], label=model_name, 
               color=color, marker=marker, markersize=3, linewidth=lw, linestyle=linestyle)
    
    ax.set_xlabel('Forecast Lead (months)', fontsize=11)
    ax.set_ylabel('CRPS (lower is better)', fontsize=11)
    ax.set_title('Continuous Ranked Probability Score Comparison', fontsize=12)
    ax.set_xlim([0, 21])
    
    # Adjust legend based on number of items
    ncol = 3 if len(results) > 15 else 2 if len(results) > 8 else 1
    legend_fontsize = 7 if len(results) > 15 else 8
    ax.legend(fontsize=legend_fontsize, ncol=ncol, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/stochastic_crps_comparison.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/stochastic_crps_comparison.png")


def plot_spread_skill_comparison(results, out_dir):
    """Plot spread vs RMSE for all models."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {
        'XRO': '#FF1744',
        'NXRO-Res': '#4CAF50',
        'NXRO-Graph': '#78909C',
        'NXRO-Attentive': '#EC407A',
        'NXRO-RO+Diag': '#FF6F00',
        'NXRO-Linear': '#2196F3',
    }
    
    for model_name, df in results.items():
        color = colors.get(model_name, None)
        
        # Plot both spread and RMSE
        ax.plot(df['lead'], df['spread'], label=f'{model_name} Spread', 
               color=color, linestyle='-', linewidth=2, alpha=0.7)
        ax.plot(df['lead'], df['rmse_mean'], label=f'{model_name} RMSE', 
               color=color, linestyle='--', linewidth=1.5, alpha=0.9)
    
    ax.set_xlabel('Forecast Lead (months)', fontsize=11)
    ax.set_ylabel('Spread / RMSE (C)', fontsize=11)
    ax.set_title('Ensemble Spread vs RMSE (Calibration Check)', fontsize=12)
    ax.set_xlim([0, 21])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/stochastic_spread_vs_rmse.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/stochastic_spread_vs_rmse.png")


def plot_calibration_ratio(results, out_dir):
    """Plot spread/RMSE ratio (should be near 1.0 for good calibration)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {
        'XRO': '#FF1744',
        'NXRO-Res': '#4CAF50',
        'NXRO-Graph': '#78909C',
        'NXRO-Attentive': '#EC407A',
        'NXRO-RO+Diag': '#FF6F00',
        'NXRO-Linear': '#2196F3',
    }
    
    for model_name, df in results.items():
        color = colors.get(model_name, None)
        ratio = df['spread'] / df['rmse_mean']
        marker = 'o' if 'XRO' in model_name and model_name != 'XRO' else 's'
        lw = 2.5 if model_name == 'XRO' else 2
        
        ax.plot(df['lead'], ratio, label=model_name, 
               color=color, marker=marker, markersize=4, linewidth=lw)
    
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, 
              label='Perfect calibration', alpha=0.5)
    ax.set_xlabel('Forecast Lead (months)', fontsize=11)
    ax.set_ylabel('Spread / RMSE Ratio', fontsize=11)
    ax.set_title('Ensemble Calibration (Ratio = 1.0 is ideal)', fontsize=12)
    ax.set_xlim([0, 21])
    ax.set_ylim([0.5, 1.5])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/stochastic_calibration_ratio.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/stochastic_calibration_ratio.png")


def plot_metric_rankings(results, out_dir, top_n=10):
    """Create bar plots ranking models by CRPS and RMSE (top N only)."""
    # Compute average metrics across leads
    summary = []
    
    for model_name, df in results.items():
        avg_crps = float(np.nanmean(df['crps'].values))
        avg_rmse = float(np.nanmean(df['rmse_mean'].values))
        
        summary.append({
            'Model': model_name,
            'Avg_CRPS': avg_crps,
            'Avg_RMSE': avg_rmse,
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Create 1x2 subplot (only CRPS and RMSE)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. CRPS ranking (lower is better) - Top N only
    df_sorted = summary_df.sort_values('Avg_CRPS').head(top_n)
    colors_crps = ['red' if 'XRO' == m else 'green' for m in df_sorted['Model']]
    
    axes[0].barh(range(len(df_sorted)), df_sorted['Avg_CRPS'], color=colors_crps, alpha=0.7)
    axes[0].set_yticks(range(len(df_sorted)))
    axes[0].set_yticklabels(df_sorted['Model'], fontsize=10)
    axes[0].set_xlabel('Average CRPS (lower is better)', fontsize=11)
    axes[0].set_title(f'Ranking by CRPS (Top {top_n})', fontsize=12)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add values
    for i, val in enumerate(df_sorted['Avg_CRPS']):
        axes[0].text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # 2. Ensemble-mean RMSE ranking (lower is better) - Top N only
    df_sorted = summary_df.sort_values('Avg_RMSE').head(top_n)
    colors_rmse = ['red' if 'XRO' == m else 'green' for m in df_sorted['Model']]
    
    axes[1].barh(range(len(df_sorted)), df_sorted['Avg_RMSE'], color=colors_rmse, alpha=0.7)
    axes[1].set_yticks(range(len(df_sorted)))
    axes[1].set_yticklabels(df_sorted['Model'], fontsize=10)
    axes[1].set_xlabel('Average Ensemble-Mean RMSE (C)', fontsize=11)
    axes[1].set_title(f'Ranking by Ensemble-Mean RMSE (Top {top_n})', fontsize=12)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    for i, val in enumerate(df_sorted['Avg_RMSE']):
        axes[1].text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/stochastic_rankings_all_metrics.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/stochastic_rankings_all_metrics.png")


def plot_top_model_forecasts(results, base_dir, out_dir, top_n=5):
    """Plot stochastic forecast examples for top N models including XRO."""
    # Get top models by CRPS
    summary = []
    for model_name, df in results.items():
        avg_crps = float(np.nanmean(df['crps'].values))
        summary.append({'Model': model_name, 'Avg_CRPS': avg_crps})
    
    summary_df = pd.DataFrame(summary).sort_values('Avg_CRPS')
    
    # Always include XRO if present
    top_models = []
    if 'XRO' in summary_df['Model'].values:
        top_models.append('XRO')
        other_models = summary_df[summary_df['Model'] != 'XRO'].head(top_n - 1)
    else:
        other_models = summary_df.head(top_n)
    
    top_models.extend(other_models['Model'].tolist())
    
    print(f"\n  Plotting forecasts for top {len(top_models)} models:")
    for m in top_models:
        print(f"    - {m}")
    print()
    
    # Load forecast NetCDF files for top models
    for model_name in top_models:
        print(f"  Processing {model_name}...")
        try:
            # Determine file path based on model name
            if model_name == 'XRO':
                fcst_path = f'{base_dir}/xro_baseline/xro_stochastic_fcst.nc'
                model_label = 'XRO'
            else:
                # Extract base model and stage
                base_model = model_name.split()[0]
                
                # Determine suffix from stage
                if '(S2+S3)' in model_name:
                    suffix = '_sim_noise_stage2'
                elif '(S2)' in model_name:
                    suffix = '_stage2'
                elif '(S3)' in model_name:
                    suffix = '_sim_noise'
                else:
                    suffix = ''
                
                # Map model names to directories
                model_dirs = {
                    'Res': 'res',
                    'Graph': 'graphpyg/gcn_k3',
                    'Attentive': 'attentive',
                    'RO+Diag': 'rodiag',
                    'Linear': 'linear',
                }
                
                model_dir = model_dirs.get(base_model, base_model.lower())
                
                # Handle graph special case
                if base_model == 'Graph':
                    fcst_path = f'{base_dir}/{model_dir}/NXRO_GRAPHPYG_GCN_K3_stochastic{suffix}_forecasts.nc'
                else:
                    fcst_path = f'{base_dir}/{model_dir}/NXRO_{base_model.upper()}_stochastic{suffix}_forecasts.nc'
                
                model_label = model_name
            
            print(f"    Looking for: {fcst_path}")
            if not os.path.exists(fcst_path):
                print(f"    [!] Forecast file not found, skipping")
                continue
            
            # Load and plot sample forecast
            fcst_ds = xr.open_dataset(fcst_path)
            obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
            
            # Pick a sample initialization (e.g., 2002-01 for out-of-sample)
            init_date = '2002-01'
            if init_date not in fcst_ds.init.values:
                init_date = str(fcst_ds.init.values[0])[:7]  # Use first available
            
            # Plot forecast
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            fcst_var = fcst_ds['Nino34'].sel(init=init_date)
            fcst_mean = fcst_var.mean('member').squeeze()
            fcst_std = fcst_var.std('member').squeeze()
            
            lead_months = fcst_mean.lead.values
            # Convert to numpy arrays and flatten
            fcst_mean_vals = fcst_mean.values.flatten()
            fcst_std_vals = fcst_std.values.flatten()
            
            ax.plot(lead_months, fcst_mean_vals, color='blue', lw=2, label='Ensemble Mean')
            ax.fill_between(lead_months, fcst_mean_vals - fcst_std_vals, fcst_mean_vals + fcst_std_vals,
                           alpha=0.3, color='blue', label='±1 Std Dev')
            
            # Overlay observations
            init_time = pd.Timestamp(init_date + '-01')
            obs_times = [init_time + pd.DateOffset(months=int(L)) for L in lead_months]
            obs_vals = []
            for t in obs_times:
                try:
                    val = obs_ds['Nino34'].sel(time=t, method='nearest').values
                    obs_vals.append(float(val))
                except:
                    obs_vals.append(np.nan)
            
            ax.plot(lead_months, obs_vals, color='black', lw=2, marker='o', 
                   markersize=4, label='Observed', alpha=0.8)
            
            ax.set_xlabel('Forecast Lead (months)', fontsize=11)
            ax.set_ylabel('Nino3.4 SSTA (C)', fontsize=11)
            ax.set_title(f'{model_label}: Stochastic Forecast (Init: {init_date})', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p')
            out_path = f'{out_dir}/forecast_{safe_name}.png'
            plt.savefig(out_path, dpi=300)
            plt.close()
            
            print(f"    ✓ Saved: {out_path}")
            
        except Exception as e:
            import traceback
            print(f"    [!] Error plotting {model_name}:")
            print(f"    {str(e)}")
            traceback.print_exc()
            continue


def plot_ensemble_mean_skill(results, out_dir):
    """Plot ensemble-mean ACC and RMSE curves."""
    # Load ensemble-mean forecasts
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    colors = {
        'XRO': '#FF1744',
        'NXRO-Res': '#4CAF50',
        'NXRO-Graph': '#78909C',
        'NXRO-Attentive': '#EC407A',
        'NXRO-RO+Diag': '#FF6F00',
        'NXRO-Linear': '#2196F3',
    }
    
    # Plot RMSE
    for model_name, df in results.items():
        color = colors.get(model_name, None)
        marker = 'o' if 'XRO' in model_name and model_name != 'XRO' else 's'
        lw = 2.5 if model_name == 'XRO' else 2
        
        axes[0].plot(df['lead'], df['rmse_mean'], label=model_name, 
                    color=color, marker=marker, markersize=4, linewidth=lw)
    
    axes[0].set_xlabel('Forecast Lead (months)', fontsize=11)
    axes[0].set_ylabel('Ensemble-Mean RMSE (C)', fontsize=11)
    axes[0].set_title('Ensemble-Mean Forecast Skill: RMSE', fontsize=12)
    axes[0].set_xlim([0, 21])
    axes[0].set_ylim([0, 1.0])
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot ACC (if available)
    has_acc = any('acc_mean' in df.columns for df in results.values())
    
    if has_acc:
        for model_name, df in results.items():
            if 'acc_mean' in df.columns:
                color = colors.get(model_name, None)
                marker = 'o' if 'XRO' in model_name and model_name != 'XRO' else 's'
                lw = 2.5 if model_name == 'XRO' else 2
                
                axes[1].plot(df['lead'], df['acc_mean'], label=model_name, 
                            color=color, marker=marker, markersize=4, linewidth=lw)
        
        axes[1].set_xlabel('Forecast Lead (months)', fontsize=11)
        axes[1].set_ylabel('Ensemble-Mean ACC', fontsize=11)
        axes[1].set_title('Ensemble-Mean Forecast Skill: ACC', fontsize=12)
        axes[1].set_xlim([0, 21])
        axes[1].set_ylim([0.2, 1.0])
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'ACC data not available in stochastic eval files', 
                    ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/stochastic_ensemble_mean_skill.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/stochastic_ensemble_mean_skill.png")


def print_summary_table(results, out_dir):
    """Print comprehensive summary table."""
    print("\n" + "="*80)
    print("STOCHASTIC ENSEMBLE COMPARISON SUMMARY")
    print("="*80)
    
    # Compute summary statistics
    summary = []
    
    for model_name, df in results.items():
        avg_crps = float(np.nanmean(df['crps'].values))
        avg_rmse = float(np.nanmean(df['rmse_mean'].values))
        avg_spread = float(np.nanmean(df['spread'].values))
        ratio = avg_spread / avg_rmse if avg_rmse > 0 else np.nan
        
        # Coverage at 80% if available
        if 'interval' in df.columns and 'coverage' in df.columns:
            cov_80_rows = df[df['interval'] == 0.8]
            avg_cov80 = float(np.nanmean(cov_80_rows['coverage'].values)) if len(cov_80_rows) > 0 else np.nan
        else:
            avg_cov80 = np.nan
        
        summary.append({
            'Model': model_name,
            'CRPS': avg_crps,
            'RMSE_Mean': avg_rmse,
            'Spread': avg_spread,
            'Spread/RMSE': ratio,
            'Cov@80%': avg_cov80
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Rankings
    print("\nRanking by CRPS (lower is better):")
    print("-"*80)
    crps_ranked = summary_df.sort_values('CRPS')
    for i, row in enumerate(crps_ranked.itertuples(), 1):
        print(f"  {i}. {row.Model:20s} CRPS={row.CRPS:.4f}")
    
    print("\nRanking by Ensemble-Mean RMSE (lower is better):")
    print("-"*80)
    rmse_ranked = summary_df.sort_values('RMSE_Mean')
    for i, row in enumerate(rmse_ranked.itertuples(), 1):
        print(f"  {i}. {row.Model:20s} RMSE={row.RMSE_Mean:.4f} C")
    
    print("\nRanking by Calibration (closer to 1.0 is better):")
    print("-"*80)
    summary_df['Cal_Error'] = abs(summary_df['Spread/RMSE'] - 1.0)
    cal_ranked = summary_df.sort_values('Cal_Error')
    for i, row in cal_ranked.iterrows():
        ratio_val = row['Spread/RMSE']
        print(f"  {i+1}. {row['Model']:20s} Ratio={ratio_val:.4f}")
    
    if not summary_df['Cov@80%'].isna().all():
        print("\nRanking by Coverage@80% (closer to 0.80 is better):")
        print("-"*80)
        summary_df['Cov_Error'] = abs(summary_df['Cov@80%'] - 0.80)
        cov_ranked = summary_df[summary_df['Cov@80%'].notna()].sort_values('Cov_Error')
        for i, row in cov_ranked.iterrows():
            cov_val = row['Cov@80%']
            if not np.isnan(cov_val):
                print(f"  {i+1}. {row['Model']:20s} Coverage={cov_val:.4f}")
    
    # Complete table
    print("\n" + "-"*80)
    print("Complete Metrics Table:")
    print("-"*80)
    print(summary_df.to_string(index=False))
    
    # Save CSV
    csv_out = f'{out_dir}/stochastic_summary_metrics.csv'
    summary_df.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved summary table: {csv_out}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare Stochastic Ensemble Performance')
    parser.add_argument('--results_dir', type=str, default='results_out_of_sample',
                       help='Results directory')
    parser.add_argument('--compare_stages', action='store_true',
                       help='Compare all stages (post-hoc, S2, S3, S2+S3) instead of just most recent')
    args = parser.parse_args()
    
    print("="*80)
    print("STOCHASTIC ENSEMBLE COMPARISON")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    if args.compare_stages:
        print(f"Mode: COMPARE ALL STAGES (post-hoc, S2, S3, S2+S3)")
    else:
        print(f"Mode: Compare models (most recent run only)")
    print()
    
    # Load results
    print("Loading stochastic evaluation results...")
    results = load_stochastic_results(args.results_dir, compare_stages=args.compare_stages)
    
    if len(results) == 0:
        print("[X] No stochastic results found!")
        print("Run evaluate_stochastic_top5.sh first")
        return
    
    print(f"\nLoaded {len(results)} models/variants")
    print()
    
    # Create output directory
    if args.compare_stages:
        out_dir = f'{args.results_dir}/rankings/stochastic_stages_comparison'
    else:
        out_dir = f'{args.results_dir}/rankings/stochastic_comparison'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Only generate ranking plot and forecast visualizations
    plot_metric_rankings(results, out_dir, top_n=10)
    
    print("\nGenerating forecast visualizations for top models...")
    plot_top_model_forecasts(results, args.results_dir, out_dir, top_n=5)
    
    # Print summary
    print_summary_table(results, out_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput files in: {out_dir}/")
    print("  - stochastic_rankings_all_metrics.png (CRPS & RMSE rankings, top 10)")
    print("  - forecast_*.png (forecast visualizations for top 5 models + XRO)")
    print("  - stochastic_summary_metrics.csv (complete metrics table)")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

