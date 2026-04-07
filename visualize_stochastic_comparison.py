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

def load_stochastic_results(base_dir, compare_stages=False, compare_arp=False, include_two_stage=False):
    """Load stochastic evaluation results for all models.
    
    Args:
        base_dir: Base results directory
        compare_stages: If True, load all stages (post-hoc, stage2, sim_noise, etc.)
        compare_arp: If True, load different AR(p) variants (AR1, AR2, etc.)
        include_two_stage: If True, load both single-stage and two-stage trained models
    """
    results = {}
    import re
    
    # XRO baseline
    # Note: XRO baseline is AR(1) only in current implementation
    xro_pattern = f'{base_dir}/xro_baseline/*_stochastic_eval_lead_metrics.csv'
    xro_matches = glob.glob(xro_pattern)
    if xro_matches:
        label = 'XRO (AR1)' if compare_arp else 'XRO'
        results[label] = pd.read_csv(xro_matches[0])
        print(f"  Loaded {label}: {xro_matches[0]}")
    
    # NXRO models
    model_dirs = {
        'Res': f'{base_dir}/res',
        'Graph': f'{base_dir}/graphpyg',
        'Attentive': f'{base_dir}/attentive',
        'RO+Diag': f'{base_dir}/rodiag',
        'Linear': f'{base_dir}/linear',
        'Neural': f'{base_dir}/neural',
    }
    
    for model_name, model_dir in model_dirs.items():
        # Find stochastic eval CSV (look for _lead_metrics.csv specifically)
        # Pattern fixed to match AR1 files without intermediate tag (e.g., NXRO_RES_stochastic_eval_lead_metrics.csv)
        pattern = f'{model_dir}/**/*stochastic*eval_lead_metrics.csv'
        matches = glob.glob(pattern, recursive=True)
        
        if not matches:
            # Silent skip if not found (common if looking for specific variants)
            continue
        
        if compare_arp:
            # Load AR(p) variants
            for match in matches:
                basename = os.path.basename(match)
                
                # Skip sim noise / stage 2 for pure AR comparison
                if '_sim_noise' in basename or '_stage2' in basename:
                    continue

                # Extract p from ..._stochastic_arp{p}_eval...
                arp_match = re.search(r'_arp(\d+)_', basename)
                
                if arp_match:
                    p = arp_match.group(1)
                    stage_label = f'{model_name} (AR{p})'
                else:
                    # Default is AR1 (standard stochastic file without arp tag)
                    stage_label = f'{model_name} (AR1)'
                
                results[stage_label] = pd.read_csv(match)
                print(f"  Loaded {stage_label}: {match}")

        elif compare_stages:
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
                elif '_arp' in basename:
                    continue # Skip AR(p) files when comparing stages
                else:
                    stage_label = f'{model_name} (Post-hoc)'
                
                results[stage_label] = pd.read_csv(match)
                print(f"  Loaded {stage_label}: {match}")
        else:
            # Use most recent only - filter for AR1 baseline (no stage/arp variants)
            # Filter out ARP, stage, and sim_noise files to get the basic AR1 model
            standard_matches = [m for m in matches 
                              if '_arp' not in os.path.basename(m) 
                              and '_stage' not in os.path.basename(m)
                              and '_sim_noise' not in os.path.basename(m)]
            
            if include_two_stage:
                # Separate single-stage and two-stage models
                # Two-stage models have '_extra_data' in their path or specific naming
                single_stage_matches = [m for m in standard_matches 
                                       if '_extra_data' not in m and 'finetuned' not in m]
                two_stage_matches = [m for m in standard_matches 
                                    if '_extra_data' in m or 'finetuned' in m]
                
                # Load single-stage if available
                if single_stage_matches:
                    eval_file = max(single_stage_matches, key=os.path.getmtime)
                    results[f'NXRO-{model_name}'] = pd.read_csv(eval_file)
                    print(f"  Loaded NXRO-{model_name}: {eval_file}")
                
                # Load two-stage if available
                if two_stage_matches:
                    eval_file = max(two_stage_matches, key=os.path.getmtime)
                    results[f'NXRO-{model_name} (Two-Stage)'] = pd.read_csv(eval_file)
                    print(f"  Loaded NXRO-{model_name} (Two-Stage): {eval_file}")
            else:
                # Original behavior: load most recent only
                if standard_matches:
                    eval_file = max(standard_matches, key=os.path.getmtime)
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
        'Neural': '#9C27B0',
    }
    
    # Linestyles for stages
    stage_styles = {
        'Post-hoc': '-',
        'S2': '--',
        'S3': '-.',
        'S2+S3': ':',
        # AR styles
        'AR1': '-',
        'AR2': '--',
        'AR3': '-.',
        'AR4': ':',
    }
    
    for model_name, df in results.items():
        # Extract base model name and stage/AR
        base_model = model_name.split()[0].replace('NXRO-', '')
        
        # Determine color and linestyle
        color = base_colors.get(base_model, None)
        
        linestyle = '-'
        # Check for stages
        if '(S2+S3)' in model_name: linestyle = stage_styles['S2+S3']
        elif '(S2)' in model_name: linestyle = stage_styles['S2']
        elif '(S3)' in model_name: linestyle = stage_styles['S3']
        elif '(Post-hoc)' in model_name: linestyle = stage_styles['Post-hoc']
        # Check for AR
        elif '(AR1)' in model_name: linestyle = stage_styles['AR1']
        elif '(AR2)' in model_name: linestyle = stage_styles['AR2']
        elif '(AR3)' in model_name: linestyle = stage_styles['AR3']
        elif '(AR4)' in model_name: linestyle = stage_styles['AR4']
        
        marker = 'o' if 'XRO' in model_name else 's'
        lw = 2.5 if 'XRO' in model_name else 2
        
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
        'NXRO-Neural': '#9C27B0',
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
        'NXRO-Neural': '#9C27B0',
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


def plot_top_model_forecasts(results, base_dir, out_dir, top_n=5, shared_ylim=True):
    """Plot stochastic forecast examples for top N models including XRO.
    
    Args:
        results: Dictionary of model results
        base_dir: Base directory for forecast files
        out_dir: Output directory for plots
        top_n: Number of top models to plot
        shared_ylim: If True, use the same y-axis range for all plots
    """
    # Get top models by CRPS
    summary = []
    for model_name, df in results.items():
        avg_crps = float(np.nanmean(df['crps'].values))
        summary.append({'Model': model_name, 'Avg_CRPS': avg_crps})
    
    summary_df = pd.DataFrame(summary).sort_values('Avg_CRPS')
    
    # Always include XRO if present
    top_models = []
    # Check if any XRO variant is in the summary
    xro_variants = [m for m in summary_df['Model'].values if 'XRO' in m and 'NXRO' not in m]
    
    if xro_variants:
        top_models.extend(xro_variants)
        # Filter out XRO from others
        other_models = summary_df[~summary_df['Model'].isin(xro_variants)].head(top_n - len(xro_variants))
    else:
        other_models = summary_df.head(top_n)
    
    top_models.extend(other_models['Model'].tolist())
    
    print(f"\n  Plotting forecasts for top {len(top_models)} models:")
    for m in top_models:
        print(f"    - {m}")
    print()
    
    # Helper function to get forecast path
    def get_forecast_path(model_name, base_dir):
        if 'XRO' in model_name and 'NXRO' not in model_name:
            return f'{base_dir}/xro_baseline/xro_stochastic_fcst.nc', model_name
        
        # Extract base model and stage
        base_model = model_name.split()[0]
        
        # Determine suffix from stage or AR
        if '(S2+S3)' in model_name:
            suffix = '_sim_noise_stage2'
        elif '(S2)' in model_name:
            suffix = '_stage2'
        elif '(S3)' in model_name:
            suffix = '_sim_noise'
        elif '(AR' in model_name:
            import re
            p_match = re.search(r'\(AR(\d+)\)', model_name)
            if p_match:
                p = int(p_match.group(1))
                suffix = f'_arp{p}' if p > 1 else ''
            else:
                suffix = ''
        else:
            suffix = ''
        
        # Map model names to directories
        model_dirs = {
            'Res': 'res',
            'Graph': 'graphpyg/gcn_k3',
            'Attentive': 'attentive',
            'RO+Diag': 'rodiag',
            'Linear': 'linear',
            'Neural': 'neural',
        }
        
        model_dir = model_dirs.get(base_model, base_model.lower())
        
        # Handle graph special case
        if base_model == 'Graph':
            fcst_path = f'{base_dir}/{model_dir}/NXRO_GRAPHPYG_GCN_K3_stochastic{suffix}_forecasts.nc'
        elif base_model == 'Neural' and 'Two-Stage' in model_name:
            # Two-stage models may have _extra_data suffix
            fcst_path = f'{base_dir}/{model_dir}/NXRO_{base_model.upper()}_stochastic_extra_data_forecasts.nc'
            # Fallback if not found
            if not os.path.exists(fcst_path):
                fcst_path = f'{base_dir}/{model_dir}/NXRO_{base_model.upper()}_stochastic{suffix}_forecasts.nc'
        else:
            fcst_path = f'{base_dir}/{model_dir}/NXRO_{base_model.upper()}_stochastic{suffix}_forecasts.nc'
        
        return fcst_path, model_name
    
    # First pass: collect all data to determine shared y-axis range (if shared_ylim=True)
    all_plot_data = []
    global_ymin, global_ymax = np.inf, -np.inf
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    
    for model_name in top_models:
        try:
            fcst_path, model_label = get_forecast_path(model_name, base_dir)
            
            if not os.path.exists(fcst_path):
                continue
            
            fcst_ds = xr.open_dataset(fcst_path)
            
            # Pick a sample initialization
            init_date = '1979-01'
            if init_date not in fcst_ds.init.values:
                init_date = str(fcst_ds.init.values[0])[:7]
            
            fcst_var = fcst_ds['Nino34'].sel(init=init_date)
            fcst_mean = fcst_var.mean('member').squeeze()
            fcst_std = fcst_var.std('member').squeeze()
            
            lead_months = fcst_mean.lead.values
            fcst_mean_vals = fcst_mean.values.flatten()
            fcst_std_vals = fcst_std.values.flatten()
            
            # Get observations
            init_time = pd.Timestamp(init_date + '-01')
            obs_times = [init_time + pd.DateOffset(months=int(L)) for L in lead_months]
            obs_vals = []
            for t in obs_times:
                try:
                    val = obs_ds['Nino34'].sel(time=t, method='nearest').values
                    obs_vals.append(float(val))
                except:
                    obs_vals.append(np.nan)
            
            # Store data for plotting
            all_plot_data.append({
                'model_name': model_name,
                'model_label': model_label,
                'fcst_path': fcst_path,
                'init_date': init_date,
                'lead_months': lead_months,
                'fcst_mean_vals': fcst_mean_vals,
                'fcst_std_vals': fcst_std_vals,
                'obs_vals': obs_vals,
            })
            
            # Update global y-axis range
            if shared_ylim:
                y_vals = np.concatenate([
                    fcst_mean_vals - fcst_std_vals,
                    fcst_mean_vals + fcst_std_vals,
                    [v for v in obs_vals if not np.isnan(v)]
                ])
                global_ymin = min(global_ymin, np.nanmin(y_vals))
                global_ymax = max(global_ymax, np.nanmax(y_vals))
            
            fcst_ds.close()
            
        except Exception as e:
            print(f"    [!] Error loading {model_name}: {str(e)}")
            continue
    
    # Add some padding to y-axis range
    if shared_ylim and len(all_plot_data) > 0:
        y_range = global_ymax - global_ymin
        global_ymin -= 0.1 * y_range
        global_ymax += 0.1 * y_range
        print(f"  Using shared y-axis range: [{global_ymin:.2f}, {global_ymax:.2f}]")
    
    # Second pass: plot all models
    for data in all_plot_data:
        model_name = data['model_name']
        print(f"  Processing {model_name}...")
        print(f"    Looking for: {data['fcst_path']}")
        
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(data['lead_months'], data['fcst_mean_vals'], color='blue', lw=2, label='Ensemble Mean')
            ax.fill_between(data['lead_months'], 
                           data['fcst_mean_vals'] - data['fcst_std_vals'], 
                           data['fcst_mean_vals'] + data['fcst_std_vals'],
                           alpha=0.3, color='blue', label='±1 Std Dev')
            
            ax.plot(data['lead_months'], data['obs_vals'], color='black', lw=2, marker='o', 
                   markersize=4, label='Observed', alpha=0.8)
            
            ax.set_xlabel('Forecast Lead (months)', fontsize=11)
            ax.set_ylabel('Nino3.4 SSTA (C)', fontsize=11)
            ax.set_title(f'{data["model_label"]}: Stochastic Forecast (Init: {data["init_date"]})', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            
            # Apply shared y-axis limits if enabled
            if shared_ylim:
                ax.set_ylim(global_ymin, global_ymax)
            
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
    
    obs_ds.close()


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
        'NXRO-Neural': '#9C27B0',
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


def plot_forecast_plumes(results, base_dir, out_dir, init_dates=None):
    """Plot detailed forecast plumes for specific initialization dates.
    
    Similar to XRO_example.py plot_forecast_plume, showing:
    - Ensemble mean with ±1 std dev bands
    - Observations with proper time axis
    - Multiple models in subplots
    
    Args:
        results: Dictionary of model results
        base_dir: Base results directory
        out_dir: Output directory for plots
        init_dates: List of initialization dates (e.g., ['1997-04', '2022-09'])
    """
    import datetime
    from dateutil.relativedelta import relativedelta
    import matplotlib.dates as mdates
    
    if init_dates is None:
        init_dates = ['1997-04', '1997-12', '2022-09']
    
    # Load observations
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    
    # Get top models by CRPS
    summary = []
    for model_name, df in results.items():
        avg_crps = float(np.nanmean(df['crps'].values))
        summary.append({'Model': model_name, 'Avg_CRPS': avg_crps})
    
    summary_df = pd.DataFrame(summary).sort_values('Avg_CRPS')
    
    # Select top models (exclude XRO baseline, handle separately)
    top_nxro_models = summary_df[~summary_df['Model'].str.contains('XRO') | summary_df['Model'].str.contains('NXRO')].head(3)['Model'].tolist()
    # Note: The above logic is slightly flawed if XRO is not in the name.
    # Better: Exclude pure XRO variants
    top_nxro_models = [m for m in summary_df['Model'].values if 'NXRO' in m or ('XRO' not in m)][:3]
    
    print(f"\n  Plotting forecast plumes for dates: {init_dates}")
    print(f"  Models: {top_nxro_models}")
    
    for init_date in init_dates:
        print(f"\n  Processing initialization date: {init_date}")
        
        # Create figure with subplots for each model + XRO
        n_models = len(top_nxro_models) + 1  # +1 for XRO
        fig, axes = plt.subplots(n_models, 1, figsize=(10, 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        # Plot XRO first (baseline)
        try:
            xro_path = f'{base_dir}/xro_baseline/xro_stochastic_fcst.nc'
            if os.path.exists(xro_path):
                fcst_ds = xr.open_dataset(xro_path)
                ax = axes[0]
                
                # Get forecast data
                fcst_var = fcst_ds['Nino34'].sel(init=init_date)
                fcst_mean = fcst_var.mean('member').squeeze()
                fcst_std = fcst_var.std('member').squeeze()
                
                nlead = len(fcst_mean.lead)
                
                # Setup time axis
                xdate_init = datetime.datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
                xdate_strt = xdate_init + relativedelta(months=-2)
                xdate_last = xdate_init + relativedelta(months=nlead-1)
                xtime_fcst = [xdate_init + relativedelta(months=int(i)) for i in range(nlead)]
                
                # Plot forecast with uncertainty
                ax.plot(xtime_fcst, fcst_mean.values, c='red', marker='.', lw=2.5, 
                       label='XRO (AR1) Ensemble Mean')
                ax.fill_between(xtime_fcst, 
                               fcst_mean.values - fcst_std.values, 
                               fcst_mean.values + fcst_std.values,
                               fc='red', alpha=0.3, label='±1 Std Dev')
                
                # Plot observations
                sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
                ax.plot(sel_obs.time.values, sel_obs.values, c='black', 
                       marker='o', markersize=3, lw=2, label='Observed', alpha=0.8)
                
                # Formatting
                ax.axhline(0, c='gray', ls='-', lw=0.5, alpha=0.5)
                ax.axhline(0.5, c='red', ls='--', lw=1, alpha=0.3)
                ax.axhline(-0.5, c='blue', ls='--', lw=1, alpha=0.3)
                
                ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
                ax.set_xlim([xdate_strt, xdate_last])
                ax.set_ylim([-3, 3])
                ax.set_ylabel('Nino3.4 SSTA (°C)', fontsize=10)
                ax.set_title(f'XRO (AR1) Baseline - Init: {init_date}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9, loc='upper left')
                ax.grid(True, alpha=0.3)
                
                fcst_ds.close()
        except Exception as e:
            print(f"    [!] Error plotting XRO: {str(e)}")
        
        # Plot NXRO models
        for idx, model_name in enumerate(top_nxro_models):
            ax_idx = idx + 1
            try:
                # Determine file path - strip "NXRO-" prefix if present
                base_model = model_name.replace('NXRO-', '').split()[0]
                
                # Determine suffix from stage or AR
                if '(S2+S3)' in model_name:
                    suffix = '_sim_noise_stage2'
                elif '(S2)' in model_name:
                    suffix = '_stage2'
                elif '(S3)' in model_name:
                    suffix = '_sim_noise'
                elif '(AR' in model_name:
                    import re
                    p_match = re.search(r'\(AR(\d+)\)', model_name)
                    if p_match:
                        p = int(p_match.group(1))
                        suffix = f'_arp{p}' if p > 1 else ''
                    else:
                        suffix = ''
                else:
                    suffix = '_stage2'  # Default to stage2 for most recent if unclear
                
                # Map model names to directories
                model_dirs = {
                    'Res': 'res',
                    'Graph': 'graphpyg/gcn_k3',
                    'Attentive': 'attentive',
                    'RO+Diag': 'rodiag',
                    'Linear': 'linear',
                    'Neural': 'neural',
                }
                
                model_dir = model_dirs.get(base_model, base_model.lower())
                
                # Construct file path
                if base_model == 'Graph':
                    fcst_path = f'{base_dir}/{model_dir}/NXRO_GRAPHPYG_GCN_K3_stochastic{suffix}_forecasts.nc'
                elif base_model == 'Neural' and 'Two-Stage' in model_name:
                    # Two-stage models may have _extra_data suffix
                    fcst_path = f'{base_dir}/{model_dir}/NXRO_{base_model.upper()}_stochastic_extra_data_forecasts.nc'
                    # Fallback if not found
                    if not os.path.exists(fcst_path):
                        fcst_path = f'{base_dir}/{model_dir}/NXRO_{base_model.upper()}_stochastic{suffix}_forecasts.nc'
                else:
                    fcst_path = f'{base_dir}/{model_dir}/NXRO_{base_model.upper()}_stochastic{suffix}_forecasts.nc'
                
                if not os.path.exists(fcst_path):
                    print(f"    [!] Forecast file not found: {fcst_path}")
                    continue
                
                fcst_ds = xr.open_dataset(fcst_path)
                ax = axes[ax_idx]
                
                # Get forecast data
                fcst_var = fcst_ds['Nino34'].sel(init=init_date)
                fcst_mean = fcst_var.mean('member').squeeze()
                fcst_std = fcst_var.std('member').squeeze()
                
                nlead = len(fcst_mean.lead)
                
                # Setup time axis
                xdate_init = datetime.datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
                xdate_strt = xdate_init + relativedelta(months=-2)
                xdate_last = xdate_init + relativedelta(months=nlead-1)
                xtime_fcst = [xdate_init + relativedelta(months=int(i)) for i in range(nlead)]
                
                # Determine color
                colors = {
                    'Res': '#4CAF50',
                    'Graph': '#78909C',
                    'Attentive': '#EC407A',
                    'RO+Diag': '#FF6F00',
                    'Linear': '#2196F3',
                    'Neural': '#9C27B0',
                }
                color = colors.get(base_model, '#666666')
                
                # Plot forecast with uncertainty
                ax.plot(xtime_fcst, fcst_mean.values, c=color, marker='.', lw=2.5,
                       label=f'{model_name} Ensemble Mean')
                ax.fill_between(xtime_fcst,
                               fcst_mean.values - fcst_std.values,
                               fcst_mean.values + fcst_std.values,
                               fc=color, alpha=0.3, label='±1 Std Dev')
                
                # Plot observations
                sel_obs = obs_ds['Nino34'].sel(time=slice(xdate_strt, xdate_last))
                ax.plot(sel_obs.time.values, sel_obs.values, c='black',
                       marker='o', markersize=3, lw=2, label='Observed', alpha=0.8)
                
                # Formatting
                ax.axhline(0, c='gray', ls='-', lw=0.5, alpha=0.5)
                ax.axhline(0.5, c='red', ls='--', lw=1, alpha=0.3)
                ax.axhline(-0.5, c='blue', ls='--', lw=1, alpha=0.3)
                
                ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
                ax.set_xlim([xdate_strt, xdate_last])
                ax.set_ylim([-3, 3])
                ax.set_ylabel('Nino3.4 SSTA (°C)', fontsize=10)
                ax.set_title(f'{model_name} - Init: {init_date}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9, loc='upper left')
                ax.grid(True, alpha=0.3)
                
                fcst_ds.close()
                
            except Exception as e:
                import traceback
                print(f"    [!] Error plotting {model_name}: {str(e)}")
                traceback.print_exc()
        
        plt.tight_layout()
        safe_date = init_date.replace('-', '_')
        out_path = f'{out_dir}/plume_comparison_{safe_date}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {out_path}")
    
    obs_ds.close()


def plot_spring_barrier_forecasts(results, base_dir, out_dir, selected_models=None):
    """
    Plot forecasts specifically designed to test the "spring barrier" phenomenon.
    
    The spring barrier refers to the difficulty in making skillful ENSO predictions
    across the boreal spring (March-May), when predictability drops significantly.
    
    We test forecasts initialized in winter (Dec-Feb) that must predict through
    spring to capture the development and peak of major El Niño events:
    - 1982-83 El Niño (peaked Dec 1982 - Jan 1983)
    - 1997-98 El Niño (peaked Nov 1997 - Jan 1998) 
    - 2015-16 El Niño (peaked Nov 2015 - Feb 2016)
    
    Style: Each model in its own subplot with ensemble mean + uncertainty bands.
    
    Args:
        results: Dictionary of model results
        base_dir: Base results directory
        out_dir: Output directory for plots
        selected_models: List of model names to plot
    """
    import datetime
    from dateutil.relativedelta import relativedelta
    import matplotlib.dates as mdates
    
    # Spring barrier test cases - Major El Niño events (historical + out-of-sample)
    # Initialize in winter (Jan) to test prediction through spring barrier to peak
    spring_barrier_cases = [
        # Historical events (in-sample period)
        {
            'name': '1982-83 El Niño',
            'init_date': '1982-01',
            'event_peak': 'Dec 1982 - Jan 1983',
        },
        {
            'name': '1997-98 El Niño', 
            'init_date': '1997-01',
            'event_peak': 'Nov 1997 - Jan 1998',
        },
        # Out-of-sample period events
        {
            'name': '2009-10 El Niño',
            'init_date': '2009-01',
            'event_peak': 'Nov 2009 - Jan 2010',
        },
        {
            'name': '2015-16 El Niño', 
            'init_date': '2015-01',
            'event_peak': 'Nov 2015 - Feb 2016',
        },
        {
            'name': '2018-19 El Niño',
            'init_date': '2018-01',
            'event_peak': 'Oct 2018 - Feb 2019',
        },
    ]
    
    # Load observations
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    
    # Fixed model list: XRO, NXRO-MLP (Res), NXRO-Graph, NXRO-Attentive, NXRO-Linear
    # Based on actual files found in results_out_of_sample
    model_configs = [
        {'key': 'XRO', 'display': 'XRO Baseline', 'dir': 'xro_baseline', 
         'file_patterns': ['xro_stochastic_fcst.nc'],
         'color': 'red', 'is_xro': True},
        {'key': 'Res', 'display': 'NXRO-MLP', 'dir': 'res', 
         'file_patterns': ['NXRO_RES_stochastic_forecasts.nc', 'NXRO_RES_stochastic_stage2_forecasts.nc'],
         'color': '#4CAF50', 'is_xro': False},
        {'key': 'Graph', 'display': 'NXRO-Graph', 'dir': 'graphpyg/gcn_k3', 
         'file_patterns': ['NXRO_GRAPHPYG_GCN_K3_stochastic_forecasts.nc'],
         'color': '#78909C', 'is_xro': False},
        {'key': 'Attentive', 'display': 'NXRO-Attentive', 'dir': 'attentive', 
         'file_patterns': ['NXRO_ATTENTIVE_stochastic_forecasts.nc', 'NXRO_ATTENTIVE_stochastic_stage2_forecasts.nc'],
         'color': '#EC407A', 'is_xro': False},
        {'key': 'Linear', 'display': 'NXRO-Linear', 'dir': 'linear', 
         'file_patterns': ['NXRO_linear_stochastic_forecasts.nc', 'NXRO_LINEAR_stochastic_forecasts.nc'],
         'color': '#2196F3', 'is_xro': False},
    ]
    
    print("\n" + "="*80)
    print("SPRING BARRIER FORECAST VISUALIZATION")
    print("="*80)
    print("Testing prediction skill across the boreal spring barrier (Mar-May)")
    print("for major El Niño events initialized in winter (Jan)")
    print(f"Models: {[m['display'] for m in model_configs]}")
    print()
    
    # Helper function to find forecast file
    def find_forecast_file(model_config):
        """Find the stochastic forecast NetCDF file for a model."""
        # Try all file patterns for this model
        for pattern in model_config['file_patterns']:
            path = f'{base_dir}/{model_config["dir"]}/{pattern}'
            if os.path.exists(path):
                return path
        return None
    
    # Generate plot for each El Niño event
    for case in spring_barrier_cases:
        init_date = case['init_date']
        event_name = case['name']
        
        print(f"\n  Processing {event_name} (Init: {init_date})...")
        
        # Create figure with subplots - one per model (like plume comparison)
        n_models = len(model_configs)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        # Setup time axis
        try:
            xdate_init = datetime.datetime.strptime(init_date + '-01', "%Y-%m-%d").date()
        except:
            print(f"    [!] Invalid init date: {init_date}")
            continue
        
        xdate_strt = xdate_init + relativedelta(months=-2)
        xdate_last = xdate_init + relativedelta(months=20)
        
        # Plot each model in its own subplot
        for ax_idx, model_config in enumerate(model_configs):
            ax = axes[ax_idx]
            
            fcst_path = find_forecast_file(model_config)
            if fcst_path is None:
                print(f"    [!] Forecast file not found for {model_config['display']}")
                ax.text(0.5, 0.5, f"No data for {model_config['display']}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{model_config['display']} - Init: {init_date}", fontsize=11, fontweight='bold')
                continue
            
            try:
                fcst_ds = xr.open_dataset(fcst_path)
                
                # Get forecast data - use sel directly (like plot_forecast_plumes does)
                # The init coordinate may be datetime objects, not strings
                fcst_var = fcst_ds['Nino34'].sel(init=init_date)
                fcst_mean = fcst_var.mean('member').squeeze()
                fcst_std = fcst_var.std('member').squeeze()
                
                nlead = len(fcst_mean.lead)
                xtime_fcst = [xdate_init + relativedelta(months=int(i)) for i in range(nlead)]
                
                # Plot ensemble mean with uncertainty band
                color = model_config['color']
                ax.plot(xtime_fcst, fcst_mean.values, c=color, marker='.', lw=2.5,
                       label=f'{model_config["display"]} Ensemble Mean')
                ax.fill_between(xtime_fcst,
                               fcst_mean.values - fcst_std.values,
                               fcst_mean.values + fcst_std.values,
                               fc=color, alpha=0.3, label='±1 Std Dev')
                
                fcst_ds.close()
                
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
                ax.set_title(f'{model_config["display"]} - Init: {init_date}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9, loc='upper left')
                ax.grid(True, alpha=0.3)
                
                print(f"    ✓ Plotted {model_config['display']}")
                
            except KeyError as e:
                print(f"    [!] Init date {init_date} not in {model_config['display']} forecast")
                ax.text(0.5, 0.5, f"Init date {init_date} not available", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{model_config["display"]} - Init: {init_date}', fontsize=11, fontweight='bold')
            except Exception as e:
                print(f"    [!] Error plotting {model_config['display']}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:50]}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{model_config["display"]} - Init: {init_date}', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'Spring Barrier Test: {event_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space at top for suptitle
        
        safe_name = event_name.replace(' ', '_').replace('-', '_')
        out_path = f'{out_dir}/spring_barrier_{safe_name}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {out_path}")
    
    obs_ds.close()
    print("\n  Spring barrier visualization complete!")


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
    parser.add_argument('--compare_arp', action='store_true',
                       help='Compare different AR(p) noise models (AR1, AR2, etc.)')
    parser.add_argument('--include_two_stage', action='store_true',
                       help='Include two-stage trained models (synthetic pre-training + fine-tuning)')
    parser.add_argument('--plot_plumes', action='store_true',
                       help='Generate detailed forecast plume plots for specific dates')
    parser.add_argument('--plume_dates', type=str, nargs='+',
                       default=['1997-04', '1997-12', '2022-09'],
                       help='Initialization dates for plume plots (format: YYYY-MM)')
    parser.add_argument('--spring_barrier', action='store_true',
                       help='Generate spring barrier test visualizations for major El Niño events (1982-83, 1997-98, 2015-16)')
    parser.add_argument('--no_shared_ylim', action='store_true',
                       help='Disable shared y-axis range across forecast plots (default: shared)')
    args = parser.parse_args()
    
    print("="*80)
    print("STOCHASTIC ENSEMBLE COMPARISON")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    if args.compare_stages:
        print(f"Mode: COMPARE ALL STAGES (post-hoc, S2, S3, S2+S3)")
    elif args.compare_arp:
        print(f"Mode: COMPARE AR(p) VARIANTS (AR1, AR2, etc.)")
    elif args.include_two_stage:
        print(f"Mode: Compare models (including two-stage trained models)")
    else:
        print(f"Mode: Compare models (most recent run only)")
    print()
    
    # Load results
    print("Loading stochastic evaluation results...")
    results = load_stochastic_results(args.results_dir, compare_stages=args.compare_stages, 
                                     compare_arp=args.compare_arp, include_two_stage=args.include_two_stage)
    
    if len(results) == 0:
        print("[X] No stochastic results found!")
        print("Run evaluate_stochastic_top5.sh first")
        return
    
    print(f"\nLoaded {len(results)} models/variants")
    print()
    
    # Create output directory
    if args.compare_stages:
        out_dir = f'{args.results_dir}/rankings/stochastic_stages_comparison'
    elif args.compare_arp:
        out_dir = f'{args.results_dir}/rankings/stochastic_arp_comparison'
    elif args.include_two_stage:
        out_dir = f'{args.results_dir}/rankings/stochastic_comparison_with_two_stage'
    else:
        out_dir = f'{args.results_dir}/rankings/stochastic_comparison'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Only generate ranking plot and forecast visualizations
    plot_metric_rankings(results, out_dir, top_n=10)
    
    print("\nGenerating forecast visualizations for top models...")
    plot_top_model_forecasts(results, args.results_dir, out_dir, top_n=5, 
                             shared_ylim=not args.no_shared_ylim)
    
    # Generate plume plots if requested
    if args.plot_plumes:
        print("\nGenerating detailed forecast plume plots...")
        plot_forecast_plumes(results, args.results_dir, out_dir, init_dates=args.plume_dates)
    
    # Generate spring barrier visualizations if requested
    if args.spring_barrier:
        print("\nGenerating spring barrier test visualizations...")
        plot_spring_barrier_forecasts(results, args.results_dir, out_dir)
    
    # Print summary
    print_summary_table(results, out_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput files in: {out_dir}/")
    print("  - stochastic_rankings_all_metrics.png (CRPS & RMSE rankings, top 10)")
    print("  - forecast_*.png (forecast visualizations for top 5 models + XRO)")
    if args.plot_plumes:
        print("  - plume_comparison_*.png (detailed forecast plumes for specified dates)")
    if args.spring_barrier:
        print("  - spring_barrier_*.png (spring barrier test for 1982-83, 1997-98, 2015-16 El Niño)")
        print("  - spring_barrier_summary_comparison.png (combined comparison)")
    print("  - stochastic_summary_metrics.csv (complete metrics table)")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

