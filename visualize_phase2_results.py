#!/usr/bin/env python
"""
Visualize Phase 2 Results: Training Strategy Optimization

This script analyzes training strategy experiments for top graph models from Phase 1.

Usage:
    python visualize_phase2_results.py
    python visualize_phase2_results.py --results_dir results_out_of_sample
"""

import os
import re
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_graph_model_name(model_name):
    """Extract topology, k, and GNN type from model name."""
    name_lower = model_name.lower()
    
    if 'gat' in name_lower:
        gnn_type = 'GAT'
    elif 'gcn' in name_lower or 'graph' in name_lower:
        gnn_type = 'GCN'
    else:
        gnn_type = 'Unknown'
    
    k_match = re.search(r'k(\d+)', name_lower)
    k_val = int(k_match.group(1)) if k_match else None
    
    if 'pearson' in name_lower:
        topology = 'Pearson'
    elif 'spearman' in name_lower:
        topology = 'Spearman'
    elif 'mi' in name_lower and 'resmix' not in name_lower:
        topology = 'MI'
    elif 'xcorr' in name_lower:
        topology = 'XCorr-Max'
    elif 'xro' in name_lower or 'graph' in name_lower:
        topology = 'XRO'
    else:
        topology = 'Unknown'
    
    return topology, k_val, gnn_type


def load_rankings(csv_path):
    """Load ranking CSV and filter for graph models."""
    df = pd.read_csv(csv_path)
    
    graph_mask = (df['Model'].str.contains('Graph', case=False, na=False) | 
                  df['Model'].str.contains('graphpyg', case=False, na=False))
    exclude_mask = (df['Model'].str.contains(r'RO\+Diag', case=False, na=False) |
                   df['Model'].str.contains('ResidualMix', case=False, na=False) |
                   df['Model'].str.contains(r'^Linear$', case=False, na=False) |
                   df['Model'].str.contains(r'^Ro$', case=False, na=False) |
                   df['Model'].str.contains('Neural', case=False, na=False))
    
    graph_df = df[graph_mask & ~exclude_mask].copy()
    
    graph_df['Topology'] = graph_df['Model'].apply(lambda x: parse_graph_model_name(x)[0])
    graph_df['K'] = graph_df['Model'].apply(lambda x: parse_graph_model_name(x)[1])
    graph_df['GNN_Type'] = graph_df['Model'].apply(lambda x: parse_graph_model_name(x)[2])
    
    return graph_df


def identify_config_variants(df):
    """Group models by base configuration and identify training variants."""
    df = df.copy()
    
    # Create base config identifier
    def get_base_config(model_name):
        name_lower = model_name.lower()
        # Remove training variant indicators
        for suffix in ['ws', 'fixl', 'fixro', 'fixdiag', 'fixnl', 'fixphysics']:
            name_lower = name_lower.replace(f'_{suffix}', '')
            name_lower = name_lower.replace(f' {suffix}', '')
        return name_lower
    
    df['Base_Config'] = df['Model'].apply(get_base_config)
    
    # Identify training variant
    def get_training_variant(model_name):
        name_lower = model_name.lower()
        if 'fixl' in name_lower:
            return 'FixL'
        elif ' ws' in name_lower or '_ws' in name_lower:
            return 'WS'
        else:
            return 'Base'
    
    df['Training_Variant'] = df['Model'].apply(get_training_variant)
    
    return df


def plot_improvement_over_phase1(df, out_dir):
    """Plot performance improvement for each base configuration."""
    df_variants = identify_config_variants(df)
    
    # Group by base config
    configs = df_variants.groupby(['Topology', 'K', 'GNN_Type'])
    
    improvements = []
    
    for (topo, k, gnn), group in configs:
        if len(group) < 2:
            continue
        
        base_rmse = group[group['Training_Variant'] == 'Base']['Mean_RMSE_Test']
        best_rmse = group['Mean_RMSE_Test'].min()
        
        if len(base_rmse) > 0:
            base_val = base_rmse.values[0]
            improvement = base_val - best_rmse
            best_variant = group.loc[group['Mean_RMSE_Test'].idxmin(), 'Training_Variant']
            
            improvements.append({
                'Config': f'{topo} k={k} {gnn}',
                'Base_RMSE': base_val,
                'Best_RMSE': best_rmse,
                'Improvement': improvement,
                'Best_Variant': best_variant
            })
    
    if not improvements:
        print("  [!] Not enough training variants to show improvement")
        return
    
    imp_df = pd.DataFrame(improvements).sort_values('Improvement', ascending=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(imp_df))
    colors = ['green' if imp > 0 else 'red' for imp in imp_df['Improvement']]
    
    bars = ax.barh(x, imp_df['Improvement']*1000, color=colors, alpha=0.7)  # Convert to millidegrees
    ax.set_yticks(x)
    ax.set_yticklabels(imp_df['Config'], fontsize=8)
    ax.set_xlabel('RMSE Improvement (millidegrees C)', fontsize=11)
    ax.set_title('Phase 2 Training Optimization: Improvement over Phase 1 Baseline', fontsize=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val, variant) in enumerate(zip(bars, imp_df['Improvement'], imp_df['Best_Variant'])):
        label_x = val*1000 + (0.5 if val > 0 else -0.5)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
               f'{val*1000:.1f} ({variant})', ha='left' if val > 0 else 'right', 
               va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase2_improvement_over_phase1.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase2_improvement_over_phase1.png")


def plot_training_variant_comparison(df, out_dir):
    """Compare training strategy variants."""
    df_variants = identify_config_variants(df)
    
    # Group by training variant
    variant_stats = df_variants.groupby('Training_Variant').agg({
        'Mean_RMSE_Test': ['mean', 'std', 'min', 'count'],
        'Mean_ACC_Test': ['mean', 'std', 'max']
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE comparison
    variants = variant_stats['Training_Variant'].values
    rmse_means = variant_stats['Mean_RMSE_Test']['mean'].values
    rmse_stds = variant_stats['Mean_RMSE_Test']['std'].values
    counts = variant_stats['Mean_RMSE_Test']['count'].values
    
    x = np.arange(len(variants))
    axes[0].bar(x, rmse_means, yerr=rmse_stds, alpha=0.7, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{v}\n(n={c})' for v, c in zip(variants, counts)])
    axes[0].set_ylabel('Test RMSE (C)', fontsize=11)
    axes[0].set_title('RMSE by Training Strategy', fontsize=12)
    axes[0].axhline(0.567, color='green', linestyle='--', alpha=0.7, label='Target (Res)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # ACC comparison
    acc_means = variant_stats['Mean_ACC_Test']['mean'].values
    acc_stds = variant_stats['Mean_ACC_Test']['std'].values
    
    axes[1].bar(x, acc_means, yerr=acc_stds, alpha=0.7, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{v}\n(n={c})' for v, c in zip(variants, counts)])
    axes[1].set_ylabel('Test ACC', fontsize=11)
    axes[1].set_title('ACC by Training Strategy', fontsize=12)
    axes[1].axhline(0.628, color='green', linestyle='--', alpha=0.7, label='Target (Res)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase2_training_strategy_comparison.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase2_training_strategy_comparison.png")


def plot_phase2_top_models(df, out_dir, top_n=10):
    """Show top N models from Phase 2."""
    top_df = df.nsmallest(top_n, 'Mean_RMSE_Test')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    x = np.arange(len(top_df))
    colors = []
    for model in top_df['Model'].values:
        if 'GAT' in model.upper():
            colors.append('tab:orange')
        else:
            colors.append('tab:blue')
    
    bars = ax.barh(x, top_df['Mean_RMSE_Test'], color=colors, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{int(rank)}. {model[:50]}" for rank, model in 
                        zip(top_df['Rank'], top_df['Model'])], fontsize=7)
    ax.set_xlabel('Test RMSE (C)', fontsize=11)
    ax.set_title(f'Top {top_n} Graph Models After Phase 2', fontsize=12)
    target_line = ax.axvline(0.567, color='green', linestyle='--', linewidth=2, label='NXRO-Res (Target)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', alpha=0.7, label='GCN'),
        Patch(facecolor='tab:orange', alpha=0.7, label='GAT'),
        target_line
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase2_top{top_n}_models.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase2_top{top_n}_models.png")


def print_summary_table(df, phase1_best_rmse=None):
    """Print Phase 2 summary statistics."""
    print("\n" + "="*80)
    print("PHASE 2 RESULTS SUMMARY")
    print("="*80)
    
    # Overall best
    best_rmse_idx = df['Mean_RMSE_Test'].idxmin()
    best_rmse = df.loc[best_rmse_idx]
    
    print("\nBest Graph Model After Phase 2:")
    print(f"  Model: {best_rmse['Model']}")
    print(f"  Topology: {best_rmse['Topology']}, k={best_rmse['K']}, GNN={best_rmse['GNN_Type']}")
    print(f"  Test RMSE: {best_rmse['Mean_RMSE_Test']:.4f} C")
    print(f"  Test ACC: {best_rmse['Mean_ACC_Test']:.4f}")
    print(f"  Overall Rank: {int(best_rmse['Rank'])}")
    
    # Check against target
    gap_to_target = best_rmse['Mean_RMSE_Test'] - 0.567
    if gap_to_target <= 0:
        print(f"\n  [SUCCESS] Beats NXRO-Res target by {-gap_to_target:.4f} C!")
    else:
        print(f"\n  [GAP] Still {gap_to_target:.4f} C away from target")
    
    # Improvement from Phase 1
    if phase1_best_rmse is not None:
        phase2_improvement = phase1_best_rmse - best_rmse['Mean_RMSE_Test']
        print(f"\n  Phase 1 best: {phase1_best_rmse:.4f} C")
        print(f"  Phase 2 best: {best_rmse['Mean_RMSE_Test']:.4f} C")
        print(f"  Improvement: {phase2_improvement:.4f} C ({phase2_improvement/phase1_best_rmse*100:.1f}%)")
    
    # Top 10
    print("\n" + "-"*80)
    print("Top 10 Graph Models After Phase 2:")
    print("-"*80)
    top10 = df.nsmallest(10, 'Mean_RMSE_Test')[['Rank', 'Model', 'Topology', 'K', 'GNN_Type', 
                                                  'Mean_ACC_Test', 'Mean_RMSE_Test']]
    print(top10.to_string(index=False))
    
    # Training variant statistics
    df_variants = identify_config_variants(df)
    print("\n" + "-"*80)
    print("Performance by Training Strategy:")
    print("-"*80)
    variant_stats = df_variants.groupby('Training_Variant').agg({
        'Mean_RMSE_Test': ['mean', 'min', 'count'],
        'Mean_ACC_Test': ['mean', 'max']
    })
    print(variant_stats.to_string())
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase 2 Training Strategy Results')
    parser.add_argument('--results_dir', type=str, default='results_out_of_sample',
                       help='Results directory')
    parser.add_argument('--phase1_csv', type=str, default=None,
                       help='Phase 1 results CSV (for baseline comparison)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top models to display')
    args = parser.parse_args()
    
    # Find ranking CSV
    ranking_csvs = glob.glob(f'{args.results_dir}/rankings/all_variants_ranked_*_out_of_sample.csv')
    
    if not ranking_csvs:
        print(f"[X] No ranking CSV found in {args.results_dir}/rankings/")
        print("Run ranking script first: python rank_all_variants_out_of_sample.py")
        return
    
    # Use most recent
    csv_path = max(ranking_csvs, key=os.path.getmtime)
    print(f"Loading rankings from: {csv_path}")
    
    # Load and filter graph models
    graph_df = load_rankings(csv_path)
    
    print(f"\nFound {len(graph_df)} graph models in rankings")
    
    # Load Phase 1 baseline if provided
    phase1_best_rmse = None
    if args.phase1_csv and os.path.exists(args.phase1_csv):
        phase1_df = load_rankings(args.phase1_csv)
        if len(phase1_df) > 0:
            phase1_best_rmse = phase1_df['Mean_RMSE_Test'].min()
            print(f"Phase 1 best RMSE: {phase1_best_rmse:.4f} C")
    
    # Create output directory
    out_dir = f'{args.results_dir}/rankings/phase2_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating Phase 2 visualizations...")
    
    plot_improvement_over_phase1(graph_df, out_dir)
    plot_training_variant_comparison(graph_df, out_dir)
    plot_phase2_top_models(graph_df, out_dir, top_n=args.top_n)
    
    # Print summary
    print_summary_table(graph_df, phase1_best_rmse)
    
    # Save filtered dataframe
    csv_out = f'{out_dir}/phase2_graph_models.csv'
    graph_df.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved results: {csv_out}")
    
    print("\n" + "="*80)
    print("PHASE 2 VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput files in: {out_dir}/")
    print("  - phase2_improvement_over_phase1.png")
    print("  - phase2_training_strategy_comparison.png")
    print(f"  - phase2_top{args.top_n}_models.png")
    print("  - phase2_graph_models.csv")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

