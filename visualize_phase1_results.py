#!/usr/bin/env python
"""
Visualize Phase 1 Results: Graph Topology Search

This script analyzes and visualizes the results of Phase 1 graph topology exploration,
helping identify the best (topology, k, GNN_type) combination.

Usage:
    python visualize_phase1_results.py
    python visualize_phase1_results.py --results_dir results_all_outsample
"""

import os
import re
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def parse_graph_model_name(model_name):
    """Extract topology, k, and GNN type from model name."""
    name_lower = model_name.lower()
    
    # Determine GNN type
    if 'gat' in name_lower:
        gnn_type = 'GAT'
    elif 'gcn' in name_lower or 'graph' in name_lower:
        gnn_type = 'GCN'
    else:
        gnn_type = 'Unknown'
    
    # Extract k value
    k_match = re.search(r'k(\d+)', name_lower)
    if k_match:
        k_val = int(k_match.group(1))
    else:
        k_val = None
    
    # Determine topology
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
    
    # Filter for graph models only (not Graph in name for variants like RO+Diag)
    graph_mask = (df['Model'].str.contains('Graph', case=False, na=False) | 
                  df['Model'].str.contains('graphpyg', case=False, na=False))
    # Exclude non-graph models that happen to have 'graph' in path
    exclude_mask = (df['Model'].str.contains('RO+Diag', case=False, na=False) |
                   df['Model'].str.contains('ResidualMix', case=False, na=False) |
                   df['Model'].str.contains('^Linear$', case=False, na=False) |
                   df['Model'].str.contains('^Ro$', case=False, na=False) |
                   df['Model'].str.contains('Neural', case=False, na=False))
    
    graph_df = df[graph_mask & ~exclude_mask].copy()
    
    # Parse model properties
    graph_df['Topology'] = graph_df['Model'].apply(lambda x: parse_graph_model_name(x)[0])
    graph_df['K'] = graph_df['Model'].apply(lambda x: parse_graph_model_name(x)[1])
    graph_df['GNN_Type'] = graph_df['Model'].apply(lambda x: parse_graph_model_name(x)[2])
    
    return graph_df


def plot_topology_comparison(df, out_dir):
    """Plot performance by topology type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter valid entries
    valid_df = df[df['Topology'] != 'Unknown'].copy()
    
    # Group by topology
    topology_stats = valid_df.groupby('Topology').agg({
        'Mean_RMSE_Test': ['mean', 'std', 'min'],
        'Mean_ACC_Test': ['mean', 'std', 'max']
    }).reset_index()
    
    # Plot RMSE
    topologies = topology_stats['Topology'].values
    rmse_means = topology_stats['Mean_RMSE_Test']['mean'].values
    rmse_stds = topology_stats['Mean_RMSE_Test']['std'].values
    rmse_mins = topology_stats['Mean_RMSE_Test']['min'].values
    
    x = np.arange(len(topologies))
    axes[0].bar(x, rmse_means, yerr=rmse_stds, alpha=0.7, capsize=5, label='Mean +/- Std')
    axes[0].scatter(x, rmse_mins, color='red', s=100, marker='*', label='Best', zorder=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(topologies, rotation=45, ha='right')
    axes[0].set_ylabel('Test RMSE (C)', fontsize=11)
    axes[0].set_xlabel('Graph Topology', fontsize=11)
    axes[0].set_title('RMSE by Topology Type', fontsize=12)
    axes[0].axhline(0.586, color='green', linestyle='--', alpha=0.7, label='NXRO-Linear (Target)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot ACC
    acc_means = topology_stats['Mean_ACC_Test']['mean'].values
    acc_stds = topology_stats['Mean_ACC_Test']['std'].values
    acc_maxs = topology_stats['Mean_ACC_Test']['max'].values
    
    axes[1].bar(x, acc_means, yerr=acc_stds, alpha=0.7, capsize=5, label='Mean +/- Std')
    axes[1].scatter(x, acc_maxs, color='red', s=100, marker='*', label='Best', zorder=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(topologies, rotation=45, ha='right')
    axes[1].set_ylabel('Test ACC', fontsize=11)
    axes[1].set_xlabel('Graph Topology', fontsize=11)
    axes[1].set_title('ACC by Topology Type', fontsize=12)
    axes[1].axhline(0.615, color='green', linestyle='--', alpha=0.7, label='NXRO-Linear (Target)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase1_topology_comparison.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase1_topology_comparison.png")


def plot_sparsity_analysis(df, out_dir):
    """Plot performance vs sparsity (k) for each topology."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    valid_df = df[(df['K'].notna()) & (df['Topology'] != 'Unknown')].copy()
    
    # Plot RMSE vs k for each topology
    for topology in valid_df['Topology'].unique():
        topo_df = valid_df[valid_df['Topology'] == topology].sort_values('K')
        if len(topo_df) > 1:
            axes[0].plot(topo_df['K'], topo_df['Mean_RMSE_Test'], 
                        marker='o', label=topology, linewidth=2, markersize=6)
    
    axes[0].axhline(0.586, color='green', linestyle='--', alpha=0.7, label='Target (Linear)')
    axes[0].set_xlabel('Top-k Neighbors (Sparsity)', fontsize=11)
    axes[0].set_ylabel('Test RMSE (C)', fontsize=11)
    axes[0].set_title('RMSE vs Sparsity by Topology', fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 11])
    
    # Plot ACC vs k
    for topology in valid_df['Topology'].unique():
        topo_df = valid_df[valid_df['Topology'] == topology].sort_values('K')
        if len(topo_df) > 1:
            axes[1].plot(topo_df['K'], topo_df['Mean_ACC_Test'], 
                        marker='o', label=topology, linewidth=2, markersize=6)
    
    axes[1].axhline(0.615, color='green', linestyle='--', alpha=0.7, label='Target (Linear)')
    axes[1].set_xlabel('Top-k Neighbors (Sparsity)', fontsize=11)
    axes[1].set_ylabel('Test ACC', fontsize=11)
    axes[1].set_title('ACC vs Sparsity by Topology', fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 11])
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase1_sparsity_analysis.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase1_sparsity_analysis.png")


def plot_gcn_vs_gat(df, out_dir):
    """Compare GCN vs GAT performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    valid_df = df[df['GNN_Type'].isin(['GCN', 'GAT'])].copy()
    
    # Group by GNN type
    gnn_stats = valid_df.groupby('GNN_Type').agg({
        'Mean_RMSE_Test': ['mean', 'std', 'min'],
        'Mean_ACC_Test': ['mean', 'std', 'max'],
        'Model': 'count'
    }).reset_index()
    
    # RMSE comparison
    gnn_types = gnn_stats['GNN_Type'].values
    rmse_means = gnn_stats['Mean_RMSE_Test']['mean'].values
    rmse_stds = gnn_stats['Mean_RMSE_Test']['std'].values
    counts = gnn_stats['Model']['count'].values
    
    x = np.arange(len(gnn_types))
    bars = axes[0].bar(x, rmse_means, yerr=rmse_stds, alpha=0.7, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{t}\n(n={c})' for t, c in zip(gnn_types, counts)])
    axes[0].set_ylabel('Test RMSE (C)', fontsize=11)
    axes[0].set_title('GCN vs GAT: RMSE', fontsize=12)
    axes[0].axhline(0.567, color='green', linestyle='--', alpha=0.7, label='Target (Res)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, rmse_means)):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + rmse_stds[i], 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ACC comparison
    acc_means = gnn_stats['Mean_ACC_Test']['mean'].values
    acc_stds = gnn_stats['Mean_ACC_Test']['std'].values
    
    bars = axes[1].bar(x, acc_means, yerr=acc_stds, alpha=0.7, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{t}\n(n={c})' for t, c in zip(gnn_types, counts)])
    axes[1].set_ylabel('Test ACC', fontsize=11)
    axes[1].set_title('GCN vs GAT: ACC', fontsize=12)
    axes[1].axhline(0.628, color='green', linestyle='--', alpha=0.7, label='Target (Res)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, acc_means)):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + acc_stds[i], 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase1_gcn_vs_gat.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase1_gcn_vs_gat.png")


def plot_heatmap(df, out_dir):
    """Create heatmap of (topology, k) combinations."""
    valid_df = df[(df['K'].notna()) & (df['Topology'] != 'Unknown')].copy()
    
    if len(valid_df) == 0:
        print("  [!] No valid data for heatmap")
        return
    
    # Create pivot tables for RMSE and ACC
    pivot_rmse = valid_df.pivot_table(values='Mean_RMSE_Test', 
                                       index='Topology', columns='K', aggfunc='min')
    pivot_acc = valid_df.pivot_table(values='Mean_ACC_Test', 
                                      index='Topology', columns='K', aggfunc='max')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE heatmap (lower is better, use reversed colormap)
    sns.heatmap(pivot_rmse, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                ax=axes[0], cbar_kws={'label': 'Test RMSE (C)'}, 
                vmin=0.58, vmax=0.65, linewidths=0.5)
    axes[0].set_title('Test RMSE by (Topology, k)', fontsize=12)
    axes[0].set_xlabel('Top-k Neighbors', fontsize=11)
    axes[0].set_ylabel('Topology', fontsize=11)
    
    # ACC heatmap (higher is better)
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=axes[1], cbar_kws={'label': 'Test ACC'}, 
                vmin=0.55, vmax=0.65, linewidths=0.5)
    axes[1].set_title('Test ACC by (Topology, k)', fontsize=12)
    axes[1].set_xlabel('Top-k Neighbors', fontsize=11)
    axes[1].set_ylabel('Topology', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase1_topology_k_heatmap.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase1_topology_k_heatmap.png")


def plot_top_models_bar(df, out_dir, top_n=10):
    """Bar chart of top N graph models."""
    top_df = df.nsmallest(top_n, 'Mean_RMSE_Test')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(top_df))
    colors = ['tab:blue' if gnn == 'GCN' else 'tab:orange' for gnn in top_df['GNN_Type']]
    
    bars = ax.barh(x, top_df['Mean_RMSE_Test'], color=colors, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{rank}. {model}" for rank, model in 
                        zip(top_df['Rank'], top_df['Model'])], fontsize=8)
    ax.set_xlabel('Test RMSE (C)', fontsize=11)
    ax.set_title(f'Top {top_n} Graph Models (Lower RMSE is Better)', fontsize=12)
    ax.axvline(0.586, color='green', linestyle='--', linewidth=2, alpha=0.7, label='NXRO-Linear (Target)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tab:blue', alpha=0.7, label='GCN'),
                      Patch(facecolor='tab:orange', alpha=0.7, label='GAT'),
                      ax.get_legend().legendHandles[0]]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase1_top{top_n}_models.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase1_top{top_n}_models.png")


def plot_scatter_topology_k(df, out_dir):
    """Scatter plot showing all (topology, k, GNN) combinations."""
    valid_df = df[(df['K'].notna()) & (df['Topology'] != 'Unknown')].copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each topology with different markers
    topologies = valid_df['Topology'].unique()
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, topology in enumerate(topologies):
        topo_df = valid_df[valid_df['Topology'] == topology]
        
        # Separate GCN and GAT
        gcn_df = topo_df[topo_df['GNN_Type'] == 'GCN']
        gat_df = topo_df[topo_df['GNN_Type'] == 'GAT']
        
        marker = markers[i % len(markers)]
        
        if len(gcn_df) > 0:
            ax.scatter(gcn_df['Mean_ACC_Test'], gcn_df['Mean_RMSE_Test'], 
                      marker=marker, s=100, alpha=0.7, label=f'{topology} GCN',
                      edgecolors='black', linewidths=1)
        
        if len(gat_df) > 0:
            ax.scatter(gat_df['Mean_ACC_Test'], gat_df['Mean_RMSE_Test'], 
                      marker=marker, s=100, alpha=0.7, label=f'{topology} GAT',
                      facecolors='none', edgecolors='black', linewidths=2)
    
    # Mark target (NXRO-Res, rank 1 in ORAS5-only out-of-sample)
    ax.scatter([0.628], [0.567], marker='*', s=500, color='green', 
              edgecolors='darkgreen', linewidths=2, label='NXRO-Res (Target)', zorder=100)
    
    # Best graph model
    best_idx = df['Mean_RMSE_Test'].idxmin()
    if best_idx in valid_df.index:
        best_row = valid_df.loc[best_idx]
        ax.scatter([best_row['Mean_ACC_Test']], [best_row['Mean_RMSE_Test']], 
                  marker='*', s=400, color='red', edgecolors='darkred', 
                  linewidths=2, label='Best Graph', zorder=99)
    
    ax.set_xlabel('Test ACC (higher is better)', fontsize=11)
    ax.set_ylabel('Test RMSE (C) (lower is better)', fontsize=11)
    ax.set_title('Graph Models: ACC vs RMSE Trade-off', fontsize=12)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/phase1_acc_vs_rmse_scatter.png', dpi=300)
    plt.close()
    print(f"  [OK] Saved: {out_dir}/phase1_acc_vs_rmse_scatter.png")


def print_summary_table(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("PHASE 1 RESULTS SUMMARY")
    print("="*80)
    
    # Overall best
    best_rmse_idx = df['Mean_RMSE_Test'].idxmin()
    best_acc_idx = df['Mean_ACC_Test'].idxmax()
    
    print("\nBest by RMSE:")
    best_rmse = df.loc[best_rmse_idx]
    print(f"  Model: {best_rmse['Model']}")
    print(f"  Topology: {best_rmse['Topology']}, k={best_rmse['K']}, GNN={best_rmse['GNN_Type']}")
    print(f"  Test RMSE: {best_rmse['Mean_RMSE_Test']:.4f} C")
    print(f"  Test ACC: {best_rmse['Mean_ACC_Test']:.4f}")
    print(f"  Rank: {int(best_rmse['Rank'])}")
    
    gap_to_target = best_rmse['Mean_RMSE_Test'] - 0.567
    if gap_to_target <= 0:
        print(f"  [SUCCESS] Beats NXRO-Res by {-gap_to_target:.4f} C!")
    else:
        print(f"  [GAP] Still {gap_to_target:.4f} C away from target")
    
    print("\nBest by ACC:")
    best_acc = df.loc[best_acc_idx]
    print(f"  Model: {best_acc['Model']}")
    print(f"  Topology: {best_acc['Topology']}, k={best_acc['K']}, GNN={best_acc['GNN_Type']}")
    print(f"  Test ACC: {best_acc['Mean_ACC_Test']:.4f}")
    print(f"  Test RMSE: {best_acc['Mean_RMSE_Test']:.4f} C")
    print(f"  Rank: {int(best_acc['Rank'])}")
    
    # Statistics by topology
    print("\n" + "-"*80)
    print("Performance by Topology Type:")
    print("-"*80)
    topo_stats = df.groupby('Topology').agg({
        'Mean_RMSE_Test': ['mean', 'min', 'count'],
        'Mean_ACC_Test': ['mean', 'max']
    })
    print(topo_stats.to_string())
    
    # GCN vs GAT
    print("\n" + "-"*80)
    print("GCN vs GAT Comparison:")
    print("-"*80)
    gnn_stats = df.groupby('GNN_Type').agg({
        'Mean_RMSE_Test': ['mean', 'min'],
        'Mean_ACC_Test': ['mean', 'max'],
        'Model': 'count'
    })
    print(gnn_stats.to_string())
    
    # Top 10
    print("\n" + "-"*80)
    print("Top 10 Graph Models:")
    print("-"*80)
    top10 = df.nsmallest(10, 'Mean_RMSE_Test')[['Rank', 'Model', 'Topology', 'K', 'GNN_Type', 
                                                  'Mean_ACC_Test', 'Mean_RMSE_Test']]
    print(top10.to_string(index=False))
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase 1 Graph Topology Search Results')
    parser.add_argument('--results_dir', type=str, default='results_out_of_sample',
                       help='Results directory')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top models to display')
    args = parser.parse_args()
    
    # Find ranking CSV
    ranking_csvs = glob.glob(f'{args.results_dir}/rankings/all_variants_ranked_*_out_of_sample.csv')
    
    if not ranking_csvs:
        print(f"[X] No ranking CSV found in {args.results_dir}/rankings/")
        print("Run ranking script first: python rank_all_variants_outsample.py")
        return
    
    # Use most recent
    csv_path = max(ranking_csvs, key=os.path.getmtime)
    print(f"Loading rankings from: {csv_path}")
    
    # Load and filter graph models
    graph_df = load_rankings(csv_path)
    
    print(f"\nFound {len(graph_df)} graph models in rankings")
    print(f"Topologies: {graph_df['Topology'].unique()}")
    print(f"K values: {sorted(graph_df['K'].dropna().unique())}")
    print(f"GNN types: {graph_df['GNN_Type'].unique()}")
    
    # Create output directory
    out_dir = f'{args.results_dir}/rankings/phase1_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_topology_comparison(graph_df, out_dir)
    plot_sparsity_analysis(graph_df, out_dir)
    plot_gcn_vs_gat(graph_df, out_dir)
    plot_heatmap(graph_df, out_dir)
    plot_top_models_bar(graph_df, out_dir, top_n=args.top_n)
    
    # Print summary
    print_summary_table(graph_df)
    
    # Save filtered dataframe
    csv_out = f'{out_dir}/phase1_graph_models_only.csv'
    graph_df.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved filtered results: {csv_out}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput files in: {out_dir}/")
    print("  - phase1_topology_comparison.png")
    print("  - phase1_sparsity_analysis.png")
    print("  - phase1_gcn_vs_gat.png")
    print("  - phase1_topology_k_heatmap.png")
    print(f"  - phase1_top{args.top_n}_models.png")
    print("  - phase1_graph_models_only.csv")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

