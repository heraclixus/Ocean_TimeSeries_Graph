#!/usr/bin/env python3
"""
Model Explainability Script for NXRO Graph Models.

This script extracts and visualizes the learned interactions between climate modes
from NXRO-GCN, NXRO-GAT, and NXRO-Attention models.

Visualizations include:
1. Seasonal linear coupling matrices L(t) for different months
2. Graph structure (fixed or learned adjacency)
3. Attention patterns (for GAT and Attention models)
4. Comparison of how each model weighs different climate indices
5. Learned graph discovery (for learnable graph models)

Usage:
    python explain_models.py [--output_dir plots/explainability]
    python explain_models.py --learnable_graph_ckpt path/to/checkpoint.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import xarray as xr

# Import model classes
from nxro.models import NXROAttentiveModel, NXROGraphPyGModel, fourier_time_embedding
from graph_construction import get_or_build_stat_knn_graph


def load_checkpoint(ckpt_path: str) -> Tuple[dict, list]:
    """Load checkpoint and return state_dict and var_order."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    var_order = ckpt.get('var_order', None)
    return state_dict, var_order


def load_attentive_model(ckpt_path: str) -> Tuple[NXROAttentiveModel, list]:
    """Load NXRO-Attentive model from checkpoint."""
    state_dict, var_order = load_checkpoint(ckpt_path)
    
    # Infer model parameters from state_dict
    n_vars = state_dict['L_basis'].shape[1]
    k_max = (state_dict['L_basis'].shape[0] - 1) // 2
    
    # Infer d from attention weights (handle both naming conventions)
    if 'Wq.weight' in state_dict:
        d = state_dict['Wq.weight'].shape[0]
    elif 'W_Q.weight' in state_dict:
        d = state_dict['W_Q.weight'].shape[0]
    else:
        d = 16  # default
    
    model = NXROAttentiveModel(n_vars=n_vars, k_max=k_max, d=d, dropout=0.0)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, var_order


def load_gcn_model(ckpt_path: str, top_k: int = 2) -> Tuple[NXROGraphPyGModel, list, torch.Tensor]:
    """Load NXRO-GCN model from checkpoint."""
    state_dict, var_order = load_checkpoint(ckpt_path)
    
    # Infer model parameters
    n_vars = state_dict['L_basis'].shape[1]
    k_max = (state_dict['L_basis'].shape[0] - 1) // 2
    hidden = state_dict['conv1.lin.weight'].shape[0]
    
    # Rebuild edge_index from statistical KNN graph
    A, _ = get_or_build_stat_knn_graph(
        data_path='data/XRO_indices_oras5_train.csv',
        train_start='1979-01', train_end='2001-12',
        var_order=var_order, method='pearson', top_k=top_k
    )
    
    # Convert adjacency to edge_index
    V = A.shape[0]
    A2 = A.clone()
    A2.fill_diagonal_(0.0)
    edges = []
    for i in range(V):
        vals, idx = torch.topk(A2[i], k=min(top_k, V - 1))
        for j in idx.tolist():
            if i != j and A2[i, j] > 0:
                edges.append([i, j])
                edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    
    model = NXROGraphPyGModel(n_vars=n_vars, k_max=k_max, edge_index=edge_index, 
                               hidden=hidden, dropout=0.0, use_gat=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, var_order, A


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def get_seasonal_L_matrix(L_basis: torch.Tensor, month: int, k_max: int) -> np.ndarray:
    """Compute L(t) for a specific month using Fourier basis."""
    # month: 1-12
    t_years = torch.tensor([(month - 1) / 12.0], dtype=torch.float32)  # fraction of year
    emb = fourier_time_embedding(t_years, k_max).float()  # [1, n_basis]
    L_t = torch.einsum('bk,kuv->buv', emb, L_basis.float())  # [1, n_vars, n_vars]
    return L_t[0].detach().numpy()


def plot_seasonal_L_matrices(L_basis: torch.Tensor, var_order: list, k_max: int,
                             output_path: str, title_prefix: str = ''):
    """Plot L matrices for selected months (Jan, Apr, Jul, Oct)."""
    months = [1, 4, 7, 10]
    month_names = ['January', 'April', 'July', 'October']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Compute global vmin/vmax for consistent colorbar
    all_L = [get_seasonal_L_matrix(L_basis, m, k_max) for m in months]
    vmax = max(np.abs(L).max() for L in all_L)
    vmin = -vmax
    
    cmap = sns.diverging_palette(250, 15, s=75, l=40, as_cmap=True)
    
    for idx, (month, name) in enumerate(zip(months, month_names)):
        L = get_seasonal_L_matrix(L_basis, month, k_max)
        
        im = axes[idx].imshow(L, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[idx].set_xticks(range(len(var_order)))
        axes[idx].set_yticks(range(len(var_order)))
        axes[idx].set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
        axes[idx].set_yticklabels(var_order, fontsize=9)
        axes[idx].set_title(f'{name}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Source Variable', fontsize=11)
        axes[idx].set_ylabel('Target Variable (dX/dt)', fontsize=11)
        
        # Add grid
        for edge, spine in axes[idx].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    fig.suptitle(f'{title_prefix}Seasonal Linear Coupling Matrix L(t)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Coupling Strength', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_L_diagonal_seasonality(L_basis: torch.Tensor, var_order: list, k_max: int,
                                output_path: str, title_prefix: str = ''):
    """Plot diagonal elements of L across all months (damping rates)."""
    months = np.arange(1, 13)
    n_vars = len(var_order)
    
    # Compute diagonal for each month
    diag_values = np.zeros((12, n_vars))
    for m in months:
        L = get_seasonal_L_matrix(L_basis, m, k_max)
        diag_values[m-1] = np.diag(L)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use distinct colors for each variable
    colors = plt.cm.tab20(np.linspace(0, 1, n_vars))
    
    for i, var in enumerate(var_order):
        ax.plot(months, diag_values[:, i], 'o-', label=var, color=colors[i], 
                linewidth=2, markersize=6)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('L_ii (Damping Rate)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title_prefix}Seasonal Damping Rates (Diagonal of L)', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_graph_structure(adjacency: torch.Tensor, var_order: list, output_path: str,
                        title: str = 'GCN Graph Structure'):
    """Visualize the graph adjacency matrix and network."""
    A = adjacency.numpy()
    n_vars = len(var_order)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Adjacency heatmap
    cmap = sns.light_palette("steelblue", as_cmap=True)
    im = axes[0].imshow(A, cmap=cmap, aspect='auto')
    axes[0].set_xticks(range(n_vars))
    axes[0].set_yticks(range(n_vars))
    axes[0].set_xticklabels(var_order, rotation=45, ha='right', fontsize=10)
    axes[0].set_yticklabels(var_order, fontsize=10)
    axes[0].set_title('Adjacency Matrix (Edge Weights)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Target', fontsize=11)
    axes[0].set_ylabel('Source', fontsize=11)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label='Connection Strength')
    
    # 2. Network graph visualization
    ax = axes[1]
    
    # Use circular layout
    angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
    radius = 1.0
    pos = {i: (radius * np.cos(a - np.pi/2), radius * np.sin(a - np.pi/2)) 
           for i, a in enumerate(angles)}
    
    # Draw edges with width proportional to weight
    edge_weights = []
    for i in range(n_vars):
        for j in range(n_vars):
            if A[i, j] > 0 and i != j:
                edge_weights.append((i, j, A[i, j]))
    
    if edge_weights:
        max_weight = max(w for _, _, w in edge_weights)
        for i, j, w in edge_weights:
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            alpha = 0.3 + 0.7 * (w / max_weight)
            linewidth = 1 + 3 * (w / max_weight)
            ax.plot([x0, x1], [y0, y1], 'b-', alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Draw nodes
    node_colors = plt.cm.Set3(np.linspace(0, 1, n_vars))
    for i, var in enumerate(var_order):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.12, color=node_colors[i], ec='black', 
                            linewidth=2, zorder=2)
        ax.add_patch(circle)
        # Label outside the circle
        label_x = x * 1.25
        label_y = y * 1.25
        ha = 'center'
        if x > 0.1:
            ha = 'left'
        elif x < -0.1:
            ha = 'right'
        ax.text(label_x, label_y, var, fontsize=10, ha=ha, va='center', fontweight='bold')
    
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Graph Network (K=2 Neighbors)', fontsize=13, fontweight='bold')
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def compute_attention_pattern(model: NXROAttentiveModel, x: torch.Tensor, 
                              month: int) -> np.ndarray:
    """Compute attention weights for a given input and month."""
    with torch.no_grad():
        # Time embedding
        t_years = torch.tensor([(month - 1) / 12.0], dtype=torch.float32)
        
        # Compute Q, K
        x_input = x.float().unsqueeze(-1)  # [n_vars, 1]
        Q = model.Wq(x_input)  # [n_vars, d]
        K = model.Wk(x_input)  # [n_vars, d]
        
        # Compute raw attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(model.d)  # [n_vars, n_vars]
        
        # Apply mask
        mask = model.attn_mask.float()
        masked_scores = scores.squeeze() + (1 - mask) * (-1e9)
        
        # Softmax
        attn_weights = torch.softmax(masked_scores, dim=-1)
        
    return attn_weights.numpy()


def plot_attention_patterns(model: NXROAttentiveModel, var_order: list, 
                           output_path: str, sample_states: Optional[Dict] = None):
    """Visualize attention patterns for different months and states."""
    n_vars = len(var_order)
    
    # Create sample inputs: neutral, El Niño-like, La Niña-like
    if sample_states is None:
        neutral = torch.zeros(n_vars)
        
        # El Niño: warm Nino34, positive T/H
        elnino = torch.zeros(n_vars)
        for i, v in enumerate(var_order):
            if 'Nino' in v or v in ['T', 'H']:
                elnino[i] = 1.5
            elif v == 'WWV':
                elnino[i] = 0.5
        
        # La Niña: cold Nino34
        lanina = torch.zeros(n_vars)
        for i, v in enumerate(var_order):
            if 'Nino' in v or v in ['T', 'H']:
                lanina[i] = -1.5
            elif v == 'WWV':
                lanina[i] = -0.5
        
        sample_states = {
            'Neutral': neutral,
            'El Niño': elnino,
            'La Niña': lanina
        }
    
    months = [1, 4, 7, 10]
    month_names = ['Jan', 'Apr', 'Jul', 'Oct']
    
    fig, axes = plt.subplots(len(sample_states), len(months), 
                             figsize=(14, 3*len(sample_states)))
    
    cmap = sns.light_palette("coral", as_cmap=True)
    
    for row, (state_name, x) in enumerate(sample_states.items()):
        for col, (month, mname) in enumerate(zip(months, month_names)):
            ax = axes[row, col] if len(sample_states) > 1 else axes[col]
            
            attn = compute_attention_pattern(model, x, month)
            
            im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(n_vars))
            ax.set_yticks(range(n_vars))
            
            if row == len(sample_states) - 1:
                ax.set_xticklabels(var_order, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticklabels([])
            
            if col == 0:
                ax.set_yticklabels(var_order, fontsize=8)
                ax.set_ylabel(state_name, fontsize=11, fontweight='bold')
            else:
                ax.set_yticklabels([])
            
            if row == 0:
                ax.set_title(mname, fontsize=12, fontweight='bold')
    
    fig.suptitle('Attention Weights: Which Variables Each Variable Attends To\n(Row=Query, Col=Key)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_attention_mask_and_gate(model: NXROAttentiveModel, var_order: list, 
                                 output_path: str):
    """Visualize the attention mask and seasonal gate α(t)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Attention mask
    mask = model.attn_mask.numpy()
    im = axes[0].imshow(mask, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[0].set_xticks(range(len(var_order)))
    axes[0].set_yticks(range(len(var_order)))
    axes[0].set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(var_order, fontsize=9)
    axes[0].set_title('Attention Mask\n(1=Allowed, 0=Blocked)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key (Source)', fontsize=10)
    axes[0].set_ylabel('Query (Target)', fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    
    # 2. Seasonal gate α(t)
    months = np.arange(1, 13)
    alpha_values = []
    
    with torch.no_grad():
        for m in months:
            t_years = torch.tensor([(m - 1) / 12.0], dtype=torch.float32)
            k_max = model.k_max
            emb = fourier_time_embedding(t_years, k_max).float()
            alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, model.alpha_w.float()))
            alpha_values.append(alpha.item())
    
    axes[1].bar(months, alpha_values, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xticks(months)
    axes[1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    axes[1].set_xlabel('Month', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('α(t) Gate Value', fontsize=11, fontweight='bold')
    axes[1].set_title('Seasonal Attention Gate α(t)\n(How much attention contributes)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_gcn_gate(model: NXROGraphPyGModel, output_path: str):
    """Visualize the seasonal gate α(t) for GCN model."""
    months = np.arange(1, 13)
    alpha_values = []
    
    with torch.no_grad():
        for m in months:
            t_years = torch.tensor([(m - 1) / 12.0], dtype=torch.float32)
            k_max = model.k_max
            emb = fourier_time_embedding(t_years, k_max).float()
            alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, model.alpha_w.float()))
            alpha_values.append(alpha.item())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(months, alpha_values, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('α(t) Gate Value', fontsize=12, fontweight='bold')
    ax.set_title('Seasonal GNN Gate α(t)\n(How much GNN output contributes to dynamics)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_variable_importance_comparison(attn_model: NXROAttentiveModel,
                                       gcn_adjacency: torch.Tensor,
                                       var_order: list, output_path: str):
    """Compare variable importance between Attention and GCN models."""
    n_vars = len(var_order)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. GCN: Sum of incoming edge weights (how much information each var receives)
    A = gcn_adjacency.numpy()
    incoming = A.sum(axis=0)  # sum over sources
    outgoing = A.sum(axis=1)  # sum over targets
    
    x = np.arange(n_vars)
    width = 0.35
    
    axes[0].bar(x - width/2, incoming, width, label='Incoming', color='steelblue', alpha=0.7)
    axes[0].bar(x + width/2, outgoing, width, label='Outgoing', color='coral', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(var_order, rotation=45, ha='right', fontsize=10)
    axes[0].set_xlabel('Variable', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Sum of Edge Weights', fontsize=11, fontweight='bold')
    axes[0].set_title('GCN: Variable Connectivity', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Attention: Average attention received (across months and neutral state)
    attn_received = np.zeros(n_vars)
    attn_given = np.zeros(n_vars)
    
    x_neutral = torch.zeros(n_vars)
    for month in range(1, 13):
        attn = compute_attention_pattern(attn_model, x_neutral, month)
        attn_received += attn.sum(axis=0)  # sum over queries (how much attention each var receives)
        attn_given += attn.sum(axis=1)  # sum over keys (how much attention each var gives)
    
    attn_received /= 12
    attn_given /= 12
    
    axes[1].bar(x - width/2, attn_received, width, label='Received', color='steelblue', alpha=0.7)
    axes[1].bar(x + width/2, attn_given, width, label='Given', color='coral', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(var_order, rotation=45, ha='right', fontsize=10)
    axes[1].set_xlabel('Variable', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Average Attention (across months)', fontsize=11, fontweight='bold')
    axes[1].set_title('Attention: Variable Importance', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Comparison: How Variables Interact in Each Model', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_nino34_drivers(attn_model: NXROAttentiveModel, gcn_adjacency: torch.Tensor,
                        L_basis_attn: torch.Tensor, L_basis_gcn: torch.Tensor,
                        var_order: list, output_path: str):
    """
    Focus on Nino34: What drives it according to each model?
    This is key for explainability.
    """
    n_vars = len(var_order)
    nino34_idx = var_order.index('Nino34') if 'Nino34' in var_order else 0
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Month colors
    month_colors = plt.cm.twilight(np.linspace(0, 1, 12))
    
    # 1. GCN - Graph connections TO Nino34
    ax = axes[0, 0]
    A = gcn_adjacency.numpy()
    incoming_to_nino = A[:, nino34_idx]  # edges pointing to Nino34
    
    bars = ax.bar(range(n_vars), incoming_to_nino, color='steelblue', alpha=0.7, edgecolor='black')
    bars[nino34_idx].set_color('red')
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Edge Weight', fontsize=11, fontweight='bold')
    ax.set_title('GCN: What Variables Influence Nino34?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Attention - What Nino34 attends to (averaged)
    ax = axes[0, 1]
    attn_from_nino = np.zeros(n_vars)
    x_neutral = torch.zeros(n_vars)
    for month in range(1, 13):
        attn = compute_attention_pattern(attn_model, x_neutral, month)
        attn_from_nino += attn[nino34_idx, :]  # row = Nino34 as query
    attn_from_nino /= 12
    
    bars = ax.bar(range(n_vars), attn_from_nino, color='coral', alpha=0.7, edgecolor='black')
    bars[nino34_idx].set_color('red')
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
    ax.set_title('Attention: What Nino34 Attends To?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Linear L - Coupling TO Nino34 by month (Attention model)
    ax = axes[1, 0]
    k_max_attn = (L_basis_attn.shape[0] - 1) // 2
    
    for m in range(1, 13):
        L = get_seasonal_L_matrix(L_basis_attn, m, k_max_attn)
        coupling_to_nino = L[nino34_idx, :]  # row = how others affect Nino34
        ax.plot(range(n_vars), coupling_to_nino, 'o-', color=month_colors[m-1], 
                alpha=0.7, linewidth=1.5, markersize=4)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('L[Nino34, j] Coupling', fontsize=11, fontweight='bold')
    ax.set_title('Attention Model: Linear Coupling to Nino34\n(colored by month)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add month legend
    sm = plt.cm.ScalarMappable(cmap='twilight', norm=plt.Normalize(1, 12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label='Month')
    cbar.set_ticks([1, 4, 7, 10])
    cbar.set_ticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
    
    # 4. Linear L - Coupling TO Nino34 by month (GCN model)
    ax = axes[1, 1]
    k_max_gcn = (L_basis_gcn.shape[0] - 1) // 2
    
    for m in range(1, 13):
        L = get_seasonal_L_matrix(L_basis_gcn, m, k_max_gcn)
        coupling_to_nino = L[nino34_idx, :]
        ax.plot(range(n_vars), coupling_to_nino, 'o-', color=month_colors[m-1], 
                alpha=0.7, linewidth=1.5, markersize=4)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('L[Nino34, j] Coupling', fontsize=11, fontweight='bold')
    ax.set_title('GCN Model: Linear Coupling to Nino34\n(colored by month)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap='twilight', norm=plt.Normalize(1, 12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label='Month')
    cbar.set_ticks([1, 4, 7, 10])
    cbar.set_ticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
    
    fig.suptitle('What Drives Nino34? Model Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_gcn_layer_weights(model: NXROGraphPyGModel, output_path: str):
    """Visualize GCN layer weights."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Conv1 weights: [hidden, 1]
    conv1_w = model.conv1.lin.weight.detach().numpy()
    ax = axes[0]
    ax.imshow(conv1_w, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Input (1D)', fontsize=11)
    ax.set_ylabel('Hidden Units', fontsize=11)
    ax.set_title(f'GCN Layer 1 Weights\n({conv1_w.shape[0]} hidden units)', 
                 fontsize=12, fontweight='bold')
    
    # Conv2 weights: [1, hidden]
    conv2_w = model.conv2.lin.weight.detach().numpy()
    ax = axes[1]
    ax.imshow(conv2_w, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Hidden Units', fontsize=11)
    ax.set_ylabel('Output (1D)', fontsize=11)
    ax.set_title(f'GCN Layer 2 Weights\n(Output projection)', 
                 fontsize=12, fontweight='bold')
    
    fig.suptitle('GCN Internal Weights', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_attention_projection_weights(model: NXROAttentiveModel, output_path: str):
    """Visualize attention projection matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Wq, Wk, Wv, Wo
    weights = [
        ('Wq (Query)', model.Wq.weight.detach().numpy()),
        ('Wk (Key)', model.Wk.weight.detach().numpy()),
        ('Wv (Value)', model.Wv.weight.detach().numpy()),
        ('Wo (Output)', model.Wo.weight.detach().numpy().T),
    ]
    
    for ax, (name, w) in zip(axes.flatten(), weights):
        im = ax.imshow(w, cmap='coolwarm', aspect='auto')
        ax.set_title(f'{name}\nshape: {w.shape}', fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle('Attention Projection Weights', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# LEARNABLE GRAPH VISUALIZATION
# ============================================================================

def load_learnable_graph_model(ckpt_path: str) -> Tuple[any, list, np.ndarray]:
    """Load a model with learnable graph and extract the learned adjacency."""
    from nxro.models import NXROGraphModel
    
    state_dict, var_order = load_checkpoint(ckpt_path)
    
    # Check if this is a learnable graph model (has A_param)
    if 'A_param' in state_dict:
        n_vars = state_dict['A_param'].shape[0]
        k_max = (state_dict['L_basis'].shape[0] - 1) // 2
        
        model = NXROGraphModel(n_vars=n_vars, k_max=k_max, use_fixed_graph=False)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Extract learned adjacency (apply ReLU as in forward pass)
        A_learned = torch.relu(model.A_param).detach().numpy()
        
        return model, var_order, A_learned
    else:
        raise ValueError("Checkpoint does not contain learnable graph (A_param)")


def plot_learned_vs_initial_graph(A_learned: np.ndarray, A_initial: np.ndarray, 
                                   var_order: list, output_path: str):
    """Compare learned adjacency matrix to initial (correlation-based) graph."""
    n_vars = len(var_order)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Normalize for comparison
    A_learned_norm = A_learned / (A_learned.max() + 1e-8)
    A_initial_norm = A_initial / (A_initial.max() + 1e-8)
    
    cmap = sns.light_palette("steelblue", as_cmap=True)
    
    # 1. Initial graph
    im1 = axes[0].imshow(A_initial_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[0].set_xticks(range(n_vars))
    axes[0].set_yticks(range(n_vars))
    axes[0].set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(var_order, fontsize=9)
    axes[0].set_title('Initial Graph (Correlation-based)', fontsize=12, fontweight='bold')
    fig.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 2. Learned graph
    im2 = axes[1].imshow(A_learned_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(range(n_vars))
    axes[1].set_yticks(range(n_vars))
    axes[1].set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(var_order, fontsize=9)
    axes[1].set_title('Learned Graph (After Training)', fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 3. Difference (what changed)
    diff = A_learned_norm - A_initial_norm
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    cmap_diff = sns.diverging_palette(250, 15, s=75, l=40, as_cmap=True)
    im3 = axes[2].imshow(diff, cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
    axes[2].set_xticks(range(n_vars))
    axes[2].set_yticks(range(n_vars))
    axes[2].set_xticklabels(var_order, rotation=45, ha='right', fontsize=9)
    axes[2].set_yticklabels(var_order, fontsize=9)
    axes[2].set_title('Change (Learned - Initial)', fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar.set_label('Strengthened ↑ / Weakened ↓', fontsize=10)
    
    fig.suptitle('Learned Graph Structure Discovery', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_discovered_connections(A_learned: np.ndarray, var_order: list, 
                                 output_path: str, threshold_percentile: float = 75):
    """
    Visualize the strongest learned connections as a network diagram.
    This highlights discovered interdependencies.
    """
    n_vars = len(var_order)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Threshold to show only strong connections
    A_np = A_learned.copy()
    np.fill_diagonal(A_np, 0)  # Remove self-loops for visualization
    threshold = np.percentile(A_np[A_np > 0], threshold_percentile) if A_np.max() > 0 else 0
    
    # 1. Heatmap of learned adjacency
    ax = axes[0]
    cmap = sns.light_palette("darkblue", as_cmap=True)
    im = ax.imshow(A_np, cmap=cmap, aspect='auto')
    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(var_order, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(var_order, fontsize=10)
    ax.set_title('Learned Adjacency Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Target Variable', fontsize=11)
    ax.set_ylabel('Source Variable', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Connection Strength')
    
    # 2. Network visualization of strong connections
    ax = axes[1]
    
    # Circular layout
    angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
    radius = 1.0
    pos = {i: (radius * np.cos(a - np.pi/2), radius * np.sin(a - np.pi/2)) 
           for i, a in enumerate(angles)}
    
    # Draw edges for strong connections
    edge_list = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and A_np[i, j] > threshold:
                edge_list.append((i, j, A_np[i, j]))
    
    if edge_list:
        max_weight = max(w for _, _, w in edge_list)
        for i, j, w in edge_list:
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            alpha = 0.3 + 0.7 * (w / max_weight)
            linewidth = 1 + 4 * (w / max_weight)
            # Draw arrow
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='->', color='steelblue', 
                                      alpha=alpha, lw=linewidth))
    
    # Draw nodes
    node_colors = plt.cm.Set3(np.linspace(0, 1, n_vars))
    for i, var in enumerate(var_order):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.12, color=node_colors[i], ec='black', 
                            linewidth=2, zorder=10)
        ax.add_patch(circle)
        # Label
        label_x = x * 1.3
        label_y = y * 1.3
        ha = 'center'
        if x > 0.1:
            ha = 'left'
        elif x < -0.1:
            ha = 'right'
        ax.text(label_x, label_y, var, fontsize=11, ha=ha, va='center', fontweight='bold')
    
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Discovered Connections\n(Top {100-threshold_percentile:.0f}% strongest)', 
                 fontsize=13, fontweight='bold')
    
    fig.suptitle('Learned Graph: Discovered Climate Mode Interactions', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def analyze_discovered_patterns(A_learned: np.ndarray, var_order: list, output_path: str):
    """
    Analyze and summarize discovered patterns from learned graph.
    Creates a summary plot and table of key findings.
    """
    n_vars = len(var_order)
    A_np = A_learned.copy()
    np.fill_diagonal(A_np, 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Top connections bar plot
    ax = axes[0, 0]
    connections = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and A_np[i, j] > 0:
                connections.append((var_order[i], var_order[j], A_np[i, j]))
    
    connections.sort(key=lambda x: -x[2])
    top_n = min(15, len(connections))
    
    if connections:
        labels = [f'{c[0]}→{c[1]}' for c in connections[:top_n]]
        values = [c[2] for c in connections[:top_n]]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
        
        y_pos = np.arange(top_n)
        ax.barh(y_pos, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Connection Strength', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Discovered Connections', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Incoming connections (what influences each variable)
    ax = axes[0, 1]
    incoming = A_np.sum(axis=0)  # Sum over sources
    sorted_idx = np.argsort(incoming)[::-1]
    
    colors = plt.cm.Greens(np.linspace(0.3, 0.8, n_vars))
    ax.barh(range(n_vars), incoming[sorted_idx], color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels([var_order[i] for i in sorted_idx], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Total Incoming Connection Strength', fontsize=11, fontweight='bold')
    ax.set_title('Variables Most Influenced by Others', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Outgoing connections (what each variable influences)
    ax = axes[1, 0]
    outgoing = A_np.sum(axis=1)  # Sum over targets
    sorted_idx = np.argsort(outgoing)[::-1]
    
    colors = plt.cm.Oranges(np.linspace(0.3, 0.8, n_vars))
    ax.barh(range(n_vars), outgoing[sorted_idx], color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels([var_order[i] for i in sorted_idx], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Total Outgoing Connection Strength', fontsize=11, fontweight='bold')
    ax.set_title('Variables That Influence Others Most', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Asymmetry analysis (A[i,j] vs A[j,i])
    ax = axes[1, 1]
    asymmetry = []
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if A_np[i, j] > 0 or A_np[j, i] > 0:
                diff = A_np[i, j] - A_np[j, i]
                total = A_np[i, j] + A_np[j, i]
                if total > 0:
                    asymmetry.append((var_order[i], var_order[j], diff, total))
    
    if asymmetry:
        asymmetry.sort(key=lambda x: -abs(x[2]))
        top_asym = asymmetry[:min(10, len(asymmetry))]
        
        labels = []
        values = []
        for a in top_asym:
            if a[2] > 0:
                labels.append(f'{a[0]}→{a[1]} > {a[1]}→{a[0]}')
            else:
                labels.append(f'{a[1]}→{a[0]} > {a[0]}→{a[1]}')
            values.append(a[2])
        
        colors = ['coral' if v > 0 else 'steelblue' for v in values]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, np.abs(values), color=colors, edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Asymmetry Magnitude', fontsize=11, fontweight='bold')
        ax.set_title('Asymmetric Relationships\n(Directional dependencies)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Analysis of Discovered Climate Mode Interactions', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    
    # Print key findings to console
    print("\n" + "="*60)
    print("KEY DISCOVERIES FROM LEARNED GRAPH")
    print("="*60)
    
    if connections:
        print("\nTop 5 strongest connections:")
        for i, (src, tgt, w) in enumerate(connections[:5], 1):
            print(f"  {i}. {src} → {tgt}: {w:.4f}")
    
    print("\nMost influential variables (outgoing):")
    for i in np.argsort(outgoing)[::-1][:3]:
        print(f"  {var_order[i]}: {outgoing[i]:.4f}")
    
    print("\nMost influenced variables (incoming):")
    for i in np.argsort(incoming)[::-1][:3]:
        print(f"  {var_order[i]}: {incoming[i]:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Explain NXRO-GCN and NXRO-Attention models')
    parser.add_argument('--attention_ckpt', type=str, 
                       default='results_out_of_sample/attentive/nxro_attentive_best.pt',
                       help='Path to NXRO-Attention checkpoint')
    parser.add_argument('--gcn_ckpt', type=str,
                       default='results_out_of_sample/graphpyg/gcn_k2/nxro_graphpyg_gcn_k2_best.pt',
                       help='Path to NXRO-GCN checkpoint')
    parser.add_argument('--learnable_graph_ckpt', type=str, default=None,
                       help='Path to learnable graph model checkpoint (for discovery analysis)')
    parser.add_argument('--output_dir', type=str, default='plots/explainability',
                       help='Output directory for plots')
    parser.add_argument('--top_k', type=int, default=2,
                       help='K neighbors for GCN graph')
    parser.add_argument('--discovery_only', action='store_true',
                       help='Only run learnable graph discovery analysis')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("NXRO Model Explainability")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    
    print(f"  Loading Attention model from: {args.attention_ckpt}")
    attn_model, var_order_attn = load_attentive_model(args.attention_ckpt)
    print(f"    Variables: {var_order_attn}")
    
    print(f"  Loading GCN model from: {args.gcn_ckpt}")
    gcn_model, var_order_gcn, gcn_adjacency = load_gcn_model(args.gcn_ckpt, top_k=args.top_k)
    print(f"    Variables: {var_order_gcn}")
    
    # Use the same var_order (they should match)
    var_order = var_order_attn if var_order_attn else var_order_gcn
    
    print("\n" + "="*60)
    print("Generating Explainability Plots...")
    print("="*60)
    
    # 1. Seasonal Linear Coupling Matrices
    print("\n[1/8] Seasonal Linear Coupling Matrices L(t)...")
    plot_seasonal_L_matrices(
        attn_model.L_basis, var_order, attn_model.k_max,
        f'{args.output_dir}/attention_L_matrices.png',
        'NXRO-Attention: '
    )
    plot_seasonal_L_matrices(
        gcn_model.L_basis, var_order, gcn_model.k_max,
        f'{args.output_dir}/gcn_L_matrices.png',
        'NXRO-GCN: '
    )
    
    # 2. Diagonal Elements (Damping Rates)
    print("\n[2/8] Seasonal Damping Rates...")
    plot_L_diagonal_seasonality(
        attn_model.L_basis, var_order, attn_model.k_max,
        f'{args.output_dir}/attention_damping_rates.png',
        'NXRO-Attention: '
    )
    plot_L_diagonal_seasonality(
        gcn_model.L_basis, var_order, gcn_model.k_max,
        f'{args.output_dir}/gcn_damping_rates.png',
        'NXRO-GCN: '
    )
    
    # 3. Graph Structure (GCN)
    print("\n[3/8] GCN Graph Structure...")
    plot_graph_structure(
        gcn_adjacency, var_order,
        f'{args.output_dir}/gcn_graph_structure.png',
        'NXRO-GCN: Graph Structure (K=2 Pearson)'
    )
    
    # 4. Attention Patterns
    print("\n[4/8] Attention Patterns...")
    plot_attention_patterns(
        attn_model, var_order,
        f'{args.output_dir}/attention_patterns.png'
    )
    
    # 5. Attention Mask and Gate
    print("\n[5/8] Attention Mask and Seasonal Gate...")
    plot_attention_mask_and_gate(
        attn_model, var_order,
        f'{args.output_dir}/attention_mask_gate.png'
    )
    
    # 6. GCN Seasonal Gate
    print("\n[6/8] GCN Seasonal Gate...")
    plot_gcn_gate(
        gcn_model,
        f'{args.output_dir}/gcn_seasonal_gate.png'
    )
    
    # 7. Variable Importance Comparison
    print("\n[7/8] Variable Importance Comparison...")
    plot_variable_importance_comparison(
        attn_model, gcn_adjacency, var_order,
        f'{args.output_dir}/variable_importance_comparison.png'
    )
    
    # 8. Nino34 Drivers - Key for Explainability
    print("\n[8/8] What Drives Nino34? (Key Explainability Plot)...")
    plot_nino34_drivers(
        attn_model, gcn_adjacency,
        attn_model.L_basis, gcn_model.L_basis,
        var_order,
        f'{args.output_dir}/nino34_drivers.png'
    )
    
    # Bonus: Internal weights
    print("\n[Bonus] Internal Layer Weights...")
    plot_gcn_layer_weights(gcn_model, f'{args.output_dir}/gcn_layer_weights.png')
    plot_attention_projection_weights(attn_model, f'{args.output_dir}/attention_projection_weights.png')
    
    print("\n" + "="*60)
    print(f"All explainability plots saved to: {args.output_dir}/")
    print("="*60)
    
    # ========================================================================
    # LEARNABLE GRAPH DISCOVERY (if checkpoint provided)
    # ========================================================================
    if args.learnable_graph_ckpt:
        print("\n" + "="*60)
        print("LEARNABLE GRAPH DISCOVERY ANALYSIS")
        print("="*60)
        
        try:
            print(f"\nLoading learnable graph model from: {args.learnable_graph_ckpt}")
            learned_model, learned_var_order, A_learned = load_learnable_graph_model(args.learnable_graph_ckpt)
            print(f"  Variables: {learned_var_order}")
            
            # Get initial graph for comparison
            A_initial, _ = get_or_build_stat_knn_graph(
                data_path='data/XRO_indices_oras5_train.csv',
                train_start='1979-01', train_end='2001-12',
                var_order=learned_var_order, method='pearson', top_k=args.top_k
            )
            A_initial_np = A_initial.numpy()
            
            # Generate discovery plots
            print("\n[Discovery 1/3] Learned vs Initial Graph Comparison...")
            plot_learned_vs_initial_graph(
                A_learned, A_initial_np, learned_var_order,
                f'{args.output_dir}/discovery_learned_vs_initial.png'
            )
            
            print("\n[Discovery 2/3] Discovered Connections Network...")
            plot_discovered_connections(
                A_learned, learned_var_order,
                f'{args.output_dir}/discovery_connections_network.png'
            )
            
            print("\n[Discovery 3/3] Pattern Analysis...")
            analyze_discovered_patterns(
                A_learned, learned_var_order,
                f'{args.output_dir}/discovery_pattern_analysis.png'
            )
            
        except Exception as e:
            print(f"  [!] Learnable graph analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary of key findings
    print("\n" + "="*60)
    print("KEY EXPLAINABILITY SUMMARY")
    print("="*60)
    
    # GCN edges
    A = gcn_adjacency.numpy()
    nino34_idx = var_order.index('Nino34') if 'Nino34' in var_order else 0
    incoming = [(var_order[i], A[i, nino34_idx]) for i in range(len(var_order)) if A[i, nino34_idx] > 0]
    incoming.sort(key=lambda x: -x[1])
    
    print("\nGCN: Variables connected to Nino34:")
    for var, weight in incoming[:5]:
        print(f"  {var}: {weight:.3f}")
    
    # Attention
    x_neutral = torch.zeros(len(var_order))
    attn_avg = np.zeros(len(var_order))
    for m in range(1, 13):
        attn = compute_attention_pattern(attn_model, x_neutral, m)
        attn_avg += attn[nino34_idx, :]
    attn_avg /= 12
    
    attn_sorted = sorted(enumerate(attn_avg), key=lambda x: -x[1])
    print("\nAttention: What Nino34 attends to (avg):")
    for idx, weight in attn_sorted[:5]:
        print(f"  {var_order[idx]}: {weight:.3f}")


if __name__ == '__main__':
    main()
