# NXRO Results Folder Structure

This document describes the folder structure for storing trained model checkpoints, visualizations, and forecasts for all NXRO variants and XRO baselines.

## Quick Start: Running All Variants

### Option 1: Run All 32 Variants (Comprehensive)
```bash
# Train all 32 variants from README summary table
bash run_all_variants_complete.sh --epochs 1000 --device cuda --test

# Skip base variants if already trained
bash run_all_variants_complete.sh --epochs 1000 --device cuda --skip_base

# Skip warm-start variants (only train base)
bash run_all_variants_complete.sh --epochs 1000 --device cuda --skip_warmstart
```

### Option 2: Run Only Base Variants (9 variants)
```bash
# ORAS5 only
bash run_all_nxro_ora5.sh --epochs 1000 --device cuda --test

# With extra training data
bash run_all_nxro.sh --epochs 1000 --device cuda --test
```

### Option 3: Run Only Warm-Start Variants (23 variants)
```bash
bash run_all_warmstart_variants.sh --epochs 1000 --device cuda --test
```

**Recommended workflow**:
1. First run base variants: `bash run_all_nxro.sh --epochs 1000 --device cuda --test`
2. Then run warm-start ablations: `bash run_all_variants_complete.sh --epochs 1000 --device cuda --skip_base`
3. Compare results: `python XRO_variants.py --test`

## Overview

All results are saved under the `results/` directory with subdirectories organized by model variant. Each variant has its own folder containing:
- Model checkpoints (`.pt` files)
- Training curves (PNG)
- Forecast skill plots (ACC/RMSE)
- Seasonal synchronization plots
- Forecast plumes
- Stochastic ensemble diagnostics (if `--stochastic` enabled)

## Root Directory Structure

```
results/
├── xro_fit_warmstart.nc          # XRO fit for warm-start initialization
├── NXRO_observed_Nino34.png      # Reference observation plot
├── variants_rmse_skill_test_base.png      # Cross-variant RMSE comparison
├── variants_acc_skill_test_base.png       # Cross-variant ACC comparison
├── variants_rank_rmse_heatmap_test_base.png  # RMSE rank heatmap
├── variants_overall_rank_rmse_bar_test_base.png  # Average RMSE rank
├── variants_overall_rank_acc_bar_test_base.png   # Average ACC rank
├── summary/                      # Cross-model comparison CSVs/plots
├── graphs/                       # Pre-computed graph adjacency matrices
├── hyperparam_search/            # Hyperparameter search results
├── linear/                       # Variant 1, 1a
├── ro/                           # Variant 2, 2a, 2a-Fix*
├── rodiag/                       # Variant 3, 3a, 3a-Fix*
├── res/                          # Variant 4, 4a
├── res_fullxro/                  # Variant 4b (NEW)
├── neural/                       # Variant 5
├── attentive/                    # Variant 5a, 5a-WS, 5a-FixL
├── graph/                        # Variant 5b, 5b-WS, 5b-FixL (multiple graph types)
├── neural_phys/                  # Variant 5c
├── resmix/                       # Variant 5d, 5d-WS, 5d-Fix*
├── bilinear/                     # Deprecated (poor performance)
├── graphpyg/                     # Graph variants with PyTorch Geometric
└── ...
```

## Per-Variant Folder Structure

Each variant folder (e.g., `results/linear/`, `results/rodiag/`) contains:

### 1. Model Checkpoints

**Random initialization (base variants)**:
```
nxro_<model>_best.pt              # Best model by validation RMSE
nxro_<model>_best_test.pt         # Best model by test RMSE (if --test flag)
```

**Warm-start variants** (naming convention to be added):
```
nxro_<model>_ws_best.pt           # Warm-start, train all
nxro_<model>_fixL_best.pt         # Warm-start, freeze linear
nxro_<model>_fixRO_best.pt        # Warm-start, freeze RO
nxro_<model>_fixDiag_best.pt      # Warm-start, freeze diagonal
nxro_<model>_fixNL_best.pt        # Warm-start, freeze nonlinear (RO+Diag)
nxro_<model>_fixAll_best.pt       # All frozen (pure XRO)
nxro_<model>_fixPhysics_best.pt   # Freeze all physics (for ResidualMix)
```

Checkpoint file format (PyTorch dict):
```python
{
    'state_dict': model.state_dict(),  # Model parameters
    'var_order': var_order,            # Variable names list
}
```

### 2. Training Diagnostics

```
NXRO_<model>_training_curves.png         # Train/test RMSE vs epoch
NXRO_<model>_training_curves_test.png    # If --test flag
```

### 3. Forecast Skill Plots

```
NXRO_<model>_acc.png                # ACC vs forecast lead
NXRO_<model>_rmse.png               # RMSE vs forecast lead
NXRO_<model>_acc_test.png           # Test period ACC (if --test)
NXRO_<model>_rmse_test.png          # Test period RMSE (if --test)
```

### 4. Seasonal Synchronization

```
NXRO_<model>_seasonal_synchronization.png      # Monthly stddev curves
NXRO_<model>_seasonal_synchronization_test.png
```

### 5. Forecast Plumes

```
NXRO_<model>_plume_<YYYY-MM>_<init_idx>.png   # Individual plume plots
```

### 6. Stochastic Ensemble Diagnostics (if `--stochastic`)

```
NXRO_<model>_stochastic_forecasts.nc       # NetCDF with ensemble forecasts
nxro_<model>_stochastic_noise.npz         # Fitted noise parameters
NXRO_<model>_stochastic_acc.png           # Ensemble mean ACC
NXRO_<model>_stochastic_rmse.png          # Ensemble mean RMSE
NXRO_<model>_stochastic_coverage_heatmap.png  # Coverage diagnostics
NXRO_<model>_stochastic_spread_rmse.png   # Spread-skill relationship
NXRO_<model>_stochastic_crps.png          # CRPS score
NXRO_<model>_stochastic_reliability.png   # Reliability diagram
<model>_stochastic_lead_metrics.csv        # Spread, RMSE, CRPS, Brier
<model>_stochastic_coverage.csv            # Coverage percentages
<model>_stochastic_interval_width.csv      # Interval widths
```

## Special Folder Structures

### Graph Variants (`results/graph/`)

Graph variants have sub-folders for different graph construction methods:

```
results/graph/
├── fixed_xro/                    # XRO-derived graph (default)
│   ├── nxro_graph_best.pt
│   ├── NXRO_graph_acc.png
│   └── ...
├── fixed_stat_pearson_k3/        # Pearson correlation k-NN (k=3)
│   └── ...
├── fixed_stat_spearman_k3/       # Spearman correlation k-NN
│   └── ...
├── fixed_stat_mi_k3/             # Mutual information k-NN
│   └── ...
├── fixed_stat_xcorr_max_k3/      # Cross-correlation k-NN
│   └── ...
├── learned_xro_l1<lambda>/       # Learned adjacency from XRO prior
│   └── ...
└── learned_stat_<method>_k<K>_l1<lambda>/  # Learned from statistical prior
    └── ...
```

### PyG Graph Variants (`results/graphpyg/`)

```
results/graphpyg/
├── gcn_k3/                       # GCN with k=3 neighbors
│   └── ...
├── gat_k3/                       # GAT with k=3 neighbors
│   └── ...
└── <convtype>_<method>_k<K>/     # Various combinations
    └── ...
```

### Hyperparameter Search (`results/hyperparam_search/`)

```
results/hyperparam_search/
├── grid_search_results.csv       # All configurations sorted by performance
├── best_config.json              # Best configuration details
├── search_grid.json              # Grid specification
└── grid_search.log               # Execution log
```

### Graph Adjacency Matrices (`results/graphs/`)

Pre-computed and cached graph structures:

```
results/graphs/
├── xro_adj_XRO_indices_oras5_<dates>_norm<N>_th<T>_<hash>.npz
├── stat_adj_<dataset>_<method>_k<K>_b<bins>_lag<L>_<hash>.npz
└── ...
```

Format: NumPy `.npz` file with keys:
- `'adjacency'`: Adjacency matrix [n_vars, n_vars]
- `'edge_index'` (optional): PyG format [2, n_edges]
- Metadata: method, topk, normalization settings

## Naming Conventions for Warm-Start Variants

### File Suffix Conventions

To distinguish warm-start and freezing variants, we use the following suffixes:

| Variant Type | Checkpoint Name | Example |
|--------------|----------------|---------|
| **Base (random)** | `nxro_<model>_best.pt` | `nxro_linear_best.pt` |
| **Warm-start all** | `nxro_<model>_ws_best.pt` | `nxro_linear_ws_best.pt` |
| **Freeze linear** | `nxro_<model>_fixL_best.pt` | `nxro_ro_fixL_best.pt` |
| **Freeze RO** | `nxro_<model>_fixRO_best.pt` | `nxro_ro_fixRO_best.pt` |
| **Freeze diagonal** | `nxro_<model>_fixDiag_best.pt` | `nxro_rodiag_fixDiag_best.pt` |
| **Freeze RO+Diag** | `nxro_<model>_fixNL_best.pt` | `nxro_rodiag_fixNL_best.pt` |
| **Freeze all physics** | `nxro_<model>_fixPhysics_best.pt` | `nxro_resmix_fixPhysics_best.pt` |
| **Freeze all (baseline)** | `nxro_<model>_fixAll_best.pt` | `nxro_rodiag_fixAll_best.pt` |

### Plot Suffix Conventions

Plots follow the same naming pattern:
- `NXRO_<model>_<suffix>_acc.png`
- `NXRO_<model>_<suffix>_rmse.png`
- `NXRO_<model>_<suffix>_training_curves.png`
- etc.

Examples:
- `NXRO_linear_ws_acc.png` - Variant 1a ACC skill
- `NXRO_rodiag_fixL_rmse.png` - Variant 3a-FixL RMSE skill
- `NXRO_resmix_fixPhysics_training_curves.png` - Variant 5d-FixPhysics training

## Variant-to-Folder Mapping

| Variant ID | Folder | Checkpoint Suffix | Description |
|------------|--------|-------------------|-------------|
| **1** | `results/linear/` | `_best.pt` | Random init |
| **1a** | `results/linear/` | `_ws_best.pt` | Warm-start |
| **2** | `results/ro/` | `_best.pt` | Random init |
| **2a** | `results/ro/` | `_ws_best.pt` | Warm-start all |
| **2a-FixL** | `results/ro/` | `_fixL_best.pt` | Freeze linear |
| **2a-FixRO** | `results/ro/` | `_fixRO_best.pt` | Freeze RO |
| **2a-FixAll** | `results/ro/` | `_fixAll_best.pt` | No training |
| **3** | `results/rodiag/` | `_best.pt` | Random init |
| **3a** | `results/rodiag/` | `_ws_best.pt` | Warm-start all |
| **3a-FixL** | `results/rodiag/` | `_fixL_best.pt` | Freeze linear |
| **3a-FixRO** | `results/rodiag/` | `_fixRO_best.pt` | Freeze RO |
| **3a-FixDiag** | `results/rodiag/` | `_fixDiag_best.pt` | Freeze diagonal |
| **3a-FixNL** | `results/rodiag/` | `_fixNL_best.pt` | Freeze nonlinear |
| **3a-FixAll** | `results/rodiag/` | `_fixAll_best.pt` | No training |
| **4** | `results/res/` | `_best.pt` | Random init |
| **4a** | `results/res/` | `_fixL_best.pt` | Freeze linear |
| **4b** | `results/res_fullxro/` | `_best.pt` | Frozen full XRO + MLP |
| **5** | `results/neural/` | `_best.pt` | Pure MLP |
| **5a** | `results/attentive/` | `_best.pt` | Random init |
| **5a-WS** | `results/attentive/` | `_ws_best.pt` | Warm-start |
| **5a-FixL** | `results/attentive/` | `_fixL_best.pt` | Freeze linear |
| **5b** | `results/graph/fixed_xro/` | `_best.pt` | Random init |
| **5b-WS** | `results/graph/fixed_xro/` | `_ws_best.pt` | Warm-start |
| **5b-FixL** | `results/graph/fixed_xro/` | `_fixL_best.pt` | Freeze linear |
| **5c** | `results/neural_phys/` | `_best.pt` | Regularized MLP |
| **5d** | `results/resmix/` | `_best.pt` | Random init |
| **5d-WS** | `results/resmix/` | `_ws_best.pt` | Warm-start all |
| **5d-FixL** | `results/resmix/` | `_fixL_best.pt` | Freeze linear |
| **5d-FixRO** | `results/resmix/` | `_fixRO_best.pt` | Freeze RO |
| **5d-FixDiag** | `results/resmix/` | `_fixDiag_best.pt` | Freeze diagonal |
| **5d-FixNL** | `results/resmix/` | `_fixNL_best.pt` | Freeze RO+Diag |
| **5d-FixPhysics** | `results/resmix/` | `_fixPhysics_best.pt` | Freeze all physics |

## Example: Variant 3a-FixL (Freeze Linear, Train RO+Diag)

```
results/rodiag/
├── nxro_rodiag_fixL_best.pt                          # Model checkpoint
├── NXRO_rodiag_fixL_training_curves.png              # Training curves
├── NXRO_rodiag_fixL_acc.png                          # ACC skill curve
├── NXRO_rodiag_fixL_rmse.png                         # RMSE skill curve
├── NXRO_rodiag_fixL_seasonal_synchronization.png     # Seasonal stddev
├── NXRO_rodiag_fixL_plume_2022-09_0.png              # Forecast plume (Sep 2022)
├── NXRO_rodiag_fixL_plume_2024-12_1.png              # Forecast plume (Dec 2024)
└── ... (additional forecasts and diagnostics)
```

## Example: Variant 4b (Frozen Full XRO + Residual)

```
results/res_fullxro/
├── nxro_res_fullxro_best.pt                          # Model checkpoint
├── NXRO_res_fullxro_training_curves.png              # Training curves
├── NXRO_res_fullxro_acc.png                          # ACC skill curve
├── NXRO_res_fullxro_rmse.png                         # RMSE skill curve
├── NXRO_res_fullxro_seasonal_synchronization.png     # Seasonal stddev
└── ... (forecasts and diagnostics)
```

## Loading Saved Models

### Python Example

```python
import torch
import xarray as xr
from nxro.models import NXROLinearModel, NXRORODiagModel, NXROResidualMixModel
from utils.xro_utils import init_nxro_from_xro

# Load a base variant (random init)
checkpoint = torch.load('results/linear/nxro_linear_best.pt')
model = NXROLinearModel(n_vars=10, k_max=2)
model.load_state_dict(checkpoint['state_dict'])
var_order = checkpoint['var_order']

# Load a warm-start variant (variant 1a)
checkpoint_ws = torch.load('results/linear/nxro_linear_ws_best.pt')
# For inference, initialization doesn't matter; just load state_dict
model_ws = NXROLinearModel(n_vars=10, k_max=2)
model_ws.load_state_dict(checkpoint_ws['state_dict'])

# Load variant with freezing (variant 3a-FixL)
checkpoint_fixL = torch.load('results/rodiag/nxro_rodiag_fixL_best.pt')
# To recreate with same frozen structure (for continued training):
xro_fit = xr.open_dataset('results/xro_fit_warmstart.nc')
init_dict = init_nxro_from_xro(xro_fit, k_max=2, include_ro=True, include_diag=True)
init_params = {k+'_init': v for k, v in init_dict.items()}
model_fixL = NXRORODiagModel(n_vars=10, k_max=2, **init_params, freeze_linear=True)
model_fixL.load_state_dict(checkpoint_fixL['state_dict'])
```

## Summary CSV Files

Cross-variant comparison CSVs are saved in `results/summary/` or root `results/`:

### From `XRO_variants.py` (original comparison script):
```
variants_overall_rank_rmse_test_base.csv  # RMSE ranks table
variants_overall_rank_acc_test_base.csv   # ACC ranks table
variants_rank_rmse_test_base.csv          # RMSE by lead
variants_rank_acc_test_base.csv           # ACC by lead
```

### From `xro_variant_comparisons.py` (new systematic comparison):

**Within-category comparisons** (one set per category):
```
results/summary/Cat1_Linear_within_category_acc.png         # ACC plot
results/summary/Cat1_Linear_within_category_rmse.png        # RMSE plot
results/summary/Cat1_Linear_within_category_acc.csv         # ACC values
results/summary/Cat1_Linear_within_category_rmse.csv        # RMSE values

results/summary/Cat2_RO_within_category_*.png/csv           # Category 2
results/summary/Cat3_RODiag_within_category_*.png/csv       # Category 3
results/summary/Cat4_Res_within_category_*.png/csv          # Category 4
results/summary/Cat5a_Attentive_within_category_*.png/csv   # Category 5a
results/summary/Cat5d_ResidualMix_within_category_*.png/csv # Category 5d
```

**Between-category comparisons** (best from each category):
```
results/summary/between_category_best_acc.png   # ACC comparison
results/summary/between_category_best_rmse.png  # RMSE comparison
```

**Summary statistics**:
```
results/summary/variant_summary_statistics.csv       # All variants with mean ACC/RMSE
results/summary/variant_summary_statistics_test.csv  # Test period version
```

CSV format:
- Columns: Model names (for comparisons) or Category/Variant/Mean_ACC/Mean_RMSE (for summary)
- Rows: Forecast leads (0-21 months) or individual variants
- Values: Skill metric or rank

## Disk Space Estimates

Per variant (approximate):
- Model checkpoint: ~1-10 MB (depends on model size)
- Plots (PNG): ~50-100 KB each, ~10-20 plots → ~1-2 MB
- Stochastic forecasts NetCDF: ~10-50 MB (if enabled)

**Total for all 32 variants**: ~1-5 GB (base) + up to ~50 GB (if all stochastic)

## Best Practices

1. **Consistent naming**: Use suffixes (`_ws`, `_fixL`, etc.) to distinguish variants
2. **Version control**: Checkpoint files should be in `.gitignore` (large binaries)
3. **Metadata**: Always save `var_order` with checkpoint for reproducibility
4. **Test mode**: Use `--test` flag to save separate checkpoints for test-optimized models
5. **Cleanup**: Remove intermediate checkpoints or old experiments to save space
6. **Documentation**: Update this file if new variant types or folder structures are added

## Current Implementation Status

✅ **Implemented**:
- Folder structure for base variants (1-5, 5a-5d)
- Model checkpoint saving
- Plot generation (training curves, skill, seasonal sync)
- Graph variant subfolders
- Stochastic diagnostics

🚧 **In Progress**:
- Automatic suffix naming for warm-start variants (requires NXRO_train.py updates)
- Centralized comparison across all 32 variants
- Automated result aggregation scripts

✅ **Implemented**:

1. **`xro_variant_comparisons.py`** - Systematic variant comparison plots:
   - Within-category comparisons (e.g., all variant 1 sub-variants, all variant 3 sub-variants)
   - Between-category comparisons (best from each main category)
   - Saves to `results/summary/` with organized naming

   Usage:
   ```bash
   # Compare all available variants (training period)
   python xro_variant_comparisons.py --eval_period train
   
   # Compare using test-period checkpoints
   python xro_variant_comparisons.py --test --eval_period test
   ```

2. **`rank_all_variants.py`** - Comprehensive ranking across ALL variants:
   - Discovers all checkpoint files (all 32 variants if trained)
   - Loads and evaluates each independently
   - Ranks ALL variants together (not just one per model type)
   - Displays top N performers with plots
   - Saves complete ranking table

   Usage:
   ```bash
   # Rank all variants, show top 10 (default)
   python rank_all_variants.py --test --eval_period train
   
   # Show top 5, rank by RMSE only
   python rank_all_variants.py --top_n 5 --metric rmse --test --eval_period train
   
   # Rank by ACC only
   python rank_all_variants.py --top_n 10 --metric acc --test --eval_period train
   
   # Combined ranking (ACC + RMSE)
   python rank_all_variants.py --top_n 10 --metric combined --test
   ```

   Outputs:
   - `results/rankings/all_variants_ranked_<metric>_eval_<period>.csv` - Full ranking table
   - `results/rankings/top<N>_acc_eval_<period>.png` - ACC comparison plot
   - `results/rankings/top<N>_rmse_eval_<period>.png` - RMSE comparison plot

📋 **Planned**:
- `analyze_warmstart_convergence.py` - Compare training curves: warm-start vs. random
- `analyze_freezing_importance.py` - Rank component importance via Fix* ablations
- `compare_variants_pairwise.py` - Direct pairwise comparisons (e.g., V3 vs. V3a, V3a vs. V3a-FixL)

## Quick Reference: Finding Results for Each Variant

| Variant | Folder | Filename Pattern | Trainable Components |
|---------|--------|------------------|----------------------|
| 1 | `results/linear/` | `nxro_linear_best.pt` | L (random) |
| 1a | `results/linear/` | `nxro_linear_ws_best.pt` | L (warm-start) |
| 2 | `results/ro/` | `nxro_ro_best.pt` | L+RO (random) |
| 2a | `results/ro/` | `nxro_ro_ws_best.pt` | L+RO (warm-start) |
| 2a-FixL | `results/ro/` | `nxro_ro_fixL_best.pt` | RO only |
| 2a-FixRO | `results/ro/` | `nxro_ro_fixRO_best.pt` | L only |
| 2a-FixAll | N/A | (use XRO directly) | None |
| 3 | `results/rodiag/` | `nxro_rodiag_best.pt` | L+RO+Diag (random) |
| 3a | `results/rodiag/` | `nxro_rodiag_ws_best.pt` | L+RO+Diag (warm-start) |
| 3a-FixL | `results/rodiag/` | `nxro_rodiag_fixL_best.pt` | RO+Diag |
| 3a-FixRO | `results/rodiag/` | `nxro_rodiag_fixRO_best.pt` | L+Diag |
| 3a-FixDiag | `results/rodiag/` | `nxro_rodiag_fixDiag_best.pt` | L+RO |
| 3a-FixNL | `results/rodiag/` | `nxro_rodiag_fixNL_best.pt` | L only |
| 3a-FixAll | N/A | (use XRO directly) | None |
| 4 | `results/res/` | `nxro_res_best.pt` | L+MLP (random) |
| 4a | `results/res/` | `nxro_res_fixL_best.pt` | MLP only |
| 4b | `results/res_fullxro/` | `nxro_res_fullxro_best.pt` | MLP only (XRO frozen) |
| 5 | `results/neural/` | `nxro_neural_best.pt` | All MLP |
| 5a | `results/attentive/` | `nxro_attentive_best.pt` | L+Attn (random) |
| 5a-WS | `results/attentive/` | `nxro_attentive_ws_best.pt` | L+Attn (warm-start) |
| 5a-FixL | `results/attentive/` | `nxro_attentive_fixL_best.pt` | Attn only |
| 5b | `results/graph/fixed_xro/` | `nxro_graph_best.pt` | L+Graph (random) |
| 5b-WS | `results/graph/fixed_xro/` | `nxro_graph_ws_best.pt` | L+Graph (warm-start) |
| 5b-FixL | `results/graph/fixed_xro/` | `nxro_graph_fixL_best.pt` | Graph only |
| 5c | `results/neural_phys/` | `nxro_neural_phys_best.pt` | All MLP (regularized) |
| 5d | `results/resmix/` | `nxro_resmix_best.pt` | L+RO+Diag+MLP (random) |
| 5d-WS | `results/resmix/` | `nxro_resmix_ws_best.pt` | L+RO+Diag+MLP (warm-start) |
| 5d-FixL | `results/resmix/` | `nxro_resmix_fixL_best.pt` | RO+Diag+MLP |
| 5d-FixRO | `results/resmix/` | `nxro_resmix_fixRO_best.pt` | L+Diag+MLP |
| 5d-FixDiag | `results/resmix/` | `nxro_resmix_fixDiag_best.pt` | L+RO+MLP |
| 5d-FixNL | `results/resmix/` | `nxro_resmix_fixNL_best.pt` | L+MLP |
| 5d-FixPhysics | `results/resmix/` | `nxro_resmix_fixPhysics_best.pt` | MLP only |

**Note**: If `--test` flag is used, replace `_best.pt` with `_best_test.pt`

