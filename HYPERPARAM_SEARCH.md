# Hyperparameter Grid Search for Graph-based NXRO Models

This document explains the hyperparameter search tools for optimizing graph-based NXRO models.

## Overview

Three complementary scripts are provided for hyperparameter optimization:

1. **`run_graph_hyperparam_search.py`** - Python-based grid search with detailed tracking
2. **`run_graph_grid_search.sh`** - Bash-based exhaustive grid search
3. **`run_graph_quick_search.sh`** - Bash-based focused search for faster iteration

All scripts focus on **graph-based models only** and use **base ORAS5 data** (no extra training data).

## Why Grid Search for Graph Models?

Graph-based NXRO models introduce several new hyperparameters beyond the standard NXRO models:

- **Graph structure choice**: XRO-based vs. statistical k-NN (Pearson, Spearman, MI, xcorr)
- **Graph sparsity**: top-k neighbors (affects computational cost and generalization)
- **GNN architecture**: For PyG models, choice between GCN and GAT layers
- **Learned adjacency**: Whether to learn graph structure with L1 sparsity regularization

Combined with standard hyperparameters (hidden_dim, learning_rate, rollout_k), this creates a large search space.

## Selection Criterion

**Test RMSE on Niño3.4** averaged across all forecast lead times (1-24 months).

This metric directly measures operational forecast skill on the primary ENSO index during the held-out test period.

## Grid Specifications

### Full Grid (`run_graph_grid_search.sh`)

**Total configurations: ~10,800**

| Parameter | Values |
|-----------|--------|
| Model type | `graph`, `graph_pyg` |
| Graph structure | `xro`, `pearson`, `spearman`, `mi`, `xcorr_max` |
| GNN type (PyG only) | `gcn`, `gat` |
| top_k | 1, 2, 3, 5, 7, 10 |
| hidden_dim | 32, 64, 128, 256 |
| learning_rate | 1e-4, 5e-4, 1e-3, 5e-3 |
| rollout_k | 1, 2, 3 |
| use_learned_graph | False, True |
| l1_lambda (if learned) | 1e-4, 5e-4, 1e-3, 5e-3 |

**Estimated time**: 5-10 days on single GPU (at 2000 epochs each)

### Quick Grid (`run_graph_quick_search.sh`)

**Total configurations: ~500**

Focused on most promising ranges based on prior experience:

| Parameter | Values (reduced) |
|-----------|------------------|
| Graph structure | `xro`, `pearson`, `spearman` (drop MI, xcorr) |
| top_k | 3, 5, 7 |
| hidden_dim | 64, 128 |
| learning_rate | 5e-4, 1e-3 |
| rollout_k | 1, 2 |
| l1_lambda | 1e-3, 5e-3 |

**Estimated time**: 1-2 days on single GPU (at 1000 epochs each)

## Usage Examples

### 1. Quick Search (Recommended for First Run)

```bash
# Run focused grid search
bash run_graph_quick_search.sh --epochs 1000 --device cuda --test

# Results saved to results/hyperparam_search_quick/
# - results.csv: all configs sorted by test RMSE
# - quick_search.log: full execution log
```

### 2. Full Grid Search

```bash
# WARNING: This will take a long time!
bash run_graph_grid_search.sh --epochs 2000 --device cuda --test

# Results saved to results/hyperparam_search/
# - grid_search_results.csv
# - best_config.txt
# - grid_search.log
```

### 3. Python Version (Most Flexible)

```bash
# Run with custom output directory
python run_graph_hyperparam_search.py \
    --epochs 2000 \
    --device cuda \
    --test \
    --output_dir results/my_custom_search

# Dry run to preview commands
python run_graph_hyperparam_search.py \
    --epochs 2000 \
    --test \
    --dry_run
```

### 4. Validation Period Search

To search on validation period (2015-2024) instead of test:

```bash
# Omit --test flag
bash run_graph_quick_search.sh --epochs 1000 --device cuda
```

## Output Files

All scripts produce similar outputs:

### CSV Results Table

```csv
config_name,test_rmse,model_type,graph_structure,gnn_type,top_k,hidden_dim,learning_rate,rollout_k,use_learned_graph,l1_lambda
graphpyg_gat_pearson_k5_h128_lr0.001_r2,0.3245,graph_pyg,stat_pearson,gat,5,128,0.001,2,False,
graph_learned_xro_l10.001_h64_lr0.0005_r1,0.3312,graph,xro,,3,64,0.0005,1,True,0.001
...
```

### Best Configuration

`best_config.json` or `best_config.txt`:
```json
{
  "config_name": "graphpyg_gat_pearson_k5_h128_lr0.001_r2",
  "test_rmse": 0.3245,
  "model_type": "graph_pyg",
  "graph_structure": "stat_pearson",
  "gnn_type": "gat",
  "top_k": 5,
  "hidden_dim": 128,
  "learning_rate": 0.001,
  "rollout_k": 2,
  "use_learned_graph": false,
  "l1_lambda": null
}
```

### Execution Log

`grid_search.log` contains:
- Start/end timestamps for each experiment
- Training commands executed
- Success/failure status
- Any error messages

## Analyzing Results

### Quick Look at Top Models

```bash
# View top 10 from CSV
head -11 results/hyperparam_search/grid_search_results.csv | column -t -s,

# Or use Python
python -c "
import pandas as pd
df = pd.read_csv('results/hyperparam_search/grid_search_results.csv')
print(df.head(10)[['config_name', 'test_rmse']])
"
```

### Visualize Best Models

```bash
# Compare all graph models
python XRO_variants.py --test --data_filter base --graph_filter graph

# Or use dedicated graph comparison script
python XRO_graph_variants.py --test --data_filter base
```

### Extract Insights

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results/hyperparam_search/grid_search_results.csv')

# Which graph structure is best?
df.groupby('graph_structure')['test_rmse'].mean().sort_values()

# Best top_k value?
df.groupby('top_k')['test_rmse'].mean().sort_values()

# Fixed vs learned graph?
df.groupby('use_learned_graph')['test_rmse'].mean()

# GNN type comparison (PyG models only)
pyg_df = df[df['model_type'] == 'graph_pyg']
pyg_df.groupby('gnn_type')['test_rmse'].mean()

# Scatter plot: hidden_dim vs test_rmse
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='hidden_dim', y='test_rmse', hue='graph_structure', alpha=0.6)
plt.title('Hidden Dimension vs Test RMSE by Graph Structure')
plt.savefig('results/hyperparam_search/hidden_dim_analysis.png', dpi=300)
```

## Strategies for Efficient Search

### 1. Coarse-to-Fine Strategy

```bash
# Phase 1: Quick search to identify promising regions
bash run_graph_quick_search.sh --epochs 500 --device cuda --test

# Phase 2: Analyze results, identify best 2-3 configurations
# Then retrain with more epochs
python NXRO_train.py --model graph_pyg \
    --epochs 5000 \
    --hidden_dim 128 \
    --lr 0.001 \
    --rollout_k 2 \
    --top_k 5 \
    --gat \
    --graph_stat_method pearson \
    --device cuda --test
```

### 2. Focused Sub-Grids

Edit the grid arrays in the bash scripts to narrow the search:

```bash
# In run_graph_quick_search.sh, modify:
declare -a GRAPH_STRUCTURES=("pearson")  # Only Pearson
declare -a TOP_K_VALUES=(5 7)            # Only k=5,7
declare -a HIDDEN_DIMS=(128)             # Only 128
```

### 3. Parallel Execution

Split the grid across multiple GPUs or machines:

```bash
# Machine 1: GCN models
for struct in xro pearson spearman; do
    for k in 3 5 7; do
        # ... train GCN variants
    done
done

# Machine 2: GAT models
for struct in xro pearson spearman; do
    for k in 3 5 7; do
        # ... train GAT variants
    done
done
```

### 4. Early Stopping

For very large grids, implement early stopping:

```bash
# Add to NXRO_train.py calls:
--patience 50 --min_delta 0.001
```

This will stop training if validation loss doesn't improve for 50 epochs.

## Common Issues and Solutions

### Issue: Grid search takes too long

**Solution**: Start with quick search, or reduce epochs for initial screening:
```bash
bash run_graph_quick_search.sh --epochs 500 --device cuda
```

### Issue: Out of memory

**Solution**: Reduce `hidden_dim` or `rollout_k`:
```bash
# Focus on smaller models
declare -a HIDDEN_DIMS=(32 64)
```

### Issue: Some configs fail to train

**Solution**: Check `grid_search.log` for specific errors. Common issues:
- Learning rate too high → add smaller values like `1e-4`
- Rollout too long → reduce to `rollout_k=1`
- Graph too dense → increase k-NN sparsity (lower k)

### Issue: Results CSV is empty

**Solution**: Ensure models are being saved with proper naming. Check:
```bash
ls -lh results/graph/*/
ls -lh results/graphpyg/*/
```

## Advanced: Custom Grid

To create your own custom grid, edit `run_graph_hyperparam_search.py`:

```python
def build_hyperparameter_grid() -> List[Dict]:
    grid = {
        'model_type': ['graph_pyg'],  # Only PyG models
        'graph_structure': ['stat_pearson'],  # Only Pearson
        'gnn_type': [None, 'gcn', 'gat'],
        'top_k': [3, 5, 7, 10],
        'hidden_dim': [64, 128, 256],
        'learning_rate': [1e-3, 5e-3],
        'rollout_k': [1, 2],
        'use_learned_graph': [False],  # No learned graphs
        'l1_lambda': [None],
    }
    # ... rest of function
```

## References

For background on the graph models and structures, see:
- Main README: Graph construction methods (XRO-based, statistical k-NN)
- `graph_construction.py`: Implementation of statistical interaction metrics
- `nxro/models.py`: Graph model architectures
- `nxro/train.py`: Training procedures for graph models

## Questions?

If you encounter issues or have questions about the hyperparameter search:
1. Check `grid_search.log` for execution details
2. Verify model checkpoints exist in `results/graph/` and `results/graphpyg/`
3. Try a minimal dry run: `bash run_graph_quick_search.sh --epochs 10 --dry_run`

