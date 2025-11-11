#!/bin/bash
#
# Phase 2: Training Strategy Optimization for Top Graph Models
#
# This script takes the top N graph configurations from Phase 1 and optimizes
# their training strategies to maximize performance.
#
# Usage:
#   bash xro_graph_variants_explore_phase2.sh --top_n 3
#   bash xro_graph_variants_explore_phase2.sh --top_n 10 --quick
#
# Options:
#   --top_n N     Number of top Phase 1 models to optimize (default: 3)
#   --quick       Run reduced experiment set per model (4 instead of 18)
#   --device      Device to use (default: auto)
#

set -euo pipefail

TOP_N=3
QUICK_MODE=false
DEVICE=auto
BASE_DIR="results_out_of_sample"
PHASE1_CSV="${BASE_DIR}/rankings/all_variants_ranked_rmse_out_of_sample.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --top_n) TOP_N="$2"; shift 2 ;;
    --quick) QUICK_MODE=true; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "================================================================================"
echo "     PHASE 2: TRAINING STRATEGY OPTIMIZATION"
echo "================================================================================"
echo "Configuration:"
echo "  Dataset: ORAS5 only"
echo "  Top N models from Phase 1: ${TOP_N}"
echo "  Quick mode: ${QUICK_MODE}"
echo "  Device: ${DEVICE}"
echo "  Results: ${BASE_DIR}/"
echo "================================================================================"
echo ""

# Check if Phase 1 results exist
if [[ ! -f "$PHASE1_CSV" ]]; then
  echo "[ERROR] Phase 1 results not found: $PHASE1_CSV"
  echo "Run Phase 1 first: ./xro_graph_variants_explore_phase1.sh"
  exit 1
fi

# Extract top N graph models from Phase 1
echo "Extracting top ${TOP_N} graph models from Phase 1 results..."

# Read CSV and filter for graph models, get top N
TOP_MODELS=$(python3 << PYEOF
import pandas as pd
import re

df = pd.read_csv('$PHASE1_CSV')

# Filter graph models
graph_mask = (df['Model'].str.contains('Graph', case=False, na=False) | 
              df['Model'].str.contains('graphpyg', case=False, na=False))
exclude = (df['Model'].str.contains('RO\+Diag', case=False, na=False) |
           df['Model'].str.contains('ResidualMix', case=False, na=False))
graph_df = df[graph_mask & ~exclude].head($TOP_N)

# Extract configuration for each
for idx, row in graph_df.iterrows():
    model_name = row['Model'].lower()
    
    # Parse topology
    if 'pearson' in model_name:
        topology = 'pearson'
    elif 'spearman' in model_name:
        topology = 'spearman'
    elif 'mi' in model_name:
        topology = 'mi'
    elif 'xcorr' in model_name:
        topology = 'xcorr_max'
    else:
        topology = 'xro'
    
    # Parse k
    k_match = re.search(r'k(\d+)', model_name)
    k_val = int(k_match.group(1)) if k_match else 3
    
    # Parse GNN type
    gnn_type = 'gat' if 'gat' in model_name else 'gcn'
    
    rank = int(row['Rank'])
    rmse = row['Mean_RMSE_Test']
    
    print(f"{rank}|{topology}|{k_val}|{gnn_type}|{rmse:.4f}")

PYEOF
)

if [[ -z "$TOP_MODELS" ]]; then
  echo "[ERROR] Could not extract top models from Phase 1 results"
  exit 1
fi

echo "$TOP_MODELS" | while IFS='|' read -r rank topology k gnn_type rmse; do
  echo "  Rank $rank: topology=$topology, k=$k, GNN=$gnn_type, RMSE=$rmse C"
done

echo ""
echo "Starting Phase 2 optimization for these ${TOP_N} configurations..."
echo ""

# Counter for tracking progress
total_experiments=0

# Iterate over top models
echo "$TOP_MODELS" | while IFS='|' read -r rank topology k gnn_type rmse; do
  
  echo "================================================================================"
  echo "Optimizing Rank ${rank} Model: ${topology} (k=${k}, ${gnn_type})"
  echo "================================================================================"
  echo "Baseline RMSE: ${rmse} C"
  echo ""
  
  # Build base command
  BASE_CMD="python NXRO_train_out_of_sample.py --model graph_pyg --device $DEVICE"
  
  # Add topology flags
  if [[ "$topology" == "xro" ]]; then
    CMD_BASE="$BASE_CMD --top_k $k"
  else
    CMD_BASE="$BASE_CMD --graph_stat_method $topology --graph_stat_topk $k"
  fi
  
  # Add GAT flag if needed
  if [[ "$gnn_type" == "gat" ]]; then
    CMD_BASE="$CMD_BASE --gat"
  fi
  
  if [[ "$QUICK_MODE" == "true" ]]; then
    # Quick mode: 4 key experiments per model
    echo "Running 4 quick experiments..."
    
    # 1. Warm-start linear
    echo "  [1/4] Warm-start test..."
    $CMD_BASE --warm_start ${BASE_DIR}/xro_fit_warmstart.nc --epochs 1500
    
    # 2. Best learning rate
    echo "  [2/4] Learning rate 5e-4..."
    $CMD_BASE --lr 5e-4 --epochs 1500
    
    # 3. 2-step rollout
    echo "  [3/4] 2-step rollout..."
    $CMD_BASE --rollout_k 2 --epochs 1500
    
    # 4. Higher weight decay
    echo "  [4/4] Weight decay 5e-4..."
    $CMD_BASE --weight_decay 5e-4 --epochs 1500
    
    total_experiments=$((total_experiments + 4))
    
  else
    # Full mode: 18 experiments per model
    echo "Running 18 full experiments..."
    
    # 2.1: Warm-start variants (3 runs)
    echo "  [Warm-start 1/3] Random init baseline..."
    $CMD_BASE --epochs 1500
    
    echo "  [Warm-start 2/3] Warm-start linear, train all..."
    $CMD_BASE --warm_start ${BASE_DIR}/xro_fit_warmstart.nc --epochs 1500
    
    echo "  [Warm-start 3/3] Freeze linear, train graph only..."
    $CMD_BASE --warm_start ${BASE_DIR}/xro_fit_warmstart.nc --freeze linear --epochs 1500
    
    # 2.2: Learning rate tuning (5 runs)
    echo "  [LR 1/5] lr=5e-4..."
    $CMD_BASE --lr 5e-4 --epochs 1500
    
    echo "  [LR 2/5] lr=1e-3 (baseline)..."
    $CMD_BASE --lr 1e-3 --epochs 1500
    
    echo "  [LR 3/5] lr=2e-3..."
    $CMD_BASE --lr 2e-3 --epochs 1500
    
    echo "  [LR 4/5] lr=5e-3..."
    $CMD_BASE --lr 5e-3 --epochs 1500
    
    echo "  [LR 5/5] lr=1e-2..."
    $CMD_BASE --lr 1e-2 --epochs 1500
    
    # 2.3: Multi-step rollout (4 runs)
    echo "  [Rollout 1/4] 1-step (baseline)..."
    $CMD_BASE --rollout_k 1 --epochs 1500
    
    echo "  [Rollout 2/4] 2-step..."
    $CMD_BASE --rollout_k 2 --epochs 1500
    
    echo "  [Rollout 3/4] 3-step..."
    $CMD_BASE --rollout_k 3 --epochs 1500
    
    echo "  [Rollout 4/4] 5-step..."
    $CMD_BASE --rollout_k 5 --epochs 1500
    
    # 2.4: Epoch count (3 runs)
    echo "  [Epochs 1/3] 1000 epochs..."
    $CMD_BASE --epochs 1000
    
    echo "  [Epochs 2/3] 1500 epochs (baseline)..."
    $CMD_BASE --epochs 1500
    
    echo "  [Epochs 3/3] 2000 epochs..."
    $CMD_BASE --epochs 2000
    
    # 2.5: Batch size (3 runs)
    echo "  [Batch 1/3] batch=64..."
    $CMD_BASE --batch_size 64 --epochs 1500
    
    echo "  [Batch 2/3] batch=128 (baseline)..."
    $CMD_BASE --batch_size 128 --epochs 1500
    
    echo "  [Batch 3/3] batch=256..."
    $CMD_BASE --batch_size 256 --epochs 1500
    
    total_experiments=$((total_experiments + 18))
  fi
  
  echo ""
  echo "Completed optimization for Rank ${rank} model"
  echo ""
  
done

# Final ranking
echo "================================================================================"
echo "Re-ranking all models after Phase 2..."
echo "================================================================================"
python rank_all_variants_out_of_sample.py --top_n 20 --metric rmse --force

echo ""
echo "================================================================================"
echo "PHASE 2 COMPLETE!"
echo "================================================================================"
echo ""
echo "Total experiments run: ${total_experiments}"
echo "Results saved to: ${BASE_DIR}/"
echo "Rankings: ${BASE_DIR}/rankings/"
echo ""
echo "Next steps:"
echo "  1. Visualize: python visualize_phase2_results.py"
echo "  2. Check if target achieved (Test RMSE <= 0.567 C)"
echo "  3. If not, proceed to Phase 3 (regularization tuning)"
echo ""
echo "================================================================================"

