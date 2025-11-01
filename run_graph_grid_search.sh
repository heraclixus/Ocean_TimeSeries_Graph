#!/bin/bash

# Grid Search for Graph-based NXRO Models (Base Data Only)
# Selection based on test RMSE for Nino3.4
#
# Usage:
#   bash run_graph_grid_search.sh [--epochs 2000] [--device auto] [--test] [--dry_run]

set -euo pipefail

# Default parameters
EPOCHS=2000
DEVICE=auto
TEST=""
DRY_RUN=false
OUTPUT_DIR="results/hyperparam_search"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --test)
      TEST="--test"; shift ;;
    --dry_run)
      DRY_RUN=true; shift ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/grid_search.log"

echo "=====================================================================" | tee "${LOG_FILE}"
echo "GRAPH MODEL HYPERPARAMETER GRID SEARCH" | tee -a "${LOG_FILE}"
echo "=====================================================================" | tee -a "${LOG_FILE}"
echo "Epochs: ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "Device: ${DEVICE}" | tee -a "${LOG_FILE}"
echo "Test mode: ${TEST:-false}" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "=====================================================================" | tee -a "${LOG_FILE}"

# Hyperparameter grids
declare -a GRAPH_STRUCTURES=("xro" "pearson" "spearman" "mi" "xcorr_max")
declare -a TOP_K_VALUES=(1 2 3 5 7 10)
declare -a HIDDEN_DIMS=(32 64 128 256)
declare -a LEARNING_RATES=(1e-4 5e-4 1e-3 5e-3)
declare -a ROLLOUT_K=(1 2 3)
declare -a L1_LAMBDAS=(1e-4 5e-4 1e-3 5e-3)

TOTAL_CONFIGS=0
RUN_COUNT=0

# Function to run a single experiment
run_experiment() {
  local cmd="$1"
  local desc="$2"
  
  ((TOTAL_CONFIGS++))
  
  echo "" | tee -a "${LOG_FILE}"
  echo "[${TOTAL_CONFIGS}] ${desc}" | tee -a "${LOG_FILE}"
  echo "Command: ${cmd}" | tee -a "${LOG_FILE}"
  
  if [ "${DRY_RUN}" = true ]; then
    echo "  [DRY RUN - skipping]" | tee -a "${LOG_FILE}"
    return
  fi
  
  if eval "${cmd}" >> "${LOG_FILE}" 2>&1; then
    echo "  ✓ Completed" | tee -a "${LOG_FILE}"
    ((RUN_COUNT++))
  else
    echo "  ✗ Failed" | tee -a "${LOG_FILE}"
  fi
}

echo ""
echo "Starting grid search..."
echo ""

# ============================================================================
# 1. NXROGraphModel with Fixed Adjacency
# ============================================================================

echo "--- NXROGraphModel (Fixed Adjacency) ---" | tee -a "${LOG_FILE}"

for graph_struct in "${GRAPH_STRUCTURES[@]}"; do
  for k in "${TOP_K_VALUES[@]}"; do
    for hdim in "${HIDDEN_DIMS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for rollout in "${ROLLOUT_K[@]}"; do
          
          # Build command
          cmd="python NXRO_train.py --model graph --epochs ${EPOCHS} --device ${DEVICE} ${TEST}"
          cmd="${cmd} --hidden_dim ${hdim} --lr ${lr} --rollout_k ${rollout}"
          
          if [ "${graph_struct}" != "xro" ]; then
            cmd="${cmd} --graph_stat_method ${graph_struct} --graph_stat_topk ${k}"
            desc="graph_fixed_stat_${graph_struct}_k${k}_h${hdim}_lr${lr}_r${rollout}"
          else
            desc="graph_fixed_xro_h${hdim}_lr${lr}_r${rollout}"
          fi
          
          run_experiment "${cmd}" "${desc}"
        done
      done
    done
  done
done

# ============================================================================
# 2. NXROGraphModel with Learned Adjacency
# ============================================================================

echo ""
echo "--- NXROGraphModel (Learned Adjacency) ---" | tee -a "${LOG_FILE}"

for graph_struct in "${GRAPH_STRUCTURES[@]}"; do
  for k in "${TOP_K_VALUES[@]}"; do
    for hdim in "${HIDDEN_DIMS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for rollout in "${ROLLOUT_K[@]}"; do
          for l1 in "${L1_LAMBDAS[@]}"; do
            
            # Build command
            cmd="python NXRO_train.py --model graph --epochs ${EPOCHS} --device ${DEVICE} ${TEST}"
            cmd="${cmd} --hidden_dim ${hdim} --lr ${lr} --rollout_k ${rollout}"
            cmd="${cmd} --graph_learned --graph_l1 ${l1}"
            
            if [ "${graph_struct}" != "xro" ]; then
              cmd="${cmd} --graph_stat_method ${graph_struct} --graph_stat_topk ${k}"
              desc="graph_learned_stat_${graph_struct}_k${k}_l1${l1}_h${hdim}_lr${lr}_r${rollout}"
            else
              desc="graph_learned_xro_l1${l1}_h${hdim}_lr${lr}_r${rollout}"
            fi
            
            run_experiment "${cmd}" "${desc}"
          done
        done
      done
    done
  done
done

# ============================================================================
# 3. NXROGraphPyGModel (GCN)
# ============================================================================

echo ""
echo "--- NXROGraphPyGModel (GCN) ---" | tee -a "${LOG_FILE}"

for graph_struct in "${GRAPH_STRUCTURES[@]}"; do
  for k in "${TOP_K_VALUES[@]}"; do
    for hdim in "${HIDDEN_DIMS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for rollout in "${ROLLOUT_K[@]}"; do
          
          # Build command
          cmd="python NXRO_train.py --model graph_pyg --epochs ${EPOCHS} --device ${DEVICE} ${TEST}"
          cmd="${cmd} --hidden_dim ${hdim} --lr ${lr} --rollout_k ${rollout} --top_k ${k}"
          
          if [ "${graph_struct}" != "xro" ]; then
            cmd="${cmd} --graph_stat_method ${graph_struct}"
            desc="graphpyg_gcn_${graph_struct}_k${k}_h${hdim}_lr${lr}_r${rollout}"
          else
            desc="graphpyg_gcn_k${k}_h${hdim}_lr${lr}_r${rollout}"
          fi
          
          run_experiment "${cmd}" "${desc}"
        done
      done
    done
  done
done

# ============================================================================
# 4. NXROGraphPyGModel (GAT)
# ============================================================================

echo ""
echo "--- NXROGraphPyGModel (GAT) ---" | tee -a "${LOG_FILE}"

for graph_struct in "${GRAPH_STRUCTURES[@]}"; do
  for k in "${TOP_K_VALUES[@]}"; do
    for hdim in "${HIDDEN_DIMS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for rollout in "${ROLLOUT_K[@]}"; do
          
          # Build command
          cmd="python NXRO_train.py --model graph_pyg --epochs ${EPOCHS} --device ${DEVICE} ${TEST}"
          cmd="${cmd} --hidden_dim ${hdim} --lr ${lr} --rollout_k ${rollout} --top_k ${k} --gat"
          
          if [ "${graph_struct}" != "xro" ]; then
            cmd="${cmd} --graph_stat_method ${graph_struct}"
            desc="graphpyg_gat_${graph_struct}_k${k}_h${hdim}_lr${lr}_r${rollout}"
          else
            desc="graphpyg_gat_k${k}_h${hdim}_lr${lr}_r${rollout}"
          fi
          
          run_experiment "${cmd}" "${desc}"
        done
      done
    done
  done
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=====================================================================" | tee -a "${LOG_FILE}"
echo "GRID SEARCH COMPLETE" | tee -a "${LOG_FILE}"
echo "=====================================================================" | tee -a "${LOG_FILE}"
echo "Total configurations: ${TOTAL_CONFIGS}" | tee -a "${LOG_FILE}"
echo "Successfully run: ${RUN_COUNT}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "=====================================================================" | tee -a "${LOG_FILE}"

# Extract and rank results
if [ "${DRY_RUN}" = false ]; then
  echo ""
  echo "Analyzing results..."
  python - << 'EOF'
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import re

results_dir = Path('results')
output_dir = Path('results/hyperparam_search')

# Find all graph model checkpoints
graph_ckpts = []
graph_ckpts.extend(results_dir.glob('graph/*/nxro_graph_*_best*.pt'))
graph_ckpts.extend(results_dir.glob('graphpyg/*/nxro_graphpyg_*_best*.pt'))

# Also check flat structure (legacy)
graph_ckpts.extend(results_dir.glob('nxro_graph_*_best*.pt'))
graph_ckpts.extend(results_dir.glob('nxro_graphpyg_*_best*.pt'))

results = []

for ckpt_path in graph_ckpts:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Extract test_loss (MSE) -> RMSE
        if 'test_loss' in ckpt:
            test_rmse = float(np.sqrt(ckpt['test_loss']))
        else:
            continue
        
        # Extract config from filename
        name = ckpt_path.stem
        
        results.append({
            'checkpoint': ckpt_path.name,
            'test_rmse': test_rmse,
            'path': str(ckpt_path),
        })
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        continue

if not results:
    print("No results found!")
    exit(0)

# Sort by RMSE
df = pd.DataFrame(results)
df = df.sort_values('test_rmse')

# Save
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / 'grid_search_results.csv'
df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to {csv_path}")

# Print top 10
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS (by test RMSE)")
print("="*80)
for i, (idx, row) in enumerate(df.head(10).iterrows()):
    print(f"\n{i+1}. {row['checkpoint']}")
    print(f"   Test RMSE: {row['test_rmse']:.4f}")
    print(f"   Path: {row['path']}")

# Save best
best_row = df.iloc[0]
with open(output_dir / 'best_config.txt', 'w') as f:
    f.write(f"Best configuration:\n")
    f.write(f"  Checkpoint: {best_row['checkpoint']}\n")
    f.write(f"  Test RMSE: {best_row['test_rmse']:.4f}\n")
    f.write(f"  Path: {best_row['path']}\n")

print("\n✓ Best config saved to results/hyperparam_search/best_config.txt")
EOF
fi

echo ""
echo "Done!"

