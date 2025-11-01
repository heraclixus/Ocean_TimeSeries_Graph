#!/bin/bash

# Quick Grid Search for Graph Models (Focused Hyperparameter Space)
# Uses a smaller, carefully selected grid for faster iteration
#
# Usage:
#   bash run_graph_quick_search.sh [--epochs 1000] [--device auto] [--test]

set -euo pipefail

EPOCHS=1000
DEVICE=auto
TEST=""
OUTPUT_DIR="results/hyperparam_search_quick"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --test) TEST="--test"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/quick_search.log"

echo "=====================================================================" | tee "${LOG_FILE}"
echo "GRAPH MODEL QUICK HYPERPARAMETER SEARCH" | tee -a "${LOG_FILE}"
echo "=====================================================================" | tee -a "${LOG_FILE}"

# Focused grid (based on typical good values)
declare -a GRAPH_STRUCTURES=("xro" "pearson" "spearman")
declare -a TOP_K_VALUES=(3 5 7)
declare -a HIDDEN_DIMS=(64 128)
declare -a LEARNING_RATES=(5e-4 1e-3)
declare -a ROLLOUT_K=(1 2)
declare -a L1_LAMBDAS=(1e-3 5e-3)

TOTAL=0
SUCCESS=0

run_exp() {
  local cmd="$1"
  local desc="$2"
  ((TOTAL++))
  echo ""
  echo "[${TOTAL}] ${desc}" | tee -a "${LOG_FILE}"
  echo "${cmd}" | tee -a "${LOG_FILE}"
  if eval "${cmd}" >> "${LOG_FILE}" 2>&1; then
    echo "  ✓" | tee -a "${LOG_FILE}"
    ((SUCCESS++))
  else
    echo "  ✗" | tee -a "${LOG_FILE}"
  fi
}

common="--epochs ${EPOCHS} --device ${DEVICE} ${TEST}"

# 1. Fixed graph models
for struct in "${GRAPH_STRUCTURES[@]}"; do
  for k in "${TOP_K_VALUES[@]}"; do
    for h in "${HIDDEN_DIMS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for r in "${ROLLOUT_K[@]}"; do
          if [ "${struct}" = "xro" ]; then
            run_exp "python NXRO_train.py --model graph ${common} --hidden_dim ${h} --lr ${lr} --rollout_k ${r}" \
                    "graph_fixed_xro_h${h}_lr${lr}_r${r}"
          else
            run_exp "python NXRO_train.py --model graph ${common} --hidden_dim ${h} --lr ${lr} --rollout_k ${r} --graph_stat_method ${struct} --graph_stat_topk ${k}" \
                    "graph_fixed_${struct}_k${k}_h${h}_lr${lr}_r${r}"
          fi
        done
      done
    done
  done
done

# 2. Learned graph models (only Pearson prior)
for k in "${TOP_K_VALUES[@]}"; do
  for h in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for r in "${ROLLOUT_K[@]}"; do
        for l1 in "${L1_LAMBDAS[@]}"; do
          run_exp "python NXRO_train.py --model graph ${common} --hidden_dim ${h} --lr ${lr} --rollout_k ${r} --graph_learned --graph_l1 ${l1} --graph_stat_method pearson --graph_stat_topk ${k}" \
                  "graph_learned_pearson_k${k}_l1${l1}_h${h}_lr${lr}_r${r}"
        done
      done
    done
  done
done

# 3. PyG GCN
for struct in "${GRAPH_STRUCTURES[@]}"; do
  for k in "${TOP_K_VALUES[@]}"; do
    for h in "${HIDDEN_DIMS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for r in "${ROLLOUT_K[@]}"; do
          if [ "${struct}" = "xro" ]; then
            run_exp "python NXRO_train.py --model graph_pyg ${common} --hidden_dim ${h} --lr ${lr} --rollout_k ${r} --top_k ${k}" \
                    "graphpyg_gcn_k${k}_h${h}_lr${lr}_r${r}"
          else
            run_exp "python NXRO_train.py --model graph_pyg ${common} --hidden_dim ${h} --lr ${lr} --rollout_k ${r} --top_k ${k} --graph_stat_method ${struct}" \
                    "graphpyg_gcn_${struct}_k${k}_h${h}_lr${lr}_r${r}"
          fi
        done
      done
    done
  done
done

# 4. PyG GAT
for struct in "${GRAPH_STRUCTURES[@]}"; do
  for k in "${TOP_K_VALUES[@]}"; do
    for h in "${HIDDEN_DIMS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for r in "${ROLLOUT_K[@]}"; do
          if [ "${struct}" = "xro" ]; then
            run_exp "python NXRO_train.py --model graph_pyg ${common} --hidden_dim ${h} --lr ${lr} --rollout_k ${r} --top_k ${k} --gat" \
                    "graphpyg_gat_k${k}_h${h}_lr${lr}_r${r}"
          else
            run_exp "python NXRO_train.py --model graph_pyg ${common} --hidden_dim ${h} --lr ${lr} --rollout_k ${r} --top_k ${k} --gat --graph_stat_method ${struct}" \
                    "graphpyg_gat_${struct}_k${k}_h${h}_lr${lr}_r${r}"
          fi
        done
      done
    done
  done
done

echo ""
echo "=====================================================================" | tee -a "${LOG_FILE}"
echo "SEARCH COMPLETE: ${SUCCESS}/${TOTAL} succeeded" | tee -a "${LOG_FILE}"
echo "=====================================================================" | tee -a "${LOG_FILE}"

# Analyze results
python - << 'EOF'
import numpy as np
import pandas as pd
import torch
from pathlib import Path

results_dir = Path('results')
output_dir = Path('results/hyperparam_search_quick')

ckpts = []
ckpts.extend(results_dir.glob('graph/*/nxro_graph_*_best*.pt'))
ckpts.extend(results_dir.glob('graphpyg/*/nxro_graphpyg_*_best*.pt'))
ckpts.extend(results_dir.glob('nxro_graph_*_best*.pt'))
ckpts.extend(results_dir.glob('nxro_graphpyg_*_best*.pt'))

results = []
for p in ckpts:
    try:
        ckpt = torch.load(p, map_location='cpu')
        if 'test_loss' in ckpt:
            rmse = float(np.sqrt(ckpt['test_loss']))
            results.append({'checkpoint': p.name, 'test_rmse': rmse, 'path': str(p)})
    except:
        pass

if results:
    df = pd.DataFrame(results).sort_values('test_rmse')
    df.to_csv(output_dir / 'results.csv', index=False)
    print("\n" + "="*80)
    print("TOP 10 MODELS")
    print("="*80)
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        print(f"{i+1}. {row['checkpoint']:60s} RMSE={row['test_rmse']:.4f}")
    print(f"\n✓ Results: {output_dir / 'results.csv'}")
else:
    print("No results found.")
EOF

echo "Done!"

