#!/bin/bash

set -euo pipefail

# Usage: bash run_all_nxro_ora5.sh [--stochastic] [--members 100] [--rollout_k 1] [--epochs 2000] [--device auto] [--test] [--topk 3]
# Note: This script trains on ORAS5 only (no extra training NetCDFs).
#
# NOTE: This script runs BASE VARIANTS ONLY (random initialization).
# For warm-start variants and freezing ablations, see run_all_warmstart_variants.sh
# which runs all 32 variants from the README summary table.

STOCH=""
MEMBERS=100
ROLLOUT=1
EPOCHS=50
DEVICE=auto
TEST=""
TOPK=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stochastic)
      STOCH="--stochastic"; shift ;;
    --members)
      MEMBERS="$2"; shift 2 ;;
    --rollout_k)
      ROLLOUT="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --test)
      TEST="--test"; shift ;;
    --topk)
      TOPK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

common="--epochs ${EPOCHS} --rollout_k ${ROLLOUT} --device ${DEVICE} ${STOCH} ${TEST} --members ${MEMBERS}"

echo "Running NXRO (ORAS5-only) variants with: ${common}"

python NXRO_train.py --model linear ${common}
python NXRO_train.py --model ro ${common}
python NXRO_train.py --model rodiag ${common}
python NXRO_train.py --model res ${common}
python NXRO_train.py --model neural ${common}
python NXRO_train.py --model neural_phys ${common}
python NXRO_train.py --model resmix ${common}
python NXRO_train.py --model bilinear ${common}
python NXRO_train.py --model attentive ${common}

# Graph ODE variants
# 1) Fixed XRO-based graph (default)
python NXRO_train.py --model graph ${common}
# 2) Fixed statistical KNN priors
python NXRO_train.py --model graph ${common} --graph_stat_method pearson --graph_stat_topk ${TOPK}
python NXRO_train.py --model graph ${common} --graph_stat_method spearman --graph_stat_topk ${TOPK}
python NXRO_train.py --model graph ${common} --graph_stat_method mi --graph_stat_topk ${TOPK}
python NXRO_train.py --model graph ${common} --graph_stat_method xcorr_max --graph_stat_topk ${TOPK}
# 3) Learned adjacency with L1 sparsity (initialized from prior)
python NXRO_train.py --model graph ${common} --graph_learned --graph_l1 1e-3
python NXRO_train.py --model graph ${common} --graph_learned --graph_l1 1e-3 --graph_stat_method pearson --graph_stat_topk 3

# PyG Graph ODE (GCN/GAT) with k-NN graph, for all priors
# XRO prior
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK}
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --gat
# Statistical priors
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method pearson
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method pearson --gat
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method spearman
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method spearman --gat
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method mi
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method mi --gat
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method xcorr_max
python NXRO_train.py --model graph_pyg ${common} --top_k ${TOPK} --graph_stat_method xcorr_max --gat

echo "Done."


