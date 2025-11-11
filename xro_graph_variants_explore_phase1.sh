#!/bin/bash
#
# Phase 1: Graph Topology Search for NXRO-Graph Optimization
#
# This script systematically explores different graph structures to find
# the optimal topology for beating NXRO-Linear (current rank 1).
#
# Total: 25 experiments
# Estimated time: 25-30 hours on single GPU
# Can be parallelized across multiple GPUs
#

set -euo pipefail

EPOCHS=1500
DEVICE=auto
BASE_DIR="results_out_of_sample"

echo "================================================================================"
echo "     PHASE 1: GRAPH TOPOLOGY SEARCH (25 Experiments)"
echo "================================================================================"
echo "Configuration:"
echo "  Dataset: ORAS5 only (single dataset)"
echo "  Train: 1979-01 to 2001-12"
echo "  Test: 2002-01 to 2022-12"
echo "  Epochs: ${EPOCHS}"
echo "  Device: ${DEVICE}"
echo "  Results: ${BASE_DIR}/"
echo "  Goal: Find topology that beats NXRO-Linear (Test RMSE 0.567 C)"
echo "================================================================================"
echo ""

# Part A: XRO-derived topology with GCN (5 runs)
echo "================================================================================"
echo "Part A: XRO-Derived Topology with GCN (5 runs)"
echo "================================================================================"
echo "Testing sparsity levels: k=1,2,3,5,7"
echo ""

for k in 1 2 3 5 7; do
  echo "[A.$k] XRO topology, GCN, k=$k"
  python NXRO_train_out_of_sample.py \
    --model graph_pyg \
    --top_k $k \
    --epochs $EPOCHS \
    --device $DEVICE
  echo ""
done

echo "Part A complete: 5/25 experiments done"
echo ""

# Part B: XRO-derived topology with GAT (5 runs)
echo "================================================================================"
echo "Part B: XRO-Derived Topology with GAT (5 runs)"
echo "================================================================================"
echo "Testing sparsity levels: k=1,2,3,5,7"
echo ""

for k in 1 2 3 5 7; do
  echo "[B.$k] XRO topology, GAT, k=$k"
  python NXRO_train_out_of_sample.py \
    --model graph_pyg \
    --gat \
    --top_k $k \
    --epochs $EPOCHS \
    --device $DEVICE
  echo ""
done

echo "Part B complete: 10/25 experiments done"
echo ""

# Part C: Statistical topologies with GCN (4 runs)
echo "================================================================================"
echo "Part C: Statistical Topologies with GCN (4 runs)"
echo "================================================================================"
echo "Testing methods: Pearson, Spearman, MI, XCorr-Max (all with k=3)"
echo ""

METHODS=(pearson spearman mi xcorr_max)
for method in "${METHODS[@]}"; do
  echo "[C.$method] $method topology, GCN, k=3"
  python NXRO_train_out_of_sample.py \
    --model graph_pyg \
    --graph_stat_method $method \
    --graph_stat_topk 3 \
    --epochs $EPOCHS \
    --device $DEVICE
  echo ""
done

echo "Part C complete: 14/25 experiments done"
echo ""

# Part D: Statistical topologies with GAT (4 runs)
echo "================================================================================"
echo "Part D: Statistical Topologies with GAT (4 runs)"
echo "================================================================================"
echo "Testing methods: Pearson, Spearman, MI, XCorr-Max (all with k=3)"
echo ""

for method in "${METHODS[@]}"; do
  echo "[D.$method] $method topology, GAT, k=3"
  python NXRO_train_out_of_sample.py \
    --model graph_pyg \
    --gat \
    --graph_stat_method $method \
    --graph_stat_topk 3 \
    --epochs $EPOCHS \
    --device $DEVICE
  echo ""
done

echo "Part D complete: 18/25 experiments done"
echo ""

# Part E: Best statistical method with varied k (assume MI, 7 runs)
# Note: You may want to check results from C/D first and adjust this
echo "================================================================================"
echo "Part E: Best Statistical Method (MI assumed) with Varied Sparsity (7 runs)"
echo "================================================================================"
echo "Testing MI with k=1,2,5,7,10 using GAT"
echo ""

for k in 1 2 5 7 10; do
  echo "[E.$k] MI topology, GAT, k=$k"
  python NXRO_train_out_of_sample.py \
    --model graph_pyg \
    --gat \
    --graph_stat_method mi \
    --graph_stat_topk $k \
    --epochs $EPOCHS \
    --device $DEVICE
  echo ""
done

echo "Part E complete: 25/25 experiments done"
echo ""

# Ranking
echo "================================================================================"
echo "Ranking all models..."
echo "================================================================================"
python rank_all_variants_out_of_sample.py --top_n 5 --metric rmse --force

echo ""
echo "================================================================================"
echo "PHASE 1 COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: ${BASE_DIR}/"
echo "Rankings: ${BASE_DIR}/rankings/"
echo ""
echo "Next steps:"
echo "  1. Review rankings: cat ${BASE_DIR}/rankings/all_variants_ranked_rmse_all_outsample.csv"
echo "  2. Visualize results: python visualize_phase1_results.py"
echo "  3. Identify BEST_TOPOLOGY, BEST_K, BEST_GNN"
echo "  4. Proceed to Phase 2 with best configuration"
echo ""
echo "================================================================================"

