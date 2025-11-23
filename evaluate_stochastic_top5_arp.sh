#!/bin/bash
#
# Evaluate Stochastic Ensembles for Top NXRO Models with AR(p) Noise
#
# Usage:
#   ./evaluate_stochastic_top5_arp.sh --p 2
#   ./evaluate_stochastic_top5_arp.sh --p 3 --members 100
#

set -euo pipefail

# Ensure conda environment is active
# source $(conda info --base)/etc/profile.d/conda.sh || true
# conda activate graph || true

MEMBERS=100
DEVICE=auto
AR_P=1
BASE_DIR="results_out_of_sample"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --members) MEMBERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --p) AR_P="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "================================================================================"
echo "STOCHASTIC ENSEMBLE EVALUATION (AR-$AR_P): Top NXRO Models"
echo "================================================================================"
echo "Configuration:"
echo "  Dataset: ORAS5 only"
echo "  Train: 1979-01 to 2001-12"
echo "  Test: 2002-01 to 2022-12"
echo "  Ensemble members: ${MEMBERS}"
echo "  Device: ${DEVICE}"
echo "  Noise Model: Seasonal AR(${AR_P}) on model residuals"
echo "  Results: ${BASE_DIR}/"
echo "================================================================================"
echo ""

# Note: XRO Baseline in current codebase is fixed to AR(1). 
# We use NXRO-Linear as the linear baseline for AR(p) comparison.

export MEMBERS

# Top NXRO models
echo "================================================================================"
echo "[1/4] NXRO-Res (Rank 1)"
echo "================================================================================"
python NXRO_train_out_of_sample.py --model res \
  --stochastic --members $MEMBERS --device $DEVICE --ar_p $AR_P
echo ""

echo "================================================================================"
echo "[2/4] NXRO-Graph Fixed XRO (Rank 2)"
echo "================================================================================"
echo "Using default XRO topology with k=3, GCN"
python NXRO_train_out_of_sample.py --model graph_pyg --top_k 3 \
  --stochastic --members $MEMBERS --device $DEVICE --ar_p $AR_P
echo ""

echo "================================================================================"
echo "[3/4] NXRO-Attentive (Rank 3)"
echo "================================================================================"
python NXRO_train_out_of_sample.py --model attentive \
  --stochastic --members $MEMBERS --device $DEVICE --ar_p $AR_P
echo ""

echo "================================================================================"
echo "[4/4] NXRO-Linear (Rank 5 / Linear Baseline)"
echo "================================================================================"
python NXRO_train_out_of_sample.py --model linear \
  --stochastic --members $MEMBERS --device $DEVICE --ar_p $AR_P
echo ""

echo "================================================================================"
echo "STOCHASTIC AR(${AR_P}) EVALUATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: ${BASE_DIR}/{model}/*_stochastic_arp${AR_P}_*"
echo ""

