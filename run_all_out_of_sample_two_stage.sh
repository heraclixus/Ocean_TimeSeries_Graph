#!/bin/bash

set -euo pipefail

# Master script to run Two-Stage NXRO Training (Synthetic Pre-train -> ORAS5 Fine-tune)
# for the top performing variants and full ODE models.
#
# Usage: bash run_all_out_of_sample_two_stage.sh [OPTIONS]
#
# Options:
#   --epochs N           Number of epochs per stage (default: 1500)
#   --device DEVICE      Device to use: cpu, cuda, mps, auto (default: auto)
#   --eval_all_datasets  Evaluate on all available datasets (ORAS5, ERA5, GODAS, etc.)

EPOCHS=1500
DEVICE=auto
EVAL_ALL_DATASETS=""
XRO_FIT_FILE="results_out_of_sample/xro_fit_warmstart.nc"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --eval_all_datasets) EVAL_ALL_DATASETS="--eval_all_datasets"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Update XRO fit file path if eval_all_datasets is enabled
if [[ -n "$EVAL_ALL_DATASETS" ]]; then
    XRO_FIT_FILE="results_all_outsample/xro_fit_warmstart.nc"
fi

echo "================================================================================"
echo "      TWO-STAGE OUT-OF-SAMPLE EXPERIMENT (Synthetic -> ORAS5)"
echo "================================================================================"
echo "Configuration:"
echo "  Train period: 1979-01 to 2001-12 (23 years)"
echo "  Test period: 2002-01 to 2022-12 (21 years)"
echo "  Epochs per stage: ${EPOCHS}"
echo "  Device: ${DEVICE}"
echo "  Eval all datasets: ${EVAL_ALL_DATASETS:-disabled}"
echo "  XRO fit file (for frozen/warm-start): ${XRO_FIT_FILE}"
echo "  Synthetic Data: Auto-detected (data/XRO_indices_*_preproc.nc)"
echo "================================================================================"
echo ""

common_args="--epochs ${EPOCHS} --device ${DEVICE} ${EVAL_ALL_DATASETS} --two_stage --extra_train_nc auto"

# ============================================================================
# STEP 0: Fit XRO for Warm-Start (if not exists)
# ============================================================================
if [[ ! -f "$XRO_FIT_FILE" ]]; then
    echo "STEP 0: Fitting baseline XRO model on train period (1979-2001)..."
    echo "------------------------------------------------------------"
    python -c "
import xarray as xr
from XRO.core import XRO
import os

# Load data - use TRAIN period only for fitting
obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
train_ds = obs_ds.sel(time=slice('1979-01', '2001-12'))

# Fit XRO (ac_order=2, with RO and diagonal terms)
xro_model = XRO(ncycle=12, ac_order=2)
xro_fit = xro_model.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])

# Save fit
import os
os.makedirs(os.path.dirname('${XRO_FIT_FILE}'), exist_ok=True)
xro_fit.to_netcdf('${XRO_FIT_FILE}')
print(f'✓ XRO fit saved to ${XRO_FIT_FILE}')
"
    echo ""
else
    echo "STEP 0: Found existing XRO fit file at ${XRO_FIT_FILE}"
    echo ""
fi

# ============================================================================
# RUN 12 MODELS (Two-Stage)
# ============================================================================

echo "[1/12] NXRO-Res (Rank 1)"
python NXRO_train_out_of_sample.py --model res ${common_args}
echo ""

echo "[2/12] NXRO-Graph (Fixed XRO) (Rank 2)"
python NXRO_train_out_of_sample.py --model graph ${common_args}
echo ""

echo "[3/12] NXRO-Attentive (Rank 3)"
python NXRO_train_out_of_sample.py --model attentive ${common_args}
echo ""

echo "[4/12] NXRO-RO+Diag (Rank 4)"
python NXRO_train_out_of_sample.py --model rodiag ${common_args}
echo ""

echo "[5/12] NXRO-Linear (Rank 5)"
python NXRO_train_out_of_sample.py --model linear ${common_args}
echo ""

echo "[6/12] NXRO-NeuralODE (Rank 6)"
python NXRO_train_out_of_sample.py --model neural ${common_args}
echo ""

echo "[7/12] NXRO-RO+Diag-FixNL (Rank 8)"
python NXRO_train_out_of_sample.py --model rodiag ${common_args} --warm_start ${XRO_FIT_FILE} --freeze ro,diag
echo ""

echo "[8/12] NXRO-RO+Diag-FixRO (Rank 9)"
python NXRO_train_out_of_sample.py --model rodiag ${common_args} --warm_start ${XRO_FIT_FILE} --freeze ro
echo ""

echo "[9/12] NXRO-ResidualMix-FixRO (Rank 10)"
python NXRO_train_out_of_sample.py --model resmix ${common_args} --warm_start ${XRO_FIT_FILE} --freeze ro
echo ""

echo "[10/12] NXRO-Graph (Learned Adjacency)"
python NXRO_train_out_of_sample.py --model graph ${common_args} --graph_learned --graph_l1 0.01
echo ""

echo "[11/12] NXRO-GraphPyG (GAT)"
python NXRO_train_out_of_sample.py --model graph_pyg ${common_args} --gat
echo ""

echo "[12/12] NXRO-PhysReg (NeuralODE + Physics Reg)"
python NXRO_train_out_of_sample.py --model neural_phys ${common_args} --jac_reg 1e-4
echo ""

# ============================================================================
# RANKING (Two-Stage)
# ============================================================================
echo ""
echo "Ranking all two-stage models (out-of-sample)..."
python rank_all_variants_out_of_sample.py \
    --top_n 12 \
    --metric combined \
    --two_stage \
    --force

# ============================================================================
# COMPARISON
# ============================================================================
echo ""
echo "Generating comparison between Single-Stage and Two-Stage models..."
python compare_single_vs_two_stage.py --metric combined

# ============================================================================
# SUMMARY
# ============================================================================
echo "================================================================================"
echo "                    ✓ TWO-STAGE EXPERIMENT COMPLETE!"
echo "================================================================================"
echo "Results saved to: results_out_of_sample/ (or results_all_outsample/)"
echo ""
