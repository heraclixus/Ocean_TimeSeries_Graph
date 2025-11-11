#!/bin/bash

set -euo pipefail

# Master script to run ALL 32 NXRO variants with out-of-sample setup
# Train: 1979-2001, Test: 2002-2022
# Results saved to results_out_of_sample/
#
# Usage: bash run_all_out_of_sample.sh [OPTIONS]
#
# Options:
#   --epochs N           Number of epochs for base variants (default: 1500)
#   --warmup_epochs M    Number of epochs for warm-start variants (default: 100)
#   --device DEVICE      Device to use: cpu, cuda, mps, auto (default: auto)
#   --skip_base          Skip base variants (random initialization)
#   --skip_warmstart     Skip warm-start and freezing ablations
#   --eval_all_datasets  Evaluate on all available datasets (ORAS5, ERA5, GODAS, etc.)
#
EPOCHS=1500
WARMUP_EPOCHS=100
DEVICE=auto
SKIP_BASE=false
SKIP_WARMSTART=false
EVAL_ALL_DATASETS=""
XRO_FIT_FILE="results_out_of_sample/xro_fit_warmstart.nc"  # Will be overridden if eval_all_datasets is set

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --warmup_epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --skip_base) SKIP_BASE=true; shift ;;
    --skip_warmstart) SKIP_WARMSTART=true; shift ;;
    --eval_all_datasets) EVAL_ALL_DATASETS="--eval_all_datasets"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Update XRO fit file path if eval_all_datasets is enabled
if [[ -n "$EVAL_ALL_DATASETS" ]]; then
    XRO_FIT_FILE="results_all_outsample/xro_fit_warmstart.nc"
fi

echo "================================================================================"
echo "           OUT-OF-SAMPLE EXPERIMENT: ALL 32 NXRO VARIANTS"
echo "================================================================================"
echo "Configuration:"
echo "  Train period: 1979-01 to 2001-12 (23 years)"
echo "  Test period: 2002-01 to 2022-12 (21 years)"
echo "  Epochs (base variants): ${EPOCHS}"
echo "  Epochs (warm-start variants): ${WARMUP_EPOCHS}"
echo "  Device: ${DEVICE}"
echo "  Skip base: ${SKIP_BASE}"
echo "  Skip warmstart: ${SKIP_WARMSTART}"
echo "  Eval all datasets: ${EVAL_ALL_DATASETS:-disabled}"
echo "  XRO fit file: ${XRO_FIT_FILE}"
echo ""
echo "This will train 30 variants (2 FixAll variants use XRO directly)."
echo "  - 9 base variants: ${EPOCHS} epochs each (random initialization)"
echo "  - 21 warm-start variants: ${WARMUP_EPOCHS} epochs each (good initialization)"
echo "================================================================================"
echo ""

common_base="--epochs ${EPOCHS} --device ${DEVICE} ${EVAL_ALL_DATASETS}"
common_warmstart="--epochs ${WARMUP_EPOCHS} --device ${DEVICE} ${EVAL_ALL_DATASETS}"

# ============================================================================
# STEP 0: Fit XRO for Warm-Start (on training period only: 1979-2001)
# ============================================================================
if [[ "$SKIP_WARMSTART" == "false" ]]; then
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
print(f'  Lcoef shape: {xro_fit[\"Lcoef\"].shape}')
print(f'  NROT_Lcoef available: {\"NROT_Lcoef\" in xro_fit}')
print(f'  NLb_Lcoef available: {\"NLb_Lcoef\" in xro_fit}')
"
    echo ""
fi

# ============================================================================
# GROUP 1: BASE VARIANTS (Random Initialization) - 9 variants
# ============================================================================
if [[ "$SKIP_BASE" == "false" ]]; then
    echo "================================================================================"
    echo "GROUP 1: BASE VARIANTS (Random Initialization) - 9 variants"
    echo "================================================================================"
    echo ""

    echo "[1/9] Variant 1: NXRO-Linear (random)"
    python NXRO_train_out_of_sample.py --model linear ${common_base}
    echo ""

    echo "[2/9] Variant 2: NXRO-RO (random)"
    python NXRO_train_out_of_sample.py --model ro ${common_base}
    echo ""

    echo "[3/9] Variant 3: NXRO-RO+Diag (random)"
    python NXRO_train_out_of_sample.py --model rodiag ${common_base}
    echo ""

    echo "[4/9] Variant 4: NXRO-Res (random)"
    python NXRO_train_out_of_sample.py --model res ${common_base}
    echo ""

    echo "[5/9] Variant 5: NXRO-NeuralODE (random)"
    python NXRO_train_out_of_sample.py --model neural ${common_base}
    echo ""

    echo "[6/9] Variant 5a: NXRO-Attentive (random)"
    python NXRO_train_out_of_sample.py --model attentive ${common_base}
    echo ""

    echo "[7/9] Variant 5b: NXRO-Graph (random)"
    python NXRO_train_out_of_sample.py --model graph ${common_base}
    echo ""

    echo "[8/9] Variant 5c: NXRO-PhysReg (random)"
    python NXRO_train_out_of_sample.py --model neural_phys ${common_base}
    echo ""

    echo "[9/9] Variant 5d: NXRO-ResidualMix (random)"
    python NXRO_train_out_of_sample.py --model resmix ${common_base}
    echo ""
    
    echo "✓ Group 1 complete: 9 base variants trained"
    echo ""
fi

# ============================================================================
# GROUP 2: WARM-START VARIANTS (No Freezing) - 6 variants
# ============================================================================
if [[ "$SKIP_WARMSTART" == "false" ]]; then
    echo "================================================================================"
    echo "GROUP 2: WARM-START VARIANTS (Train All Parameters) - 6 variants"
    echo "================================================================================"
    echo ""

    echo "[1/6] Variant 1a: NXRO-Linear-WS"
    python NXRO_train_out_of_sample.py --model linear ${common_warmstart} --warm_start ${XRO_FIT_FILE}
    echo ""

    echo "[2/6] Variant 2a: NXRO-RO-WS"
    python NXRO_train_out_of_sample.py --model ro ${common_warmstart} --warm_start ${XRO_FIT_FILE}
    echo ""

    echo "[3/6] Variant 3a: NXRO-RO+Diag-WS"
    python NXRO_train_out_of_sample.py --model rodiag ${common_warmstart} --warm_start ${XRO_FIT_FILE}
    echo ""

    echo "[4/6] Variant 5a-WS: NXRO-Attentive-WS"
    python NXRO_train_out_of_sample.py --model attentive ${common_warmstart} --warm_start ${XRO_FIT_FILE}
    echo ""

    echo "[5/6] Variant 5b-WS: NXRO-Graph-WS"
    python NXRO_train_out_of_sample.py --model graph ${common_warmstart} --warm_start ${XRO_FIT_FILE}
    echo ""

    echo "[6/6] Variant 5d-WS: NXRO-ResidualMix-WS"
    python NXRO_train_out_of_sample.py --model resmix ${common_warmstart} --warm_start ${XRO_FIT_FILE}
    echo ""
    
    echo "✓ Group 2 complete: 6 warm-start variants trained"
    echo ""

    # ============================================================================
    # GROUP 3: VARIANT 2 FREEZING ABLATIONS - 2 variants
    # ============================================================================
    echo "================================================================================"
    echo "GROUP 3: VARIANT 2 FREEZING ABLATIONS - 2 variants"
    echo "================================================================================"
    echo ""

    echo "[1/2] Variant 2a-FixL: Freeze linear, train RO"
    python NXRO_train_out_of_sample.py --model ro ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze linear
    echo ""

    echo "[2/2] Variant 2a-FixRO: Freeze RO, train linear"
    python NXRO_train_out_of_sample.py --model ro ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze ro
    echo ""
    
    echo "✓ Group 3 complete: 2 variant-2 freezing ablations trained"
    echo ""

    # ============================================================================
    # GROUP 4: VARIANT 3 FREEZING ABLATIONS - 4 variants
    # ============================================================================
    echo "================================================================================"
    echo "GROUP 4: VARIANT 3 FREEZING ABLATIONS - 4 variants"
    echo "================================================================================"
    echo ""

    echo "[1/4] Variant 3a-FixL: Freeze linear, train RO+Diag"
    python NXRO_train_out_of_sample.py --model rodiag ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze linear
    echo ""

    echo "[2/4] Variant 3a-FixRO: Freeze RO, train linear+Diag"
    python NXRO_train_out_of_sample.py --model rodiag ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze ro
    echo ""

    echo "[3/4] Variant 3a-FixDiag: Freeze Diag, train linear+RO"
    python NXRO_train_out_of_sample.py --model rodiag ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze diag
    echo ""

    echo "[4/4] Variant 3a-FixNL: Freeze RO+Diag, train linear only"
    python NXRO_train_out_of_sample.py --model rodiag ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze ro,diag
    echo ""
    
    echo "✓ Group 4 complete: 4 variant-3 freezing ablations trained"
    echo ""

    # ============================================================================
    # GROUP 5: VARIANT 4 WARM-START - 2 variants
    # ============================================================================
    echo "================================================================================"
    echo "GROUP 5: VARIANT 4 WARM-START - 2 variants"
    echo "================================================================================"
    echo ""

    echo "[1/2] Variant 4a: NXRO-Res-WS-FixL (frozen XRO linear + trainable MLP)"
    python NXRO_train_out_of_sample.py --model res ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze linear
    echo ""

    echo "[2/2] Variant 4b: NXRO-Res-FullXRO (frozen full XRO + trainable MLP)"
    python NXRO_train_out_of_sample.py --model res_fullxro ${common_warmstart} --warm_start ${XRO_FIT_FILE}
    echo ""
    
    echo "✓ Group 5 complete: 2 variant-4 warm-start variants trained"
    echo ""

    # ============================================================================
    # GROUP 6: VARIANT 5a/5b FREEZING - 2 variants
    # ============================================================================
    echo "================================================================================"
    echo "GROUP 6: VARIANT 5a/5b FREEZING - 2 variants"
    echo "================================================================================"
    echo ""

    echo "[1/2] Variant 5a-FixL: NXRO-Attentive with frozen linear"
    python NXRO_train_out_of_sample.py --model attentive ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze linear
    echo ""

    echo "[2/2] Variant 5b-FixL: NXRO-Graph with frozen linear"
    python NXRO_train_out_of_sample.py --model graph ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze linear
    echo ""
    
    echo "✓ Group 6 complete: 2 intermediate variant freezing ablations trained"
    echo ""

    # ============================================================================
    # GROUP 7: VARIANT 5d FREEZING ABLATIONS - 5 variants
    # ============================================================================
    echo "================================================================================"
    echo "GROUP 7: VARIANT 5d FREEZING ABLATIONS - 5 variants"
    echo "================================================================================"
    echo ""

    echo "[1/5] Variant 5d-FixL: Freeze linear, train RO+Diag+MLP"
    python NXRO_train_out_of_sample.py --model resmix ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze linear
    echo ""

    echo "[2/5] Variant 5d-FixRO: Freeze RO, train linear+Diag+MLP"
    python NXRO_train_out_of_sample.py --model resmix ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze ro
    echo ""

    echo "[3/5] Variant 5d-FixDiag: Freeze Diag, train linear+RO+MLP"
    python NXRO_train_out_of_sample.py --model resmix ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze diag
    echo ""

    echo "[4/5] Variant 5d-FixNL: Freeze RO+Diag, train linear+MLP"
    python NXRO_train_out_of_sample.py --model resmix ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze ro,diag
    echo ""

    echo "[5/5] Variant 5d-FixPhysics: Freeze all physics (L+RO+Diag), train MLP only"
    python NXRO_train_out_of_sample.py --model resmix ${common_warmstart} --warm_start ${XRO_FIT_FILE} --freeze linear,ro,diag
    echo ""
    
    echo "✓ Group 7 complete: 5 variant-5d freezing ablations trained"
    echo ""
fi

# ============================================================================
# RANKING
# ============================================================================
echo "Ranking all variants (out-of-sample)..."
python rank_all_variants_out_of_sample.py \
    --top_n 10 \
    --metric combined \
    --force

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "                         COMPLETE SUMMARY"
echo "================================================================================"
echo ""
echo "Variants Trained (Out-of-Sample Setup):"
if [[ "$SKIP_BASE" == "false" ]]; then
    echo "  ✓ Group 1 (Base/Random): 9 variants"
else
    echo "  ⊘ Group 1 (Base/Random): Skipped"
fi

if [[ "$SKIP_WARMSTART" == "false" ]]; then
    echo "  ✓ Group 2 (Warm-start): 6 variants"
    echo "  ✓ Group 3 (Variant 2 freezing): 2 variants"
    echo "  ✓ Group 4 (Variant 3 freezing): 4 variants"
    echo "  ✓ Group 5 (Variant 4 warm-start): 2 variants"
    echo "  ✓ Group 6 (Variant 5a/5b freezing): 2 variants"
    echo "  ✓ Group 7 (Variant 5d freezing): 5 variants"
else
    echo "  ⊘ Groups 2-7 (Warm-start): Skipped"
fi

echo ""
echo "Total Variants: 32 (30 trained + 2 pure XRO)"
echo ""
echo "Results saved to: results_out_of_sample/"
echo "Rankings saved to: results_out_of_sample/rankings/"
echo ""
echo "To view rankings:"
echo "  cat results_out_of_sample/rankings/all_variants_ranked_combined_out_of_sample.csv"
echo ""
echo "================================================================================"
echo "                    ✓ OUT-OF-SAMPLE EXPERIMENT COMPLETE!"
echo "================================================================================"

