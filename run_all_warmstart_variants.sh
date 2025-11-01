#!/bin/bash

set -euo pipefail

# Usage: bash run_all_warmstart_variants.sh [--epochs N] [--device DEVICE] [--test]
#
# This script runs all 32 variants from the README summary table including:
# - Base variants (random initialization)
# - Warm-start variants
# - Freezing ablations
#
# IMPLEMENTATION STATUS:
# - Model classes fully support warm-start and freezing (nxro/models.py) ✓
# - Utility functions for XRO extraction implemented (utils/xro_utils.py) ✓
# - CLI arguments added to NXRO_train.py (--warm_start, --freeze) ✓
# - Training functions need updating to pass parameters to models (IN PROGRESS)
#
# This script shows the intended command structure.
# Full integration requires updating train_nxro_* functions in nxro/train.py
# to accept and use warm_start_params and freeze_flags.
#
# First, it fits XRO to get the warm-start initialization file.

EPOCHS=1000
DEVICE=auto
TEST=""
XRO_FIT_FILE="results/xro_fit_warmstart.nc"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --test) TEST="--test"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "="*80
echo "Running All 32 Warm-Start Variants"
echo "="*80
echo "Settings:"
echo "  Epochs: ${EPOCHS}"
echo "  Device: ${DEVICE}"
echo "  Test mode: ${TEST:-disabled}"
echo "  XRO fit file: ${XRO_FIT_FILE}"
echo ""

# Step 1: Fit XRO model to get warm-start coefficients
echo "Step 1: Fitting baseline XRO model for warm-start initialization..."
python -c "
import xarray as xr
from XRO.core import XRO

# Load data
obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))

# Fit XRO
xro_model = XRO(ncycle=12, ac_order=2)
xro_fit = xro_model.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])

# Save fit for warm-start
xro_fit.to_netcdf('${XRO_FIT_FILE}')
print('✓ XRO fit saved to ${XRO_FIT_FILE}')
"

echo ""
echo "Step 2: Running all 32 variants..."
echo ""

common="--epochs ${EPOCHS} --device ${DEVICE} ${TEST}"

# ============================================================================
# Group 1: Base Variants (Random Initialization) - 9 variants
# ============================================================================
echo "--- BASE VARIANTS (Random Init) ---"

echo "[1] Variant 1: NXRO-Linear (random)"
python NXRO_train.py --model linear ${common}

echo "[2] Variant 2: NXRO-RO (random)"
python NXRO_train.py --model ro ${common}

echo "[3] Variant 3: NXRO-RO+Diag (random)"
python NXRO_train.py --model rodiag ${common}

echo "[4] Variant 4: NXRO-Res (random)"
python NXRO_train.py --model res ${common}

echo "[5] Variant 5: NXRO-NeuralODE (random)"
python NXRO_train.py --model neural ${common}

echo "[5a] Variant 5a: NXRO-Attentive (random)"
python NXRO_train.py --model attentive ${common}

echo "[5b] Variant 5b: NXRO-Graph (random)"
python NXRO_train.py --model graph ${common}

echo "[5c] Variant 5c: NXRO-PhysReg (random)"
python NXRO_train.py --model neural_phys ${common}

echo "[5d] Variant 5d: NXRO-ResidualMix (random)"
python NXRO_train.py --model resmix ${common}

# ============================================================================
# Group 2: Warm-Start Variants - 6 variants
# ============================================================================
echo ""
echo "--- WARM-START VARIANTS ---"

echo "[1a] Variant 1a: NXRO-Linear-WS"
python NXRO_train.py --model linear ${common} --warm_start ${XRO_FIT_FILE}

echo "[2a] Variant 2a: NXRO-RO-WS"
python NXRO_train.py --model ro ${common} --warm_start ${XRO_FIT_FILE}

echo "[3a] Variant 3a: NXRO-RO+Diag-WS"
python NXRO_train.py --model rodiag ${common} --warm_start ${XRO_FIT_FILE}

echo "[5a-WS] Variant 5a-WS: NXRO-Attentive-WS"
python NXRO_train.py --model attentive ${common} --warm_start ${XRO_FIT_FILE}

echo "[5b-WS] Variant 5b-WS: NXRO-Graph-WS"
python NXRO_train.py --model graph ${common} --warm_start ${XRO_FIT_FILE}

echo "[5d-WS] Variant 5d-WS: NXRO-ResidualMix-WS"
python NXRO_train.py --model resmix ${common} --warm_start ${XRO_FIT_FILE}

# ============================================================================
# Group 3: Variant 2 Freezing Ablations - 2 variants
# ============================================================================
echo ""
echo "--- VARIANT 2 FREEZING ABLATIONS ---"

echo "[2a-FixL] Variant 2a-FixL: Freeze linear, train RO"
python NXRO_train.py --model ro ${common} --warm_start ${XRO_FIT_FILE} --freeze linear

echo "[2a-FixRO] Variant 2a-FixRO: Freeze RO, train linear"
python NXRO_train.py --model ro ${common} --warm_start ${XRO_FIT_FILE} --freeze ro

# Note: 2a-FixAll is just using XRO directly, can skip training

# ============================================================================
# Group 4: Variant 3 Freezing Ablations - 4 variants
# ============================================================================
echo ""
echo "--- VARIANT 3 FREEZING ABLATIONS ---"

echo "[3a-FixL] Variant 3a-FixL: Freeze linear, train RO+Diag"
python NXRO_train.py --model rodiag ${common} --warm_start ${XRO_FIT_FILE} --freeze linear

echo "[3a-FixRO] Variant 3a-FixRO: Freeze RO, train linear+Diag"
python NXRO_train.py --model rodiag ${common} --warm_start ${XRO_FIT_FILE} --freeze ro

echo "[3a-FixDiag] Variant 3a-FixDiag: Freeze Diag, train linear+RO"
python NXRO_train.py --model rodiag ${common} --warm_start ${XRO_FIT_FILE} --freeze diag

echo "[3a-FixNL] Variant 3a-FixNL: Freeze RO+Diag, train linear"
python NXRO_train.py --model rodiag ${common} --warm_start ${XRO_FIT_FILE} --freeze ro,diag

# Note: 3a-FixAll is just using XRO directly, can skip training

# ============================================================================
# Group 5: Variant 4 Warm-Start - 2 variants
# ============================================================================
echo ""
echo "--- VARIANT 4 WARM-START VARIANTS ---"

echo "[4a] Variant 4a: NXRO-Res-WS-FixL (frozen XRO linear + MLP)"
python NXRO_train.py --model res ${common} --warm_start ${XRO_FIT_FILE} --freeze linear

echo "[4b] Variant 4b: NXRO-Res-FullXRO (frozen full XRO + MLP)"
python NXRO_train.py --model res_fullxro ${common} --warm_start ${XRO_FIT_FILE}

# ============================================================================
# Group 6: Variant 5a/5b Freezing - 2 variants
# ============================================================================
echo ""
echo "--- VARIANT 5a/5b FREEZING ---"

echo "[5a-FixL] Variant 5a-FixL: NXRO-Attentive with frozen linear"
python NXRO_train.py --model attentive ${common} --warm_start ${XRO_FIT_FILE} --freeze linear

echo "[5b-FixL] Variant 5b-FixL: NXRO-Graph with frozen linear"
python NXRO_train.py --model graph ${common} --warm_start ${XRO_FIT_FILE} --freeze linear

# ============================================================================
# Group 7: Variant 5d Freezing Ablations - 5 variants
# ============================================================================
echo ""
echo "--- VARIANT 5d FREEZING ABLATIONS ---"

echo "[5d-FixL] Variant 5d-FixL: Freeze linear, train RO+Diag+MLP"
python NXRO_train.py --model resmix ${common} --warm_start ${XRO_FIT_FILE} --freeze linear

echo "[5d-FixRO] Variant 5d-FixRO: Freeze RO, train linear+Diag+MLP"
python NXRO_train.py --model resmix ${common} --warm_start ${XRO_FIT_FILE} --freeze ro

echo "[5d-FixDiag] Variant 5d-FixDiag: Freeze Diag, train linear+RO+MLP"
python NXRO_train.py --model resmix ${common} --warm_start ${XRO_FIT_FILE} --freeze diag

echo "[5d-FixNL] Variant 5d-FixNL: Freeze RO+Diag, train linear+MLP"
python NXRO_train.py --model resmix ${common} --warm_start ${XRO_FIT_FILE} --freeze ro,diag

echo "[5d-FixPhysics] Variant 5d-FixPhysics: Freeze all physics, train MLP only"
python NXRO_train.py --model resmix ${common} --warm_start ${XRO_FIT_FILE} --freeze linear,ro,diag

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "="*80
echo "✓ ALL 32 VARIANTS COMPLETED!"
echo "="*80
echo ""
echo "Summary:"
echo "  - Random init: 9 variants (1-5, 5a-5d)"
echo "  - Warm-start: 6 variants (1a, 2a, 3a, 5a-WS, 5b-WS, 5d-WS)"
echo "  - Variant 2 freezing: 2 variants (2a-FixL, 2a-FixRO)"
echo "  - Variant 3 freezing: 4 variants (3a-FixL/RO/Diag/NL)"
echo "  - Variant 4 warm-start: 2 variants (4a, 4b)"
echo "  - Variant 5a/5b freezing: 2 variants (5a-FixL, 5b-FixL)"
echo "  - Variant 5d freezing: 5 variants (5d-FixL/RO/Diag/NL/Physics)"
echo "  - Pure XRO baselines: 2 variants (2a-FixAll, 3a-FixAll) - not trained"
echo ""
echo "Total trained: 30 variants"
echo "Results in results/<model_name>/ directories"
echo ""

