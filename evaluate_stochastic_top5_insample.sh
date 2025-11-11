#!/bin/bash
#
# Evaluate Stochastic Ensembles for Top 5 In-Sample Models + XRO Baseline
#
# This script runs stochastic ensemble forecasts for the top 5 deterministic
# models using in-sample training (full 1979-2022 data) and compares against
# XRO baseline for probabilistic skill.
#
# Usage:
#   ./evaluate_stochastic_top5_insample.sh
#   ./evaluate_stochastic_top5_insample.sh --members 200 --device cuda
#   ./evaluate_stochastic_top5_insample.sh --stage2  # Use likelihood-based noise optimization
#   ./evaluate_stochastic_top5_insample.sh --sim     # Use simulation-based noise
#   ./evaluate_stochastic_top5_insample.sh --sim --stage2  # Combine both (sim noise + likelihood opt)
#

set -euo pipefail

MEMBERS=100
DEVICE=auto
STAGE2_FLAG=""
SIM_NOISE_FLAG=""
BASE_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --members) MEMBERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --stage2) STAGE2_FLAG="--train_noise_stage2"; shift ;;
    --sim) SIM_NOISE_FLAG="--use_sim_noise"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "================================================================================"
echo "STOCHASTIC ENSEMBLE EVALUATION (IN-SAMPLE): Top 5 Models + XRO Baseline"
echo "================================================================================"
echo "Configuration:"
echo "  Dataset: ORAS5 only"
echo "  Train: 1979-01 to 2022-12 (full dataset, in-sample)"
echo "  Ensemble members: ${MEMBERS}"
echo "  Device: ${DEVICE}"
echo "  Stage 2 training: ${STAGE2_FLAG:-disabled (post-hoc fitting)}"
echo "  Simulation noise: ${SIM_NOISE_FLAG:-disabled (model residuals)}"
echo "  Results: ${BASE_DIR}/"
echo "================================================================================"
echo ""

# Generate XRO stochastic baseline
echo "================================================================================"
echo "[0/5] Generating XRO Baseline (Stochastic)"
echo "================================================================================"
echo ""

python3 << 'PYEOF'
import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import os
from XRO.core import XRO
from utils.xro_utils import calc_forecast_skill, evaluate_stochastic_ensemble

print("Fitting XRO model on full dataset (in-sample)...")
obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))

xro = XRO(ncycle=12, ac_order=2)
xro_fit = xro.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])

print("Generating deterministic forecast...")
xro_fcst = xro.reforecast(fit_ds=xro_fit, init_ds=obs_ds, n_month=21, 
                          ncopy=1, noise_type='zero')

print(f"Generating {os.environ.get('MEMBERS', 100)}-member stochastic ensemble...")
xro_fcst_stoc = xro.reforecast(fit_ds=xro_fit, init_ds=obs_ds, n_month=21, 
                               ncopy=int(os.environ.get('MEMBERS', 100)), noise_type='red')

# Save
os.makedirs('results/xro_baseline', exist_ok=True)
xro_fcst.to_netcdf('results/xro_baseline/xro_deterministic_fcst.nc')
xro_fcst_stoc.to_netcdf('results/xro_baseline/xro_stochastic_fcst.nc')
xro_fit.to_netcdf('results/xro_baseline/xro_fit.nc')

# Evaluate ensemble
print("Evaluating stochastic ensemble...")
eval_df = evaluate_stochastic_ensemble(xro_fcst_stoc, obs_ds, var='Nino34',
                                      out_prefix='results/xro_baseline/xro_stochastic_eval')

# Compute ensemble-mean skill (in-sample)
xro_mean = xro_fcst_stoc.mean('member')
acc_mean = calc_forecast_skill(xro_mean, obs_ds, metric='acc', is_mv3=True,
                               by_month=False, verify_periods=slice('1979-01', '2022-12'))
rmse_mean = calc_forecast_skill(xro_mean, obs_ds, metric='rmse', is_mv3=True,
                                by_month=False, verify_periods=slice('1979-01', '2022-12'))

print(f"XRO Stochastic Results (In-Sample):")
print(f"  Ensemble-mean ACC: {float(np.nanmean(acc_mean['Nino34'].values)):.4f}")
print(f"  Ensemble-mean RMSE: {float(np.nanmean(rmse_mean['Nino34'].values)):.4f} C")
print(f"  Avg CRPS: {float(np.nanmean(eval_df['crps'].values)):.4f}")

print("XRO baseline complete!")
PYEOF

export MEMBERS
echo ""

# Top 5 NXRO models (based on in-sample rankings)
echo "================================================================================"
echo "[1/5] NXRO-Res (Rank 1)"
echo "================================================================================"
python NXRO_train.py --model res \
  --stochastic --members $MEMBERS --device $DEVICE $STAGE2_FLAG $SIM_NOISE_FLAG
echo ""

echo "================================================================================"
echo "[2/5] NXRO-Graph Fixed XRO (Rank 2)"
echo "================================================================================"
echo "Using default XRO topology with k=3, GCN"
python NXRO_train.py --model graph_pyg --top_k 3 \
  --stochastic --members $MEMBERS --device $DEVICE $STAGE2_FLAG $SIM_NOISE_FLAG
echo ""

echo "================================================================================"
echo "[3/5] NXRO-Attentive (Rank 3)"
echo "================================================================================"
python NXRO_train.py --model attentive \
  --stochastic --members $MEMBERS --device $DEVICE $STAGE2_FLAG $SIM_NOISE_FLAG
echo ""

echo "================================================================================"
echo "[4/5] NXRO-RO+Diag (Rank 4)"
echo "================================================================================"
python NXRO_train.py --model rodiag \
  --stochastic --members $MEMBERS --device $DEVICE $STAGE2_FLAG $SIM_NOISE_FLAG
echo ""

echo "================================================================================"
echo "[5/5] NXRO-Linear (Rank 5)"
echo "================================================================================"
python NXRO_train.py --model linear \
  --stochastic --members $MEMBERS --device $DEVICE $STAGE2_FLAG $SIM_NOISE_FLAG
echo ""

echo "================================================================================"
echo "STOCHASTIC EVALUATION COMPLETE (IN-SAMPLE)!"
echo "================================================================================"
echo ""
echo "All models evaluated with ${MEMBERS}-member ensembles"
echo "Results saved to: ${BASE_DIR}/{model}/*_stochastic_*"
echo ""
echo "Next steps:"
echo "  1. Visualize: python visualize_stochastic_comparison.py --results_dir results"
echo "  2. Compare stages: python visualize_stochastic_comparison.py --results_dir results --compare_stages"
echo "  3. Review plume plots in each model directory"
echo "  4. Compare CRPS, spread-skill, coverage across models"
echo ""
echo "================================================================================"

