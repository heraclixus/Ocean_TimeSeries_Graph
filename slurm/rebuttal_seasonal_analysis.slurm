#!/bin/bash
#SBATCH --job-name=nxro_season
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/seasonal_analysis_%j.out
#SBATCH --error=slurm/logs/seasonal_analysis_%j.err

# Re-run reforecasts from saved checkpoints and compute per-season RMSE
set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

echo "=== Seasonal analysis: gate ablation ==="
${PYTHON} scripts/reforecast_seasonal_analysis.py --experiment gate_ablation --device cuda

echo "=== Seasonal analysis: multiseed ==="
${PYTHON} scripts/reforecast_seasonal_analysis.py --experiment multiseed --device cuda

echo "=== Done ==="
