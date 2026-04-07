#!/bin/bash
#SBATCH --job-name=nxro_baselines
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/classical_baselines_%j.out
#SBATCH --error=slurm/logs/classical_baselines_%j.err

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

echo "=== Running classical baselines ==="
${PYTHON} scripts/run_classical_baselines.py

echo "=== Done ==="
