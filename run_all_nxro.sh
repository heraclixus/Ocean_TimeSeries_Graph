#!/bin/bash

set -euo pipefail

# Usage: bash run_all_nxro.sh [--stochastic] [--members 100] [--rollout_k 1] [--epochs 2000] [--device auto]

STOCH=""
MEMBERS=100
ROLLOUT=1
EPOCHS=2000
DEVICE=auto

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
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

common="--epochs ${EPOCHS} --rollout_k ${ROLLOUT} --device ${DEVICE} ${STOCH} --members ${MEMBERS}"

echo "Running NXRO variants with: ${common}"

python NXRO_train.py --model linear ${common}
python NXRO_train.py --model ro ${common}
python NXRO_train.py --model rodiag ${common}
python NXRO_train.py --model res ${common}
python NXRO_train.py --model neural ${common}
python NXRO_train.py --model neural_phys ${common}
python NXRO_train.py --model resmix ${common}
python NXRO_train.py --model bilinear ${common}
python NXRO_train.py --model attentive ${common}
python NXRO_train.py --model graph ${common}
python NXRO_train.py --model graph_pyg ${common} --top_k 3
python NXRO_train.py --model graph_pyg ${common} --top_k 3 --gat
echo "Done."