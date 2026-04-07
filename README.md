# Neural Residual Extended Recharge Oscillator (NXRO) for ENSO Forecasting

Code for the paper: **"Neural Hybrid Residual XRO Models for ENSO Forecasting"** (KDD 2026).

## Overview

NXRO is a hybrid physics-ML framework for El Nino-Southern Oscillation (ENSO) forecasting. It decomposes ocean dynamics into a physics-based seasonal linear operator and a learned nonlinear neural residual:

$$\frac{dX}{dt} = L_\theta(t) \cdot X + \alpha(t) \cdot R_\phi(X, t)$$

where $L_\theta(t)$ is a seasonally modulated linear operator (from the Extended Recharge Oscillator), $R_\phi$ is a neural correction (MLP, Attention, or GNN), and $\alpha(t)$ is a learned seasonal gate.

## Project Structure

```
nxro/                  # Core NXRO model code
  models.py            #   Model architectures (Linear, MLP, Attentive, GNN, Transformer)
  train.py             #   Training loops with val split support
  eval.py              #   Evaluation metrics
  data.py              #   Data loading (ORAS5, CESM2-LENS)
  stochastic.py        #   Stochastic noise fitting and ensemble forecasting

XRO/                   # Physics-based XRO baseline
  core.py              #   XRO model (closed-form regression)

src/                   # Additional baselines
  baseline_models/     #   ARIMA, GP, Neural ODE, Graph ODE, etc.
  cgode/, lgode/, pgode/  # Coupled/Latent/Partial Graph ODE baselines

data/                  # Ocean time series indices
  XRO_indices_oras5.nc #   Primary dataset (10 climate indices, 1979-2024)

KDD_ENSO_tex/          # Paper LaTeX source
tex/rebuttal/          # Rebuttal responses and figures
```

## Key Files

| File | Description |
|------|-------------|
| `NXRO_train_out_of_sample.py` | Main entry point for out-of-sample experiments |
| `run_utils.py` | Wrapper functions for training and evaluation |
| `graph_construction.py` | Teleconnection graph construction |
| `utils/xro_utils.py` | Forecast skill metrics (RMSE, ACC, CRPS) |

## Data

10 monthly climate indices from ORAS5 reanalysis (1979-2024):

| Index | Description |
|-------|-------------|
| Nino34 | El Nino 3.4 SST anomaly |
| WWV | Warm Water Volume (thermocline depth) |
| NPMM / SPMM | North/South Pacific Meridional Mode |
| IOB / IOD / SIOD | Indian Ocean Basin / Dipole / Subtropical Dipole |
| TNA / ATL3 / SASD | Tropical North Atlantic / Atlantic Nino / South Atlantic |

## Quick Start

### Training

```bash
# Train NXRO-Attentive (best model) with proper train/val/test split
python NXRO_train_out_of_sample.py \
    --model attentive \
    --seed 42 \
    --train_start 1979-01 --train_end 2001-12 \
    --val_start 1996-01 --val_end 2001-12 \
    --test_start 2002-01 --test_end 2022-12 \
    --epochs 2000 --batch_size 128 --lr 1e-3 \
    --extra_train_nc none \
    --stochastic --members 100 --train_noise_stage2

# Other models: --model {linear, res, graph_pyg, pure_neural_ode, pure_transformer}
```

### Reproducing Paper Results

```bash
# Run all core models with 10 seeds (requires SLURM cluster)
sbatch slurm/rebuttal_multiseed.slurm

# Aggregate results
python scripts/aggregate_rebuttal_results.py --experiment multiseed
```

## Results

All NXRO variants outperform the XRO baseline (0.605 avg RMSE) under strict train/val/test evaluation with 10 random seeds:

| Model | Test RMSE | +/- std | vs XRO |
|-------|-----------|---------|--------|
| NXRO-Attentive | **0.555** | 0.003 | -8.3% |
| NXRO-GNN | **0.557** | 0.000 | -8.0% |
| NXRO-MLP | **0.577** | 0.017 | -4.6% |
| Transformer | 0.676 | 0.025 | +11.8% |
| Neural ODE | 0.782 | 0.018 | +29.2% |

## Citation

```bibtex
@inproceedings{xu2026nxro,
  title={Neural Hybrid Residual XRO Models for ENSO Forecasting},
  author={Xu, Fred and Lu, Kezhou and Kondrashov, Dmitri and Chen, Gang and Sun, Yizhou},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

## License

This project is for research purposes.
