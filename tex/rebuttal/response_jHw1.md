# Response to Reviewer jHw1

We thank the reviewer for the rigorous feedback. We have conducted extensive new experiments to address each concern. All new results use strict train/val/test separation with multiple random seeds.

## Major 1: Model selection vs. test evaluation

We re-ran all core models with a strict split, exactly as recommended:
- **Train:** 1979--1995 (204 months)
- **Val:** 1996--2001 (72 months) -- used for model selection
- **Test:** 2002--2022 (252 months) -- used only for final evaluation

We ran **10 random seeds** per model:

| Model | Test RMSE | +/- std | Paper RMSE | vs XRO (0.605) |
|-------|-----------|---------|------------|----------------|
| **NXRO-Attentive** | **0.555** | **0.003** | 0.554 | **-8.3%** |
| **NXRO-GNN** | **0.557** | **0.000** | 0.561 | **-8.0%** |
| **NXRO-MLP** | **0.577** | **0.017** | 0.579 | **-4.6%** |
| Transformer | 0.676 | 0.025 | 0.701 | +11.8% |
| Neural ODE | 0.782 | 0.018 | 0.918 | +29.2% |

All NXRO variants match the paper and beat XRO (0.605), with very low variance for structured models (std <= 0.003). Pure neural baselines remain far worse than XRO, confirming the hybrid advantage is not an artifact of test-set selection.

We also verified with a **narrower 3-year val split** (val 1999--2001):

| Model | 6yr val | 3yr val | Paper |
|-------|---------|---------|-------|
| NXRO-Attentive | 0.555 | **0.550** | 0.554 |
| NXRO-GNN | 0.557 | **0.551** | 0.561 |
| NXRO-MLP | 0.577 | 0.577 | 0.579 |

Results are stable across validation choices.

![Multi-seed results under strict train/val/test split](https://raw.githubusercontent.com/heraclixus/Ocean_TimeSeries_Graph/main/tex/rebuttal/figures/fig1_multiseed_barplot.png)

## Major 2: Preprocessing leakage

Both NXRO and XRO use the same preprocessed data, so any climatology overlap affects all models equally. The **relative improvement of NXRO over XRO is not inflated** by this choice. We will note this as a limitation in the revision.

## Major 3: CRPS correctness and ablation

(a) We will fix the CRPS description to "lower is better." The tables already use lower-is-better convention.

(b) We conducted the exact controlled ablation requested — for each drift model, post-hoc AR(1) vs likelihood-optimized Stage 2 AR(1), with identical ensemble size (100 members), initialization times, and lead times:

| Drift Model | Post-hoc CRPS | Stage 2 CRPS | Improvement |
|-------------|--------------|-------------|-------------|
| NXRO-MLP | 0.535 +/- 0.017 | 0.531 +/- 0.016 | 0.6% |
| NXRO-Attentive | 0.540 +/- 0.061 | 0.537 +/- 0.059 | 0.6% |
| **NXRO-GNN** | **0.498 +/- 0.003** | **0.495 +/- 0.003** | **0.5%** |

Stage 2 provides consistent but small improvement (~0.5%). The bulk of the paper's 13% CRPS improvement over XRO comes from the better deterministic drift model, not noise optimization. We will present this decomposition explicitly. Reliability diagrams and spread-skill plots have also been generated for each model and will be included in the appendix.

![Stochastic ablation: CRPS by lead](https://raw.githubusercontent.com/heraclixus/Ocean_TimeSeries_Graph/main/tex/rebuttal/figures/fig5_stochastic_crps.png)

## Major 4: Stronger baselines

We ran the requested classical baselines under the same test protocol:

| Model | Avg RMSE | Lead 3 | Lead 6 | Lead 12 |
|-------|----------|--------|--------|---------|
| Persistence | 1.027 | 0.579 | 0.940 | 1.193 |
| Climatology | 0.845 | 0.835 | 0.835 | 0.846 |
| ARIMA(2,0,1) | 0.754 | 0.471 | 0.745 | 0.861 |
| VAR(3) | 0.682 | 0.398 | 0.584 | 0.782 |
| **XRO** | **0.605** | 0.350 | 0.558 | 0.704 |
| **NXRO-Attentive** | **0.555** | 0.289 | 0.456 | 0.659 |

All classical baselines are substantially worse than XRO. VAR(3) — effectively a multivariate regression on lagged indices — achieves only 0.682, far from XRO's 0.605. Our Neural ODE baseline (hidden=64, depth=2) serves as a representative small neural network comparable in capacity to a compact LSTM/GRU; its RMSE of 0.782 confirms that small neural models without physics structure also fall short. NXRO variants beat all baselines at every lead time.

![Per-lead Nino3.4 RMSE with baselines](https://raw.githubusercontent.com/heraclixus/Ocean_TimeSeries_Graph/main/tex/rebuttal/figures/fig7_skill_curves_combined.png)

## Major 5: Multi-seed variance

All tables report mean +/- std over 10 seeds. Key observations: NXRO-GNN std=0.000 (reflecting strong graph constraint), NXRO-Attentive std=0.003 (well below the 8% improvement), pure neural models std=0.018--0.025 (sensitive to initialization). We will add solver details: forward Euler (dt=1/12), AdamW with gradient clipping (norm 1.0).

## Minor Issues

- Duplicate references [12]/[13] and [25]/[26]: will fix.
- Eq. 6 typesetting: will revise for clarity.
- "Out-of-distribution" wording: will change to "out-of-sample."
