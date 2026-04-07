# Response to Reviewer cGpF

We thank the reviewer for the thorough feedback. Below we address each concern with new experimental evidence.

## W1: Lack of AI architectural innovation

The novelty lies in the **hybrid decomposition itself** -- preserving physics-based linear dynamics while learning nonlinear corrections. Under rigorous validation (train/val/test split, 10 seeds), all NXRO variants beat XRO (0.605): Attentive (0.555 +/- 0.003), GNN (0.557 +/- 0.000), MLP (0.577 +/- 0.017). The structured variants have 6x lower variance than MLP, showing that our domain-specific choices -- the attention mask restricting interactions to T/H, the graph sparsity encoding observed teleconnections, and the seasonal gate -- provide crucial regularization beyond standard techniques. We will articulate these as architectural innovations.

## W2: Interpretability contradiction

We provide direct evidence of $R_\phi$'s physical role via a **seasonal gate ablation** (5 seeds each, with and without the gate):

**RMSE by initialization season (all leads):**

| Model | Condition | DJF | MAM | JJA | SON |
|-------|-----------|-----|-----|-----|-----|
| Attentive | with gate | 0.726 | **0.723** | 0.657 | 0.606 |
| Attentive | no gate | 0.874 | **1.155** | 0.996 | 0.844 |
| GNN | with gate | 0.637 | **0.725** | 0.668 | 0.606 |
| GNN | no gate | 0.726 | **0.850** | 0.721 | 0.658 |

The gate **selectively activates $R_\phi$ during spring/summer** (MAM/JJA), precisely when linear dynamics encounter the Spring Predictability Barrier. Without the gate, $R_\phi$ overcorrects year-round, degrading all seasons. This is not post-hoc alignment — it is a mechanistic role evidenced by ablation: $R_\phi$ compensates for nonlinear dynamics that intensify during ENSO's spring phase transition.

![Seasonal gate ablation by season](https://raw.githubusercontent.com/heraclixus/Ocean_TimeSeries_Graph/main/tex/rebuttal/figures/fig3_seasonal_gate_ablation.png)

## W3: Insufficient addressing of SPB

**Spring barrier focus — MAM initializations, leads 3--9 months:**

| Model | with gate | no gate | Degradation |
|-------|-----------|---------|-------------|
| Attentive | 0.780 | 1.198 | **+54%** |
| GNN | 0.815 | 0.899 | **+10%** |

Per-lead RMSE for MAM initializations:

| Lead | Attn (gate) | Attn (no gate) | GNN (gate) | GNN (no gate) |
|------|-------------|----------------|------------|---------------|
| 3 | 0.503 | 0.603 | 0.515 | 0.519 |
| 6 | 0.821 | 1.205 | 0.868 | 0.939 |
| 9 | 0.900 | 1.657 | 0.932 | 1.114 |
| 12 | 0.611 | 1.188 | 0.575 | 0.799 |

The gate effect grows with lead time (6--12 months), exactly where SPB manifests.

![Spring barrier per-lead detail](https://raw.githubusercontent.com/heraclixus/Ocean_TimeSeries_Graph/main/tex/rebuttal/figures/fig4_spring_barrier_detail.png)

## W4: Limited baselines

We added classical baselines. All are substantially worse than XRO:

| Model | Avg Nino3.4 RMSE |
|-------|-----------------|
| Persistence | 1.027 |
| ARIMA(2,0,1) | 0.754 |
| VAR(3) | 0.682 |
| XRO | 0.605 |
| **NXRO-Attentive** | **0.555** |

VAR(3) — the strongest classical alternative — achieves only 0.682, far from XRO's 0.605 which uses seasonal modulation and physics-informed coupling. We will discuss recent ENSO-specific models (Zhang et al.) as additional context.

## W5: Transfer learning

We will soften the claim to be specific to CESM2, discuss mitigations (domain adaptation, bias correction), and note consistency with known CESM2 biases in CMIP6 literature.

## On architectural innovation

ENSO operates on basin-averaged indices, not spatial fields — standard equivariance is not directly applicable. The seasonal gate and attention mask are domain-specific innovations encoding the annual cycle and recharge-oscillator coupling. Energy-conserving dynamics are promising future work.
