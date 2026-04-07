# Response to Reviewer w9cG

We thank the reviewer for the constructive feedback. Below we address each concern.

## W1: Neural residual vs. improved optimization

We re-ran all core models under a strict train/val/test split (train 1979--1995, val 1996--2001, test 2002--2022) with 10 random seeds:

| Model | Test RMSE (mean +/- std) | vs XRO |
|-------|--------------------------|--------|
| XRO | 0.605 | baseline |
| NXRO-Attentive | 0.555 +/- 0.003 | **-8.3%** |
| NXRO-GNN | 0.557 +/- 0.000 | **-8.0%** |

The improvement comes from **two synergistic sources**: (1) end-to-end optimization of L(t) jointly across all variables via gradient descent, producing better-conditioned dynamics for multi-step forecasting (vs XRO's per-equation regression); and (2) the neural residual $R_\phi$ capturing nonlinear dynamics L(t) cannot represent.

A **seasonal gate ablation** demonstrates (2): removing the gate degrades spring-initialized forecasts by **+54%** for NXRO-Attentive (MAM RMSE: 0.780 → 1.198), proving $R_\phi$ learns targeted corrections rather than redundant dynamics. The two contributions are complementary: optimization provides the bulk of the accuracy gain, while the neural residual adds **interpretability** (learned teleconnection graphs, seasonal gate patterns) and **targeted skill improvements** in seasons where linear dynamics are insufficient.

## W2: Identifiability of the neural residual

$R_\phi$ is modulated by a seasonal gate $\alpha(t) \in [0,1]$ with learned values averaging 0.3--0.5, so L(t) remains the primary driver. The gate ablation reveals what $R_\phi$ captures:

| Model | Condition | MAM RMSE (leads 3--9) | All-season RMSE |
|-------|-----------|----------------------|-----------------|
| NXRO-Attentive | with gate | **0.780** | **0.555** |
| NXRO-Attentive | no gate | 1.198 (+54%) | 0.558 |
| NXRO-GNN | with gate | **0.815** | **0.557** |
| NXRO-GNN | no gate | 0.899 (+10%) | 0.557 |

Without the gate, $R_\phi$ overwhelms the linear physics during spring, degrading forecasts. The gate modulates $R_\phi$'s influence seasonally, preventing it from absorbing dynamics that L(t) handles well in other seasons. This directly addresses the identifiability concern: the gate ensures L(t) and $R_\phi$ operate in complementary regimes.

## W3: Three variants feel like three different methods

The three share the **same formulation**: $dX/dt = L_\theta(t) X + \alpha(t) R_\phi(X, t)$, differing only in $R_\phi$'s parameterization. Following the strict train/val/test protocol, all beat XRO:

| R_phi variant | Params | Test RMSE (10 seeds) | Beats XRO? |
|---------------|--------|----------------------|------------|
| MLP | 634 | 0.577 +/- 0.017 | **Yes (-4.6%)** |
| Attention | ~750 | 0.555 +/- 0.003 | **Yes (-8.3%)** |
| GNN | ~850 | 0.557 +/- 0.000 | **Yes (-8.0%)** |

They represent a **spectrum from less to more structured**. The MLP has higher variance (0.017) than Attention/GNN (<=0.003), demonstrating that structured inductive biases provide both better accuracy and greater stability. Rather than three competing methods, this is a systematic study of how much structure helps in small-data regimes.

![Validation-selected performance](https://raw.githubusercontent.com/heraclixus/Ocean_TimeSeries_Graph/main/tex/rebuttal/figures/fig6_decomposition.png)

## Q1: What does the residual learn?

$R_\phi$ primarily captures **spring-specific nonlinear dynamics** that L(t) misses. The seasonal gate peaks in MAM and SON, corresponding to ENSO's phase-locking to the annual cycle. The GNN's learned adjacency reveals the strongest nonlinear interactions connect ENSO with IOD, IOB, and ATL3 — consistent with recent observational studies on strengthened Atlantic-ENSO teleconnections.

## Q2: Strategies for simulation data

We will soften the claim to be specific to CESM2 and discuss strategies including domain adaptation (e.g., optimal transport alignment) and bias correction (e.g., quantile mapping) as promising future directions.
