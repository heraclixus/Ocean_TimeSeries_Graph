# Response to Reviewer w9cG

We thank the reviewer for the constructive feedback. Below we address each concern.

## W1: Neural residual vs. improved optimization

We isolate the two sources of improvement by comparing three levels under a strict train/val/test split (10 seeds):

| Model | What it has | Test RMSE | Gain source |
|-------|-------------|-----------|-------------|
| XRO | L(t) via OLS regression | 0.605 | — |
| NXRO-Linear | L(t) via end-to-end gradient descent | 0.560 +/- 0.001 | Optimization: **-7.4%** |
| NXRO-Attentive | L(t) + neural $R_\phi$ | 0.555 +/- 0.003 | + Residual: **-0.8%** |
| NXRO-GNN | L(t) + neural $R_\phi$ | 0.557 +/- 0.000 | + Residual: **-0.5%** |

NXRO-Linear uses the same seasonal linear operator as XRO but optimizes it jointly across all variables via gradient descent on multi-step rollout loss, rather than XRO's per-equation regression. This accounts for ~90% of the improvement. The neural residual adds a further ~1%, but its primary value is **interpretability** (learned teleconnection graphs, seasonal gate patterns) and **targeted skill at the Spring Predictability Barrier**: removing the seasonal gate degrades MAM forecasts by +54% for Attentive.

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
