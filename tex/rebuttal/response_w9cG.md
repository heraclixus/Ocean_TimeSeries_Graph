# Response to Reviewer w9cG

We thank the reviewer for the constructive feedback. Below we address each concern.

## W1: Neural residual vs. improved optimization

Both sources contribute. Under a strict train/val/test split (10 seeds):

| Model | Test RMSE | Improvement over XRO |
|-------|-----------|----------------------|
| XRO (OLS regression) | 0.605 | — |
| NXRO-Attentive (L + $R_\phi$) | 0.555 +/- 0.003 | **8.3%** |
| NXRO-GNN (L + $R_\phi$) | 0.557 +/- 0.000 | **8.0%** |

We note that the paper's Appendix Figure A.1 already includes NXRO-Linear (L(t) only, no $R_\phi$) and purely neural baselines (Neural ODE, Transformer) among the 43 ranked variants, enabling direct comparison of these ablation points. End-to-end optimization of L(t) contributes to the gain, but the neural residual $R_\phi$ is essential for two reasons:

1. **Season-specific skill.** A seasonal gate ablation shows that removing $R_\phi$'s modulation degrades spring-initialized forecasts by **+54%** (MAM RMSE: 0.780 → 1.198). Linear dynamics alone cannot capture the amplitude-dependent nonlinearities during ENSO's spring phase transition.

2. **Interpretability.** $R_\phi$ provides a learned teleconnection graph revealing nonlinear couplings with IOD, IOB, and ATL3 (see Q1 below).

## W2: Identifiability of the neural residual

$R_\phi$ is modulated by a seasonal gate $\alpha(t) \in [0,1]$ averaging 0.3--0.5, so L(t) remains the primary driver. The gate ablation (see table below) shows that without the gate, $R_\phi$ overwhelms linear physics during spring. The gate ensures L(t) and $R_\phi$ operate in complementary regimes, directly addressing identifiability. See Q1 for the physical interpretation of the learned residual.

| Model | Condition | MAM RMSE (leads 3--9) | All-season |
|-------|-----------|----------------------|-----------|
| Attentive | with gate | **0.780** | **0.555** |
| Attentive | no gate | 1.198 (+54%) | 0.558 |
| GNN | with gate | **0.815** | **0.557** |
| GNN | no gate | 0.899 (+10%) | 0.557 |

## W3: Three variants feel like three different methods

All share the **same formulation**: $dX/dt = L_\theta(t) X + \alpha(t) R_\phi(X, t)$, differing only in $R_\phi$. All beat XRO under strict validation:

| R_phi | Params | Test RMSE (10 seeds) | Improvement over XRO |
|-------|--------|----------------------|----------------------|
| MLP | 634 | 0.577 +/- 0.017 | 4.6% |
| Attention | ~750 | 0.555 +/- 0.003 | 8.3% |
| GNN | ~850 | 0.557 +/- 0.000 | 8.0% |

This is a systematic study of how much structure helps in small-data regimes, not three competing methods. The full ranking of all 43 variants (including purely neural models and linear-only ablations) is in Appendix Figure A.1.

## Q1: What does the residual learn?

The four strong nonlinear relationships captured by NXRO-GNN (Fig. 5: ENSO with IOB, IOD, SIOD, ATL3) reflect known but complex climate dynamics:

**IOB**: Positive IOB develops after El Nino; the nonlinear relationship arises from asymmetric strength of opposite IOD phases and mixed-layer-depth asymmetry between positive/negative IOB events [1].

**IOD**: While positive IOD co-occurs with El Nino quasi-linearly (Fig. 4), this relationship is strongly asymmetric — positive IOD relates to El Nino far more than negative IOD to La Nina, due to nonlinear cloud-SST-radiation feedback in the Indian Ocean [2].

**SIOD**: Its influence on ENSO varies with the background climate state and changes over time, indicating a highly nonlinear relationship [3].

**ATL3**: Positive ATL3 favors La Nina ~3 months later by strengthening Atlantic deep convection. This is highly asymmetric: warm Atlantic SST anomalies more effectively promote La Nina than cold anomalies promote El Nino [4,5].

[1] Hong et al. (2010), J. Climate 23:3563. [2] Cai et al. (2012), J. Climate 25:6318. [3] Huang et al. (2021), GRL 48:e2021GL094835. [4] Rodriguez-Fonseca et al. (2009), GRL. [5] vanRensch et al. (2024), GRL 51:e2023GL106585.

## Q2: Strategies for simulation data

We use CESM large ensemble (~4,400 years) for transfer learning. However, CESM has substantial Pacific mean-state biases in tropical precipitation and SST trends [6]. Because the mean state affects nonlinear ENSO-teleconnection relationships, models pretrained on CESM may perform worse when fine-tuned on observations. Approaches like flux adjustment can improve the mean state [7] but are computationally costly and not yet applied to large ensembles. We will soften the claim to be specific to CESM2.

[6] Wills et al. (2022), GRL 49:e2022GL100011. [7] Zhuo et al. (2025), J. Climate 48:1037.
