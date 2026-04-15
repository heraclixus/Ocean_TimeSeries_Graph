# Response to Reviewer cGpF

We thank the reviewer for the thorough feedback. Below we address each concern.

## W1: Lack of AI architectural innovation

The novelty lies in the **hybrid decomposition** — preserving physics-based linear dynamics while learning nonlinear corrections. Under rigorous validation (train/val/test split, 10 seeds), all NXRO variants beat XRO (0.605): Attentive (0.555 +/- 0.003), GNN (0.557 +/- 0.000), MLP (0.577 +/- 0.017). The structured variants have 6x lower variance than MLP, showing that domain-specific choices (attention mask on T/H, graph sparsity, seasonal gate) provide crucial regularization. We will articulate these as architectural innovations.

## W2: Interpretability contradiction

We provide direct evidence of $R_\phi$'s physical role via **seasonal gate ablation** (5 seeds each):

| Model | Condition | DJF | MAM | JJA | SON |
|-------|-----------|-----|-----|-----|-----|
| Attentive | with gate | 0.726 | **0.723** | 0.657 | 0.606 |
| Attentive | no gate | 0.874 | **1.155** | 0.996 | 0.844 |
| GNN | with gate | 0.637 | **0.725** | 0.668 | 0.606 |
| GNN | no gate | 0.726 | **0.850** | 0.721 | 0.658 |

The gate selectively activates $R_\phi$ during spring/summer, precisely when linear dynamics encounter the SPB. The four strong nonlinear relationships captured by NXRO-GNN (Fig. 5: ENSO with IOB, IOD, SIOD, ATL3) reflect known climate dynamics: IOB nonlinearity arises from asymmetric IOD phases and mixed-layer-depth asymmetry [1]; the IOD-ENSO relationship is strongly asymmetric due to nonlinear cloud-SST-radiation feedback [2]; SIOD influence on ENSO varies with background climate state [3]; and ATL3 asymmetrically favors La Nina via tropospheric temperature response differences [4,5]. These are not post-hoc alignments — they are mechanistic dynamics captured by the learned residual.

[1] Hong et al. (2010), J.Clim 23:3563. [2] Cai et al. (2012), J.Clim 25:6318. [3] Huang et al. (2021), GRL 48:e2021GL094835. [4] Rodriguez-Fonseca et al. (2009), GRL. [5] vanRensch et al. (2024), GRL 51:e2023GL106585.

## W3: Insufficient addressing of SPB

MAM initializations, leads 3--9 months:

| Model | with gate | no gate | Degradation |
|-------|-----------|---------|-------------|
| Attentive | 0.780 | 1.198 | **+54%** |
| GNN | 0.815 | 0.899 | **+10%** |

The gate effect grows with lead time (6--12 months), exactly where SPB manifests.

## W4: Limited baselines

We added classical baselines (Persistence: 1.027, ARIMA: 0.754, VAR(3): 0.682), all worse than XRO (0.605). NXRO is not designed to learn spatial fields — it operates on basin-averaged climate indices in a low-dimensional framework. Attention-based spatial models (e.g., Zhang et al.) address subsurface mixing for a specific ENSO type in a particular model, rather than providing a general forecast framework. Our model improves upon XRO as a general ENSO forecast tool.

## W5: Transfer learning

Mean-state biases in the tropical Pacific exist across CMIP models and affect ENSO teleconnections [6]. Flux adjustment can improve model mean states [7] but is computationally costly and not yet applied to large ensembles. A broader assessment of how climate-model bias affects transfer learning is beyond the scope of the present study. We will soften the claim to be CESM2-specific.

[6] Wills et al. (2022), GRL 49:e2022GL100011. [7] Zhuo et al. (2025), J.Clim 48:1037.

## On architectural innovation

ENSO operates on basin-averaged indices, not spatial fields — standard equivariance is not applicable. The seasonal gate and attention mask encode the annual cycle and recharge-oscillator coupling. Energy-conserving dynamics are promising future work.
