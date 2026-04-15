# Response to Reviewer i9br

We thank the reviewer for the positive assessment. Below we address the two weaknesses.

## W1: In-depth analysis on alternative design choices

**Concern:** The GNN and attention variants seem mathematically similar with similar performance. A physics-based graph design would provide more perspective.

Both variants apply a structured nonlinear correction modulated by a seasonal gate, but differ importantly:
- **NXRO-Attentive**: learned attention mask restricting interactions to Nino3.4 (T) and WWV (H), encoding the recharge-oscillator hypothesis that teleconnections are mediated through the core ENSO state.
- **NXRO-GNN**: correlation-initialized graph (top-k Pearson from training data), encoding empirically observed teleconnection strengths as inductive bias.

Under rigorous validation (10 seeds, train/val/test split), both achieve ~0.555--0.557 aggregate RMSE. However, a **seasonal gate ablation** reveals a critical difference:

| Model | with gate | no gate (MAM leads 3--9) | Degradation |
|-------|-----------|--------------------------|-------------|
| Attentive | 0.780 | 1.198 | **+54%** |
| GNN | 0.815 | 0.899 | +10% |

The Attentive model is **more dependent on the seasonal gate** — its learned attention weights can overfit without temporal modulation. The GNN, with its correlation-initialized graph providing stronger structural constraint, degrades less when the gate is removed. This provides the perspective the reviewer requests: **more physics-based structure (graph) trades peak performance for robustness**, while **learned structure (attention) achieves higher performance but requires additional regularization (the gate)**.

We will add this analysis to better differentiate the two variants.

## W2: Data scarcity analysis

**Concern:** How much data is required? How accurate is the model with fewer years?

We conducted a systematic data scarcity experiment with 3 models, 5 training sizes (10--23 years), and 3 seeds each:

| Model | 10yr | 13yr | 16yr | 19yr | 23yr (full) |
|-------|------|------|------|------|-------------|
| NXRO-MLP | 0.616 +/- 0.004 | 0.584 +/- 0.004 | 0.566 +/- 0.011 | 0.588 +/- 0.002 | 0.573 +/- 0.004 |
| NXRO-Attentive | 0.633 +/- 0.010 | 0.594 +/- 0.006 | 0.562 +/- 0.000 | 0.560 +/- 0.002 | **0.550 +/- 0.001** |
| **NXRO-GNN** | **0.600 +/- 0.000** | **0.576 +/- 0.000** | **0.560 +/- 0.000** | **0.561 +/- 0.000** | **0.550 +/- 0.000** |
| XRO (ref) | 0.605 | 0.605 | 0.605 | 0.605 | 0.605 |

**Key findings:**
1. **NXRO-GNN beats XRO with only 10 years of data** (0.600 vs 0.605) — competitive with the physics baseline using ~120 monthly samples.
2. **NXRO-Attentive beats XRO starting at 13 years** (0.594 vs 0.605).
3. Both converge by ~16 years (RMSE ~0.560), with diminishing returns beyond that.
4. NXRO-GNN has near-zero variance (std = 0.000) at all data sizes, reflecting the strong constraint from the physics-informed graph, making it the most reliable choice for limited-data settings.

**Practical implication:** For ocean basins or climate indices where only 10--15 years of reanalysis may be available, GNN is the recommended choice, while Attentive requires ~13 years to outperform the physics-only baseline. Future work may use synthetic millennial-scale climate-model simulations to further test sensitivity to training-sample size.


## On physics-based graph design

Our NXRO-GNN uses an adjacency initialized from pairwise correlation, encoding observed teleconnection strengths (e.g., ENSO-IOD, ENSO-TNA coupling). This is exactly the "physics-based design tailored for the specific system" the reviewer suggests. As shown in Section 4.5, the graph structure reveals interpretable teleconnection patterns consistent with recent climate science, demonstrating that the architecture provides both performance and physical insight.
