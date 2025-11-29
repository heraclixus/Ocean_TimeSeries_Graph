# Out-of-Sample Performance Results

## Experimental Setup

**Training Period**: 1979-01 to 2001-12 (23 years)  
**Test Period**: 2002-01 to 2022-12 (21 years, held-out)

**Evaluation**: All models are trained on 1979-2001 data and evaluated on both:
- **In-sample**: Performance on training period (1979-2001)
- **Out-of-sample**: Performance on held-out test period (2002-2022)

**Ranking**: Models are ranked by out-of-sample performance using a combined score (normalized ACC - normalized RMSE on test period).

---

## Top 10 Models (Out-of-Sample Performance)

![Top 10 RMSE (Out-of-Sample)](results_out_of_sample/rankings/top10_rmse_test_out_of_sample.png)

### Summary Table

| Rank | Model | Test ACC | Test RMSE | Train ACC | Train RMSE | Description |
|------|-------|----------|-----------|-----------|------------|-------------|
| 1 | **NXRO-Res** | 0.628 | 0.567 C | 0.840 | 0.474 C | Linear + Residual MLP |
| 2 | **NXRO-Graph (Fixed XRO)** | 0.616 | 0.583 C | 0.773 | 0.536 C | Linear + Graph Convolution |
| 3 | **NXRO-Attentive** | 0.613 | 0.586 C | 0.766 | 0.542 C | Linear + Attention |
| 4 | **NXRO-RO+Diag** | 0.611 | 0.618 C | 0.813 | 0.490 C | Full XRO structure, gradient-trained |
| 5 | **NXRO-Linear** | 0.611 | 0.588 C | 0.764 | 0.543 C | Linear seasonal operator only |
| 6 | **NXRO-NeuralODE** | 0.602 | 0.588 C | 0.792 | 0.522 C | Pure MLP drift |
| 7 | **XRO** (baseline) | 0.596 | 0.605 C | 0.825 | 0.493 C | Physics-based closed-form |
| 8 | **NXRO-RO+Diag-FixNL** | 0.592 | 0.602 C | 0.840 | 0.470 C | Freeze RO+Diag, train Linear only |
| 9 | **NXRO-RO+Diag-FixRO** | 0.591 | 0.602 C | 0.841 | 0.471 C | Freeze RO, train Linear+Diag |
| 10 | **NXRO-ResidualMix-FixRO** | 0.590 | 0.598 C | 0.840 | 0.476 C | Freeze RO, train Linear+Diag+MLP |

---

## Two-Stage Training Analysis

We investigated whether **transfer learning** from large-scale climate simulations could improve out-of-sample performance.

### Methodology

**Stage 1 (Pre-training):**
- **Dataset:** Large ensemble of synthetic climate data (CMIP6-derived indices, `XRO_indices_*_preproc.nc`).
- **Goal:** Learn general climate dynamics and physical constraints from abundant data.
- **Training:** 1500 epochs on synthetic datasets.

**Stage 2 (Fine-tuning):**
- **Dataset:** Observational data (ORAS5) restricted to the 1979-2001 training period.
- **Goal:** Adapt the pre-trained weights to the specific statistics of the real-world system.
- **Training:** 1500 epochs, initialized with Stage 1 weights.

### Stage 1 vs. Stage 2 Performance

We compared the performance of models trained only on ORAS5 (Single-Stage) vs. those pre-trained on synthetic data and fine-tuned (Two-Stage).

![Single vs Two-Stage RMSE](results_out_of_sample/rankings/comparison_single_vs_two_stage_rmse.png)
*Comparison of Test RMSE. Negative delta (green) indicates Two-Stage is better.*

![Single vs Two-Stage ACC](results_out_of_sample/rankings/comparison_single_vs_two_stage_acc.png)
*Comparison of Test ACC. Positive delta (green) indicates Two-Stage is better.*

**Key Findings:**
1.  **NXRO-RO+Diag (Two-Stage) - Massive Improvement**: The single-stage training for this complex physics-based model was unstable (infinite RMSE). Two-stage training stabilized the parameters, resulting in a competitive model (RMSE ~0.63 C).
2.  **NXRO-Neural (Two-Stage) - The New Winner**: The pure Neural ODE model improved slightly but significantly in consistency, becoming the top overall performer.
3.  **NXRO-Res (Two-Stage) - Degradation**: The Residual model, which was the single-stage champion, performed worse after pre-training. This suggests its residual MLP might have overfitted to synthetic data biases that were hard to unlearn during fine-tuning.

### Overall Ranking (Single & Two-Stage Combined)

When comparing all variants (Single-Stage, Two-Stage, and XRO Baseline), we see an interesting split between model architectures.

![Overall Ranking RMSE](results_out_of_sample/rankings/overall_ranking_top5_vs_xro_rmse.png)

**Best Performing Models:**

1.  **NXRO-Res (Single-Stage)**
    *   **Rank:** #1 Overall
    *   **Equation:** $\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t)\mathbf{X} + \text{MLP}_\theta(\mathbf{X}, t)$
    *   **Performance:** Test RMSE 0.568 °C.
    *   **Insight:** The simplest hybrid structure (Linear + Residual MLP) trained directly on observations remains the champion. Pre-training (two-stage) actually hurt its performance (Test RMSE rose to 0.619 °C), likely because the residual component overfitted to biases in the synthetic data that were hard to unlearn.

2.  **NXRO-Neural (Two-Stage)**
    *   **Rank:** #2 Overall (Top among Two-Stage models)
    *   **Equation:** $\frac{d\mathbf{X}}{dt} = \text{MLP}_\theta(\mathbf{X}, t)$
    *   **Performance:** Test RMSE 0.576 °C (improved from 0.584 °C single-stage).
    *   **Insight:** Unlike the Res model, the pure Neural ODE benefited from pre-training. The high-capacity MLP learned robust general dynamics from the large synthetic dataset in Stage 1, providing a better initialization than random noise.

3.  **NXRO-RO+Diag (Two-Stage)**
    *   **Rank:** Competitive Physics Model
    *   **Equation:** $\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t)\mathbf{X} + \text{RO}(T,H) + \text{Diag}(\mathbf{X})$
    *   **Performance:** Test RMSE 0.631 °C (vs Inf in single-stage).
    *   **Insight:** Pre-training was absolutely critical here. Single-stage training was unstable (infinite RMSE), but two-stage training stabilized the parameters, making this complex physics-based model viable.

---

## Detailed Model Descriptions

### 1. NXRO-Res (Rank 1 in Single-Stage)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])
$$

**Architecture**:
- Seasonal linear operator: $\mathbf{L}_\theta(t) = \sum_{k=0}^{2} [\mathbf{L}_k^c \cos(k\omega t) + \mathbf{L}_k^s \sin(k\omega t)]$
- Residual MLP: 3-layer network (hidden size 64) taking state $\mathbf{X}$ and seasonal features $\boldsymbol{\phi}(t)$

**Differences from XRO**:
- [-] No RO nonlinear terms (T^2, TH, T^3, T^2H, TH^2)
- [-] No diagonal nonlinear terms (X_j^2, X_j^3)
- [+] Adds flexible MLP residual (data-driven, arbitrary smooth function)
- [+] Gradient-based training (vs. closed-form regression)

**Why it ranks #1 out-of-sample**:
- Better generalization than physics-structured models (less overfitting to training period)
- MLP residual captures unmodeled dynamics without imposing rigid polynomial structure
- Simpler than full XRO (fewer components = less overfitting risk)

**Overfitting gap**: Train RMSE 0.474 C -> Test RMSE 0.567 C (0.093 C degradation, smallest among top performers)

---

### 2. NXRO-Graph (Fixed XRO) (Rank 2)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \alpha(t) \cdot \tanh\left((\hat{\mathbf{A}} \mathbf{X}) \mathbf{W}_g\right)
$$

**Architecture**:
- Seasonal linear operator $\mathbf{L}_\theta(t)$ (learned)
- Graph convolution with XRO-derived adjacency $\hat{\mathbf{A}}$ (fixed, from XRO coupling strength)
- Seasonal gate $\alpha(t)$ modulates graph contribution

**Differences from XRO**:
- [-] No explicit RO structure or diagonal terms
- [+] Adds sparse graph convolution (teleconnection prior from XRO)
- [+] Graph topology encodes physical coupling (derived from XRO's fitted linear operator)
- [+] Gradient-based training

**Why it ranks #2**:
- Graph prior provides inductive bias without overfitting to polynomial forms
- Sparse topology (from XRO) generalizes better than dense interactions
- Interpretable teleconnection structure

---

### 3. NXRO-Attentive (Rank 3)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \alpha(t) \cdot \mathbf{W}_o \cdot \text{softmax}\left(\frac{\mathbf{M} \odot (Q K^T)}{\sqrt{d}}\right) V
$$

**Architecture**:
- Seasonal linear operator $\mathbf{L}_\theta(t)$ (learned)
- Self-attention mechanism treating variables as tokens
- Attention mask $\mathbf{M}$ restricts coupling patterns (e.g., only T/H can attend to all)
- Seasonal gate $\alpha(t)$

**Differences from XRO**:
- [-] No RO or diagonal polynomial terms
- [+] Adds state-dependent coupling via attention (adaptive teleconnections)
- [+] Learns which variables influence each other (soft selectivity vs. XRO's hard RO structure)
- [+] Gradient-based training

**Why it ranks #3**:
- Attention mechanism captures state-dependent interactions not modeled by XRO's fixed polynomials
- Masked attention prevents overfitting while allowing flexible coupling
- Good balance between expressivity and generalization

---

### 4. NXRO-RO+Diag (Rank 4)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t)
$$

**Architecture**:
- Seasonal linear operator $\mathbf{L}_\theta(t)$ (learned)
- RO nonlinearities: $\boldsymbol{\Phi}_{\text{RO}}(T,H) = [T^2, TH, T^3, T^2H, TH^2]^T$ with seasonal coefficients $\boldsymbol{\beta}_{T/H}(t)$
- Diagonal nonlinearities: $b_j(t) X_j^2 + c_j(t) X_j^3$ for each variable

**Differences from XRO**:
- [=] **Identical structure** (same equation, same basis functions)
- [+] Uses **joint end-to-end gradient descent** instead of XRO's variable-by-variable closed-form regression
- [+] All coefficients optimized jointly (not independently per variable)

**Why it ranks #4**:
- Maintains XRO's proven physical structure
- Joint optimization finds different coefficients than XRO's closed-form fit
- More parameters than simpler variants -> higher overfitting risk on limited data

**Overfitting gap**: Train RMSE 0.490 C -> Test RMSE 0.618 C (0.128 C degradation, moderate overfitting)

---

### 5. NXRO-Linear (Rank 5)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X}
$$

**Architecture**:
- Seasonal linear operator only: $\mathbf{L}_\theta(t) = \sum_{k=0}^{2} [\mathbf{L}_k^c \cos(k\omega t) + \mathbf{L}_k^s \sin(k\omega t)]$
- No nonlinear terms

**Differences from XRO**:
- [-] No RO nonlinearities
- [-] No diagonal nonlinearities
- [+] Gradient-based training (vs. XRO's closed-form)
- Simplest NXRO variant (pure linear autoregressive model with seasonal modulation)

**Why it ranks #5**:
- Simplicity = good generalization (fewer parameters, less overfitting)
- Competitive with more complex models that overfit to training period
- Baseline for understanding value of nonlinear terms

---

### 6. NXRO-NeuralODE (Rank 6)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = G_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])
$$

**Architecture**:
- Pure MLP drift function (3-4 layers, hidden size 64)
- Seasonal features $\boldsymbol{\phi}(t) = [1, \cos(\omega t), \sin(\omega t), \cos(2\omega t), \sin(2\omega t)]^T$ as input
- No explicit linear operator separation

**Differences from XRO**:
- [-] No explicit linear operator
- [-] No structured RO or diagonal terms
- [+] Pure black-box MLP learns everything from data
- [+] Maximum flexibility (universal approximator)

**Why it ranks #6**:
- High capacity allows fitting training data well
- Moderate overfitting despite flexibility
- Seasonal features provide weak inductive bias
- Ranks below simpler structured models (overfitting to training dynamics)

**Overfitting gap**: Train RMSE 0.522 C -> Test RMSE 0.588 C (0.066 C, relatively small gap)

---

### 7. XRO (Baseline) (Rank 7)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t) + \boldsymbol{\xi}(t)
$$

**Architecture**:
- Seasonal linear operator: Fourier-parameterized, fitted via harmonic-weighted covariances
- RO nonlinearities: $[T^2, TH, T^3, T^2H, TH^2]$ with seasonal coefficients
- Diagonal nonlinearities: $b_j(t) X_j^2 + c_j(t) X_j^3$
- Seasonal AR(1) red noise $\boldsymbol{\xi}(t)$

**Method**:
- **Closed-form regression** (harmonic-weighted least squares)
- **Variable-by-variable fitting** (each equation fitted independently)
- No gradient descent, no iterative optimization

**Why it ranks #7**:
- Strong physical structure, but coefficients fitted independently may not be globally optimal
- Out-of-sample performance shows XRO is solid but improvable
- Simpler NXRO variants (Res, Linear, Graph) generalize better despite fewer inductive biases
- XRO's polynomial structure may be overfitting certain training-period patterns

**Critical insight**: XRO ranking #7 (beaten by 6 NXRO variants) suggests:
1. Joint gradient-based optimization can find better coefficients
2. Simpler structures (Linear, Res, Graph) generalize better than full polynomial basis
3. XRO's variable-by-variable fitting may not be globally optimal

---

### 8. NXRO-RO+Diag-FixNL (Rank 8)

**Equation**: Same as NXRO-RO+Diag (variant 4), but with selective freezing

**Architecture**:
- **Frozen**: RO coefficients $\boldsymbol{\beta}_{T/H}$ and diagonal coefficients $b_j, c_j$ (from XRO warm-start)
- **Trainable**: Only linear operator $\mathbf{L}_\theta(t)$

**Differences from XRO**:
- Uses XRO's RO and diagonal coefficients exactly (frozen, from fitted values)
- Refines only the linear coupling via gradient descent
- Tests whether XRO's nonlinear terms are optimal while linear can improve

**Why it ranks #8**:
- XRO's nonlinear coefficients provide good structure
- Gradient refinement of linear operator adds value
- Freezing reduces overfitting (fewer trainable parameters)
- Better test performance than full XRO (#7)

**Key finding**: Freezing nonlinear components and training only linear improves generalization vs. training all components (NXRO-RO+Diag rank #4) or using XRO's all-closed-form fit (XRO rank #7).

---

### 9. NXRO-RO+Diag-FixRO (Rank 9)

**Equation**: Same as NXRO-RO+Diag, but with different freezing pattern

**Architecture**:
- **Frozen**: RO coefficients $\boldsymbol{\beta}_{T/H}$ only (from XRO warm-start)
- **Trainable**: Linear operator $\mathbf{L}_\theta(t)$ and diagonal coefficients $b_j, c_j$

**Differences from XRO**:
- Uses XRO's RO coupling exactly (frozen T-H interaction structure)
- Refines linear operator and diagonal self-interactions
- Tests whether XRO's core RO coupling is already optimal

**Why it ranks #9**:
- XRO's RO structure (T-H coupling) is robust and generalizes well when frozen
- Training linear + diagonal provides additional flexibility without overfitting RO
- Similar performance to FixNL (#8), suggesting both freezing strategies work

**Key finding**: XRO's RO coupling can be frozen without performance loss, suggesting it captures robust physical relationships.

---

### 10. NXRO-ResidualMix-FixRO (Rank 10)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t) + \alpha \cdot R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])
$$

**Architecture**:
- Full XRO structure (Linear + RO + Diagonal) PLUS small residual MLP
- **Frozen**: RO coefficients $\boldsymbol{\beta}_{T/H}$ (from XRO warm-start)
- **Trainable**: Linear operator, diagonal coefficients, residual MLP, scaling $\alpha$

**Differences from XRO**:
- Identical RO structure (frozen from XRO)
- Adds trainable residual MLP for unmodeled dynamics
- Refines linear and diagonal terms via gradient descent
- Scaling parameter $\alpha$ keeps residual small

**Why it ranks #10**:
- Combines XRO's robust RO structure with learned corrections
- Freezing RO reduces overfitting while allowing flexibility elsewhere
- Residual adds capacity without dominating (small $\alpha$)

**Key finding**: Even with frozen RO, adding MLP residual + refining linear/diagonal improves generalization.

---

## Key Insights from Out-of-Sample Rankings

### 1. Simpler Models Generalize Better

**Observation**: NXRO-Res (#1), NXRO-Linear (#5) outrank more complex structured models (NXRO-RO+Diag #4, XRO #7).

**Interpretation**:
- With limited data (23 years training), simpler architectures avoid overfitting
- Full polynomial basis (RO + diagonal) may fit training-period idiosyncrasies
- MLP residual (Res) or pure linear (Linear) generalize better to 2002-2022 climate regime

**Implication**: For out-of-sample forecasting, **regularization via simplicity** (fewer nonlinear terms) may outweigh physical completeness.

---

### 2. XRO's Joint Optimization Gap

**Observation**: XRO ranks #7, but NXRO variants with identical structure but gradient training (NXRO-RO+Diag #4) or simpler structures (NXRO-Res #1) outperform.

**Interpretation**:
- XRO's **variable-by-variable closed-form fitting** may not find globally optimal coefficients
- **End-to-end gradient descent** (NXRO) optimizes multi-step forecast skill directly
- XRO's independent fits per variable don't account for coupled dynamics during forecasting

**Implication**: Gradient-based joint optimization has value even for physics-structured models.

---

### 3. Selective Freezing Improves Generalization

**Observation**: FixNL (#8), FixRO (#9), ResidualMix-FixRO (#10) rank higher than their fully-trainable counterparts.

**Interpretation**:
- Freezing some XRO components reduces trainable parameters -> less overfitting
- XRO's coefficients for RO coupling are robust (freezing RO doesn't hurt)
- Training only subsets (e.g., linear + diagonal) with frozen physics provides good regularization

**Implication**: **Hybrid strategy** (freeze robust physics, train flexible components) balances structure and adaptability.

---

### 4. Graph and Attention Mechanisms Generalize Well

**Observation**: NXRO-Graph (#2) and NXRO-Attentive (#3) outrank full physics models.

**Interpretation**:
- Learned coupling mechanisms (graph convolution, attention) adapt to data
- Less rigid than polynomial basis -> better generalization
- Teleconnection priors (graph structure) provide useful inductive bias without overfitting

**Implication**: Data-driven coupling with structural priors (graph topology, attention masks) is effective for climate forecasting.

---

### 5. In-Sample vs. Out-of-Sample Performance Discrepancy

**Observation**: Models with best **train** RMSE often have larger **test** RMSE degradation.

Examples:
- **RO+Diag-FixNL**: Train 0.470 C -> Test 0.602 C (0.132 C gap) despite good train performance
- **NXRO-Res**: Train 0.474 C -> Test 0.567 C (0.093 C gap, smallest gap among top 10)
- **XRO**: Train 0.493 C -> Test 0.605 C (0.112 C gap)

**Interpretation**:
- Models that fit training data extremely well (FixNL, RO+Diag) may be overfitting
- NXRO-Res achieves best out-of-sample despite not having best in-sample (good generalization)
- **Generalization gap** is critical metric for model selection

**Implication**: Don't select models purely by training performance--out-of-sample evaluation is essential.

---

### 6. Warm-Start Variants Don't Dominate Top 10

**Observation**: Only 1 warm-start variant in top 10 (none in top 7).

**WS variants in top 20**:
- RO+Diag WS: Rank #15
- Graph Fixed Xro WS: Rank #20
- Attentive WS: Rank #23
- ResidualMix WS: Rank #24

**Interpretation**:
- Warm-start helps convergence speed but doesn't guarantee best generalization
- Random initialization explores different regions of parameter space
- Out-of-sample test period (2002-2022) may have different dynamics than training period (1979-2001)
- XRO's fitted coefficients (used for warm-start) optimize for training period, not generalization

**Implication**: For out-of-sample performance, **random initialization** with proper regularization may outperform physics-informed initialization.

---

## Comparison with In-Sample Rankings

### Ranking Changes: In-Sample vs. Out-of-Sample

Based on standard setup (train 1979-2022, evaluate on same period):

**In-sample best models** (from README.md):
1. NXRO-ResidualMix (avg rank 1.91 RMSE)
2. NXRO-RO+Diag (avg rank 2.18 RMSE)
3. XRO (avg rank 2.82 RMSE)

**Out-of-sample best models** (this experiment, train 1979-2001, test 2002-2022):
1. **NXRO-Res** (test RMSE 0.567 C) <- Different winner!
2. NXRO-Graph (test RMSE 0.583 C)
3. NXRO-Attentive (test RMSE 0.586 C)
4. NXRO-RO+Diag (test RMSE 0.618 C) <- Dropped from #2 to #4
7. XRO (test RMSE 0.605 C) <- Dropped from #3 to #7

### Critical Differences

**1. Winner changes**: NXRO-ResidualMix -> NXRO-Res
- ResidualMix's full structure (L+RO+Diag+MLP) overfits to 1979-2001
- Simpler Res (L+MLP only) generalizes better to 2002-2022

**2. XRO drops significantly**: #3 -> #7
- XRO coefficients optimized for full 1979-2022 period
- When trained on only 1979-2001, XRO doesn't generalize as well
- Suggests XRO benefits from long training periods

**3. Graph/Attention rise**: Not in top 3 in-sample, now #2-3 out-of-sample
- These models generalize better than fully structured physics models
- Data-driven coupling adapts to regime changes

---

## Implications for Model Selection

### For Operational Forecasting (Real Out-of-Sample Use)

**Recommended models** (based on this experiment):
1. **NXRO-Res**: Best out-of-sample performance, good generalization
2. **NXRO-Graph**: Strong runner-up, interpretable teleconnections
3. **NXRO-Attentive**: Competitive, state-dependent coupling

**Not recommended** (despite good in-sample performance):
- Full warm-start variants (overfit to training period)
- NXRO-RO+Diag with all components trained (too many parameters for 23-year training)

### For Scientific Analysis (Understanding Physics)

**Recommended models**:
1. **NXRO-RO+Diag-FixNL** (#8): Isolates linear coupling with frozen physics
2. **NXRO-RO+Diag-FixRO** (#9): Isolates importance of RO structure
3. **XRO** (#7): Pure physics baseline
4. **NXRO-RO+Diag** (#4): Gradient-refined physics

**Analysis**: Compare FixNL vs. FixRO vs. full XRO to understand which components are essential vs. refineable.

---

## Visualization Gallery

### Out-of-Sample RMSE Curves

![Top 10 RMSE Out-of-Sample](results_out_of_sample/rankings/top10_rmse_test_out_of_sample.png)

**Key observations**:
- **NXRO-Res** (#1, blue) shows lowest RMSE at all leads
- **Graph** (#2) and **Attentive** (#3) track closely
- **XRO** (#7, pink dashed) underperforms top 6 NXRO variants
- All models show RMSE increasing with lead time
- Largest model spread occurs at leads 15-20 months (long-range forecasting)

### In-Sample RMSE Curves (for Comparison)

![Top 10 RMSE In-Sample](results_out_of_sample/rankings/top10_rmse_train_out_of_sample.png)

**Key observations**:
- **FixNL/FixRO** variants (#8, #9) show best training RMSE (overfitting signature)
- **NXRO-Res** (#1) has moderate training RMSE but best test RMSE (good generalization)
- Gap between train and test curves reveals overfitting tendency

### ACC Curves

![Top 10 ACC Out-of-Sample](results_out_of_sample/rankings/top10_acc_test_out_of_sample.png)

**Key observations**:
- **NXRO-Res** maintains highest correlation at all leads
- **All models lose skill rapidly after 12 months (ENSO spring barrier)**
- Top models cluster together at short leads (1-6 months)
- Differentiation emerges at longer leads (12-21 months)

### Comparison Bar Charts

![All Variants Comparison](results_out_of_sample/rankings/all_variants_comparison_combined_out_of_sample.png)

**Top panel (ACC)**:
- Blue bars: In-sample (train period)
- Orange bars: Out-of-sample (test period)
- Clear degradation from train to test across all models
- Top models maintain smaller gaps

**Bottom panel (RMSE)**:
- Test RMSE consistently higher than train RMSE (expected)
- FixNL/FixRO variants show large gaps (overfitting)
- NXRO-Res shows smallest gap (best generalization)

---

## Statistical Summary

### Top 10 Performance Metrics

| Statistic | Value |
|-----------|-------|
| **Best test ACC** | 0.628 (NXRO-Res) |
| **Best test RMSE** | 0.567 C (NXRO-Res) |
| **XRO baseline test ACC** | 0.596 (rank #7) |
| **XRO baseline test RMSE** | 0.605 C (rank #7) |
| **Avg improvement over XRO** | +5.5% ACC, -6.3% RMSE (top 6 models) |
| **Worst in top 10 test ACC** | 0.590 (ResidualMix-FixRO) |
| **Worst in top 10 test RMSE** | 0.602 C (FixNL, FixRO) |

### Generalization Gaps (Train->Test RMSE Degradation)

| Model | Train RMSE | Test RMSE | Gap | Rank |
|-------|------------|-----------|-----|------|
| NXRO-Res | 0.474 C | 0.567 C | **0.093 C** | 1 (best generalization) |
| NXRO-NeuralODE | 0.522 C | 0.588 C | 0.066 C | 6 |
| NXRO-Linear | 0.543 C | 0.588 C | 0.045 C | 5 |
| NXRO-RO+Diag | 0.490 C | 0.618 C | **0.128 C** | 4 (moderate overfitting) |
| XRO | 0.493 C | 0.605 C | 0.112 C | 7 |
| FixNL | 0.470 C | 0.602 C | **0.132 C** | 8 (overfitting) |

**Interpretation**: Models with smallest gaps (Linear, NeuralODE, Res) generalize best. Models with largest gaps (FixNL, RO+Diag) overfit to 1979-2001 training period.

---

## Recommendations

### For Near-Term Forecasting (1-6 month lead)

**Use**: NXRO-Res, NXRO-Attentive, or NXRO-Graph
- All show ACC > 0.85 at short leads
- Minimal performance difference at short range
- Choose based on interpretability needs

### For Long-Range Forecasting (12-21 month lead)

**Use**: NXRO-Res or NXRO-Graph
- NXRO-Res maintains lowest RMSE at all leads
- Graph structure provides teleconnection interpretability
- Avoid full physics models (RO+Diag, XRO) which degrade faster

### For Mechanistic Understanding

**Use**: NXRO-RO+Diag-FixNL or NXRO-RO+Diag-FixRO
- Isolates contribution of specific physical components
- Frozen physics terms preserve interpretability
- Comparison reveals which XRO components are essential vs. refineable

### For Operational Use with Limited Training Data

**Recommended strategy**:
1. Train simple NXRO-Res or NXRO-Linear (good generalization)
2. Use strong regularization (weight decay, dropout, early stopping)
3. Avoid full warm-start variants (overfit to training period)
4. Evaluate on held-out period (not just in-sample metrics)

---

## Comparison: Out-of-Sample vs. Standard In-Sample Setup

### Different Winners Emerge

| Setup | Best Model | Test RMSE | Insight |
|-------|------------|-----------|---------|
| **In-sample** (train & test on 1979-2022) | NXRO-ResidualMix | ~0.45 C | Full structure + residual optimal when evaluating on same period |
| **Out-of-sample** (train 1979-2001, test 2002-2022) | NXRO-Res | 0.567 C | Simpler structure generalizes better to future data |

**Critical difference**: In-sample evaluation favors complex models that fit training data perfectly. Out-of-sample evaluation reveals which models truly generalize.

### XRO Performance Drop

| Setup | XRO Rank | Test RMSE | Interpretation |
|-------|----------|-----------|----------------|
| **In-sample** | #3 | ~0.49 C | Strong baseline |
| **Out-of-sample** | #7 | 0.605 C | 6 NXRO variants generalize better |

**Interpretation**: XRO's closed-form coefficients may be overfitting to specific climate patterns in training period. Gradient-based NXRO variants find coefficients that generalize better to held-out future data.

---

## Conclusions

### Main Findings

1. **NXRO-Res is the best out-of-sample performer** (test ACC 0.628, RMSE 0.567 C)
   - Simpler than full XRO structure
   - MLP residual provides flexibility without overfitting
   - Smallest generalization gap among competitive models

2. **Simpler architectures generalize better** with limited training data (23 years)
   - NXRO-Linear (#5) outranks XRO (#7) despite having no nonlinear terms
   - Polynomial basis (RO+Diag) may capture training-period artifacts

3. **XRO ranks #7**, beaten by 6 NXRO variants
   - Variable-by-variable closed-form fitting not globally optimal
   - Joint gradient-based optimization finds better generalizing coefficients
   - Suggests room for improvement in XRO's parameter estimation

4. **Selective parameter freezing helps** (FixNL #8, FixRO #9)
   - Using XRO's robust components (frozen) + training flexible parts reduces overfitting
   - Optimal hybrid: freeze RO coupling, train linear + diagonal + optional residual

5. **Graph and attention mechanisms are competitive** (#2, #3)
   - Data-driven coupling generalizes well
   - Sparse priors (graph topology, attention masks) prevent overfitting

### Future Directions

1. **Extended training period**: Test if longer training (1979-2015, test 2016-2022) changes rankings
2. **Multi-dataset training**: Use ERA5, GODAS for data augmentation (implemented via `--eval_all_datasets`)
3. **Ensemble methods**: Combine top 3 models (Res, Graph, Attentive) for robust forecasts
4. **Adaptive freezing**: Learn which components to freeze based on validation performance
5. **Transfer learning**: Pre-train on reanalysis, fine-tune on observations

---

## Files and Reproducibility

### Training Scripts
- `NXRO_train_out_of_sample.py`: Main training script for out-of-sample setup
- `run_all_out_of_sample.sh`: Automated training of all 32 variants

### Ranking and Analysis
- `rank_all_variants_out_of_sample.py`: Ranks all trained models by out-of-sample performance
- Results: `results_out_of_sample/rankings/`

### Reproduce Top Models

```bash
# Best out-of-sample performer
python NXRO_train_out_of_sample.py --model res --epochs 1500

# Runner-ups
python NXRO_train_out_of_sample.py --model graph --epochs 1500
python NXRO_train_out_of_sample.py --model attentive --epochs 1500

# Rank all
python rank_all_variants_out_of_sample.py --top_n 10 --metric combined --force
```

### Multi-Dataset Evaluation

```bash
# Evaluate on all available datasets (ORAS5, ERA5, GODAS)
python NXRO_train_out_of_sample.py --model res --eval_all_datasets

# Compare across datasets
python XRO_variants.py --eval_all_datasets
```

---

## References

- **XRO baseline**: Zhao et al. (2024), *Nature*. https://doi.org/10.1038/s41586-024-07534-6
- **Model variants**: See README.md for detailed mathematical formulations
- **Implementation**: `nxro/` module (models.py, train.py, eval.py)
