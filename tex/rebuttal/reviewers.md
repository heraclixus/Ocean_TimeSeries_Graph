# w9cG

Strengths:
The paper proposes a well-motivated hybrid framework that combines a physics-based model with neural residual dynamics. This design is intuitive and makes good use of domain knowledge while still allowing the model to capture complex nonlinear interactions.
The experimental study is fairly comprehensive, comparing multiple variants of the proposed model with both physics-based and purely neural baselines, which helps demonstrate the benefits of the hybrid approach.
Weaknesses:
The paper argues that the neural residual captures nonlinear dynamics beyond the physics-based XRO model, but it is not entirely clear how much of the performance gain comes from the neural component versus improved optimization of the linear dynamics. A more detailed ablation or analysis separating these effects would make the claim stronger.
While the hybrid formulation is appealing, the paper does not fully discuss the identifiability or interpretability of the learned neural residual. In particular, it is unclear whether the residual component may absorb dynamics that should theoretically belong to the physical model, which could limit the physical interpretability of the resulting system.
The proposed approach includes three variants, namely NXRO-MLP, NXRO-Attn, and NXRO-GNN, which seems somewhat problematic to me. At this point, it no longer feels like a single method, but rather three quite different methods, since MLP, attention, and GNN are fundamentally different architectures.
Detailed Review:
Questions:

The neural residual component is intended to capture nonlinear dynamics not modeled by the physics-based XRO component. Is there any analysis showing what types of interactions or patterns the residual network actually learns, and how these relate to known climate dynamics?
The paper shows that pretraining on climate simulation data (CESM) can sometimes degrade performance due to distribution mismatch. Could the authors further analyze this phenomenon and discuss potential strategies (e.g., domain adaptation or bias correction) to better leverage simulation data?


# cGpF

Strengths:
Effective Hybrid Decomposition: The explicit separation of dynamics into a seasonally-modulated linear physical operator $L_{\theta}(t)$ and a nonlinear neural residual $R_{\phi}$ provides a strong inductive bias for small-data climate tasks.
Improved Uncertainty Calibration: The introduction of a two-stage training procedure with likelihood-optimized AR(1) noise effectively addresses the overconfidence issues found in traditional physical baselines. 3.Sample Efficiency: The framework achieves competitive RMSE results using only 276 monthly observational samples, outperforming high-capacity models like Transformers that suffer from overfitting in this regime.
Weaknesses:
Lack of AI Architectural Innovation: The AI contribution is limited to the straightforward application of standard modules (MLP, GNN, Attention) as residual terms. There is no novel architecture design tailored to the specific symmetries or physical constraints of climate dynamics.

Internal Contradiction in Interpretability Claims: The authors argue that pure AI models lack interpretability, yet their own neural residual component $R_{\phi}$ remains a "black box" without physical validation. The GNN-based basin correlations in Section 4.5 only provide post-hoc statistical confirmation of known patterns, failing to bridge the gap between the admitted opacity of neural networks and the claimed transparency of the hybrid framework.

Insufficient Addressing of the Spring Predictability Barrier (SPB): The use of Fourier features $\phi(t)$ and seasonal gates $\alpha(t)$ only makes the model time-aware but does not provide a dynamical solution to the signal-to-noise decay in spring. Specific experimental evidence or mechanisms for resolving SPB are missing.

Limited Baseline Comparisons and Benchmarking against SOTA Models: The current evaluation lacks a rigorous comparison with representative advanced AI4Science (AI4S) architectures and state-of-the-art (SOTA) ENSO-specific models.

Overgeneralized Conclusions on Transfer Learning: The claim that pre-training on synthetic data is detrimental is based solely on one climate model (CESM2) and one fine-tuning strategy, without accounting for potential model-specific biases or testing adaptive transfer methods.

Detailed Review:
Architectural Innovation Beyond Standard Modules: The current implementation of the residual term $R_{\phi}$ relies on standard MLP, GNN, or Attention blocks. In the context of AI4Science, the lack of domain-specific architectural innovation is a significant weakness. The authors should consider incorporating physical constraints directly into the neural architecture—such as equivariant layers to respect spatial symmetries or projection layers to enforce conservation laws—rather than treating the residual as a purely statistical "off-the-shelf" correction term.

Resolving the Interpretability Contradiction: The authors position the hybrid NXRO framework as a solution to the "black-box" nature of pure AI models. However, the core of the nonlinear correction resides in $R_{\phi}$, which remains uninterpretable[cite: 54, 24]. Section 4.5 merely confirms that the GNN utilizes known basin data, which is a post-hoc statistical alignment rather than a mechanistic explanation. To bridge this gap, the authors must demonstrate the "internal physical logic" of the neural term. For instance, they could analyze whether $R_{\phi}$ effectively captures specific nonlinear physical mechanisms (e.g., amplitude-dependent discharge rates or state-dependent noise) that are explicitly missing from the linear $L_{\theta}(t)$.

Empirical Validation of SPB Mitigation: While the paper claims to address the Spring Predictability Barrier (SPB), the provided seasonal features $\phi(t)$ and gates $\alpha(t)$ only grant the model time-awareness without necessarily solving the underlying dynamical instability of the spring season. The authors should perform a rigorous ablation study by removing $\alpha(t)$ or $\phi(t)$ and plotting the resulting forecast error growth curves. Specifically, they need to show that the performance gain is concentrated during the spring transition phase, thereby proving a targeted mitigation of the SPB.

Benchmarking Against State-of-the-Art AI4S and Foundation Models: Specifically, the authors should not overlook critical domain-specific advancements, such as self-attention–based three-dimensional multivariate modeling for ENSO prediction and the physics-informed insights on how subsurface mixing promotes Central Pacific-like ENSO in Intermediate Coupled Models (e.g., the work by Zhang et al.), and incorporating these as baselines or discussion points is crucial for positioning this research within the current landscape of skillful ENSO forecasting.

Broadening the Scope of Transfer Learning Analysis: The conclusion that pre-training on synthetic data leads to "negative transfer" is currently based on a single climate model (CESM2). This finding might be an artifact of the specific systematic biases in CESM2 rather than a general property of ENSO modeling. To substantiate this claim, the authors should repeat the pre-training experiment using other CMIP6 models.


# jHw1

Strengths:
Well-motivated hybrid physics–ML design for small data: Decomposing the dynamics as $dX/dt = L_\theta(t)X + R_\phi ([X, \psi(t)])$  keeps the seasonally varying linear dynamics from XRO while learning only nonlinear residual structure, which is a sensible inductive bias under scarce observations.
End-to-end multi-horizon training matches the forecasting objective: Optimizing rollout error over leads up to 21 months is more aligned with operational ENSO forecasting than one-step regression fits.
Broad and informative model comparison/ablation effort: The paper explores many variants (MLP/attention/GNN residuals; warm-start/freezing; alternative nonlinearities), providing useful empirical insight into which inductive biases help.
Interpretability-oriented components: Learned adjacency structure and seasonal gating in graph residual variants provide an interpretable lens on teleconnections, which fits AI4Science goals.
Probabilistic forecasting included with proper scoring rules: Using CRPS and spread/RMSE is appropriate; the attempt to improve calibration via a dedicated noise-fitting stage is practically relevant.
Weaknesses:
Evaluation protocol likely conflates model selection with test evaluation: Extensive architecture search (~43 variants) is reported, but there is no clearly described validation/nested evaluation for selecting architectures/hyperparameters, creating a high risk of test-set overfitting.
Potential preprocessing leakage in anomaly/climatology/trend computation: Anomalies are computed using climatology over 1979–2022 and a trend over 1979–2023, overlapping the test period and potentially leaking future information.
Probabilistic section has clarity/correctness issues: The text/caption states “higher CRPS is better,” which is incorrect (lower is better). It is also unclear whether noise-optimization improvements are fairly isolated from drift-model improvements.
Baselines may be underpowered for the small-data regime: The transformer baseline overfits (expected), but stronger classical/statistical and small-ML baselines are missing, weakening claims about the necessity of the hybrid approach.
Limited robustness/statistical reporting for relatively modest gains: Improvements (e.g., ~8–13%) are plausible but could be within variability for correlated climate time series; no clear reporting of multi-seed variance, confidence intervals, or significance testing.
Detailed Review:
Major Issues

Model selection vs. test evaluation (risk of inflated gains)
Issue: The paper emphasizes “extensive neural architecture search” with many variants and then highlights best-performing results on a single held-out test period (2002–2022). Without a clearly separated validation protocol, selecting the best architecture/hyperparameters on test can inflate improvements. In small-data, autocorrelated time series settings, even small design choices can produce non-trivial variability. If the test set is used (directly or indirectly) for selection, the reported gain relative to XRO may not generalize.

Recommendations:

Use a strict train/validation/test split (e.g., train 1979–1995, val 1996–2001, test 2002–2022) where all architecture/hyperparameter decisions are made on validation.
Preferably report rolling-origin / blocked CV across multiple contiguous periods and average skill.
Preprocessing leakage in climatology / de-trending
Issue: Appendix B states monthly anomalies subtract the 1979–2022 climatology and remove a quadratic trend fit over 1979–2023. This uses information from the evaluation period to define anomalies. This can artificially improve forecast metrics by making train and test distributions more similar (or removing low-frequency structure using future information). Even if the effect is modest, it undermines strict out-of-sample evaluation.

Recommendations:

Recompute climatology and trend using training period only, and apply those transforms to validation/test (or use an expanding-window approach).
Report whether conclusions (rankings and % improvements) hold under leakage-free preprocessing.
Probabilistic forecasting: correctness/clarity and fair comparison
Issue: The paper says higher CRPS is better, which is incorrect; lower CRPS is better (the table appears to use lower-is-better). It is not fully clear whether improvements come from (i) a better drift model, (ii) the stage-2 noise fitting, or (iii) both, and whether comparisons are controlled.

Recommendations:

Fix the CRPS description/captions.
Add a controlled ablation: for each drift model (XRO and best NXRO), compare (a) post-hoc AR(1) fitting (as in XRO [56]) vs. (b) your likelihood-optimized AR(1), holding everything else fixed (ensemble size, initialization, lead times).
Consider adding rank histograms / PIT-like diagnostics (where appropriate) or reliability diagrams to better support calibration claims.
Baselines and claims about pure DL vs hybrid models
Issue: The transformer baseline’s poor performance is plausible in small data, but the baseline suite lacks strong small-data alternatives (regularized statistical baselines or compact NNs) that are commonly competitive.

Recommendations:

Add at least a few stronger baselines, e.g., regularized VAR/LIM (ridge/lasso) on lagged indices, ARIMA/ETS (for Niño3.4 alone) or multivariate regression, a small LSTM/GRU with strong regularization and early stopping.
Ensure all baselines use the same leakage-free preprocessing and fair tuning on validation.
Robustness/statistical significance and reproducibility details
Issue: Given modest gains and high temporal autocorrelation, the paper should quantify uncertainty and provide key implementation details that affect results.

Recommendations:

Report results over multiple random seeds (mean ± std) for the main comparisons.
Provide confidence intervals via block bootstrap over initialization times or years.
Include solver/training specifics: ODE solver type, step size, whether integration uses Euler discretization, gradient method, and any stability measures.
Additional notes / minor presentation issues

Several references are duplicated (e.g., [12]/[13], [25]/[26]).
Some equations (e.g., Eq. 6) are difficult to parse as typeset; please revise for clarity.
Wording like “out-of-distribution” for a future time split may be overstated unless distribution shift is quantified.


# i9br

Strengths:
The paper is well-written and easy to follow. The paper introduces the background on the problem they are focusing and physics-based model XRO. The authors later illustrate the gap they are trying to close: data scarcity and lack of explainability.

The design of the method is well-motivated and innovative. It addresses the specific scenario with designs that model them clearly. For example, the seasonal context features and graph-based neural network that targets the limitation they aiming for. It clearly incorporates physics and explainability in it.

Comprehensive and extensive experimental results. The authors even proposed 43 variants of the model they proposed and report the results.

Weaknesses:
More in-depth analysis on the choice of the alternative design is needed.

The paper fail to address how well the model performs under data scarcity scenarios in practical applications.

Detailed Review:
In the main text, the author listed two alternative nonlinear components that uses graph-based neural network and attention-based neural network where the graph-based neural network has learnable adjacency matrix. Mathematically, these two networks seem very similar and their performance is also very similar. Maybe some specially designed graph neural network without learnable adjacency matrix is better as a comparison, e.g., more physics-based design that tailored for specific system. This could gain even more perspective as in how human knowledge could help these models.

The paper aims to address two limitations: data scarcity and explainability. However, from the experimental results, I'm not seeing how well the model addresses the data scarcity. It would be even more highlighted and more significance to examine how much data is required exactly. How accurate the model is if we don't have these many years of data.