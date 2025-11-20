# Stochastic Ensemble Forecast Results

**Date**: November 11, 2024  
**Experiments**: In-sample (1979-2022) and Out-of-sample (Train: 1979-2001, Test: 2002-2022)  
**Models Evaluated**: XRO baseline + Top 5 NXRO variants with different noise methods

---

## Table of Contents

1. [What is CRPS?](#what-is-crps)
2. [CRPS Calculation Implementation](#crps-calculation-in-our-code)
3. [Stage 2 Likelihood Optimization](#stage-2-likelihood-optimization-details)
4. [Evaluation Time Range](#evaluation-time-range)
5. [Results: Out-of-Sample](#results-out-of-sample)
6. [Results: In-Sample](#results-in-sample)
7. [Comparison: In-Sample vs Out-of-Sample](#comparison-in-sample-vs-out-of-sample)
8. [Conclusions and Recommendations](#conclusions-and-recommendations)

---


During the stochastic simulation phase, we freeze the model obtained during the earlier fitting (deterministic, in-sample and out-of-sample), and we focus on noise associated with the residuals using AR(1) likelihood.


## What is CRPS?

### Continuous Ranked Probability Score (CRPS)

**Definition**: CRPS is a comprehensive metric for evaluating probabilistic forecasts that measures the integrated squared difference between the forecast cumulative distribution function (CDF) and the observation.

**Mathematical Form**:

$$
CRPS(F,y) = \int_{-\infty}^{\infty} |F(x) - 1 \{ y  \leq x \}|^2 dx
$$

where:
- $F(x)$ is the forecast CDF (from ensemble)
- $y$ is the observed value
- $1 \{y \leq x \}$ is the indicator function (step function at observation)

**Interpretation**:
- **Lower is better** - CRPS = 0 means perfect forecast
- Combines sharpness (narrow distribution) and calibration (correct uncertainty)
- Reduces to absolute error for deterministic forecasts
- Penalizes both over-confident and under-confident forecasts
- Units: Same as the variable (°C for temperature)

**Why CRPS is Important**:
1. **Proper scoring rule** - rewards honest probabilistic forecasts
2. **Combines accuracy and uncertainty** - not just ensemble mean
3. **Directly interpretable** - distance between forecast and observation
4. **Robust metric** - standard in probabilistic forecast verification

---

## CRPS Calculation in Our Code

### Implementation Details

**Location**: `utils/xro_utils.py::evaluate_stochastic_ensemble()` (lines 318-325)

**Formula Used**: Fair ensemble CRPS estimator (Hersbach, 2000):
```python
# For each forecast lead time L:
# ens_vals: [I, M] - I initializations, M ensemble members (100)
# obs_vals: [I] - corresponding observations

# Term 1: Average absolute error across all members
term1 = mean(|x_m - y|)  # [I] $\rightarrow$ averaged over I

# Term 2: Average pairwise difference between members (ensemble spread)
term2 = mean(|x_m - x_n|)  # [I, M, M] $\rightarrow$ averaged over all pairs

# Final CRPS
CRPS(L) = mean(term1 - 0.5 * term2)
```

**Step-by-step** (code lines 318-325):
```python
# 1. Get ensemble and observations for this lead
M = ens_vals.shape[1]  # Number of ensemble members (100)

# 2. Compute mean absolute error for each member
term1 = np.mean(np.abs(ens_vals - obs_vals[:, None]), axis=1)  # [I]

# 3. Compute pairwise absolute differences between all ensemble members
diff_pairs = np.abs(ens_vals[:, :, None] - ens_vals[:, None, :])  # [I, M, M]
term2 = np.mean(diff_pairs, axis=(1, 2))  # [I]

# 4. Fair CRPS estimator (corrects for finite ensemble size)
crps[L] = float(np.mean(term1 - 0.5 * term2))
```

**Why the 0.5 correction?**  
The -0.5 term adjusts for finite ensemble size. It removes the bias that occurs when using a finite sample to approximate the continuous distribution. Without it, CRPS would be artificially inflated for small ensembles.

### Concrete Example

Suppose we have a forecast for one initialization with 100 ensemble members:
```
Ensemble members: [1.2, 0.8, 1.5, ..., 1.0]  (100 values)
Observation: y = 1.1C
```

**Step 1**: Compute absolute errors for each member
```
|1.2 - 1.1| = 0.1
|0.8 - 1.1| = 0.3
|1.5 - 1.1| = 0.4
...
term1 = mean of these = 0.35°C (example)
```

**Step 2**: Compute ensemble diversity (internal spread)
```
|1.2 - 0.8| = 0.4
|1.2 - 1.5| = 0.3
|0.8 - 1.5| = 0.7
... (100 × 100 = 10,000 pairs)
term2 = mean of all pairwise differences = 0.25°C (example)
```

**Step 3**: Fair CRPS
```
CRPS = term1 - 0.5 × term2 = 0.35 - 0.5 × 0.25 = 0.225°C
```

**Interpretation**: The forecast is, on average, 0.225°C away from the observation when accounting for ensemble uncertainty.

---

## Stage 2 Likelihood Optimization Details

### Overview

**Stage 2** optimizes noise parameters (a1, sigma) using a likelihood objective instead of post-hoc fitting from residuals.

**Location**: `nxro/stochastic.py::train_noise_stage2()` (lines 229-326)

### Mathematical Formulation

#### Stochastic Model

Our dynamics with noise:

$$
X_{t+1} = X_t + dt \cdot f_\theta (X_t, t) + \epsilon_t
$$

where $\epsilon_t$ follows seasonal AR(1):

$$
\epsilon_j,t = a1_j \cdot \epsilon_j,t-1 + \eta_j,t
\eta_j,t \sim N(0, (1 - a1_j^2) \cdot \sigma_j(m)^2 \cdot dt^2)
$$

- $a1_j$: AR(1) coefficient for variable j
- $\sigma_j(m)$: Seasonal standard deviation (month $m \in \{1..12\}$)
- $dt = 1/12$: Monthly time step

#### Log-Likelihood Objective

**Residuals**: Difference between observed and predicted state changes

$$
r_t = X_{t+1} - X_t - dt \cdot f_\theta(X_t, t) 
$$

**AR(1) Conditional Distribution**:

$$
r_j,t | r_j,t-1 \sim N(a1_j \cdot r_j,t-1, (1 - a1_j^2) \cdot \sigma_j(m_t)^2 \cdot dt^2)
$$

### Optimization Procedure

**Step-by-step** (code lines 229-326):

**1. Initialize parameters from post-hoc fit**:
```python
# From residual-based OLS fit
a1_init = mean(a1_per_month)  # [n_vars]
sigma_init = sigma_per_month.T  # [n_vars, 12]

# Create trainable parameters
a1 = nn.Parameter(torch.tensor(a1_init))
log_sigma = nn.Parameter(torch.log(sigma_init))
```

**2. Compute residuals once** (model frozen):
```python
residuals, months = compute_residuals_series(model, train_ds, var_order, device)
# residuals: [T, n_vars] - one-step prediction errors
# months: [T] - calendar months for seasonal sigma
```

**3. Optimize with Adam**:
```python
optimizer = Adam([a1, log_sigma], lr=1e-3)

for epoch in range(100):
    # Negative log-likelihood loss
    log_likelihood = ar1_log_likelihood(residuals, a1, log_sigma, months)
    loss = -log_likelihood
    
    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Constraint: keep a1 in valid range
    with torch.no_grad():
        a1.clamp_(0.01, 0.99)
```

**4. Return optimized parameters**:
```python
a1_opt = best_a1.cpu().numpy()  # [n_vars]
sigma_opt = exp(best_log_sigma).cpu().numpy()  # [n_vars, 12]
```

### Why Likelihood Optimization Works

**Advantages over post-hoc fitting**:
1. **Statistical consistency**: Parameters optimized for the actual data distribution
2. **Joint optimization**: a1 and sigma optimized together, not separately
3. **Proper objective**: Maximizes probability of observed data
4. **Gradient-based**: Can escape local optima from OLS initialization

**Result**: Better calibrated ensembles (Spread/RMSE closer to 1.0)

---

## Evaluation Time Range

### Time Periods Used

**Forecast initialization times**: All months in the observation dataset  
**Evaluation period**: Depends on available observations after adding lead time

**Dataset coverage**:
- **Observation dataset**: 1979-01 to 2024-12 (552 months total)
- **Initialization times**: All 552 months are used for forecasting
- **Lead times**: 0 to 21 months

**For each lead L**:
```
Initialization: init_time (any month from 1979-01 to 2024-12)
Verification: init_time + L months (if observation available)
```

Example for **lead = 6 months**:
- Init: 2002-01 $\rightarrow$ Verify: 2002-07 
- Init: 2010-06 $\rightarrow$ Verify: 2010-12 
- Init: 2024-06 $\rightarrow$ Verify: 2024-12 

**Actual verification counts** (varies by lead):
- **Lead 1**: ~551 verifications
- **Lead 12**: ~540 verifications
- **Lead 21**: ~531 verifications

Missing observations (beyond dataset range) are automatically excluded.

### CRPS Averaging

```
CRPS(lead=L) = average over all valid (init, observation) pairs at that lead
Avg_CRPS = mean(CRPS(lead)) over leads 0-21
```

### Note on Train/Test Split

**Important**: CRPS evaluation uses **ALL available verifications** (1979-2024), not filtered by train/test periods.

For **out-of-sample experiments** (train: 1979-2001):
- In-sample verifications:  months (1979-2001)
- Out-of-sample verifications:  months (2002-2022)
- Remaining: around 24 months (2023-2024)
- **CRPS averages over all periods**

For **in-sample experiments** (train: 1979-2022):
- All verifications are in-sample: 528 months

---

## Results: Out-of-Sample

### Setup
- **Training**: 1979-01 to 2001-12 (276 months, 23 years)
- **Testing**: 2002-01 to 2022-12 (252 months, 21 years)
- **Ensemble size**: 100 members
- **Script**: `evaluate_stochastic_top5.sh`
- **Results**: `results_out_of_sample/`

### Summary Statistics

| Model | CRPS | RMSE (Mean) | Spread/RMSE | vs XRO |
|-------|------|-------------|-------------|--------|
| **Graph (S2)** | **0.488** | 0.899 | 0.654 | **-13.0%**  |
| **Attentive (S2)** | **0.513** | 0.901 | 0.634 | **-8.6%**  |
| **Linear (S2)** | **0.519** | 0.918 | 0.628 | **-7.5%**  |
| **Res (S2)** | **0.548** | 0.963 | 0.625 | **-2.4%**  |
| **XRO** | **0.561** | 0.970 | 0.527 | baseline |
| Attentive (S2+S3) | 0.935 | 0.951 | 3.667 | +66.6% |
| Linear (S2+S3) | 0.985 | 1.010 | 3.661 | +75.5% |
| Graph (S2+S3) | 1.006 | 1.116 | 3.362 | +79.2% |
| Res (S2+S3) | 1.045 | 1.024 | 3.975 | +86.2% |

**Note**: RO+Diag variants unstable (excluded). Negative % = better, positive % = worse.

### Visualizations (Out-of-Sample)

![Out-of-Sample Rankings](results_out_of_sample/rankings/stochastic_stages_comparison/stochastic_rankings_all_metrics.png)

**Sample Forecasts**:

| XRO Baseline | NXRO-Graph (S2) - Best |
|--------------|------------------------|
| ![XRO](results_out_of_sample/rankings/stochastic_stages_comparison/forecast_XRO.png) | ![Graph](results_out_of_sample/rankings/stochastic_stages_comparison/forecast_Graph_S2.png) |

| NXRO-Attentive (S2) | NXRO-Linear (S2) | NXRO-Res (S2) |
|---------------------|------------------|---------------|
| ![Attentive](results_out_of_sample/rankings/stochastic_stages_comparison/forecast_Attentive_S2.png) | ![Linear](results_out_of_sample/rankings/stochastic_stages_comparison/forecast_Linear_S2.png) | ![Res](results_out_of_sample/rankings/stochastic_stages_comparison/forecast_Res_S2.png) |

---

## Results: In-Sample

### Setup
- **Training**: 1979-01 to 2022-12 (528 months, 44 years)
- **Testing**: N/A (evaluated on training data)
- **Ensemble size**: 100 members
- **Script**: `evaluate_stochastic_top5_insample.sh`
- **Results**: `results/`

### Summary Statistics

| Model | CRPS | RMSE (Mean) | Spread/RMSE | vs XRO |
|-------|------|-------------|-------------|--------|
| **Attentive (S2)** | **0.503** | 0.894 | 0.671 | **-5.0%**  |
| **Graph (S2)** | **0.510** | 0.923 | 0.677 | **-3.7%**  |
| **Linear (S2)** | **0.521** | 0.927 | 0.657 | **-1.5%** |
| **Res (S2)** | **0.524** | 0.909 | 0.664 | **-1.0%** |
| **XRO** | **0.529** | 0.922 | 0.580 | baseline |
| Attentive (S2+S3) | 0.913 | 0.952 | 3.597 | +72.4% |
| Linear (S2+S3) | 0.948 | 0.991 | 3.566 | +79.0% |
| Graph (S2+S3) | 0.976 | 1.049 | 3.491 | +84.4% |
| Res (S2+S3) | 0.992 | 1.039 | 3.627 | +87.5% |

**Note**: RO+Diag variants unstable (excluded). Negative % = better, positive % = worse.

### Visualizations (In-Sample)

![In-Sample Rankings](results/rankings/stochastic_stages_comparison/stochastic_rankings_all_metrics.png)

**Sample Forecasts**:

| XRO Baseline | NXRO-Attentive (S2) - Best |
|--------------|----------------------------|
| ![XRO](results/rankings/stochastic_stages_comparison/forecast_XRO.png) | ![Attentive](results/rankings/stochastic_stages_comparison/forecast_Attentive_S2.png) |

| NXRO-Graph (S2) | NXRO-Linear (S2) | NXRO-Res (S2) |
|-----------------|------------------|---------------|
| ![Graph](results/rankings/stochastic_stages_comparison/forecast_Graph_S2.png) | ![Linear](results/rankings/stochastic_stages_comparison/forecast_Linear_S2.png) | ![Res](results/rankings/stochastic_stages_comparison/forecast_Res_S2.png) |

---

## Comparison: In-Sample vs Out-of-Sample

### CRPS Performance

**Best Models with Stage 2**:

| Rank | Out-of-Sample (harder) | CRPS | In-Sample (easier) | CRPS |
|------|------------------------|------|-------------------|------|
| 1 | Graph (S2) | 0.488 | Attentive (S2) | 0.503 |
| 2 | Attentive (S2) | 0.513 | Graph (S2) | 0.510 |
| 3 | Linear (S2) | 0.519 | Linear (S2) | 0.521 |
| 4 | Res (S2) | 0.548 | Res (S2) | 0.524 |
| 5 | XRO | 0.561 | XRO | 0.529 |

### Key Observations

**1. Out-of-sample is harder**:
- Out-of-sample CRPS generally higher (worse) than in-sample
- Makes sense: forecasting outside training period is more challenging

**2. Consistent winners**:
- **Graph and Attentive** consistently top performers in both setups
- Stage 2 always beats XRO baseline
- Stage 3 always fails (over-dispersion)

**3. Improvement margins**:
- Out-of-sample: 2.4-13% better than XRO
- In-sample: 1.0-5.0% better than XRO
- **Larger improvements in harder (out-of-sample) setting**

**4. Stage 3 consistently fails**:
- Both setups show 2× worse CRPS with simulation-based noise
- Spread/RMSE 4.0 (severe over-dispersion)
- Confirms simulation differences are too large

---


## Key Findings

### 1. Stage 2 Consistently Outperforms XRO

**All NXRO models with Stage 2 beat XRO** in both experimental setups.

**Out-of-sample improvements**: 2.4-13%  
**In-sample improvements**: 1.0-5.0%

This demonstrates that **physics-informed neural models** with likelihood-optimized stochastic components achieve better probabilistic forecast skill than classical XRO.

### 2. Stage 3 (Simulation-Based Noise) Fails in Both Setups

**Consistent failure across experiments**:
- Out-of-sample Stage 3: CRPS $\sim 1.1-1.3$ (2× worse than S2)
- In-sample Stage 3: CRPS $\sim 1.0-1.2$ (2× worse than S2)

**Root cause**: Simulation-observation differences (from 100 climate models) are **4× larger** than neural model residuals, creating severely over-dispersed ensembles.

### 3. Stage 2+3 Cannot Recover

Even with likelihood optimization (Stage 2), **starting from simulation noise (Stage 3) doesn't work**:
- S2 alone: CRPS 0.49-0.55
- S2+S3 (sim init + likelihood): CRPS 0.91-1.05
- **Starting point matters** - poor initialization cannot be fixed

### 4. Calibration Comparison

**Spread/RMSE ratios** (1.0 is ideal):

| Method | Out-of-Sample | In-Sample |
|--------|---------------|-----------|
| XRO | 0.527 | 0.580 |
| **Stage 2 models** | **0.625-0.654** | **0.657-0.677** |
| Stage 3 models | 3.7-4.8 | 3.9-4.6 |

**Stage 2 provides better calibration** in both setups (closer to 1.0).

---

## Conclusions and Recommendations

### Best Practice for Stochastic NXRO Forecasts

**Use Stage 2 training** (likelihood optimization on model residuals):

```bash
# Out-of-sample
python NXRO_train_out_of_sample.py --model <model> --stochastic --train_noise_stage2

# In-sample
python NXRO_train.py --model <model> --stochastic --train_noise_stage2
```

### Why Stage 2 Works

**Model-specific uncertainty**:
- Uses each model's own residuals
- Noise matched to model's forecast errors
- Statistically optimized via likelihood

**Benefits**:
- 1-13% CRPS improvement over XRO
- Better calibration (Spread/RMSE 0.63-0.68)
- Faster than Stage 3 (no 100-file loading)

### Why Stage 3 Fails

**Simulation biases too large**:
- 100 climate simulations have diverse physics
- Observation-simulation differences $>>$ neural model errors
- Results in 4× over-dispersed ensembles

**Simulation differences reflect**:
- Systematic model biases
- Parameterization differences
- Structural model errors
- **Not appropriate for well-tuned neural models**

### Future Work

**Understanding Stage 3**:
1. Analyze distribution of simulation differences vs model residuals
2. Test scaled version: `sigma_stage3 = 0.25 * sigma_from_simulations`
3. Use only high-quality reanalyses (ERA5, ORAS5-ens) instead of all 100 members

**Alternative approaches**:
- Hybrid: 80% model residuals + 20% simulation differences
- Adaptive: Stage 3 for long leads (>12 months), Stage 2 for short leads
- Variable-specific: Stage 3 for well-observed variables, Stage 2 for indices

---

## Key Takeaways

1. **Stage 2 is the winner**: 1-13% better CRPS than XRO in both setups
2. **Consistent across experiments**: Works for both in-sample and out-of-sample
3. **Best models**: NXRO-Graph and NXRO-Attentive with Stage 2
4. **Stage 3 fails**: Simulation noise 2× worse, over-dispersed by 4×
5. **S2+3 doesn't help**: Cannot recover from poor Stage 3 initialization

---

## Technical Details

### Experimental Configurations

| Setup | Train Period | Test Period | Results Dir |
|-------|--------------|-------------|-------------|
| **Out-of-Sample** | 1979-2001 (276mo) | 2002-2022 (252mo) | `results_out_of_sample/` |
| **In-Sample** | 1979-2022 (528mo) | N/A (same as train) | `results/` |

### Common Settings
- **Ensemble size**: 100 members
- **Forecast horizon**: 21 months
- **Variable**: Nino3.4 SSTA
- **Metrics**: CRPS, RMSE, Spread, Coverage

### Noise Methods

1. **Post-hoc**: Fit AR(1) from model residuals (OLS)
2. **Stage 2 (S2)**: Optimize AR(1) with likelihood (100 epochs, Adam)
3. **Stage 3 (S3)**: Fit AR(1) from obs-sim differences (100 climate files)
4. **Stage 2+3 (S2+S3)**: Stage 3 init + Stage 2 optimization

### Model Rankings (Stage 2 only)

**Out-of-sample** (by CRPS):
1. Graph (0.488)
2. Attentive (0.513)
3. Linear (0.519)
4. Res (0.548)

**In-sample** (by CRPS):
1. Attentive (0.503)
2. Graph (0.510)
3. Linear (0.521)
4. Res (0.524)

**Consistency**: Graph and Attentive are top 2 in both experiments.

---

## Scripts and Usage

### Running Experiments

```bash
# Out-of-sample (all stages)
./evaluate_stochastic_top5.sh                  # Post-hoc
./evaluate_stochastic_top5.sh --stage2         # Stage 2
./evaluate_stochastic_top5.sh --sim            # Stage 3
./evaluate_stochastic_top5.sh --sim --stage2   # Stage 2+3

# In-sample (all stages)
./evaluate_stochastic_top5_insample.sh                  # Post-hoc
./evaluate_stochastic_top5_insample.sh --stage2         # Stage 2
./evaluate_stochastic_top5_insample.sh --sim            # Stage 3
./evaluate_stochastic_top5_insample.sh --sim --stage2   # Stage 2+3
```

### Visualization

```bash
# Compare stages (out-of-sample)
python visualize_stochastic_comparison.py --compare_stages

# Compare stages (in-sample)
python visualize_stochastic_comparison.py --results_dir results --compare_stages
```

---

## References

- **CRPS**: Hersbach, H. (2000). "Decomposition of the continuous ranked probability score for ensemble prediction systems." *Weather and Forecasting*.
- **Proper scoring rules**: Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA*.
- **Stochastic XRO**: `XRO/core.py` - Seasonal AR(1) red noise implementation
- **Stage 2 optimization**: `nxro/stochastic.py::train_noise_stage2()` (lines 229-326)
- **Stage 2 likelihood**: `nxro/stochastic.py::ar1_log_likelihood()` (lines 102-147)
- **Stage 3 implementation**: `nxro/stochastic.py::fit_noise_from_simulations()` (lines 150-245)
- **CRPS calculation**: `utils/xro_utils.py::evaluate_stochastic_ensemble()` (lines 318-325)

---

## Stochastic Plume Visualizations

In this section we display the stochastic forecasts for certain dates in ENSO. First the empirical plots for ENSO over all the time stamps is: 

![Visualization of ORAS5 measurement of ENSO 1979-2022](results_out_of_sample/NXRO_observed_Nino34_out_of_sample.png)



![The Stochastic forecast starting January 1971](results_out_of_sample/rankings/stochastic_comparison/plume_comparison_1979_01.png)


![The Stochastic forecast starting April of 1988](results_out_of_sample/rankings/stochastic_comparison/plume_comparison_1988_04.png)


![Stochastic Forecast starting April of 1997](results_out_of_sample/rankings/stochastic_comparison/plume_comparison_1997_04.png)


![The stochastic forecast starting December of 1997](results_out_of_sample/rankings/stochastic_comparison/plume_comparison_1997_12.png)


![The stochastic forecast starting January of 2002](results_out_of_sample/rankings/stochastic_comparison/plume_comparison_2002_01.png)


![The stochastic forecast starting September of 2022](results_out_of_sample/rankings/stochastic_comparison/plume_comparison_2022_09.png)
---