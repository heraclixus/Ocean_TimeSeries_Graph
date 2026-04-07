# Learnable Stochastic NXRO

## Overview

Current NXRO models use **post-hoc noise fitting** (fit drift, then fit noise from residuals). This document proposes **learnable stochastic components** with likelihood-based optimization.

---

## Current Approach: Post-Hoc Noise Fitting

### XRO/NXRO Noise Model

Seasonal AR(1) red noise per variable:
```
xi_j,t+1(m) = a1_j * xi_j,t(m) + sqrt(1 - a1_j^2) * sigma_j(m) * epsilon
```

**Parameters**:
- `a1_j`: Lag-1 autocorrelation (one value per variable)
- `sigma_j(m)`: Seasonal std dev (12 values per variable, one per month)

### Current Fitting Process

**Step 1**: Train deterministic drift (RMSE loss)
```
min_theta E[(dX/dt - f_theta(X,t))^2]
```

**Step 2**: Compute residuals
```
residual = dX/dt_observed - f_theta(X,t)
```

**Step 3**: Fit noise from residuals
```
a1_j = lag1_correlation(residual_j)
sigma_j(m) = std(residual_j[month==m])
```

**Limitation**: Noise parameters not optimized for forecast skill, just fitted from residuals.

---

## Proposed: Learnable Stochastic Components

### Key Idea

**Learn noise parameters end-to-end** or in two stages using likelihood-based losses instead of post-hoc fitting from residuals.

---

## Approach 1: Two-Stage Training with Likelihood

### Stage 1: Train Deterministic Drift (Current)

```
min_theta E[(dX/dt - f_theta(X,t))^2]
```

Result: Trained drift f_theta (NXRO-Res, NXRO-Graph, etc.)

### Stage 2: Optimize Noise Parameters on Likelihood

**Fix** drift f_theta, **optimize** noise parameters (a1, sigma) using likelihood objective.

**Formulation**:

Given trained drift f_theta, model stochastic dynamics as:
```
dX/dt = f_theta(X,t) + xi(t)
```

where xi follows AR(1) with **trainable** parameters.

**Likelihood objective**:
```
max_{a1, sigma} log P(X_train | f_theta, a1, sigma)
```

**Discrete-time approximation**:
```
X_t+1 = X_t + dt * f_theta(X_t, t) + noise_t

noise_t ~ N(0, Sigma(m, a1, sigma))

Log-likelihood:
L = -1/2 * sum_t [(X_t+1 - X_t - dt*f_theta)^T Sigma^-1 (X_t+1 - X_t - dt*f_theta)]
```

**Trainable parameters**:
- a1_j (n_vars parameters, initialized from post-hoc fit)
- log_sigma_j(m) (n_vars x 12 parameters, log-space for positivity)

**Optimization**:
```python
# Stage 1: Train drift (already done)
model = train_nxro_res(...)  # Or any variant

# Stage 2: Optimize noise
a1 = nn.Parameter(torch.tensor(a1_init))  # From residual fit
log_sigma = nn.Parameter(torch.tensor(log_sigma_init))

optimizer = torch.optim.Adam([a1, log_sigma], lr=1e-3)

for epoch in range(100):
    # Compute likelihood on training data
    noise_pred = compute_residuals(model, train_data)
    log_prob = ar1_log_likelihood(noise_pred, a1, log_sigma, months)
    loss = -log_prob
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Pros**:
- Noise parameters optimized for likelihood, not just fitted
- Can improve calibration (spread-skill relationship)
- Relatively simple (drift already trained)

**Cons**:
- Drift and noise optimized separately (not joint)
- May not find global optimum

---

## Approach 2: Neural Network for Residual Noise

### Physical Motivation

**Key insight**: Residual noise represents **model error** and **unresolved processes**, not fundamental stochasticity.

Current AR(1) assumes:
- Simple temporal correlation (lag-1 only)
- No spatial/variable correlations
- Linear relationship

**Proposal**: Learn residual noise structure with neural networks.

---

### Method 1: MLP for State-Dependent Noise

Instead of fixed sigma_j(m), learn sigma as function of state and season:

```
sigma_phi(X, t) = softplus(MLP([X, phi_season(t)]))
```

**Architecture**:
```python
class LearnableNoiseModel(nn.Module):
    def __init__(self, n_vars, k_max=2):
        # Input: state X + seasonal features phi(t)
        input_dim = n_vars + (2*k_max + 1)
        
        self.sigma_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, n_vars)
        )
    
    def forward(self, X, phi_t):
        # State-dependent sigma
        log_sigma = self.sigma_net(torch.cat([X, phi_t], dim=-1))
        sigma = torch.exp(log_sigma)  # Positive
        return sigma
```

**Training**: Optimize on likelihood (same as Stage 2, but now sigma is neural network)

**Benefit**: Captures state-dependent noise (e.g., stronger noise during El Nino)

---

### Method 2: GNN for Spatially-Correlated Noise

Variables may have correlated noise (e.g., ENSO and IOD noise coupled).

**Architecture**:
```python
class GraphNoiseModel(nn.Module):
    def __init__(self, n_vars, edge_index, k_max=2):
        self.gnn = GATConv(2*k_max+1, 16)  # Graph attention
        self.output = nn.Linear(16, 1)  # Per-variable sigma
    
    def forward(self, X, phi_t, edge_index):
        # Broadcast seasonal features to all nodes
        node_features = phi_t.repeat(n_vars, 1)  # [n_vars, 2*k_max+1]
        
        # Graph convolution
        h = self.gnn(node_features, edge_index)
        
        # Output sigma per variable
        log_sigma = self.output(h).squeeze(-1)  # [n_vars]
        sigma = torch.exp(log_sigma)
        
        return sigma
```

**Training**: Same likelihood objective, but now sigma learned via graph structure

**Benefit**: Captures spatial correlations in noise (teleconnection-dependent uncertainty)

---

### Method 3: Learning from Simulation-Observation Differences

**Key idea**: Use climate model simulations to estimate "true" noise distribution.

**Available data**:
- ORAS5 observations (truth)
- ERA5, GODAS, other reanalyses (simulations/reconstructions)
- Difference = observation - simulation ~ model error + unresolved processes

**Training signal**:
```
noise_distribution ~ (X_obs - X_sim)
```

**Implementation**:

**Step 1**: Compute simulation-observation differences
```python
obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
era5_ds = xr.open_dataset('data/XRO_indices_era5_preproc.nc')
godas_ds = xr.open_dataset('data/XRO_indices_godas_preproc.nc')

# Align times
common_times = set(obs_ds.time.values) & set(era5_ds.time.values) & set(godas_ds.time.values)

# Compute differences (observation - simulation)
diff_era5 = obs_ds - era5_ds  # Noise sample 1
diff_godas = obs_ds - godas_ds  # Noise sample 2
```

**Step 2**: Fit noise model from differences (not from model residuals)
```python
# Treat simulation differences as noise samples
all_noise_samples = concatenate([diff_era5, diff_godas, ...])

# Fit AR(1) parameters from simulation differences
a1_sim, sigma_sim = fit_ar1_from_samples(all_noise_samples)
```

**Step 3**: Use for NXRO forecasts
```python
noise_model = SeasonalAR1Noise(a1_sim, sigma_sim)
fcst = nxro_reforecast_stochastic(model, init_ds, noise_model=noise_model)
```

**Rationale**:
- Simulation-observation differences reflect real uncertainty
- More data (multiple simulations) -> better noise estimation
- Captures unresolved physics (not just fitting error)

**CLI flag**: `--use_sim_noise`

**Pros**:
- Leverages multiple climate datasets
- Noise independent of model quality (not fitted from residuals)
- May be more realistic (reflects true climate variability)

**Cons**:
- Assumes simulation errors represent noise (may not be true)
- Requires aligned multi-dataset time series

---

## Recommended Research Plan

### Phase S1: Baseline Stochastic Evaluation [IMMEDIATE]

**Objective**: Establish baseline with current post-hoc noise fitting.

```bash
./evaluate_stochastic_top5.sh
python visualize_stochastic_comparison.py
```

**Outputs**: 
- Which model has best CRPS?
- Are ensembles well-calibrated?
- Baseline for comparison with learned noise

---

### Phase S2: Two-Stage Training [IMPLEMENTED]

**Objective**: Optimize noise parameters using likelihood, keeping drift fixed.

**Implementation**: COMPLETE

**Functions** (in `nxro/stochastic.py`):
- `ar1_log_likelihood()` - Computes AR(1) log-likelihood
- `train_noise_stage2()` - Optimizes (a1, sigma) using likelihood

**CLI Flag**: `--train_noise_stage2`

**File Naming**:
- Post-hoc: `nxro_{model}_stochastic_eval_lead_metrics.csv`
- Stage 2: `nxro_{model}_stochastic_stage2_eval_lead_metrics.csv`

**Experiments** (2 runs per model, 10 total for top 5):
```bash
for model in res linear rodiag attentive; do
  # Post-hoc (baseline)
  python NXRO_train_out_of_sample.py --model $model --stochastic
  
  # Stage 2 (optimized noise)
  python NXRO_train_out_of_sample.py --model $model --stochastic --train_noise_stage2
done

# For graph models
python NXRO_train_out_of_sample.py --model graph_pyg --top_k 3 --stochastic
python NXRO_train_out_of_sample.py --model graph_pyg --top_k 3 --stochastic --train_noise_stage2
```

**Or use script**:
```bash
# Post-hoc for all top 5
./evaluate_stochastic_top5.sh

# Stage 2 for all top 5
./evaluate_stochastic_top5.sh --stage2
```

---

### Phase S3: Simulation-Based Noise Estimation [NEXT]

**Objective**: Use simulation-observation differences to estimate noise distribution.

**Implementation**:

**File**: `nxro/stochastic.py` - Add `fit_noise_from_simulations()`

```python
def fit_noise_from_simulations(obs_ds, sim_datasets, var_order, train_period):
    """
    Fit noise parameters from simulation-observation differences.
    
    Args:
        obs_ds: ORAS5 observations
        sim_datasets: List of simulation datasets (ERA5, GODAS, etc.)
        var_order: Variable names
        train_period: Time slice for fitting
    
    Returns:
        a1: [12, n_vars] - AR(1) from combined differences
        sigma: [12, n_vars] - Seasonal std dev from combined differences
    """
    all_diffs = []
    all_months = []
    
    for sim_ds in sim_datasets:
        # Align times
        common_times = list(set(obs_ds.time.values) & set(sim_ds.time.values))
        
        # Compute difference
        obs_aligned = obs_ds.sel(time=common_times)
        sim_aligned = sim_ds.sel(time=common_times)
        
        diff = obs_aligned - sim_aligned
        
        # Extract as array
        diff_array = np.stack([diff[v].values for v in var_order], axis=-1)
        months = pd.DatetimeIndex(common_times).month.values
        
        all_diffs.append(diff_array)
        all_months.append(months)
    
    # Concatenate all differences
    combined_diffs = np.concatenate(all_diffs, axis=0)
    combined_months = np.concatenate(all_months)
    
    # Fit AR(1) from combined differences
    a1, sigma = fit_seasonal_ar1_from_residuals(combined_diffs, combined_months)
    
    return a1, sigma
```

**Usage**:
```bash
python NXRO_train_out_of_sample.py --model res --stochastic --use_sim_noise
```

**Experiments** (3 models x 2 noise sources = 6 runs):
```bash
for model in res graph attentive; do
  # Model residuals (default)
  python NXRO_train_out_of_sample.py --model $model --stochastic
  
  # Simulation differences
  python NXRO_train_out_of_sample.py --model $model --stochastic --use_sim_noise
done
```

**Time**: ~1 day (implementation + experiments)

---

### Phase S4: Neural Noise Models [FUTURE]

**Objective**: Learn sigma with MLP or GNN instead of fixed AR(1).

**Implementation needed**:
1. Create `LearnableNoiseModel` (MLP-based)
2. Create `GraphNoiseModel` (GNN-based)
3. Extend Stage 2 training to optimize neural network (not just parameters)
4. Compare with fixed AR(1)

**Time**: ~1 week (substantial implementation)

---

## Implementation Priority

**Priority 1** [DO NOW]: Baseline stochastic evaluation
- Scripts ready: `evaluate_stochastic_top5.sh`, `visualize_stochastic_comparison.py`
- No coding needed
- Time: 3-6 hours

**Priority 2** [NEXT]: Two-stage training
- Requires: `train_noise_stage2.py` (moderate coding, ~1 day)
- Expected gain: 5-10% CRPS improvement
- Time: 1-2 days total

**Priority 3** [RESEARCH]: End-to-end Neural SDE
- Requires: Major refactor of training loop (complex coding, ~3-5 days)
- Expected gain: 10-20% CRPS improvement if successful
- Time: 1-2 weeks total
- Higher risk (may not converge, hyperparameter sensitive)

---

## Mathematical Formulation Details

### Two-Stage Likelihood Loss - Detailed Formulation

**Given**: 
- Trained drift f_theta (frozen)
- Training data X = (X_0, X_1, ..., X_T) with timestamps t = (t_0, t_1, ..., t_T)
- Month indices m(t) in {1, 2, ..., 12}

---

#### Generative Model

Our stochastic model for climate dynamics:

```
dX/dt = f_theta(X,t) + xi(t)
```

Discretized (Euler-Maruyama):
```
X_{t+1} = X_t + dt * f_theta(X_t, t_t) + dt * xi_t
```

where `xi_t` follows seasonal AR(1) red noise per variable:
```
xi_{j,t+1} = a1_j * xi_{j,t} + sqrt(1 - a1_j^2) * sigma_j(m(t+1)) * epsilon_{j,t+1}

epsilon_{j,t} ~ N(0, 1)  (independent white noise)
```

---

#### Single-Step Likelihood (Simplified)

**Assumption**: Treat each time step independently (ignore AR structure temporarily).

**Residual at time t**:
```
r_t = X_{t+1} - X_t - dt * f_theta(X_t, t_t)
```

This residual should follow:
```
r_{j,t} ~ N(0, (sigma_j(m(t)))^2 * dt)
```

**Log-likelihood for single step**:
```
log p(X_{t+1} | X_t, t_t, theta, sigma) = sum_j log N(r_{j,t}; 0, sigma_j(m)^2 * dt)

= -1/2 sum_j [log(2*pi) + log(sigma_j(m)^2 * dt) + r_{j,t}^2 / (sigma_j(m)^2 * dt)]

= -1/2 sum_j [log(2*pi) + 2*log(sigma_j(m)) + log(dt) + r_{j,t}^2 / (sigma_j(m)^2 * dt)]
```

**Total log-likelihood** (all timesteps):
```
log L(sigma | X, theta) = sum_t log p(X_{t+1} | X_t, t_t, theta, sigma)

= -1/2 sum_{t,j} [log(2*pi) + 2*log(sigma_j(m(t))) + log(dt) + r_{j,t}^2 / (sigma_j(m(t))^2 * dt)]
```

**Optimization problem**:
```
max_sigma log L(sigma | X, theta)

Equivalently:
min_sigma -log L(sigma | X, theta)
```

---

#### Full AR(1) Likelihood (Exact)

To properly account for AR(1) structure, we need the conditional distribution of xi_t given past xi_{<t}.

**AR(1) process**:
```
xi_{j,t} = a1_j * xi_{j,t-1} + eta_{j,t}

where eta_{j,t} ~ N(0, (1-a1_j^2) * sigma_j(m(t))^2)  (innovation)
```

**Conditional likelihood**:
```
p(xi_{j,t} | xi_{j,t-1}, a1_j, sigma_j) = N(xi_{j,t}; a1_j * xi_{j,t-1}, (1-a1_j^2) * sigma_j(m)^2)
```

**But we don't observe xi directly**, only residuals r_t = dt * xi_t.

**Approximation for small dt**:
```
r_{j,t} approx dt * xi_{j,t}

Therefore:
r_{j,t} | r_{j,t-1} ~ N(a1_j * r_{j,t-1}, dt^2 * (1-a1_j^2) * sigma_j(m)^2)
```

**Full log-likelihood** (accounting for AR structure):
```
log L(a1, sigma | r, m) = sum_{t=1}^T sum_j log p(r_{j,t} | r_{j,t-1}, a1_j, sigma_j(m(t)))

= -1/2 sum_{t=1}^T sum_j [
    log(2*pi) + log(dt^2 * (1-a1_j^2) * sigma_j(m(t))^2) + 
    (r_{j,t} - a1_j * r_{j,t-1})^2 / (dt^2 * (1-a1_j^2) * sigma_j(m(t))^2)
]
```

**Trainable parameters**:
- a1_j: [n_vars] - lag-1 autocorrelation per variable
- log_sigma_j(m): [n_vars, 12] - log seasonal std dev (log-space for positivity)

---

#### Practical Implementation

**Step 1**: Compute residuals from trained model
```python
r_t = X_{t+1} - X_t - dt * f_theta(X_t, t_t)
# Shape: [T, n_vars]
```

**Step 2**: Define likelihood function
```python
def ar1_log_likelihood(r, a1, log_sigma, months, dt=1/12):
    """
    r: [T, n_vars] residuals
    a1: [n_vars] lag-1 autocorrelation
    log_sigma: [n_vars, 12] log seasonal std dev
    months: [T] month indices (0-11)
    """
    sigma = torch.exp(log_sigma)  # [n_vars, 12]
    
    # Get sigma for each timestep
    sigma_t = sigma[:, months]  # [n_vars, T]
    
    # AR(1) innovations
    r_lag = torch.cat([torch.zeros(1, n_vars), r[:-1]], dim=0)  # [T, n_vars]
    innovations = r[:, :] - a1[None, :] * r_lag  # [T, n_vars]
    
    # AR(1) innovation variance
    var_innovation = dt**2 * (1 - a1**2) * (sigma_t**2)  # [n_vars, T]
    
    # Log-likelihood
    log_prob = -0.5 * (
        torch.log(2 * torch.pi * var_innovation) +
        innovations.T**2 / var_innovation
    )
    
    return log_prob.sum()  # Sum over all timesteps and variables
```

**Step 3**: Optimize
```python
# Initialize from post-hoc fit
a1 = nn.Parameter(torch.tensor(a1_init, dtype=torch.float32))
log_sigma = nn.Parameter(torch.tensor(np.log(sigma_init), dtype=torch.float32))

optimizer = torch.optim.Adam([a1, log_sigma], lr=1e-3)

for epoch in range(100):
    log_likelihood = ar1_log_likelihood(r, a1, log_sigma, months)
    loss = -log_likelihood  # Negative log-likelihood
    
    # Optional: Add regularization
    # loss += 0.01 * torch.sum((a1 - a1_init)**2)  # Keep close to initial
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Constraint: a1 should be in (0, 1)
    with torch.no_grad():
        a1.clamp_(0.01, 0.99)
```

---

#### Alternative: Simplified Likelihood (Whitened Residuals)

If AR(1) structure is complex, optimize sigma only with whitened residuals:

**Step 1**: Pre-whiten residuals using post-hoc a1
```python
r_whitened[t] = (r[t] - a1 * r[t-1]) / sqrt(1 - a1^2)
```

**Step 2**: Optimize sigma assuming whitened residuals are i.i.d.
```python
log L(sigma | r_whitened) = sum_{t,j} log N(r_whitened_{j,t}; 0, sigma_j(m(t))^2 * dt)

= -1/2 sum_{t,j} [log(2*pi*sigma_j(m(t))^2*dt) + r_whitened_{j,t}^2 / (sigma_j(m(t))^2*dt)]
```

**This is simpler** but assumes a1 is correct (not optimized).

---

#### Summary of Likelihood Formulations

**Option A: Full AR(1) Likelihood** (recommended)
- Joint optimization of (a1, sigma)
- Properly accounts for temporal correlation
- More complex but theoretically correct
- Use the formula above with AR(1) innovations

**Option B: Whitened Residuals Likelihood** (simpler)
- Fix a1, optimize sigma only
- Assumes pre-whitening is correct
- Simpler implementation
- Good starting point

**Option C: Independent Gaussian Likelihood** (simplest, approximate)
- Ignore AR structure entirely
- Optimize sigma assuming i.i.d. residuals
- Fast but theoretically flawed (residuals are correlated)
- May still improve over post-hoc fitting

### End-to-End Joint Loss

**Combined objective**:
```
Loss = lambda_rmse * RMSE + lambda_nll * NLL

RMSE = E[(X_t+1 - X_t - dt*f_theta(X_t,t))^2]  (deterministic skill)

NLL = -log P(X_train | f_theta, g_phi)  (stochastic quality)
```

**Hyperparameter tuning**:
- lambda_rmse = 1.0, lambda_nll = 0.1 (prioritize deterministic)
- lambda_rmse = 1.0, lambda_nll = 0.5 (balanced)
- lambda_rmse = 0.5, lambda_nll = 0.5 (equal weight)

**Validation**: Use CRPS on validation set to select lambda.

---

## Expected Improvements

### Two-Stage Training

**Conservative**: 5% CRPS improvement
- Noise optimized for likelihood instead of just fitted
- Better calibration (spread/RMSE closer to 1.0)

**Target**: 10% CRPS improvement + better coverage

### End-to-End Training

**Conservative**: 10% CRPS improvement
- Joint optimization finds better trade-off
- Drift adjusted to work well with noise

**Target**: 15-20% CRPS improvement + significantly better calibration

**Risk**: May hurt deterministic RMSE if lambda_nll too high

---

## Implementation Roadmap

### Phase S1: Baseline [DONE - Scripts Ready]

**Run**:
```bash
./evaluate_stochastic_top5.sh
python visualize_stochastic_comparison.py
```

**Outputs**: Baseline CRPS, calibration, coverage for top 5 + XRO

---

### Phase S2: Two-Stage Implementation [IMPLEMENTED]

**Status**: COMPLETE and integrated

**Implementation details**:

All functions implemented in `nxro/stochastic.py`:
- `ar1_log_likelihood()` - Computes AR(1) likelihood (lines 102-147)
- `train_noise_stage2()` - Optimizes noise parameters (lines 150-247)

**Integrated into**:
- `NXRO_train.py`
- `NXRO_train_out_of_sample.py`  
- `evaluate_stochastic_top5.sh`

**Usage**:
```python
# Automatically triggered with --train_noise_stage2 flag
python NXRO_train_out_of_sample.py --model res --stochastic --train_noise_stage2

# The training script will:
# 1. Compute residuals
# 2. Fit initial (a1, sigma) from residuals
# 3. Optimize using ar1_log_likelihood for 100 epochs
# 4. Use optimized parameters for ensemble generation
```

**Output files** (with `_stage2` suffix):
- `nxro_{model}_stochastic_stage2_noise.npz`
- `nxro_{model}_stochastic_stage2_eval_lead_metrics.csv`
- `NXRO_{model}_stochastic_stage2_forecasts.nc`

**Ready to use** - no additional coding needed!

---

### Phase S3: Simulation-Based Noise [NEXT]

**Add to**: `nxro/stochastic.py`

**Implementation**:
```python
def fit_noise_from_simulations(obs_path, sim_paths, var_order, train_period):
    # Compute obs - sim for each dataset
    # Fit AR(1) from combined differences
    # Return a1, sigma
    pass
```

**Integrate into training scripts**:
- Add `--use_sim_noise` flag
- Load multiple datasets if flag is set
- Compute differences and fit noise
- Use for ensemble generation

**Time**: 1 day

---

### Phase S4: Neural Noise Models [FUTURE]

**Create**: `nxro/learnable_noise.py`

**Major changes needed**:
1. Implement `LearnableNoiseModel` (MLP-based sigma)
2. Implement `GraphNoiseModel` (GNN-based sigma)
3. Extend `train_noise_stage2()` to handle neural networks
4. Add hyperparameter search for architecture

## Key Research Questions

### For Two-Stage Training

1. Does likelihood optimization improve CRPS over post-hoc fitting?
2. Does it improve calibration (spread/RMSE ratio)?
3. Does it improve coverage reliability?
4. Are optimized parameters significantly different from fitted ones?

### For Simulation-Based Noise

5. Is simulation-observation difference more realistic than model residuals?
6. Does multi-dataset noise improve calibration?
7. Do different simulations give consistent noise estimates?

### For Neural Noise Models

8. Can MLP/GNN learn better noise structure than AR(1)?
9. Is noise state-dependent (stronger during extreme events)?
10. Do spatial correlations in noise matter for skill?

---

## Summary

### Current Status

- **Post-hoc noise fitting** implemented and ready
- Scripts available for baseline evaluation
- All top models can generate ensembles

### Proposed Research Direction

**Two-stage training** (recommended next step):
- Optimize noise parameters using likelihood
- Keep drift fixed
- Moderate implementation effort
- Expected 5-10% CRPS improvement

**Simulation-based noise** (next step):
- Use obs - sim differences for noise estimation
- Leverages multiple climate datasets
- Low implementation effort
- Expected: Better calibration, more realistic uncertainty

**Neural noise models** (future work):
- MLP or GNN for state/spatially-dependent noise
- Moderate implementation effort
- Expected: 10-15% CRPS improvement if noise is structured

### Immediate Action

```bash
# Step 1: Baseline (run now)
./evaluate_stochastic_top5.sh
python visualize_stochastic_comparison.py

# Step 2: Test Stage 2 likelihood optimization
python NXRO_train_out_of_sample.py --model res --stochastic --train_noise_stage2

# Step 3: Implement simulation-based noise (next)
# Add fit_noise_from_simulations() and --use_sim_noise flag

# Step 4: Neural noise models (future)
# Implement MLP/GNN-based noise, research project
```

---

## References

- **XRO noise**: `XRO/core.py` (lines 322-345, 1360-1410)
- **NXRO stochastic**: `nxro/stochastic.py`
- **Ensemble evaluation**: `utils/xro_utils.py` (`evaluate_stochastic_ensemble`)
- **Likelihood methods**: Maximum Likelihood Estimation for time series
- **Multi-dataset uncertainty**: Using ensemble spread from multiple reanalyses
