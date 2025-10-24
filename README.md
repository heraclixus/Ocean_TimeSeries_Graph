# ENSO Dynamic Modeling with Coupled Neural Oscillators


The file `XRO_indices_oras5.csv` (or `XRO_indices_ora5.nc`) contains 10 time series data:
- Nino34: the El Nino Ocean Index.
- WWV: the Thermocline depth.
- NPMM: North Pacific Meridional Mode.
- SPMM: South Pacific Meridional Mode.
- IOB: 
- IOD
- SIOD 
- TNA 
- ATL3
- SASD


## Baseline Model: the XRO

Details of the XRO model can be found in XRO.md

## Stochastic forcing in XRO

The original XRO augments its deterministic monthly tendency with a stochastic term that represents unresolved atmospheric/oceanic variability (e.g., MJO, westerly wind bursts). In continuous-time shorthand:

- dX/dt = f_XRO(X, t) + ξ(t)

For monthly stepping (Δt = 1/12), this appears as additive noise in the discrete update:

- X_{t+1} = X_t + f_XRO(X_t, t) · Δt + η_t

Where η_t is drawn from a seasonally varying AR(1) process per variable (red noise):

- ξ_t = a1(m) · ξ_{t-1} + σ(m) · ε_t, with ε_t ∼ N(0, I) and m = month.

Practical details in XRO:
- Seasonal structure: both the AR(1) coefficient a1(m) and standard deviation σ(m) follow a monthly cycle; this captures the fact that noise is typically stronger and/or more persistent during certain seasons.
- Per-variable noise: by default the process is diagonal (independent across variables), although correlated noise could be used if residual diagnostics justify it.
- Estimation: fit the deterministic part first, take residuals r_t = X_{t+1} − (X_t + f_XRO(X_t, t)·Δt), and estimate month-wise AR(1) parameters {a1(m), σ(m)}. Optionally apply shrinkage to stabilize estimates in short samples.
- Use in forecasts: deterministic hindcasts set noise to zero to isolate model drift skill; stochastic reforecasts draw η_t to produce an ensemble. Spread–skill comparisons, plumes, and reliability diagnostics rely on this ensemble.

Why it matters:
- Accounts for weather “kicks” that organize ENSO events and contribute to the spring predictability barrier.
- Provides calibrated forecast spread and improves probabilistic skill even when mean deterministic skill saturates.
- Red noise (a1(m) > 0) creates persistence in forcing, better matching observed low-frequency variability than white noise.

Connection to NXRO:
- Variant 6) NXRO-Stochastic will mirror XRO’s seasonal AR(1) noise by learning {a1(m), σ(m)} (or parameterizing them via Fourier features) and sampling during reforecasts. Deterministic skill (ACC/RMSE) is still reported with noise disabled; probabilistic metrics use ensembles.

## Neural XRO (NXRO)


### Goal

Introduce a neural ODE-style alternative to XRO that preserves its core priors (seasonality, ENSO T–H coupling, sparse interactions), while enabling gradual increases in expressivity. Train with RMSE and use the same train/test split as in `XRO_example.py` (train: 1979-01 → 2022-12, test: 2023-01 → latest).

### Data and split

- Input: `data/XRO_indices_oras5.nc` (or corresponding CSV). Monthly indices with time coordinate.
- Train: `1979-01` → `2022-12`.
- Test: `2023-01` → end.
- Variables: first two states are ENSO SST proxy (`Nino34`) and warm water volume (`WWV`), others as in XRO.

### Base neural-ODE formulation with XRO priors

We model monthly dynamics as

```
dX/dt = L_θ(t) · X + N_θ(X, t)
```

with the following priors:
- Seasonal time embedding: `t` → Fourier features [cos(kωt), sin(kωt)], k = 0..K (K=2 to match XRO), ω = 2π/year.
- Linear seasonal operator `L_θ(t)`: parameterized to mirror XRO’s harmonic expansion (weights over the same Fourier basis). Optional weight decay to keep it close to XRO.
- Coupling structure: the ENSO T/H equations admit Recharge Oscillator (RO) monomials [T², T·H, T³, T²H, TH²]. Other variables default to diagonal nonlinearities, matching XRO.
- Integration: monthly step using explicit Euler (as in `XRO.simulate`), or torchdiffeq for more advanced solvers later.

### Training objective

- Primary loss: RMSE on states at monthly steps (teacher forcing by default):
  - One-step: predict `X_{t+1}` from `X_t` via `X_{t+1} = X_t + f_θ(X_t, t) · Δt` with Δt = 1/12.
  - Multi-step rollout loss (optional): accumulate RMSE over K-step trajectories to improve stability.
- Regularization: L2 on seasonal linear weights; optional penalties to keep structure close to XRO (e.g., small residuals in variants below).
- Optimizer: Adam/AdamW; early stopping on validation RMSE; gradient clipping.

### Evaluation (same metrics as XRO)

- RMSE on the test period.
- ACC and RMSE by lead using the hindcast protocol (deterministic reforecast with `noise_type='zero'`).
- Seasonal synchronization: monthly stddev curves.
- Forecast plumes for selected start dates.

### Model variants (in increasing expressivity)

1) NXRO-Linear (seasonal L only)
- `N_θ(X, t) = 0`.
- `L_θ(t)` uses the same harmonic basis as XRO; initialize from XRO’s fitted `L` for a warm start.
- Purpose: neural re-fit of XRO’s linear seasonal operator.

2) NXRO-RO (linear + RO basis with learned seasonal coefficients)
- `N_θ(X, t)` restricted to RO monomials in T/H equations with seasonal coefficients; other variables use only linear/diagonal.
- Mirrors XRO’s `maskNT/maskNH` behavior.

3) NXRO-RO+Diag (add diagonal quadratic/cubic)
- Adds seasonal diagonal b(t), c(t) per variable to match XRO’s `NLb/NLc` structure.
- Coefficients come from small MLPs on seasonal embeddings or direct Fourier-weighted parameters.

4) NXRO-Res (residual neural drift)
- `f_θ = L_θ(t)·X + N_θ^RO+Diag(X, t) + R_θ(X, t)` with a small residual MLP.
- Regularize `R_θ` to be small (L2, spectral norm) to stay close to XRO; ablate its effect.

5) NXRO-NeuralODE (general drift with structural masks)
- MLP drift with periodic time embeddings; use masks so only T/H get cross terms per RO, others stay (mostly) diagonal.
- Optionally introduce limited cross-variable attention later.

6) NXRO-Stochastic (later)
- Learn seasonal AR(1) noise parameters (`a1`, `σ(c)`) analogous to `xi_a1`/`xi_stdac` from XRO for stochastic reforecasts.

### Intermediate variants between 5) and 6)

Given limited data, we propose small, structured extensions to 5) that increase expressivity gradually while controlling capacity:

- 5a) NXRO-Bilinear (low-rank cross-variable interactions)
  - Add a bilinear term \(\sum_{k} u_k(t)\, (X^T A_k X)\) projected back to state space via a small learned map.
  - Implement with low-rank factors: \(A_k \approx P_k Q_k^T\), rank r ≪ n. Seasonal weights via Fourier embedding.
  - Constructor knobs: rank r, number of bilinear channels, L2 penalty on factors.

- 5b) NXRO-AttentiveCoupling (limited attention)
  - One lightweight attention block over variables conditioned on seasonal embedding: \(\text{Attn}(X, X; W(t))\).
  - Restrict heads=1..2, small hidden, optional mask to emphasize T/H coupling (e.g., allow keys/values primarily from T,H).
  - Constructor knobs: heads, hidden, dropout, mask_mode={'th_only','full'}.

- 5c) NXRO-GraphNeuralODE (sparse graph prior)
  - Use a fixed sparse adjacency (teleconnections) or learn a sparse adjacency with L1 penalty. One GCN layer in drift.
  - Seasonal gating of edge weights. Keep graph tiny (k-NN in correlation space or physics-derived edges).
  - Constructor knobs: num_edges, l1_lambda, use_fixed_graph.

  Implementation notes:
  - Two backends provided:
    - A minimal in-house graph drift (`NXROGraphModel`) using a normalized adjacency and a tanh-projected message.
    - A PyTorch Geometric variant (`NXROGraphPyGModel`) using GCNConv or GATConv on a tiny graph. Requires `torch_geometric`.
  - Graph options:
    - Predefined/fixed graph: from physics or from XRO coupling (e.g., threshold nonzero entries of `Lac`/`normLac`, or connect all to T/H).
    - k-NN in empirical correlation space to create a sparse adjacency (`build_edge_index_from_corr`).
    - Learned adjacency: parameterize A, constrain to nonnegative (ReLU) and penalize with L1; ensure row-normalization and self-loops.
  - Keep capacity small: 1–2 conv layers, hidden channels ≤ 16–32, minimal dropout.

- 5d) NXRO-PhysReg (regularized NeuralODE)
  - Same drift as 5) but with physics-inspired regularizers: Lipschitz/spectral norm on layers, Jacobian Frobenius or divergence penalty, small Gaussian noise injection at training.
  - Improves stability/generalization without adding parameters.

- 5e) NXRO-ResidualMix (RO+Diag residualization)
  - Compose 3) and 5): \(f_θ = L(t)X + f_{RO+Diag}(X,t) + R_θ([X,φ(t)])\) with a strong residual penalty and/or small residual scaling parameter \(\alpha\) learned or clamped (0<\alpha≪1).

Training tips for these intermediates
- Start from 5) hyperparameters; increase weight decay and residual penalties if test RMSE > train RMSE.
- Keep hidden sizes small (16–64), heads ≤ 2, ranks ≤ 2–4, and apply dropout 0.1–0.3.
- Use multi-step rollout loss (e.g., 1→K schedule) once single-step stabilizes.
- Always checkpoint by best validation RMSE; early stop if generalization gap widens.

### Implementation sketch

- `nxro/` (new module)
  - `models.py`: PyTorch modules for `L_θ(t)`, RO heads, diagonal heads, residual MLP.
  - `integrators.py`: monthly Euler step; optional torchdiffeq wrapper.
  - `train.py`: data loading from NetCDF/CSV, split as above, training loop (RMSE), validation, checkpointing.
  - `eval.py`: ACC/RMSE by lead (hindcast), seasonal stddev, plume plots.
- Initialization: optionally import XRO fit (`Lcoef`, `NLb/NLc`) and map to NXRO parameters for warm start.
- Reproducibility: set seeds; save configs and checkpoints.

### Milestones

- M0: NXRO-Linear reproduces XRO linear performance (train RMSE, ACC/RMSE curves close to XRO).
- M1: NXRO-RO matches or improves ACC at short leads while preserving seasonal synchronization.
- M2: NXRO-RO+Diag reaches parity or better RMSE; ablations show value of diagonal NL terms.
- M3: NXRO-Res shows incremental improvement with clear regularization ablation.
- M4: NXRO-NeuralODE explores broader expressivity while honoring structure masks.
- M5: NXRO-Stochastic produces calibrated plumes; coverage, spread–skill consistency.

### Notes

- Keep monthly calendar and cycle indexing identical to XRO (`ncycle=12`).
- Maintain the T/H ordering and the RO basis to leverage the physical prior.
- Prefer teacher forcing initially; introduce multi-step rollouts only after stabilization.

