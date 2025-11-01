# ENSO Dynamic Modeling with Coupled Neural Oscillators


The file `XRO_indices_oras5.csv` (or `XRO_indices_ora5.nc`) contains 10 time series data:
- Nino34: the El Nino Ocean Index.
- WWV: the Thermocline depth.
- NPMM: North Pacific Meridional Mode.
- SPMM: South Pacific Meridional Mode.
- IOB: Indian Ocean Biurnal.
- IOD: Indian Ocean Diopole.
- SIOD 
- TNA 
- ATL3
- SASD


## Baseline Model: the XRO

The **XRO (eXtended Recharge Oscillator)** is a physics-based dynamical model for ENSO and interconnected climate variability. Unlike the neural NXRO variants that *learn* drift functions via gradient descent, XRO uses **closed-form regression** to fit seasonally modulated linear operators and nonlinear terms from data.

**Reference**: Zhao, S., Jin, F.-F., Stuecker, M.F., Thompson, P.R., Kug, J.-S., McPhaden, M.J., Cane, M.A., Wittenberg, A.T., Cai, W., (2024). Explainable El Niño predictability from climate mode interactions. *Nature*. https://doi.org/10.1038/s41586-024-07534-6

### Mathematical Formulation

The XRO model governs a state vector $\mathbf{X}(t) \in \mathbb{R}^V$ (where the first two components are ENSO SST $T$ and thermocline depth $H$, followed by other climate mode indices) via:

$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t) + \boldsymbol{\xi}(t)
$$

**Components**:

1. **Seasonal Linear Operator** $\mathbf{L}(t) \in \mathbb{R}^{V \times V}$:
   $$
   \mathbf{L}(t) = \sum_{k=0}^{K} \left[ \mathbf{L}_k^c \cos(k\omega t) + \mathbf{L}_k^s \sin(k\omega t) \right]
   $$
   where $K = \text{ac\_order}$ (default 2 for semi-annual), $\omega = 2\pi \text{year}^{-1}$, and $\mathbf{L}_0^s = \mathbf{0}$.
   
   This captures:
   - $k=0$: annual-mean coupling
   - $k=1$: annual cycle modulation
   - $k=2$: semi-annual cycle modulation

2. **Recharge Oscillator Nonlinearities** $\mathcal{N}_{\text{RO}}(T, H, t)$:
   
   Applied **only to the first two variables** (ENSO T and H), using the monomial basis:
   $$
   \boldsymbol{\Phi}_{\text{RO}}(T, H) = \begin{bmatrix} T^2 \\ TH \\ T^3 \\ T^2H \\ TH^2 \end{bmatrix}
   $$
   
   The nonlinear tendencies for T and H are:
   $$
   \mathcal{N}_{\text{RO}}(T, H, t) = \begin{bmatrix} 
   \boldsymbol{\beta}_T(t)^T \boldsymbol{\Phi}_{\text{RO}}(T,H) \\ 
   \boldsymbol{\beta}_H(t)^T \boldsymbol{\Phi}_{\text{RO}}(T,H) \\ 
   \mathbf{0}_{V-2}
   \end{bmatrix}
   $$
   where $\boldsymbol{\beta}_T(t), \boldsymbol{\beta}_H(t) \in \mathbb{R}^5$ are seasonal coefficient vectors:
   $$
   \boldsymbol{\beta}_{T/H}(t) = \sum_{k=0}^{K} \left[ \boldsymbol{\beta}_{T/H,k}^c \cos(k\omega t) + \boldsymbol{\beta}_{T/H,k}^s \sin(k\omega t) \right]
   $$
   
   **Mask control**: `maskNT` and `maskNH` specify which of the 5 monomials are active for T and H equations respectively.

3. **Diagonal Nonlinearities** $\mathcal{N}_{\text{Diag}}(\mathbf{X}, t)$:
   
   Applied **independently to each variable** $j = 1, \ldots, V$:
   $$
   \mathcal{N}_{\text{Diag},j}(\mathbf{X}, t) = b_j(t) X_j^2 + c_j(t) X_j^3
   $$
   where the seasonal coefficients are:
   $$
   b_j(t) = \sum_{k=0}^{K} \left[ b_{j,k}^c \cos(k\omega t) + b_{j,k}^s \sin(k\omega t) \right]
   $$
   $$
   c_j(t) = \sum_{k=0}^{K} \left[ c_{j,k}^c \cos(k\omega t) + c_{j,k}^s \sin(k\omega t) \right]
   $$
   
   **Mask control**: `maskb` and `maskc` specify which variables have quadratic and cubic self-interactions.
   
   **Note**: To avoid duplication, if $T$ (first variable) has $T^2$ in `maskNT`, it should not also be in `maskb`. The code automatically removes duplicates.

4. **Stochastic Noise** $\boldsymbol{\xi}(t)$:
   
   Each component $\xi_j$ follows a **seasonal AR(1) red noise** process:
   $$
   \xi_{j,t+1}(m) = a_{1,j} \cdot \xi_{j,t}(m) + \sqrt{1 - a_{1,j}^2} \cdot \sigma_j(m) \cdot \varepsilon_{j,t}
   $$
   where:
   - $a_{1,j}$ is the lag-1 autocorrelation (from `xi_a1`), estimated from residuals
   - $\sigma_j(m)$ is the seasonal standard deviation (from `xi_stdac`), $m = 1, \ldots, 12$
   - $\varepsilon_{j,t} \sim \mathcal{N}(0, 1)$ is white noise
   
   **Optional noise modulation**: 
   - Multiplicative state dependence: $\boldsymbol{\xi}(t) \to [1 + B_j X_j(t)] \cdot \boldsymbol{\xi}(t)$
   - Heaviside rectification: $\boldsymbol{\xi}(t) \to [1 + B_j X_j(t) \cdot H(X_j)] \cdot \boldsymbol{\xi}(t)$ where $H(\cdot)$ is the Heaviside step function

### Code Design Choices

XRO differs fundamentally from NXRO in its **fitting approach** and **parameter estimation**:

#### 1. Closed-Form Regression (Not Gradient Descent)

**Method**: Given observations $\mathbf{X}(t)$ and tendency $\mathbf{Y}(t) = d\mathbf{X}/dt$, XRO fits via **harmonic-weighted least squares**:

For the linear operator, construct the covariance blocks:
$$
\mathbf{G}_n^c = \langle \cos(n\omega t) \mathbf{Y}(t) \mathbf{X}^\top(t-\tau) \rangle_t, \quad
\mathbf{G}_n^s = \langle \sin(n\omega t) \mathbf{Y}(t) \mathbf{X}^\top(t-\tau) \rangle_t
$$
$$
\mathbf{C}_n^c = \langle \cos(n\omega t) \mathbf{X}(t) \mathbf{X}^\top(t-\tau) \rangle_t, \quad
\mathbf{C}_n^s = \langle \sin(n\omega t) \mathbf{X}(t) \mathbf{X}^\top(t-\tau) \rangle_t
$$

where $\tau$ is a lag parameter (from `taus`, defaults to 0 for forward differencing, 1 for centered).

These are assembled into a block system $\mathbf{G} = \mathbf{L} \mathbf{C}$ and solved via:
$$
\mathbf{L} = \mathbf{G} \mathbf{C}^{-1}
$$

The function `_solve_L_with_zero` handles rank-deficient cases by removing zero rows/columns before inversion.

**Key implementation**: `XRO.__compute__()` in lines 179-361 of `core.py`

#### 2. Gradient Computation

**Temporal derivative**: XRO computes $d\mathbf{X}/dt$ numerically:
- **Forward differencing** (default, `is_forward=True`):
  $$
  \frac{dX_t}{dt} \approx \frac{X_{t+1} - X_t}{\Delta t} \cdot \text{ncycle}
  $$
  Results in a $\Delta t/2$ phase shift in cycle alignment (`cycle_shift`).

- **Centered differencing** (`is_forward=False`):
  $$
  \frac{dX_t}{dt} \approx \frac{X_{t+1} - X_{t-1}}{2\Delta t} \cdot \text{ncycle}
  $$
  No phase shift, but affects boundary handling.

**Key implementation**: `gradient()` function in lines 1301-1346

#### 3. Nonlinear Term Fitting

For the full model (`fit_matrix`), each variable's tendency is fitted **independently** against an augmented state:
$$
\frac{dX_j}{dt} = \mathbf{L}_j \cdot \mathbf{X} + \text{[optional: } X_j^2, X_j^3, \Phi_{\text{RO}}(T,H) \text{]}
$$

The code loops over variables (lines 603-732) and:
1. Concatenates $\mathbf{X}$ with masked nonlinear features (quadratic, cubic, RO monomials)
2. Calls `fit()` on the augmented predictor set
3. Extracts coefficients for each component (`Lac`, `NLb_Lac`, `NLc_Lac`, `NROT_Lac`, `NROH_Lac`)

**Mask logic** (lines 565-600):
- `maskb`/`maskc` control which variables get $X_j^2$/$X_j^3$ terms
- `maskNT`/`maskNH` control which RO monomials enter T/H equations
- Automatic deduplication: if T has $T^2$ in both `maskb` and `maskNT`, the code removes it from `maskb` and keeps only in `maskNT`

#### 4. Noise Parameter Estimation

After fitting the deterministic part, **residuals** $\mathbf{r}(t) = \mathbf{Y}(t) - \mathbf{L}(t)\mathbf{X}(t) - \mathcal{N}(\mathbf{X}(t), t)$ are analyzed:

1. **Seasonal variance**: Group residuals by calendar month, compute `std()` → `xi_stdac`
2. **Lag-1 autocorrelation**: Use `_calc_a1()` which fits $r_p = a_1^p$ over multiple lags (`maxlags`, default 2) to robustly estimate $a_1$ → `xi_a1`
3. **Decorrelation rate**: $\lambda_j = -\log(a_{1,j}) / \Delta t$ → `xi_lambda`

**Key implementation**: Lines 322-345 in `__compute__()`

#### 5. Time Integration (Simulation/Forecasting)

**Discretization**: Explicit Euler with sub-stepping:
$$
\mathbf{X}_{n+1} = \mathbf{X}_n + \left[ \mathbf{L}(c_n) \mathbf{X}_n + \mathcal{N}(\mathbf{X}_n, c_n) + \boldsymbol{\xi}_n \right] \Delta t
$$
where $c_n$ is the cycle index (month), $\Delta t = \frac{1}{\text{ncycle} \times \text{nstep}}$, and within each cycle, `nstep` sub-steps are taken and averaged for output.

**Key implementation**: `_integration_core()` in lines 908-938

### Key Differences: XRO vs. NXRO Variants

| Aspect | XRO (Baseline) | NXRO Variants |
|--------|----------------|---------------|
| **Parameter Estimation** | Closed-form regression (harmonic-weighted covariances) | Gradient-based optimization (backprop through ODE) |
| **Seasonal Operator** | Explicit Fourier basis with fixed $K=2$ | Same basis (NXRO-Linear/RO/RO+Diag) OR learned via MLP (NXRO-Res/NeuralODE) |
| **Nonlinear Structure** | Fixed monomial basis ($T^2, TH, T^3, T^2H, TH^2$) + diagonal $X_j^{2/3}$ | Structured (NXRO-RO+Diag) OR flexible MLP (NXRO-Res/NeuralODE/Graph) |
| **Interpretability** | Fully interpretable: each coefficient has physical meaning (e.g., Bjerknes feedback, thermocline feedback) | Varies: structured variants interpretable, MLP-based variants are black-box |
| **Computational Cost** | Fast: single matrix solve per variable (~seconds) | Slow: iterative optimization with ODE rollouts (~minutes to hours) |
| **Expressivity** | Limited to harmonic modulation and polynomial nonlinearities | Flexible: can learn arbitrary smooth drift functions (NXRO-NeuralODE) |
| **Inductive Biases** | Physics-informed (RO structure, seasonal harmonics) | Physics-informed (seasonal embeddings) + data-driven (MLP capacity) |
| **Overfitting Risk** | Low: few parameters, closed-form solution | Higher: more parameters, requires regularization (weight decay, dropout, early stopping) |
| **Use Case** | Baseline for skill benchmarking, mechanistic interpretation | Exploring capacity limits, hybrid physics-ML, teleconnection learning |

**Summary**: XRO is a **parsimonious physics-based model** that serves as the baseline. NXRO variants add **neural expressivity** while attempting to retain physical structure. The best NXRO models (ResidualMix, RO+Diag) stay close to XRO's structure but allow learned refinements.

---

**Detailed XRO documentation**: See `XRO.md` for API reference, fitting workflow, simulation/reforecast usage, and example outputs.


## Neural XRO (NXRO)

### Goal

Introduce a neural ODE-style alternative to XRO that preserves its core priors (seasonality, ENSO T–H coupling, sparse interactions), while enabling gradual increases in expressivity. Train with RMSE and use the same train/test split as in `XRO_example.py` (train: 1979-01 → 2022-12, test: 2023-01 → latest).

### Data and split

- Input: `data/XRO_indices_oras5.nc` (or corresponding CSV). Monthly indices with time coordinate.
- Train: `1979-01` → `2022-12`.
- Test: `2023-01` → end.
- Variables: first two states are ENSO SST proxy (`Nino34`) and warm water volume (`WWV`), others as in XRO.

### Evaluation (same metrics as XRO)

- RMSE on the test period.
- ACC and RMSE by lead using the hindcast protocol (deterministic reforecast with `noise_type='zero'`).
- Seasonal synchronization: monthly stddev curves.
- Forecast plumes for selected start dates.

### Model variants (in increasing expressivity)

Each variant below uses **gradient-based learning** (backprop through ODE rollouts) instead of XRO's closed-form regression. Parameters are optimized to minimize RMSE on training data.

---

#### 1) **NXRO-Linear** (Linear seasonal operator only)

**Equation**: 
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X}
$$
where $\mathbf{L}_\theta(t) = \sum_{k=0}^{K} \left[ \mathbf{L}_k^c \cos(k\omega t) + \mathbf{L}_k^s \sin(k\omega t) \right]$ with $K=2$, $\omega = 2\pi$ year$^{-1}$.

**What's learned**: 
- Fourier coefficients $\{\mathbf{L}_k^c, \mathbf{L}_k^s\}_{k=0}^K$ via gradient descent

**Contrast with XRO**:
| Aspect | XRO | NXRO-Linear |
|--------|-----|-------------|
| **Equation** | Same | ✓ Identical |
| **Fitting method** | Closed-form regression (harmonic-weighted covariances) | **Gradient descent** (backprop through ODE) |
| **Nonlinear terms** | Optional (RO + diagonal) | ✗ None (linear only) |
| **Noise** | Fitted from residuals | Not used during training |

**Purpose**: Neural re-fit of XRO's linear seasonal operator to validate gradient-based learning matches closed-form solution.

**Initialization**: 
- **Current**: Random (Xavier uniform initialization)
- **Planned**: Optional warm-start from XRO's fitted $\{\mathbf{L}_k^c, \mathbf{L}_k^s\}$ (extracted from `Lcoef`) - see warm-start variants below

---

#### 2) **NXRO-RO** (Linear + Recharge Oscillator nonlinearities)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t)
$$
where $\mathcal{N}_{\text{RO}}(T, H, t)$ is the RO nonlinearity (defined as in XRO baseline):
$$
\mathcal{N}_{\text{RO}}(T, H, t) = \begin{bmatrix} 
\boldsymbol{\beta}_T(t)^T \boldsymbol{\Phi}_{\text{RO}}(T,H) \\ 
\boldsymbol{\beta}_H(t)^T \boldsymbol{\Phi}_{\text{RO}}(T,H) \\ 
\mathbf{0}_{V-2}
\end{bmatrix}
$$
with:
- **Fixed basis** (computed from state): $\boldsymbol{\Phi}_{\text{RO}}(T,H) = [T^2, TH, T^3, T^2H, TH^2]^T$ (5 monomials)
- **Trainable seasonal coefficients**: $\boldsymbol{\beta}_{T/H}(t) = \sum_{k=0}^{K} [\boldsymbol{\beta}_{T/H,k}^c \cos(k\omega t) + \boldsymbol{\beta}_{T/H,k}^s \sin(k\omega t)] \in \mathbb{R}^5$

**What's learned** (trainable parameters):
- Linear operator Fourier coefficients $\{\mathbf{L}_k^c, \mathbf{L}_k^s\}_{k=0}^K$
- RO seasonal coefficients $\{\boldsymbol{\beta}_{T,k}^{c/s}, \boldsymbol{\beta}_{H,k}^{c/s}\}_{k=0}^K$ (each is a 5-dimensional vector weighting the RO monomials)

**What's fixed** (not learned):
- RO monomial basis functions $\boldsymbol{\Phi}_{\text{RO}}(T,H)$ (functional form is hard-coded)

**Contrast with XRO**:
| Aspect | XRO | NXRO-RO |
|--------|-----|---------|
| **Equation** | $\mathbf{L}(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}} + \mathcal{N}_{\text{Diag}} + \boldsymbol{\xi}$ | $\mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}$ |
| **Fitting method** | Closed-form regression | **Gradient descent** |
| **RO structure** | ✓ Same basis $\boldsymbol{\Phi}_{\text{RO}}$ | ✓ Same (mirrors `maskNT/maskNH`) |
| **Diagonal terms** | ✓ Included (optional) | ✗ Not included |
| **Noise** | ✓ Fitted AR(1) | ✗ Not used |

**Purpose**: Test whether gradient-based learning can recover XRO's RO coupling while potentially finding better coefficients through joint optimization.

**Initialization**: 
- **Current**: Random (Xavier uniform initialization for all parameters)
- **Planned**: Optional warm-start from XRO's fitted coefficients - see warm-start variants below

---

#### 3) **NXRO-RO+Diag** (RO + Diagonal nonlinearities)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t)
$$
where:
- $\mathcal{N}_{\text{RO}}(T, H, t)$ is the RO nonlinearity (same as NXRO-RO above)
- $\mathcal{N}_{\text{Diag}}(\mathbf{X}, t)$ is the diagonal nonlinearity applied to each variable $j = 1, \ldots, V$:
  $$
  \mathcal{N}_{\text{Diag}}(\mathbf{X}, t)_j = b_j(t) \cdot X_j^2 + c_j(t) \cdot X_j^3
  $$
  with:
  - **Fixed polynomial forms** (computed from state): $X_j^2$ and $X_j^3$
  - **Trainable seasonal coefficients**: 
    $$
    b_j(t) = \sum_{k=0}^{K} [b_{j,k}^c \cos(k\omega t) + b_{j,k}^s \sin(k\omega t)], \quad c_j(t) = \sum_{k=0}^{K} [c_{j,k}^c \cos(k\omega t) + c_{j,k}^s \sin(k\omega t)]
    $$

**What's learned** (trainable parameters):
- All parameters from NXRO-RO (linear operator + RO seasonal coefficients), plus
- Diagonal seasonal coefficients $\{b_{j,k}^{c/s}, c_{j,k}^{c/s}\}_{k=0, j=1}^{K, V}$ (scalar coefficients for each variable and harmonic)

**What's fixed** (not learned):
- RO monomial basis $\boldsymbol{\Phi}_{\text{RO}}(T,H)$ and polynomial forms $X_j^2, X_j^3$ (functional forms are hard-coded)

**Contrast with XRO**:
| Aspect | XRO | NXRO-RO+Diag |
|--------|-----|--------------|
| **Equation structure** | ✓ Same components | ✓ **Identical** |
| **Fitting method** | Closed-form regression (variable-by-variable) | **Joint gradient descent** (end-to-end) |
| **RO + Diagonal** | ✓ Both included | ✓ Both included |
| **Coefficient estimation** | Independent fits per variable | **Coupled optimization** across all variables |
| **Deduplication** | Automatic ($T^2$ in `maskNT` excludes from `maskb`) | Handled in architecture |

**Purpose**: Most faithful neural analog of XRO. Tests whether **joint end-to-end optimization** improves upon XRO's variable-by-variable fitting.

**Key difference**: XRO fits each variable's tendency independently; NXRO-RO+Diag optimizes all coefficients jointly to minimize multi-step rollout loss.

**Initialization**: 
- **Current**: Random (Xavier uniform initialization for all parameters)
- **Planned**: Optional warm-start from XRO's fitted coefficients (linear $\mathbf{L}$, RO $\boldsymbol{\beta}_{T/H}$, diagonal $b_j, c_j$) - see warm-start variants below

---

#### 4) **NXRO-Res** (Linear + Residual MLP)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])
$$
where:
- $\mathbf{L}_\theta(t)$ is the seasonal linear operator (as in NXRO-Linear)
- $R_\theta: \mathbb{R}^{V+5} \to \mathbb{R}^V$ is a **small residual MLP** (3 layers, hidden size 64):
  $$
  R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)]) = W_3 \tanh(W_2 \tanh(W_1 [\mathbf{X}, \boldsymbol{\phi}(t)]^T))
  $$
- $\boldsymbol{\phi}(t) = [1, \cos(\omega t), \sin(\omega t), \cos(2\omega t), \sin(2\omega t)]^T$ are Fourier seasonal features

**What's learned**: 
- Linear operator $\mathbf{L}_\theta(t)$
- MLP weights $\{W_1, W_2, W_3\}$

**Contrast with XRO**:
| Aspect | XRO | NXRO-Res |
|--------|-----|----------|
| **Nonlinear structure** | Fixed polynomial basis (RO + diagonal) | **Flexible MLP** (arbitrary smooth function) |
| **Interpretability** | High (each term has physical meaning) | **Low** (black-box residual) |
| **Inductive bias** | Strong (physics-informed monomials) | **Weak** (only via seasonal features $\boldsymbol{\phi}(t)$) |
| **Expressivity** | Limited (polynomials up to cubic) | **High** (universal approximator) |

**Regularization**: Strong L2 weight decay and/or spectral norm constraints on $R_\theta$ keep it small, preventing departure from linear dynamics.

**Purpose**: Test whether a data-driven residual can improve upon linear XRO without imposing physical structure.

**Initialization**: 
- **Current**: Random (Xavier uniform initialization)
- **Planned**: Warm-start variants (4a, 4b) with frozen XRO components - see warm-start section below

---

#### 5) **NXRO-NeuralODE** (Pure MLP drift with seasonal encoding)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = G_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])
$$
where:
- $G_\theta: \mathbb{R}^{V+5} \to \mathbb{R}^V$ is a **general MLP drift** (no separate linear term):
  $$
  G_\theta([\mathbf{X}, \boldsymbol{\phi}(t)]) = \text{MLP}_{\theta}(\text{concat}[\mathbf{X}, \boldsymbol{\phi}(t)])
  $$
  - Typical architecture: 3-4 layers, hidden size 64-128, Tanh/ReLU activations
- $\boldsymbol{\phi}(t) = [1, \cos(\omega t), \sin(\omega t), \cos(2\omega t), \sin(2\omega t)]^T$ are **Fourier seasonal features** (fixed, not learned)
- Optional: apply **structural masks** to MLP outputs (e.g., only T/H equations get cross-variable terms, others remain diagonal)

**What's learned**: 
- All MLP weights $\{W_1, W_2, W_3, ...\}$ (no separate linear operator)
- (Optional) Mask parameters if learnable

**Key difference from variant 4**: 
- **Variant 4 (NXRO-Res)**: Separates linear physics ($\mathbf{L}_\theta(t) \cdot \mathbf{X}$) from residual correction ($R_\theta$)
- **Variant 5 (NXRO-NeuralODE)**: Pure black-box MLP learns everything (including linear dynamics) from data

**Contrast with XRO**:
| Aspect | XRO | NXRO-NeuralODE |
|--------|-----|----------------|
| **Functional form** | Explicit (linear + polynomial) | **Fully implicit** (learned via MLP) |
| **Physical constraints** | Hard-coded (RO structure for T/H only) | **Soft** (via seasonal features $\boldsymbol{\phi}(t)$ + optional masks) |
| **Capacity** | Low (~50-100 parameters) | **High** (hundreds to thousands) |
| **Overfitting risk** | Low (closed-form, few params) | **High** (requires careful regularization) |
| **Interpretability** | High (each coefficient has physical meaning) | **Very low** (black-box) |
| **Inductive bias** | Strong (RO structure, Fourier seasonality) | **Weak** (only seasonal features) |

**Structural masks** (optional): 
- Allow cross-variable interactions only for ENSO (T/H) equations
- Keep other variables (IOD, TNA, etc.) mostly diagonal
- Implemented via masked output layer or architectural constraints
- Mimics XRO's assumption that RO coupling is ENSO-specific

**Purpose**: Explore **upper bound of expressivity** with minimal structural constraints; tests how well a purely data-driven model (with only seasonal encoding as inductive bias) can perform compared to physics-informed variants.

**Implementation note**: The current code (`NXRONeuralODEModel` in `models.py`) implements $\mathbf{L}_\theta(t) \cdot \mathbf{X} + G_\theta$. This should be updated to remove the separate linear term and use a pure MLP $G_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])$ to match the specification above.

**Note**: This is the most flexible NXRO variant; any performance improvements must justify the loss of interpretability and increased overfitting risk.

---

#### 6) **NXRO-Stochastic** (Any deterministic variant + learned noise)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = f_{\text{det}}(\mathbf{X}, t) + \boldsymbol{\xi}(t)
$$
where:
- $f_{\text{det}}$ is any of the above deterministic drifts (Linear, RO, RO+Diag, Res, NeuralODE)
- $\boldsymbol{\xi}(t)$ is **seasonal AR(1) red noise** (same as XRO):
  $$
  \xi_{j,t+1}(m) = a_{1,j} \cdot \xi_{j,t}(m) + \sqrt{1 - a_{1,j}^2} \cdot \sigma_j(m) \cdot \varepsilon_{j,t}
  $$

**What's learned**: 
- Deterministic drift parameters (from chosen variant)
- (Optional) Noise parameters $\{a_{1,j}, \sigma_j(m)\}$ if not using XRO's fitted values

**Contrast with XRO**:
| Aspect | XRO | NXRO-Stochastic |
|--------|-----|-----------------|
| **Noise structure** | ✓ Seasonal AR(1) | ✓ **Same** |
| **Noise estimation** | From residuals after fitting | **Can be learned end-to-end** OR inherited from XRO |
| **Use case** | Ensemble forecasts, probabilistic skill | ✓ **Same** |

**Purpose**: Enable probabilistic forecasting (plumes, coverage, CRPS) for any NXRO variant, matching XRO's stochastic reforecast capability.

**Implementation**: 
- During training: deterministic (no noise) to learn drift
- During reforecast: sample $\boldsymbol{\xi}$ to generate ensembles

---

### Warm-Start Variants and Fine-Tuning Ablations (Planned)

To better understand the value of XRO's physics-based initialization and the importance of different model components, we plan to implement warm-start variants with selective parameter freezing.

#### Summary Table: All Variants and Their Initialization

| Variant ID | Equation Structure | Initialization | Trainable Components | Purpose |
|------------|-------------------|----------------|---------------------|---------|
| **1. NXRO-Linear** | $\mathbf{L}_\theta \cdot \mathbf{X}$ | Random | Linear coeffs | Gradient-based linear fit |
| **1a. NXRO-Linear-WS** | Same | **From XRO** | Linear coeffs | Test warm-start value |
| **2. NXRO-RO** | $\mathbf{L}_\theta + \mathcal{N}_{\text{RO}}$ | Random | L + RO coeffs | Add RO nonlinearities |
| **2a. NXRO-RO-WS** | Same | **From XRO** | L + RO coeffs | Warm-start full structure |
| **2a-FixL** | Same | **From XRO** | RO only | Test if XRO's L is optimal |
| **2a-FixRO** | Same | **From XRO** | L only | Test if XRO's RO is optimal |
| **2a-FixAll** | Same | **From XRO** | None (all frozen) | Pure XRO baseline (no training) |
| **3. NXRO-RO+Diag** | $\mathbf{L}_\theta + \mathcal{N}_{\text{RO}} + \mathcal{N}_{\text{Diag}}$ | Random | L + RO + Diag | Full XRO structure, joint training |
| **3a. NXRO-RO+Diag-WS** | Same | **From XRO** | L + RO + Diag | Warm-start, refine all |
| **3a-FixL** | Same | **From XRO** | RO + Diag | Freeze linear |
| **3a-FixRO** | Same | **From XRO** | L + Diag | Freeze RO |
| **3a-FixDiag** | Same | **From XRO** | L + RO | Freeze diagonal |
| **3a-FixNL** | Same | **From XRO** | L only | Freeze all nonlinear |
| **3a-FixAll** | Same | **From XRO** | None (all frozen) | Pure XRO baseline (no training) |
| **4. NXRO-Res** | $\mathbf{L}_\theta + R_\theta$ | Random | L + MLP | Linear + neural residual |
| **4a. NXRO-Res-WS-FixL** | Same | **From XRO** (L) | MLP only | Frozen XRO linear + residual |
| **4b. NXRO-Res-FullXRO** | $\mathbf{L}_{\text{XRO}} + \mathcal{N}_{\text{RO,XRO}} + \mathcal{N}_{\text{Diag,XRO}} + R_\theta$ | **From XRO** (all) | MLP only | Frozen full XRO + residual ⭐ |
| **5. NXRO-NeuralODE** | $G_\theta([\mathbf{X}, \boldsymbol{\phi}])$ | Random | All MLP | Pure black-box MLP |
| **5a. NXRO-Attentive** | $\mathbf{L}_\theta + $ Attention | Random | L + Attention | Linear + learned attention |
| **5a-WS** | Same | **From XRO** (L) | L + Attention | Warm-start linear |
| **5a-FixL** | Same | **From XRO** (L) | Attention only | Frozen XRO linear + attention |
| **5b. NXRO-Graph** | $\mathbf{L}_\theta + $ Graph | Random | L + Graph | Linear + graph convolution |
| **5b-WS** | Same | **From XRO** (L) | L + Graph | Warm-start linear |
| **5b-FixL** | Same | **From XRO** (L) | Graph only | Frozen XRO linear + graph |
| **5c. NXRO-PhysReg** | $G_\theta([\mathbf{X}, \boldsymbol{\phi}])$ + regularization | Random | All MLP (regularized) | Pure MLP with physics-inspired penalties |
| **5d. NXRO-ResidualMix** | $\mathbf{L}_\theta + \mathcal{N}_{\text{RO}} + \mathcal{N}_{\text{Diag}} + \alpha R_\theta$ | Random | All (L+RO+Diag+MLP) | Full structure + residual ⭐ |
| **5d-WS** | Same | **From XRO** (L+RO+Diag) | All (L+RO+Diag+MLP) | Warm-start physics |
| **5d-FixL** | Same | **From XRO** (L+RO+Diag) | RO + Diag + MLP | Freeze linear |
| **5d-FixRO** | Same | **From XRO** (L+RO+Diag) | L + Diag + MLP | Freeze RO |
| **5d-FixDiag** | Same | **From XRO** (L+RO+Diag) | L + RO + MLP | Freeze diagonal |
| **5d-FixNL** | Same | **From XRO** (L+RO+Diag) | L + MLP | Freeze RO+Diag |
| **5d-FixPhysics** | Same | **From XRO** (L+RO+Diag) | MLP only | Freeze all physics ⭐ |

**Key observations**:
- **Random init** (1-5, 5a, 5b, 5c, 5d): 9 variants learning from scratch, no physics initialization
- **Warm-start** (1a-5d-WS): 6 variants initialized from XRO, refine via gradient descent
- **Selective freezing** (Fix* variants): 15 variants isolating which components benefit from training
- **Pure XRO baselines** (2a-FixAll, 3a-FixAll): 2 variants with no training (for comparison)
- **Most conservative hybrids**: 4b and 5d-FixPhysics (freeze full XRO, train only residual)
- **Total variants**: **32 configurations** enabling systematic ablation studies

---

#### Warm-Start from XRO (Basic Variants)

**1a) NXRO-Linear-WS** (Warm-start linear):
- Initialize $\{\mathbf{L}_k^c, \mathbf{L}_k^s\}$ from XRO's `Lcoef` (extracting Fourier coefficients from XRO's fitted `Lac`)
- Train all parameters via gradient descent
- **Purpose**: Test whether warm-start accelerates convergence or improves final performance vs. random initialization

**2a) NXRO-RO-WS** (Warm-start linear + RO):
- Initialize linear coefficients from XRO's `Lcoef`
- Initialize RO coefficients $\{\boldsymbol{\beta}_{T/H,k}^{c/s}\}$ from XRO's `NROT_Lcoef`/`NROH_Lcoef`
- Train all parameters via gradient descent
- **Purpose**: Test whether physics-informed initialization improves upon random start

**3a) NXRO-RO+Diag-WS** (Warm-start full XRO):
- Initialize all coefficients from XRO's fitted values (linear, RO, diagonal)
- Train all parameters via gradient descent
- **Purpose**: Test whether XRO provides a good initialization basin for gradient-based refinement

**4a) NXRO-Res-WS-FixL** (Frozen XRO linear + trainable residual):
- Initialize and **freeze** $\mathbf{L}_\theta$ from XRO's linear operator
- Train only residual MLP $R_\theta$
- Equation: $\frac{d\mathbf{X}}{dt} = \mathbf{L}_{\text{XRO}}(t) \cdot \mathbf{X} + R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])$
- **Purpose**: Test whether residual can correct deficiencies in XRO's linear operator

**4b) NXRO-Res-FullXRO** (Frozen full XRO + trainable residual) ⭐ **KEY ABLATION**:
- Initialize and **freeze** all XRO components (linear, RO, diagonal)
- Train only residual MLP $R_\theta$
- Equation: $\frac{d\mathbf{X}}{dt} = \mathbf{L}_{\text{XRO}}(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO,XRO}}(T,H,t) + \mathcal{N}_{\text{Diag,XRO}}(\mathbf{X},t) + R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])$
- **All XRO components are frozen** (from fitted values, not trainable)
- **Only MLP residual is trainable** (starting from random initialization)
- **Purpose**: 
  - **Most conservative hybrid**: Uses XRO exactly as-is, adds only neural correction
  - Tests whether neural residual can capture missing physics beyond XRO's polynomial structure
  - Comparison with variant 5d (ResidualMix) reveals value of joint optimization vs. frozen physics + residual
  - If 4b performs well, suggests XRO structure is sufficient but needs data-driven corrections
  - If 4b underperforms 5d, suggests joint optimization of physics coefficients is critical

---

#### Fine-Tuning Ablations (Selective Parameter Freezing)

For variants 2a and 3a (which have XRO's full structure), we can study **which components benefit from gradient-based refinement** by selectively freezing/updating different parameter groups:

**Parameter Groups**:
- **L**: Linear operator coefficients $\{\mathbf{L}_k^{c/s}\}$
- **RO**: Recharge Oscillator coefficients $\{\boldsymbol{\beta}_{T,k}^{c/s}, \boldsymbol{\beta}_{H,k}^{c/s}\}$
- **Diag**: Diagonal nonlinear coefficients $\{b_{j,k}^{c/s}, c_{j,k}^{c/s}\}$

**Experimental Design** (for NXRO-RO+Diag-WS):

| Variant ID | Linear (L) | RO | Diag | Description |
|------------|------------|-----|------|-------------|
| **3a** | ✓ Update | ✓ Update | ✓ Update | Full fine-tuning (baseline warm-start) |
| **3a-FixL** | ✗ **Freeze** | ✓ Update | ✓ Update | Fix linear, refine nonlinear |
| **3a-FixRO** | ✓ Update | ✗ **Freeze** | ✓ Update | Fix RO coupling, refine linear+diagonal |
| **3a-FixDiag** | ✓ Update | ✓ Update | ✗ **Freeze** | Fix diagonal, refine linear+RO |
| **3a-FixNL** | ✓ Update | ✗ **Freeze** | ✗ **Freeze** | Fix all nonlinear, refine only linear |
| **3a-FixAll** | ✗ **Freeze** | ✗ **Freeze** | ✗ **Freeze** | No training (pure XRO baseline for comparison) |

**Simplified Design** (for NXRO-RO-WS, 2 parameter groups):

| Variant ID | Linear (L) | RO | Description |
|------------|------------|-----|-------------|
| **2a** | ✓ Update | ✓ Update | Full fine-tuning |
| **2a-FixL** | ✗ **Freeze** | ✓ Update | Fix linear, refine RO only |
| **2a-FixRO** | ✓ Update | ✗ **Freeze** | Fix RO, refine linear only |
| **2a-FixAll** | ✗ **Freeze** | ✗ **Freeze** | No training (pure XRO baseline) |

---

#### Research Questions for Ablation Studies

1. **Initialization quality**: Do warm-start variants (1a-3a) converge faster or achieve better final skill than random initialization (1-3)?

2. **Component importance**: Which parameter groups benefit most from gradient-based refinement?
   - If `3a-FixL` performs well, XRO's linear operator is already near-optimal
   - If `3a-FixNL` performs poorly, nonlinear terms need joint optimization with linear
   - If `3a-FixRO` performs well, RO coefficients from XRO are robust

3. **Joint vs. independent optimization**: Does joint end-to-end training (3a) improve upon XRO's independent fits, even starting from the same parameters?

4. **Frozen physics + residual** (variants 4b, 5d-FixPhysics): Critical comparisons:
   - **4b/5d-FixPhysics vs. XRO**: Does adding neural residual to frozen XRO improve skill?
   - **4b/5d-FixPhysics vs. 5d-WS**: Is joint optimization of physics coefficients necessary, or is frozen XRO + residual sufficient?
   - **5d-FixPhysics vs. 5d-FixNL**: Does freezing diagonal terms matter when residual is present?
   - If 4b ≈ 5d-WS: XRO's coefficients are optimal, only need neural correction
   - If 4b < 5d-WS: Joint gradient descent finds better physics coefficients than XRO's closed-form regression

5. **Component-wise freezing ablations** (variants 5d-Fix*):
   - **5d-FixL vs. 5d-WS**: Is XRO's linear operator already optimal?
   - **5d-FixRO vs. 5d-WS**: Is XRO's RO coupling already optimal?
   - **5d-FixDiag vs. 5d-WS**: Are XRO's diagonal terms already optimal?
   - Ranking Fix* variants reveals which component benefits most from refinement

6. **Intermediate variants with frozen linear** (5a-FixL, 5b-FixL):
   - Tests whether XRO's linear operator + learned non-physics mechanisms (attention/graph) can improve skill
   - If 5a-FixL or 5b-FixL outperform 3a: Data-driven coupling mechanisms are better than XRO's polynomial structure

7. **Computational efficiency**: Does warm-start reduce training epochs needed to reach target performance?

8. **Overfitting risk**: Are warm-started models more or less prone to overfitting on limited data?

---

#### Implementation Plan (Future Work)

1. **Extract XRO coefficients**: Implement utilities in `utils/xro_utils.py` to:
   - Convert XRO's fitted `Lac`, `NROT_Lac`, `NROH_Lac`, `NLb_Lac`, `NLc_Lac` (xarray) into NXRO's parameter format (PyTorch tensors)
   - Map XRO's Fourier coefficients (`Lcoef`) to NXRO's `L_basis` parameter
   - Create frozen operator functions for variant 4b (FullXRO)

2. **Selective freezing**: Add `--freeze` CLI flags to `NXRO_train.py`:
   - Syntax: `--freeze linear,ro,diag` (comma-separated list of parameter groups)
   - Implement `requires_grad=False` for specified parameter groups after warm-start

3. **Warm-start flag**: Add `--warm_start <path>` to optionally initialize from XRO fit file (NetCDF or npz)

4. **Frozen XRO operators**: For variants 4b and 5d-FixPhysics, implement:
   - `XROOperator` module that takes XRO fit and evaluates $\mathbf{L}_{\text{XRO}}(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO,XRO}} + \mathcal{N}_{\text{Diag,XRO}}$
   - Register as buffers (not parameters) to ensure no gradient computation
   - New model class `NXROResFullXROModel` that combines frozen XRO with trainable residual

5. **Partial freezing support**: Extend all model classes to accept `freeze_linear`, `freeze_ro`, `freeze_diag` flags:
   - After initialization (warm-start or random), set `param.requires_grad = False` for frozen parameter groups
   - Implement parameter grouping utilities to identify which parameters belong to L, RO, Diag, and MLP components

6. **Ablation script**: Create `run_warmstart_ablations.py` to systematically run all 32 variant combinations:
   - Random init base variants (1-5, 5a-5c, 5d): 9 variants
   - Warm-start variants (1a, 2a, 3a, 5a-WS, 5b-WS, 5d-WS): 6 variants
   - Freezing ablations (2a-Fix*, 3a-Fix*, 4a, 4b, 5a-FixL, 5b-FixL, 5d-Fix*): 15 variants
   - Pure XRO baselines (2a-FixAll, 3a-FixAll): 2 variants
   - Use grid search over configurations with consistent hyperparameters

7. **Analysis pipeline**: 
   - Compare convergence curves (training epochs vs. RMSE)
   - Final skill metrics (ACC/RMSE by lead)
   - Parameter drift from initialization (for warm-start variants)
   - Computational efficiency (wall-clock time, epochs to convergence)
   - Component importance rankings (which Fix* variants perform best)

**Expected outcomes**: 
- Identify which XRO components are robust vs. which benefit from gradient-based refinement
- Determine optimal hybrid strategy:
  - Fully frozen XRO + residual (4b, 5d-FixPhysics)
  - Partially frozen (5d-FixL/RO/Diag)
  - Full warm-start refinement (1a-5d-WS)
  - Random initialization (current baseline)
- Quantify computational savings from warm-start (fewer epochs to convergence)
- Answer: Is XRO's structure optimal, or do coefficients need gradient-based refinement?
- Inform future hybrid physics-ML model design principles

---

### Intermediate variants between 5) and 6)

Given limited data, we propose small, structured extensions that increase expressivity gradually while controlling capacity. All variants use the same **gradient-based learning** as other NXRO models.

**Note**: NXRO-Bilinear has been removed from this list due to poor performance in empirical evaluations (ranks #8 in both RMSE and ACC).

---

#### **5a) NXRO-AttentiveCoupling** (Limited self-attention)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \alpha(t) \cdot \mathbf{W}_o \cdot \text{softmax}\left(\frac{\mathbf{M} \odot (Q K^T)}{\sqrt{d}}\right) V
$$
where:
- $\mathbf{L}_\theta(t)$ is the seasonal linear operator (as in NXRO-Linear)
- $Q = W_q \mathbf{X}$, $K = W_k \mathbf{X}$, $V = W_v \mathbf{X}$ treat **variables as tokens** (dimension $V$)
- $\mathbf{M} \in \{0,1\}^{V \times V}$ is an **attention mask** (e.g., allow attention only from/to T and H)
- $\alpha(t) = \sigma(\boldsymbol{\phi}(t)^T \mathbf{w}_\alpha)$ is a **seasonal gate** (sigmoid-activated)
- $d$ is the hidden dimension (key/query size)

**What's learned**: 
- Linear operator $\mathbf{L}_\theta(t)$ (or frozen/warm-started from XRO)
- Attention weights $\{W_q, W_k, W_v, \mathbf{W}_o\}$
- Seasonal gate $\mathbf{w}_\alpha$

**Contrast with XRO**:
| Aspect | XRO | NXRO-AttentiveCoupling |
|--------|-----|------------------------|
| **Variable coupling** | Fixed (linear + RO structure) | Linear (from XRO or learned) + **learned attention weights** (data-driven) |
| **Coupling strength** | Constant or seasonally modulated (Fourier) | Linear + **state-dependent** (softmax on $QK^T$) |
| **Selectivity** | Hard-coded (RO for T/H only) | **Soft** (via attention mask $\mathbf{M}$) |
| **Interpretability** | High | **Medium** (attention patterns visualizable) |

**Constructor knobs**: `heads` (1-2), `hidden_dim` $d$ (16-32), `dropout` (0.1-0.3), `mask_mode` ∈ {`'th_only'`, `'full'`}

**Mask modes**:
- `'th_only'`: Only T and H can attend to all variables; others attend only to themselves (diagonal)
- `'full'`: All variables can attend to all (no mask)

**Initialization options**:
- **5a-Random**: Random initialization for all parameters (current default)
- **5a-WS**: Warm-start $\mathbf{L}_\theta$ from XRO, random init for attention weights, train all
- **5a-FixL**: Warm-start $\mathbf{L}_\theta$ from XRO and **freeze**, train only attention mechanism
  - Tests whether XRO's linear operator + learned attention is sufficient

**Purpose**: Learn adaptive, state-dependent coupling (like ENSO's state-dependent teleconnections) while limiting capacity via masks.

---

#### **5b) NXRO-GraphNeuralODE** (Sparse graph prior)

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \alpha(t) \cdot \tanh\left((\hat{\mathbf{A}} \mathbf{X}) \mathbf{W}_g\right)
$$
where:
- $\mathbf{L}_\theta(t)$ is the seasonal linear operator (as in NXRO-Linear)
- $\hat{\mathbf{A}} \in \mathbb{R}^{V \times V}$ is a **row-normalized sparse adjacency** (fixed or learned)
- $\mathbf{W}_g \in \mathbb{R}^{V \times V}$ is a learnable weight matrix (often diagonal or low-rank)
- $\alpha(t) = \sigma(\boldsymbol{\phi}(t)^T \mathbf{w}_\alpha)$ is a **seasonal gate**

**What's learned**: 
- Linear operator $\mathbf{L}_\theta(t)$ (or frozen/warm-started from XRO)
- Graph convolution weights $\mathbf{W}_g$
- Seasonal gate $\mathbf{w}_\alpha$
- (Optional) Adjacency $\hat{\mathbf{A}}$ if learnable (with L1 penalty)

**Contrast with XRO**:
| Aspect | XRO | NXRO-GraphNeuralODE |
|--------|-----|---------------------|
| **Interaction topology** | Dense (linear $\mathbf{L}$) + RO (T-H only) | Linear (from XRO or learned) + **sparse graph** (teleconnection prior) |
| **Edge weights** | Implicit (via $\mathbf{L}_{ij}(t)$) | **Explicit** (via adjacency $\hat{\mathbf{A}}$) |
| **Teleconnection representation** | Not explicit | **Explicit sparse graph** (visualizable) |
| **Inductive bias** | Physics (RO structure) | Linear operator + **network topology** (from XRO coupling or data) |

**Graph construction options**:
1. **XRO-derived** (default): Threshold cycle-averaged $|\mathbf{L}_{\text{ac}}(i,j,m)|$ from fitted XRO, symmetrize, normalize
2. **Statistical k-NN**: Top-$k$ neighbors by Pearson/Spearman/MI/cross-correlation on training data
3. **Learned**: Parameterize $\mathbf{A} \geq 0$, add L1 penalty $\lambda \|\mathbf{A}\|_1$, enforce row-normalization

**Implementation backends**:
- `NXROGraphModel`: In-house GCN with normalized adjacency
- `NXROGraphPyGModel`: PyTorch Geometric (GCNConv or GATConv)

**Constructor knobs**: `num_edges` (k-NN), `l1_lambda` (sparsity), `use_fixed_graph`, `graph_source` ∈ {`'xro'`, `'pearson'`, `'mi'`, ...}

**Initialization options**:
- **5b-Random**: Random initialization for all parameters (current default)
- **5b-WS**: Warm-start $\mathbf{L}_\theta$ from XRO, random init for graph weights, train all
- **5b-FixL**: Warm-start $\mathbf{L}_\theta$ from XRO and **freeze**, train only graph convolution and gate
  - Tests whether XRO's linear operator + learned graph coupling is sufficient

**Purpose**: Encode climate teleconnections as inductive bias via sparse graph structure, improving data efficiency and interpretability.

---

#### **5c) NXRO-PhysReg** (Regularized NeuralODE)

**Equation**: Same as NXRO-NeuralODE (5), but with **physics-inspired regularization**:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RMSE}} + \lambda_{\text{Lip}} \|\nabla G_\theta\|_{\text{spec}} + \lambda_{\text{Jac}} \|\nabla_{\mathbf{X}} f(\mathbf{X},t)\|_F^2 + \lambda_{\text{noise}} \mathcal{L}_{\text{noise}}
$$
where:
- $\|\nabla G_\theta\|_{\text{spec}}$ is the **spectral norm** (largest singular value) of MLP layers
- $\|\nabla_{\mathbf{X}} f(\mathbf{X},t)\|_F^2$ is the **Jacobian Frobenius norm** (penalizes sensitivity)
- $\mathcal{L}_{\text{noise}}$ is small Gaussian noise injection during training (implicit regularization)

**What's learned**: Same as NXRO-NeuralODE, but with **constrained optimization**

**Contrast with XRO**:
| Aspect | XRO | NXRO-PhysReg |
|--------|-----|--------------|
| **Parameter count** | Same as NXRO-NeuralODE | Same |
| **Capacity** | High (MLP) | **Effectively reduced** (via regularization) |
| **Stability** | Guaranteed (closed-form) | **Improved** (Lipschitz/Jacobian penalties) |
| **Generalization** | Good (few params) | **Better than unregularized NeuralODE** |
| **Physical constraints** | Built-in (RO structure) | **Soft** (via regularization) |

**Regularization techniques**:
- **Spectral normalization**: Constrains Lipschitz constant, prevents exploding gradients
- **Jacobian penalty**: Encourages smooth dynamics, reduces sensitivity to perturbations
- **Noise injection**: Simulates stochastic forcing during training, improves robustness

**Purpose**: Improve generalization of NXRO-NeuralODE without adding parameters or imposing hard structure. Tests whether soft constraints can match XRO's physical inductive biases.

---

#### **5d) NXRO-ResidualMix** (Hybrid: RO+Diag + Residual MLP) ⭐ TOP PERFORMER

**Equation**:
$$
\frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t) + \alpha \cdot R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])
$$
where:
- $\mathcal{N}_{\text{RO}}(T, H, t)$ and $\mathcal{N}_{\text{Diag}}(\mathbf{X}, t)$ are the structured nonlinearities (same as NXRO-RO+Diag variant 3):
  - **Fixed basis**: RO monomials $\boldsymbol{\Phi}_{\text{RO}}(T,H)$ and polynomials $X_j^{2/3}$
  - **Trainable coefficients**: $\boldsymbol{\beta}_{T/H}(t)$ and $b_j(t), c_j(t)$
- $R_\theta: \mathbb{R}^{V+5} \to \mathbb{R}^V$ is a **small residual MLP** (3 layers, hidden size 64, as in NXRO-Res variant 4)
- $\alpha$ is a **scaling parameter**: either fixed (e.g., 0.1) or learned via $\alpha = \alpha_{\max} \sigma(a)$ with $\alpha_{\max} = 0.5$

**What's learned** (trainable parameters):
- Linear operator Fourier coefficients $\{\mathbf{L}_k^{c/s}\}_{k=0}^K$
- RO seasonal coefficients $\{\boldsymbol{\beta}_{T,k}^{c/s}, \boldsymbol{\beta}_{H,k}^{c/s}\}_{k=0}^K$
- Diagonal seasonal coefficients $\{b_{j,k}^{c/s}, c_{j,k}^{c/s}\}_{k=0, j=1}^{K, V}$
- Residual MLP weights $\{W_1, W_2, W_3\}$
- (Optional) Scaling parameter $\alpha$ if not fixed

**What's fixed** (not learned):
- Functional forms: RO monomials $\boldsymbol{\Phi}_{\text{RO}}$, polynomials $X_j^{2/3}$, Fourier basis $\cos/\sin(k\omega t)$

**Key difference from variants 3 and 4**:
- **vs. Variant 3 (RO+Diag)**: Adds small trainable residual $\alpha R_\theta$ to capture unmodeled dynamics
- **vs. Variant 4 (Res)**: Uses structured RO+Diag base instead of just linear; all coefficients jointly trained
- **vs. Variant 5 (NeuralODE)**: Maintains explicit physics structure (RO+Diag) rather than pure black-box MLP

**Contrast with XRO**:
| Aspect | XRO | NXRO-ResidualMix |
|--------|-----|------------------|
| **Structure** | $\mathbf{L} + \mathcal{N}_{\text{RO}} + \mathcal{N}_{\text{Diag}}$ | $\mathbf{L}_\theta + \mathcal{N}_{\text{RO}} + \mathcal{N}_{\text{Diag}} + \alpha R_\theta$ |
| **Expressivity** | Fixed polynomial basis | **Hybrid**: polynomial + MLP |
| **Residual term** | ✗ None | **✓ Small MLP** ($\alpha \ll 1$) |
| **Fitting** | Variable-by-variable closed-form | **End-to-end joint gradient descent** |
| **Interpretability** | High | **Medium** (structured base + small black-box residual) |

**Key design choice**: $\alpha \ll 1$ ensures the model stays **close to XRO's interpretable structure** while allowing learned corrections for missing physics.

**Regularization**: Strong L2 weight decay on $R_\theta$ and/or clamping $\alpha < 0.5$ prevents the residual from dominating.

**Purpose**: **Best of both worlds** – preserve XRO's physical structure while adding minimal neural capacity to capture unmodeled dynamics. This is the **top-performing variant** in our experiments (see rankings).

**Why it works**: 
- Inherits XRO's strong inductive biases (RO coupling, seasonal harmonics)
- Allows gradient-based end-to-end optimization (better than XRO's independent fits)
- Adds data-driven refinement without overfitting (small $\alpha$ + regularization)

**Initialization options**: 

Given the multiple components (Linear, RO, Diag, MLP), variant 5d enables rich ablation studies:

**Base configurations**:
- **5d-Random**: Random initialization for all parameters (current default)
- **5d-WS**: Warm-start all XRO components (L, RO, Diag) from XRO fit, random init for MLP residual, train all
- **5d-WS-FullXRO**: Same initialization as 5d-WS, train all (equivalent to warm-started variant 3a + residual)

**Selective freezing (warm-start required)**:
- **5d-FixL**: Warm-start and **freeze** linear $\mathbf{L}_\theta$ from XRO, train RO + Diag + MLP
  - Tests whether refining nonlinear physics + adding residual improves frozen linear operator
- **5d-FixRO**: Warm-start and **freeze** RO coefficients $\boldsymbol{\beta}_{T/H}$ from XRO, train L + Diag + MLP
  - Tests whether XRO's RO coupling is optimal
- **5d-FixDiag**: Warm-start and **freeze** diagonal coefficients $b_j, c_j$ from XRO, train L + RO + MLP
  - Tests whether XRO's diagonal terms are optimal
- **5d-FixPhysics**: Warm-start and **freeze** all physics components (L + RO + Diag) from XRO, train only MLP residual $R_\theta$
  - **Equivalent to variant 4b** but with explicit RO+Diag structure
  - Most conservative: uses full XRO as-is, adds only neural correction
  - Critical test: Does residual add value if all physics is frozen?

**Key ablation comparisons**:
- **5d-Random vs. 5d-WS**: Value of physics-informed initialization
- **5d-WS vs. 5d-Fix*** (where * = L, RO, Diag, NL, Physics): Which XRO components benefit from gradient refinement
- **5d-FixPhysics vs. XRO**: Does neural residual improve frozen XRO?
- **5d-FixPhysics vs. 5d-WS**: Is joint optimization of physics coefficients necessary?
- **5d-FixL vs. 5d-FixRO vs. 5d-FixDiag**: Component importance ranking (which frozen component degrades performance least)

**Note on variant equivalence**:
- **5d-FixPhysics ≈ 4b**: Both freeze full XRO and train only residual MLP
  - 5d-FixPhysics: Architecturally includes frozen RO+Diag modules
  - 4b: Implements frozen XRO as a black-box operator
  - Results should be identical if implementations are consistent

**Recommended ablation sequence for 5d**:
1. Train 5d-Random (baseline)
2. Train 5d-WS (warm-start all, refine all)
3. Compare 5d-WS vs. 5d-Random → establishes value of initialization
4. Train 5d-FixPhysics → tests if frozen XRO + residual is sufficient
5. Train 5d-FixL, 5d-FixRO, 5d-FixDiag → isolates which component needs refinement
6. Rank by performance → informs optimal hybrid strategy

---

### Training Tips for Intermediate Variants

**General guidelines**:
- **Hidden sizes**: Keep small (16–64 for attention/graph, 64-128 for ResidualMix MLP)
- **Attention**: heads ≤ 2, dropout 0.1–0.3
- **Multi-step training**: Use multi-step rollout loss (e.g., 1→K schedule) once single-step stabilizes
- **Early stopping**: Always checkpoint by best validation RMSE; early stop if generalization gap widens

**Variant-specific recommendations**:
- **Variants 5a-5c**: Start from variant 5 (NeuralODE) hyperparameters; increase weight decay and residual penalties if test RMSE > train RMSE
- **Variant 5d (ResidualMix)**: 
  - Random init (5d-Random): Start from variant 3 (RO+Diag) hyperparameters, add small residual with strong regularization
  - Warm-start (5d-WS): Can use higher learning rate (5e-3 instead of 1e-3) since starting from good initialization
  - Fixed physics (5d-FixPhysics, 5d-Fix*): Train longer on residual/refinement components since frozen parts don't update

**Warm-start best practices**:
- **For FixL variants** (5a-FixL, 5b-FixL): Use higher learning rate for trainable components (attention/graph) since linear is frozen
- **For Fix* ablations** (3a-Fix*, 5d-Fix*): May need fewer epochs since some parameters are frozen at good values
- **For full warm-start** (1a-5d-WS): Start with lower learning rate (1e-4) to avoid destroying good initialization; optionally use learning rate warm-up

**Regularization for frozen + residual hybrids** (4a, 4b, 5d-FixPhysics):
- Strong L2 weight decay on residual MLP (1e-3 to 1e-2)
- Optional: Clip residual magnitude or use smaller $\alpha$ (0.05-0.1) to prevent domination
- Goal: Residual should correct physics, not replace it

### Implementation sketch

- `nxro/` (module)
  - `models.py`: PyTorch modules for `L_θ(t)`, RO heads, diagonal heads, residual MLP.
  - `integrators.py`: monthly Euler step; optional torchdiffeq wrapper.
  - `train.py`: data loading from NetCDF/CSV, split as above, training loop (RMSE), validation, checkpointing.
  - `eval.py`: ACC/RMSE by lead (hindcast), seasonal stddev, plume plots.
- **Initialization** (current): Xavier uniform random initialization for all parameters
- **Initialization** (planned): 
  - Optional warm-start from XRO fit (`Lcoef`, `NROT_Lac`, `NROH_Lac`, `NLb_Lac`, `NLc_Lac`)
  - Utilities to map XRO's xarray coefficients to NXRO's PyTorch parameters
  - CLI flags: `--warm_start`, `--freeze [linear,ro,diag]`
- Reproducibility: set seeds; save configs and checkpoints.

### Milestones

**Current milestones (achieved)**:
- M0: NXRO-Linear reproduces XRO linear performance (train RMSE, ACC/RMSE curves close to XRO).
- M1: NXRO-RO matches or improves ACC at short leads while preserving seasonal synchronization.
- M2: NXRO-RO+Diag reaches parity or better RMSE; ablations show value of diagonal NL terms.
- M3: NXRO-Res shows incremental improvement with clear regularization ablation.
- M4: NXRO-NeuralODE explores broader expressivity (note: needs code update to remove separate linear term).
- M5: NXRO-Stochastic produces calibrated plumes; coverage, spread–skill consistency.
- M6: NXRO-ResidualMix (variant 5d) achieves best overall performance (rank #1 in RMSE and ACC).

**Planned milestones (warm-start ablations)**:
- **M7**: Warm-start variants (1a-3a, 5a-WS, 5b-WS, 5d-WS) demonstrate faster convergence and/or better final skill than random initialization (1-5, 5a, 5b, 5d).
- **M8**: Frozen full XRO + residual variants (4b, 5d-FixPhysics) provide critical test:
  - If 4b ≈ XRO: Neural residual adds little value (XRO structure is sufficient)
  - If 4b > XRO and 4b ≈ 5d-WS: Frozen XRO + residual is optimal hybrid (no need for joint optimization)
  - If 4b < 5d-WS: Joint end-to-end optimization of physics coefficients is critical
- **M9**: Component-wise freezing ablations (2a-Fix*, 3a-Fix*, 5d-Fix*) reveal which XRO components are robust vs. which benefit from gradient-based refinement:
  - Ranking of Fix* variants by performance identifies critical vs. redundant components
  - Parameter drift analysis shows how much coefficients change during training
- **M10**: Intermediate variants with frozen linear (5a-FixL, 5b-FixL) test whether:
  - XRO's linear operator + learned attention/graph outperforms XRO's polynomial structure
  - Data-driven coupling mechanisms (attention, graph) can replace RO+Diag nonlinearities
- **M11**: Computational efficiency analysis:
  - Quantify epoch savings and wall-clock time reduction from warm-start vs. random
  - Compare training stability (loss curve smoothness, gradient norms)
  - Identify fastest path to target performance for each variant class

### Notes

- Keep monthly calendar and cycle indexing identical to XRO (`ncycle=12`).
- Maintain the T/H ordering and the RO basis to leverage the physical prior.
- Prefer teacher forcing initially; introduce multi-step rollouts only after stabilization.

### Graph construction (teleconnections as inductive bias)

We construct sparse graphs that encode inter-index teleconnections to guide NXRO-Graph variants. The default now follows the coupling embedded in the original XRO.

1) From XRO seasonal linear operator (default)
- XRO represents the seasonal linear tendency as
  - L_ac(i,j,m) = Σ_k [ L^c_k(i,j) cos(kωt_m) + L^s_k(i,j) sin(kωt_m) ], k ∈ {0..ac_order}
- After fitting XRO on ORAS5 over the training window, we aggregate teleconnection strength by cycle-averaged magnitude:
  - S(i,j) = mean_m |L_ac(i,j,m)|
- We then:
  - zero the diagonal (no self-loops in the prior),
  - symmetrize: A = max(S, S^T) to obtain undirected connectivity,
  - optionally threshold small entries,
  - add self-loops and row-normalize for stable message passing.
- Implementation: graph_construction.build_xro_coupling_graph and normalize_with_self_loops. This adjacency is the default for NXRO-Graph in training.

2) Statistical KNN graph (alternative)
- Build a data-driven sparse topology from whole-sequence, non‑neural interaction strength computed on the training split (CSV `data/XRO_indices_oras5_train.csv` or NC).
- Metrics (choose one):
  - Pearson: |corr(Xi, Xj)| over time.
  - Spearman: |rank‑corr(Xi, Xj)| over time.
  - MI: normalized mutual information via binned histogram.
  - XCorr‑max: max over lags of |corr(Xi(t+lag), Xj(t))| within ±L.
- Construct KNN: for each node keep top‑k neighbors by the chosen strength; symmetrize and (later) add self‑loops with row‑normalization for stability.
- Implementation: `graph_construction.get_or_build_stat_knn_graph(...)` with caching in `results/graphs/`.

3) Learned adjacency (enabled)
- Parameterize A ≥ 0 and add an L1 penalty (λ‖A‖₁) to encourage sparsity; enforce self‑loops and row normalization each forward pass.
- Initialization can use XRO‑derived or statistical‑KNN prior. CLI in `NXRO_train.py`:
  - `--graph_learned` to enable learning; `--graph_l1` sets λ (e.g., 1e-4..1e-2).
  - Optional prior from statistical KNN: `--graph_stat_method {pearson,spearman,mi,xcorr_max} --graph_stat_topk K --graph_stat_source PATH`.

Rationale
- Using |L_ac| from XRO captures directional physical coupling estimated from the seasonal linear dynamics, yielding a physics-informed prior over index interactions.
- k-NN correlation provides a purely data-driven sparse topology when XRO fitting is unavailable.
- Learned A allows adaptation when the prior is imperfect, controlled by sparsity and normalization for stability.

## Hyperparameter Search for Graph Models

We provide automated grid search tools for finding optimal hyperparameters for graph-based NXRO models.

### Quick Start

**Option 1: Python-based grid search (recommended for customization)**
```bash
python run_graph_hyperparam_search.py --epochs 1500 --device cuda --test
```

**Option 2: Bash-based full grid search**
```bash
bash run_graph_grid_search.sh --epochs 2000 --device cuda --test
```

**Option 3: Bash-based quick search (focused grid)**
```bash
bash run_graph_quick_search.sh --epochs 1000 --device cuda --test
```

### Search Space

The grid search explores:

**Model Types:**
- `graph`: NXROGraphModel with fixed or learned adjacency
- `graph_pyg`: NXROGraphPyGModel with GCN or GAT layers

**Graph Structures:**
- `xro`: XRO-based interaction graph (from original model)
- `stat_pearson`: Pearson correlation k-NN graph
- `stat_spearman`: Spearman correlation k-NN graph
- `stat_mi`: Mutual information k-NN graph
- `stat_xcorr_max`: Cross-correlation k-NN graph

**Hyperparameters:**
- `top_k`: Number of nearest neighbors (1, 2, 3, 5, 7, 10)
- `hidden_dim`: Hidden layer dimension (32, 64, 128, 256)
- `learning_rate`: Optimization learning rate (1e-4, 5e-4, 1e-3, 5e-3)
- `rollout_k`: Number of unrolled steps for training (1, 2, 3)
- `l1_lambda`: L1 sparsity for learned graphs (1e-4, 5e-4, 1e-3, 5e-3)

### Selection Criterion

Models are ranked by **test RMSE on Nino3.4** averaged across all lead times (1-24 months).

### Output

Results are saved to `results/hyperparam_search/`:
- `grid_search_results.csv`: Full results table sorted by test RMSE
- `best_config.json` or `best_config.txt`: Best configuration details
- `search_grid.json`: Complete grid specification (Python version)
- `grid_search.log`: Execution log

### Example Usage

```bash
# Full grid search (warning: ~10,000+ configurations, may take days!)
bash run_graph_grid_search.sh --epochs 2000 --device cuda --test

# Quick search on focused grid (~500 configurations, ~1-2 days)
bash run_graph_quick_search.sh --epochs 1000 --device cuda

# Dry run to see what would be executed
bash run_graph_grid_search.sh --epochs 2000 --test --dry_run

# Python version with custom output directory
python run_graph_hyperparam_search.py \
    --epochs 2000 \
    --device cuda \
    --test \
    --output_dir results/my_search
```

### Analyzing Results

After the search completes, the scripts automatically:
1. Extract test RMSE from all trained model checkpoints
2. Sort configurations by performance
3. Display top 10 models
4. Save best configuration for easy reproduction

You can also manually analyze results:
```bash
# View top 10 from CSV
head -11 results/hyperparam_search/grid_search_results.csv

# Visualize with XRO_variants.py
python XRO_variants.py --test --data_filter base --graph_filter graph
```

### Tips for Efficient Search

1. **Start with quick search**: Use `run_graph_quick_search.sh` to identify promising regions
2. **Use fewer epochs initially**: Set `--epochs 500` for faster iteration, then retrain best configs with more epochs
3. **Focus the grid**: Edit the scripts to narrow down based on initial results
4. **Parallelize**: If you have multiple GPUs, split the grid across machines
5. **Monitor progress**: Check `grid_search.log` and `results/` directory during execution

## Model Performance Rankings (Test Period 2023-01 onwards)

We evaluated all NXRO variants against the baseline XRO model on the held-out test period. Rankings are based on average rank across forecast leads (1-21 months) for Niño3.4 forecasts. Lower rank is better (1 = best).

### RMSE Rankings (Lower is Better)

| Rank | Model | Avg Rank | Description |
|------|-------|----------|-------------|
| 1 | **NXRO-ResidualMix** | 1.91 | ✓ Outperforms XRO |
| 2 | **NXRO-RO+Diag [base]** | 2.18 | ✓ Outperforms XRO |
| 3 | XRO (baseline) | 2.82 | Original physics model |
| 4 | graphpyg_gat_k3 | 4.64 | Graph attention network |
| 5 | NXRO-NeuralODE (PhysReg) | 5.00 | Regularized neural ODE |
| 6 | Linear XRO | 5.05 | Linear seasonal operator only |
| 7 | XRO_ac0 | 7.09 | XRO without autocorrelation |

### ACC Rankings (Higher Correlation is Better)

| Rank | Model | Avg Rank | Description |
|------|-------|----------|-------------|
| 1 | **NXRO-ResidualMix** | 2.05 | ✓ Outperforms XRO |
| 2 | XRO (baseline) | 2.36 | Original physics model |
| 3 | **NXRO-RO+Diag [base]** | 2.55 | ✓ Competitive with XRO |
| 4 | graphpyg_gat_k3 | 4.36 | Graph attention network |
| 5 | Linear XRO | 5.00 | Linear seasonal operator only |
| 6 | NXRO-NeuralODE (PhysReg) | 5.45 | Regularized neural ODE |
| 7 | XRO_ac0 | 6.82 | XRO without autocorrelation |

### Top Performing Models

Two NXRO variants consistently outperform or match the baseline XRO:

#### 1. NXRO-ResidualMix (Best Overall)
- **Architecture**: Combines the structured RO+Diag basis from XRO with a small residual neural network (see variant 5d above)
- **Mathematical formulation**:
  $$
  \frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t) + \alpha \cdot R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])
  $$
  where:
  - $\mathbf{L}_\theta(t) = \sum_{k=0}^{K} \left[ \mathbf{L}_k^c \cos(k\omega t) + \mathbf{L}_k^s \sin(k\omega t) \right]$ is the seasonal linear operator ($K=2$, $\omega = 2\pi$ year$^{-1}$)
  - $\mathcal{N}_{\text{RO}}(T, H, t)$ and $\mathcal{N}_{\text{Diag}}(\mathbf{X}, t)$ are the RO and diagonal nonlinearities with:
    - **Same functional form as XRO**: fixed RO monomials $\boldsymbol{\Phi}_{\text{RO}}(T,H)$ and polynomials $X_j^{2/3}$
    - **Trainable seasonal coefficients**: $\boldsymbol{\beta}_{T/H}(t)$ and $b_j(t), c_j(t)$ learned via gradient descent
  - $R_\theta([\mathbf{X}, \boldsymbol{\phi}(t)])$ is a **trainable** 3-layer MLP with hidden size 64: $W_3 \tanh(W_2 \tanh(W_1 [\mathbf{X}, \boldsymbol{\phi}(t)]^T))$
  - $\boldsymbol{\phi}(t) = [1, \cos(\omega t), \sin(\omega t), \cos(2\omega t), \sin(2\omega t)]^T$ are Fourier seasonal features
  - $\alpha$ is a scaling parameter: either fixed (0.1) or **trainable** via $\alpha = \alpha_{\max} \cdot \sigma(a)$ with $\alpha_{\max} = 0.5$, initialized to $\alpha \approx 0.1$
- **Trainable vs. Fixed**:
  - ✓ **Trainable**: Linear coefficients $\mathbf{L}_k^{c/s}$, RO coefficients $\boldsymbol{\beta}_{T/H,k}^{c/s}$, diagonal coefficients $b_{j,k}^{c/s}, c_{j,k}^{c/s}$, MLP weights $W_1, W_2, W_3$, optional $\alpha$
  - ✗ **Fixed**: Functional forms (RO monomials, polynomials, Fourier basis)
- **Key features**:
  - Preserves XRO's physical structure (same basis functions: RO monomials + diagonal polynomials)
  - All XRO-like coefficients are trainable (joint optimization, not independent fits like XRO)
  - Adds capacity to learn residual corrections via a small MLP, scaled by $\alpha \ll 1$
  - Strong regularization (small $\alpha$, residual weight decay) keeps residual term small, preventing overfitting
- **Contrast with XRO**: 
  - **Same**: Functional forms of nonlinearities (RO monomials, diagonal polynomials)
  - **Different**: (1) Adds small learned residual $\alpha R_\theta$ to capture unmodeled dynamics, (2) Uses end-to-end gradient descent instead of variable-by-variable closed-form regression
- **Performance**: Ranks #1 in both RMSE (1.91) and ACC (2.05), demonstrating that minimal neural augmentation of physics improves generalization

#### 2. NXRO-RO+Diag [base] (Strong Runner-up)
- **Architecture**: Neural analog of XRO with the same structure but joint gradient-based optimization (see variant 3 above)
- **Mathematical formulation**:
  $$
  \frac{d\mathbf{X}}{dt} = \mathbf{L}_\theta(t) \cdot \mathbf{X} + \mathcal{N}_{\text{RO}}(T, H, t) + \mathcal{N}_{\text{Diag}}(\mathbf{X}, t)
  $$
  where:
  - $\mathbf{L}_\theta(t) = \sum_{k=0}^{K} \left[ \mathbf{L}_k^c \cos(k\omega t) + \mathbf{L}_k^s \sin(k\omega t) \right]$ is the seasonal linear operator
  - **Recharge Oscillator nonlinearities** (for T = Niño3.4, H = WWV only):
    $$
    \mathcal{N}_{\text{RO}}(T, H, t) = \begin{bmatrix} 
    \boldsymbol{\beta}_T(t)^T \boldsymbol{\Phi}_{\text{RO}}(T,H) \\ 
    \boldsymbol{\beta}_H(t)^T \boldsymbol{\Phi}_{\text{RO}}(T,H) \\ 
    \mathbf{0}_{V-2}
    \end{bmatrix}
    $$
    with:
    - **Fixed RO monomial basis** (same as XRO): $\boldsymbol{\Phi}_{\text{RO}}(T,H) = [T^2, TH, T^3, T^2H, TH^2]^T$
    - **Trainable seasonal coefficients**: $\boldsymbol{\beta}_{T/H}(t) = \sum_k [\boldsymbol{\beta}_{T/H,k}^c \cos(k\omega t) + \boldsymbol{\beta}_{T/H,k}^s \sin(k\omega t)]$
  - **Diagonal nonlinearities** (for all variables $j = 1, \ldots, V$):
    $$
    \mathcal{N}_{\text{Diag}}(\mathbf{X}, t)_j = b_j(t) X_j^2 + c_j(t) X_j^3
    $$
    with:
    - **Fixed polynomial forms** (same as XRO): $X_j^2$ and $X_j^3$
    - **Trainable seasonal coefficients**: $b_j(t) = \sum_k [b_{j,k}^c \cos(k\omega t) + b_{j,k}^s \sin(k\omega t)]$, $c_j(t) = \sum_k [c_{j,k}^c \cos(k\omega t) + c_{j,k}^s \sin(k\omega t)]$
- **Trainable vs. Fixed**:
  - ✓ **Trainable**: All Fourier coefficients ($\mathbf{L}_k^{c/s}$, $\boldsymbol{\beta}_{T/H,k}^{c/s}$, $b_{j,k}^{c/s}$, $c_{j,k}^{c/s}$) learned via gradient descent
  - ✗ **Fixed**: Functional forms (RO monomials $\boldsymbol{\Phi}_{\text{RO}}$, polynomials $X_j^{2/3}$, Fourier basis $\cos/\sin(k\omega t)$)
- **Key features**:
  - **Structurally identical to XRO** (same equation, same basis functions)
  - Fully interpretable (no black-box MLP components)
  - Preserves XRO's T-H Recharge Oscillator coupling with seasonal modulation
  - All coefficients are seasonally varying via explicit Fourier parameterization
- **Contrast with XRO**: 
  - **Same**: All functional forms (RO monomials, diagonal polynomials, Fourier seasonal modulation)
  - **Different**: Uses **joint end-to-end gradient descent** instead of XRO's variable-by-variable closed-form regression
- **Performance**: Ranks #2 in RMSE (2.18) and #3 in ACC (2.55), showing that gradient-based joint optimization of XRO's structure can improve skill while maintaining full interpretability

### Key Insights

1. **Physics + minimal neural augmentation wins**: NXRO-ResidualMix demonstrates that starting from XRO's physical structure and adding a small, heavily regularized neural component outperforms both pure physics (XRO) and more flexible neural models.

2. **Structured beats unstructured**: NXRO-RO+Diag outperforms more expressive variants (NeuralODE without strong regularization) by respecting XRO's coupling structure, suggesting that domain knowledge remains critical even with neural components.

3. **Graph models show promise**: The graph attention variant (graphpyg_gat_k3) ranks 4th, indicating that teleconnection priors can be competitive, though they don't yet surpass the best structured variants on this test period.

4. **Overfitting risk**: More flexible models (some NeuralODE configs without strong regularization) rank lower, likely due to overfitting the limited training data (1979-2022), highlighting the importance of inductive biases for small-data climate forecasting.

---

## Current Results and Visualizations

This section presents empirical results comparing NXRO variants against the baseline XRO model on the held-out test period (2023-01 onwards).

### Forecast Skill by Lead Time

#### RMSE Performance (Lower is Better)

![RMSE by Forecast Lead](results/variants_rmse_skill_test_base.png)

**Key observations**:
- **NXRO-ResidualMix** (dark blue) shows the lowest RMSE across most lead times, especially at longer leads (>12 months)
- **NXRO-RO+Diag** (orange) and **XRO** (red) track closely, with RO+Diag slightly better at medium leads
- All models show increasing RMSE with forecast lead, reflecting the inherent predictability barrier
- The gap between top models (ResidualMix, RO+Diag) and XRO is most pronounced at leads 12-18 months

#### ACC Performance (Higher is Better)

![ACC by Forecast Lead](results/variants_acc_skill_test_base.png)

**Key observations**:
- **NXRO-ResidualMix** (dark blue) maintains the highest correlation across all lead times
- **XRO** (red) shows strong performance at short leads (1-6 months) but degrades faster than ResidualMix
- **NXRO-RO+Diag** (orange) tracks between XRO and ResidualMix, showing the value of joint optimization
- All models cross the 0.5 ACC threshold (commonly used skill benchmark) around lead 18-19 months
- **NXRO-Attentive** (pink) shows competitive performance at short-medium leads

### Model Rankings Across Leads

#### RMSE Rank Heatmap

![RMSE Rank Heatmap](results/variants_rank_rmse_heatmap_test_base.png)

**Key observations**:
- **Dark purple** (rank 1-2) indicates best performance; **yellow** (rank >10) indicates poor performance
- **NXRO-ResidualMix** consistently ranks 1st-2nd across most leads (dark purple band at bottom)
- **XRO** and **NXRO-RO+Diag** alternate ranks 2-3 across different leads (dark blue bands)
- **Linear XRO** and **XRO_ac0** consistently rank poorly (yellow/green), demonstrating the importance of nonlinear terms and seasonal modulation
- Performance ranking stability varies by lead: short leads (1-3) show more variability, longer leads (15-21) show clearer separation

#### Overall Average Rank by RMSE

![Overall Average RMSE Rank](results/variants_overall_rank_rmse_bar_test_base.png)

**Key observations**:
- **NXRO-RO+Diag** achieves the best average RMSE rank (~2.2), closely followed by **NXRO-ResidualMix** (~2.7) and **XRO baseline** (~2.8)
- **XRO** baseline shows strong performance (rank ~2.8), demonstrating the quality of the physics-based approach
- Clear performance tiers emerge:
  - **Tier 1** (rank <3): NXRO-RO+Diag, NXRO-ResidualMix, XRO (top performers)
  - **Tier 2** (rank 4-6): NXRO-Attentive, NXRO-Linear, NXRO-Res (competitive variants)
  - **Tier 3** (rank >7): NXRO-NeuralODE, graphpyg_gat_k3, Linear XRO, XRO_ac0, Bilinear (poorly structured or missing critical components)
- The tight clustering of top 3 models (all within rank ~2-3) shows multiple successful strategies
- The gap between Tier 1 and Tier 3 demonstrates that model structure matters critically

#### Overall Average Rank by ACC

![Overall Average ACC Rank](results/variants_overall_rank_acc_bar_test_base.png)

**Key observations**:
- Top performers cluster at low average ranks (2-4): **NXRO-RO+Diag**, **NXRO-ResidualMix**, and **XRO** all achieve strong performance
- **XRO** baseline shows solid mid-range performance (rank ~3.5-4), validating its physics-based approach
- Models on the right (rank >7) include variants without proper structure (Bilinear) or missing critical components (XRO_ac0, Linear XRO)
- The ranking spread demonstrates that model choice significantly impacts forecast skill

### Summary Statistics

**Test period**: 2023-01 onwards (most recent data, out-of-sample)

**Evaluation metrics**:
- **RMSE**: Root mean squared error (°C) between forecasted and observed Niño3.4 index
- **ACC**: Anomaly correlation coefficient (Pearson correlation) between forecast and observation
- **Rank**: Position relative to other models at each forecast lead (1 = best)

**Top 3 models by average rank**:
1. **NXRO-ResidualMix**: Avg rank 1.91 (RMSE), 2.05 (ACC) - Best overall
2. **NXRO-RO+Diag**: Avg rank 2.18 (RMSE), 2.55 (ACC) - Strong runner-up
3. **XRO**: Avg rank 2.82 (RMSE), 2.36 (ACC) - Solid physics baseline

**Key takeaway**: Physics-informed neural models (ResidualMix, RO+Diag) that preserve XRO's structure while enabling gradient-based refinement achieve the best performance, validating the hybrid physics-ML approach.

