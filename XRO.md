## XRO: Extended Recharge Oscillator Model – Documentation

This document explains the baseline XRO model implemented in `XRO/core.py` and how it is used in `XRO_example.py` to fit, simulate and forecast ENSO-related time series (e.g., Niño3.4, WWV) and other SST modes.

### What is XRO?

XRO is an extended nonlinear Recharge Oscillator model for ENSO and interconnected climate variability. It represents the coupled system with a linear time-varying operator plus optional nonlinear terms, modulated by seasonal cycles. It includes:

- Annual mean and seasonal modulations up to semi-annual harmonics.
- Quadratic and cubic nonlinear terms for selected state variables.
- Support for red/white stochastic noise with seasonal variance.
- Simulation and reforecasting with ensemble members.

Reference: Zhao et al. (2024), Nature. DOI: 10.1038/s41586-024-07534-6

---

## Mathematical formulation


Let $\mathbf{X}(t) \in \mathbb{R}^n$ be the state vector. By convention in this repository, the first two components are
$T(t)$ (ENSO SST proxy, e.g., Niño3.4) and $H(t)$ (WWV), with optional additional indices thereafter.

The continuous-time governing equation is
$$
\frac{d\mathbf{X}}{dt} 
\;=\; \mathbf{L}(t)\,\mathbf{X}(t) \;+
\; \mathbf{\mathcal{N}}\big(\mathbf{X}(t), t\big)
\;+
\; \boldsymbol{\xi}(t),
$$
where $\mathbf{L}(t)$ is a seasonally modulated linear operator, $\mathbf{\mathcal{N}}$ collects nonlinear tendencies,
and $\boldsymbol{\xi}(t)$ is a stochastic residual (red- or white-noise) forcing.

### Seasonal linear operator

Let $\omega = 2\pi\,\text{year}^{-1}$ and $K=\text{ac\_order}$. The operator is expanded in seasonal harmonics:
$$
\mathbf{L}(t) 
\;=\; \sum_{k=0}^{K} \Big( \mathbf{L}^{c}_{k}\,\cos(k\,\omega t) 
\;+
\mathbf{L}^{s}_{k}\,\sin(k\,\omega t) \Big),
$$
with the understanding that $\mathbf{L}^{s}_{0}=\mathbf{0}$. Internally, these coefficients are stored in `Lcoef` and the
reconstructed cycle-by-cycle operator appears as `Lac[ranky, rankx, cycle]` (see Shapes).

### Nonlinear recharge-oscillator (RO) terms and diagonal nonlinearities

The model allows two kinds of nonlinearities:
- RO monomials involving only the first two state variables $T,H$ and applied specifically in their tendencies.
- Diagonal quadratic/cubic nonlinearities applied componentwise to $\mathbf{X}$.

Define the RO basis
$$\boldsymbol\phi(T,H) = \big[T^2,\; T\,H,\; T^3,\; T^2H,\; TH^2\big]^\top.$$
Then the nonlinear terms entering the $T$- and $H$-equations are
$$
\mathcal{N}_T(T,H,t) = \boldsymbol\beta_T(t)\cdot \boldsymbol\phi(T,H),\qquad
\mathcal{N}_H(T,H,t) = \boldsymbol\beta_H(t)\cdot \boldsymbol\phi(T,H),
$$
where $\boldsymbol\beta_T(t),\boldsymbol\beta_H(t)$ are seasonal coefficient vectors (masked on/off per `maskNT`, `maskNH`).

In addition, diagonal nonlinearities act on each component $X_j$:
$$
\mathcal{N}_{\text{diag},j}(\mathbf{X},t) = b_j(t)\,X_j^2 + c_j(t)\,X_j^3,\quad j=1,\dots,n.
$$
These appear in the code as `NLb_Lac` and `NLc_Lac` (after fitting) and are used in time stepping as
`b[:, None] * X**2 + c[:, None] * X**3`.

Putting these together,
$$
\mathbf{\mathcal{N}}(\mathbf{X},t) = 
\begin{bmatrix}
\mathcal{N}_T(T,H,t) \\[2pt]
\mathcal{N}_H(T,H,t) \\[2pt]
\mathbf{0}_{n-2}
\end{bmatrix}
\;+
\begin{bmatrix}
b_1(t)X_1^2+c_1(t)X_1^3 \\[2pt]
\vdots \\[2pt]
b_n(t)X_n^2+c_n(t)X_n^3
\end{bmatrix}.
$$

### Residual stochastic forcing (red/white noise)

For each component $j$ and seasonal cycle index $c$, the red-noise residual evolves as
$$
\xi_{j, m+1}(c) = a_{1,j}\,\xi_{j, m}(c)
\;+
\sqrt{1-a_{1,j}^2}\;\sigma_j(c)\,\varepsilon_{j,m+1},
$$
where $a_{1,j}$ comes from `xi_a1`, and the seasonal variance $\sigma_j(c)$ from `xi_stdac`. White noise uses $a_{1,j}=0$.
Optionally, a multiplicative factor $1 + B_j X_j$ (or Heaviside variant) can be applied to the noise (`xi_B`).

### Discrete integration used in simulation/forecast

Let $\Delta t = 1/(\text{ncycle}\times \text{nstep})$. With explicit Euler stepping over sub-steps within each cycle
index $c_k$,
$$
\mathbf{X}_{k+1}
\;=\; \mathbf{X}_{k}\;+
\Big[ \mathbf{L}(c_k)\,\mathbf{X}_k + \mathbf{\mathcal{N}}(\mathbf{X}_k, c_k) + \boldsymbol{\xi}_k \Big]\,\Delta t,
$$
and an within-cycle average is accumulated for output. Reforecasting aligns the initial condition cycle with the
target month and integrates for the requested lead.

### Linear-operator fitting

Given samples $X(t), Y(t)=dX/dt$, the code forms cosine/sine-weighted cross-covariance blocks
\[\begin{aligned}
G_n^c &= \cos(n\omega t)\,Y(t)\,X^\top(t-d), &\quad 
G_n^s &= \sin(n\omega t)\,Y(t)\,X^\top(t-d),\\
C_n^c &= \cos(n\omega t)\,X(t)\,X^\top(t-d), &\quad
C_n^s &= \sin(n\omega t)\,X(t)\,X^\top(t-d),
\end{aligned}\]
with integer lag $d$ selected from `taus`. These are assembled into a block system $G = L\,C$, which is solved for
the coefficients (handling zero rows/columns). Reconstructed operators per cycle are reported as `Lac`, while `Lcomp`
retains the harmonic decomposition. A normalized operator is optionally produced by scaling with the input stddevs
(`get_norm_fit`).

### Skill metrics (used in the example)

For a lead $\ell$ (months) and initialization index set $\mathcal{I}$, the deterministic forecast skill is computed as
\[\text{ACC}(\ell) = \operatorname{corr}_{\,i\in\mathcal{I}}\Big( X^{\text{fcst}}_{i}(\ell),\; X^{\text{obs}}(t_i+\ell) \Big),\]
\[\text{RMSE}(\ell) = \sqrt{\,\left\langle \big( X^{\text{fcst}}_{i}(\ell) - X^{\text{obs}}(t_i+\ell) \big)^2 \right\rangle_{i\in\mathcal{I}}}\, .\]
The example computes these either via `climpred` or via manual alignment of `init+lead` pairs across time.

---

## Core API (XRO/core.py)

### Class: `XRO`

Constructor
- `XRO(ncycle=12, ac_order=2, is_forward=True, taus=None, maxlags=2)`
  - **ncycle**: samples per year (12 for monthly, 52 weekly, 365 daily).
  - **ac_order**: seasonal order: 0 (annual mean), 1 (annual), 2 (semi-annual).
  - **is_forward**: gradient differencing mode (forward vs. centered).
  - **taus**: integer lags for each seasonal order; defaults based on `is_forward`.
  - **maxlags**: lags to estimate red-noise memory.

Key methods
- `fit(X, Y, time=None, is_remove_nan=True) -> xr.Dataset`
  - Linear fit for Y = L X + ξ with seasonal modulation. Returns operators and noise statistics.
  - Returns dims/vars (see Shapes below): `Lac`, `Lcomp`, `Lcoef`, `xi_std`, `xi_stdac`, `xi_a1`, `xi_lambda`, `Y_stdac`, `Yfit_stdac`, and fit products.

- `fit_matrix(X, var_names=None, maskb=None, maskc=None, maskNT=None, maskNH=None, time=None) -> xr.Dataset`
  - Full model fit including quadratic (`maskb`) and cubic (`maskc`) nonlinearities and Recharge Oscillator (RO) terms in T/H equations (`maskNT`, `maskNH`).
  - The first two states of `X` must be ENSO T (e.g., Niño3.4) and H (WWV).
  - Avoid duplicating T² and T³ in both `maskb` and `maskNT`; the code removes duplicates automatically.

- `get_norm_fit(fit_ds) -> xr.Dataset`
  - Normalizes linear operator and noise terms by input stddev for comparability across variables.

- `simulate(fit_ds, X0_ds, nyear=10, nstep=10, ncopy=1, seed=None, noise_type='red', time=None, is_xi_stdac=True, xi_B=None, is_heaviside=False) -> xr.Dataset`
  - Integrates the model forward for `nyear` with `ncopy` ensemble members.
  - Supports red or white noise with seasonal amplitude and optional noise scaling `xi_B`.

- `reforecast(fit_ds, init_ds, n_month=12, nstep=10, ncopy=1, seed=None, noise_type='red', is_xi_stdac=True, xi_B=None, is_heaviside=False) -> xr.Dataset`
  - Rolling reforecast initialized from each time point of `init_ds` for `n_month` leads.
  - Sets output dims to `init`, `lead`, and optionally `member`.

- `get_RO_parameters(fit_ds) -> xr.Dataset`
  - Convenience to compute Recharge Oscillator parameters from the fitted operator (with metadata and units).

- `set_NRO_annualmean(fit_ds) -> xr.Dataset`
  - Replaces nonlinear RO seasonal components with their annual mean for analysis/simplification.

Utilities (selected)
- `gen_noise(stddev, nyear, ncopy, seed, noise_type, a1)` – seasonal variance noise generator.
- `variable_xarray_to_model(xr_ds, ncycle)` – flattens `xr.Dataset` to model array with time index.
- `variable_model_to_xarray(model_X, var_names)` – the inverse of the above.
- `gradient(arr, axis, is_forward, ncycle)` – computes dX/dt in year⁻¹.

### Shapes and Coordinates

Returned from `fit`/`fit_matrix` (key variables):
- `Lac`: [ranky, rankx, cycle] – seasonally modulated linear operator.
- `Lcomp`: [ranky, rankx, cycle, ac_rank] – operator components by seasonal rank.
- `Lcoef`: [ranky, rankx, cossin] – harmonic coefficients.
- `xi_a1`: [ranky] – red-noise AR(1) per variable.
- `xi_std`: [ranky, cycle] – overall residual stddev.
- `xi_stdac`: [ranky, cycle] – seasonal residual stddev.
- `Y_stdac`, `Yfit_stdac`: [ranky, cycle] – seasonal stddev of Y and fit.

Simulation/Reforecast outputs:
- `simulate` returns a dataset with one variable per state (e.g., `Nino34`, `WWV`), dims `[time, member]`.
- `reforecast` returns dims `[init, lead]` (and `member` if `ncopy>1`). Lead has attribute `units='months'`.

---

## Example Workflow (XRO_example.py)

The example demonstrates end-to-end usage: fit multiple model variants, simulate stochastic variability, examine seasonal synchronization, and forecast skill.

### 1) Load data and select training range

- `obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')`
- `train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))`
- The dataset contains monthly indices; `time` must be decodable. The order of variables matters: first two must be ENSO SST and WWV.

### 2) Fit model variants

- Control model with seasonal cycles: `XRO(ncycle=12, ac_order=2)` and `fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2','TH'])`.
- No seasonal cycles: `XRO(ncycle=12, ac_order=0)`.
- Linear variant: `fit_matrix(train_ds, maskb=[], maskNT=[])`.

Notes on masks
- `maskb`: quadratic terms enabled for listed variables.
- `maskc`: cubic terms enabled for listed variables.
- `maskNT`, `maskNH`: RO terms included in T/H tendency equations respectively (`['T2', 'TH', 'T3', 'T2H', 'TH2']`).

### 3) Stochastic simulation

```
XROac2_sim = XROac2.simulate(fit_ds=XROac2_fit,
                              X0_ds=train_ds.isel(time=0),
                              nyear=100, ncopy=100,
                              is_xi_stdac=False, seed=2000)
```

Outputs an ensemble dataset with dims `time` and `member`. The example plots several members vs. the linear version to contrast irregular interannual variability.

### 4) Seasonal synchronization

Compute seasonal standard deviation of observations and simulations:
- `stddev_obs = train_ds.groupby('time.month').std('time')`
- `stddev_XROac2 = XROac2_sim.groupby('time.month').std('time')`
- Plot `stddev` of a selected variable (e.g., `Nino34`) versus month. If using Matplotlib, pass NumPy arrays for axes (e.g., `stddev_obs.month.values`) to avoid unit conversion issues.

### 5) Reforecasts and skill

Deterministic reforecast:
```
XROac2_fcst = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds,
                                n_month=21, ncopy=1, noise_type='zero')
```

Forecast skill (two options):
1. Using `climpred` (recommended if available):
   - Create `HindcastEnsemble(XROac2_fcst.sel(init=slice('1979-01','2022-12')))`
   - Add observations and call `.verify(...)` for correlation (ACC) or RMSE.
   - Note: `climpred` API may differ by version; some versions require `add_observations`, others pass `observations` directly to `.verify()`.

2. Manual computation (fallback):
   - For each lead L, align forecasts at `init` with observations at `time=init+L months` and compute correlation or RMSE across `init`.

The example creates ACC and RMSE skill plots for `Nino34` across leads.

### 6) Stochastic reforecast plumes

```
XROac2_fcst_stoc = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds,
                                     n_month=21, ncopy=100, noise_type='red')
```

For a list of initialization dates, plot deterministic, ensemble mean ± std envelope, and observations to form plumes.

---

## Tips and Troubleshooting

- **Variable order**: The first two variables in training data must be ENSO T (e.g., `Nino34`) and H (WWV), per model formulation.

- **Time decoding**: Ensure NetCDF `time` decodes to real datetimes (`xr.open_dataset(..., decode_times=True)`).

- **climpred API differences**: Depending on version:
  - Older: `HindcastEnsemble(...).add_observations(obs).verify(...)`
  - Newer: `HindcastEnsemble(...).verify(observations=obs, ...)`
  - If neither is available, compute skill manually by aligning `init + lead` times with observations.

- **Matplotlib xarray conversion**: When passing xarray `DataArray` to Matplotlib (e.g., `month`), convert to NumPy with `.values` for x and y to avoid `ConversionError`.

- **Nonlinear masks**: Avoid duplicating identical nonlinear forms in both `maskb/maskc` and `maskNT/maskNH`. The implementation prevents double counting for TNT terms.

- **Noise settings**: `simulate` and `reforecast` can use `noise_type='zero'` for deterministic runs or `'red'/'white'` for stochastic; `seed` ensures reproducibility.

---

## Minimal Usage Example

```
import xarray as xr
from XRO.core import XRO

# 1) Load and select training range
obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))

# 2) Fit control model (seasonal, nonlinear)
model = XRO(ncycle=12, ac_order=2)
fit_ds = model.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2','TH'])

# 3) Deterministic reforecast (21 months)
fcst = model.reforecast(fit_ds=fit_ds, init_ds=obs_ds,
                        n_month=21, ncopy=1, noise_type='zero')

# fcst dims: init x lead (and member if ncopy>1)
```

---

## Outputs from the Example Script

- `XRO_simulation.png`: Sample members from stochastic simulation.
- `XRO_seasonal_synchronization.png`: Seasonal stddev by calendar month (obs vs. model variants).
- `XRO_forecast_skill.png`: ACC vs. lead for `Nino34`.
- `XRO_forecast_skill_rmse.png`: RMSE vs. lead for `Nino34`.
- `XRO_forecast_plume_*.png`: Forecast plumes for selected initialization dates.

---

## Notes on Internals

- The linear operator `L(t)` is constructed as a sum of harmonics (cos/sin) up to `ac_order`, enabling seasonal modulation.
- Nonlinear RO forms include `[T², T·H, T³, T²H, TH²]` that can be toggled for T/H equations.
- Red-noise parameters are inferred from residuals; `xi_a1` yields decorrelation rate `lambda = -log(a1)/Δt`.
- All fit outputs are returned as `xarray` objects with explicit dims/coords for clarity and composability.

---

## Environment & Dependencies

Required libraries include `numpy`, `xarray`, `matplotlib`, and optionally `climpred` for skill verification. For NetCDF, ensure proper time decoding support (e.g., `cftime`, `nc-time-axis`).

---

If you have questions or hit edge cases, please open an issue or contact the model authors (see header in `XRO/core.py`).


