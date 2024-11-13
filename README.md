# Dynamic System Modeling for Ocean Indices

Model the spatial-temporal interaction between ocean indices: https://psl.noaa.gov/data/climateindices/list/, which are important indicators for atmospheric and ocean dynamics, espcially for forecasting El Nino. 

# Dataset 
Multivariate time series in `data/indices_ocean_19_timeseries.csv` that includes 19 time series sampled from 1951 to 2023, with interval of one month. 
Currently, we use year 2010 as the cutoff for train and test split; for time series forecasting, we use a window size of 12 (one year forecast) and training window of 24 (2 years). 


## Feature Sets 

Different stages of feature sets to use for the multivariate time series forecasting problems

### ENSO & Pacific

- nina1: Extreme Eastern Tropical Pacific SST
- nina3: Eastern Tropical Pacific SST
- nina34: East Central Tropical Pacific SST
- nina4: Central Tropical Pacific SST
- pacwarm: Pacific Warmpool Area Average
- soi: Southern Oscillation Index
- tni: Indices of El Niño Evolution
- whwp: Western Hemisphere Warm Pool

### SST: Atlantic
- ammsst: Atlantic Meridional Mode
- tna: Tropical Northern Atlantic Index
- tsa: Tropical Southern Atlantic Index
- amo: Atlantic Multidecadal Oscillation

### Teleconnections
- ea: Eastern Atlantic/Western Russia
- epo: East Pacific/North Pacific Oscillation
- nao: North Atlantic Oscillation
- noi: Northern Oscillation Index
- pna: Pacific North American Index
- wp: Western Pacific Index 
- pdo: Pacific Decadal Oscillation

### Atmosphere and Solar 
- qbo: Quasi-Biennial Oscillation
- solar: Solar Flux



# Models 
We want to study the dynamical interaction between different ocean indices, using a Graph Neural Network (GNN) and perform __symbolic regression__ on the learned GNN to extract interpretable dynamical system information. Baselines: 
- [ ] Latent Graph ODE
- [ ] Fourier Neural Operator (FNO)
- [ ] Spherical Fourier Neural Operator (SFNO)
- [ ] Multivariate Linear Regression
- [ ] Traditional Time Series Methods (ARIMA, GARCH, etc.)
