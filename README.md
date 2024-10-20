# Dynamic System Modeling for Ocean Indices

Model the spatial-temporal interaction between ocean indices: https://psl.noaa.gov/data/climateindices/list/, which are important indicators for atmospheric and ocean dynamics, espcially for forecasting El Nino. 

# Dataset 
Multivariate time series in `data/indices_ocean_19_timeseries.csv` that includes 19 time series sampled from 1951 to 2023, with interval of one month. 

# Models 
We want to study the dynamical interaction between different ocean indices, using a Graph Neural Network (GNN) and perform __symbolic regression__ on the learned GNN to extract interpretable dynamical system information. Baselines: 
- [] Latent Graph ODE
- [] Fourier Neural Operator (FNO)
- [] Spherical Fourier Neural Operator (SFNO)
- [] Multivariate Linear Regression
