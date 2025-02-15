# Dynamic System Modeling for ENSO Forecast

Forecasting ENSO using PCA Modes

# Dataset 
Multivariate time series in `data/sst_pcs.mat` that includes 20 PCs for ENSO.


# Models 
We want to study the dynamical interaction between PCs using a Graph Neural Network (GNN) based models, we use the following baseline models. 
- Latent Graph ODE (LGODE), [Huang, 2020](https://github.com/ZijieH/LG-ODE)
- Prototypical Graph ODE (PGODE), [Luo, 2024](https://proceedings.mlr.press/v235/luo24b.html)
- Spectral Temporal Graph Neural Network (StemGNN), [Cao, 2021](https://github.com/microsoft/StemGNN)
- Multivariate Time Series GNN (MTGNN), [Wu, 2020](https://github.com/nnzhan/MTGNN)
- Adaptive Graph Convolutional Recurrent Network (AGRN), [Bai, 2020](https://github.com/LeiBAI/AGCRN)
- FourierGNN (FGNN), [Yi, 2023](https://github.com/aikunyi/FourierGNN)

 