import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import LinearMean
from gpytorch.kernels import RBFKernel, MultitaskKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from pygtemporal_models.pyg_temp_dataset import SSTDatasetLoader, inverse_normalize, batch_data_to_timeseries
from utils_pca import reconstruct_enso
import numpy as np
import os
from baseline_models.utils import plot_gp_training_loss
import matplotlib.pyplot as plt

class MultivariateGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_dimensions):
        """
        Multivariate Gaussian Process Model for time series forecasting
        
        Args:
            train_x (torch.Tensor): Training input timestamps
            train_y (torch.Tensor): Training target values
            likelihood (gpytorch.likelihoods): GP likelihood
            num_dimensions (int): Number of dimensions/features in time series
        """
        super().__init__(train_x, train_y, likelihood)
        self.num_dimensions = num_dimensions
        
        # Mean function (linear trend)
        self.mean_module = LinearMean(input_size=1)
        
        # Base kernel (RBF/Gaussian kernel)
        self.base_kernel = RBFKernel()
        
        # Multitask kernel (handles multiple dimensions)
        self.covar_module = MultitaskKernel(
            self.base_kernel,
            num_tasks=num_dimensions,
            rank=1  # Low rank for better computational efficiency
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

class TimeSeriesGP:
    def __init__(self, input_dim, forecast_horizon):
        """
        Gaussian Process for time series forecasting
        
        Args:
            input_dim (int): Number of features/dimensions
        """
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        self.is_fitted = False
        self.models = []  # Will be initialized during fit
        self.optimizers = []  # Will be initialized during fit
        
    def fit(self, x, y=None, n_iter=100):
        """
        Fit GP model for each dimension
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
            y (torch.Tensor, optional): Target tensor of shape (B, N, H)
            n_iter (int): Number of optimization iterations
        """
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        if y is not None and hasattr(y, 'cpu'):
            y = y.cpu().numpy()
        
        # Initialize models and optimizers with training data
        self.models = []
        self.optimizers = []
        
        # Store metrics
        losses_per_dim = [[] for _ in range(self.input_dim)]
        rmses = []
        rmses_reconstructed = []
        
        # Create save directory
        save_path = "results/baseline_models/gp_pcs=20"
        os.makedirs(save_path, exist_ok=True)
        
        # First initialize all models
        print("Initializing models...")
        for dim in range(self.input_dim):
            train_x = torch.arange(x.shape[-1], dtype=torch.float32)
            train_y = torch.from_numpy(x[0, dim, :]).float()
            
            model = GPModel(train_x, train_y)
            self.models.append(model)
            self.optimizers.append(
                torch.optim.Adam(model.parameters(), lr=0.1)
            )
        
        # Then train all models
        print("Training models...")
        for i in range(n_iter):
            # Train each dimension
            for dim in range(self.input_dim):
                model = self.models[dim]
                optimizer = self.optimizers[dim]
                
                model.train()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                
                optimizer.zero_grad()
                output = model(model.train_inputs[0])
                loss = -mll(output, model.train_targets)
                losses_per_dim[dim].append(loss.item())
                loss.backward()
                optimizer.step()
            
            # Every 100 iterations, compute RMSE across all dimensions
            if i % 100 == 0:
                print(f'Iteration {i}/{n_iter}')
                for dim in range(self.input_dim):
                    print(f'Dimension {dim} Loss: {losses_per_dim[dim][-1]:.3f}')
                
                # Compute RMSE if targets are provided
                if y is not None:
                    with torch.no_grad():
                        # Make predictions with current model state
                        pred_mean, _ = self.predict(x, return_std=True)
                        print(f"pred_mean = {pred_mean.shape}")
                        print(f"x = {x.shape}")
                        print(f"y = {y.shape}")
                        # Convert predictions to time series format
                        pred_np = batch_data_to_timeseries(pred_mean, n_pcs=self.input_dim)
                        target_np = batch_data_to_timeseries(y[0, :, :self.forecast_horizon], n_pcs=self.input_dim)
                        
                        # Inverse normalize if needed
                        if hasattr(self, 'dataloader') and self.dataloader.use_normalization:
                            pred_np = inverse_normalize(pred_np, self.dataloader._max, self.dataloader._min)
                            target_np = inverse_normalize(target_np, self.dataloader._max, self.dataloader._min)
                        
                        # Compute RMSE for PCs
                        rmse = np.sqrt(np.mean((pred_np - target_np)**2))
                        rmses.append(rmse)
                        
                        # Compute RMSE for reconstructed ENSO
                        nino34, nino34_pred = reconstruct_enso(
                            pcs=pred_np,
                            real_pcs=target_np,
                            top_n_pcs=self.input_dim
                        )
                        rmse_recon = np.sqrt(np.mean((nino34 - nino34_pred)**2))
                        rmses_reconstructed.append(rmse_recon)
                        
                        print(f'Current RMSE (PCs): {rmse:.3f}, RMSE (ENSO): {rmse_recon:.3f}')
        
        self.is_fitted = True
        
        # Plot training metrics
        plot_gp_training_loss(save_path, losses_per_dim)
        
        if y is not None:
            # Plot RMSE curves
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(0, n_iter, 100), rmses, label='RMSE (PCs)')
            plt.plot(np.arange(0, n_iter, 100), rmses_reconstructed, label='RMSE (ENSO)')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('GP Training RMSE')
            plt.legend()
            plt.savefig(os.path.join(save_path, "gp_training_rmse.png"))
            plt.close()
        
    def predict(self, x, return_std=True):
        """
        Generate forecasts
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
            return_std (bool): Whether to return standard deviations
            
        Returns:
            tuple or np.ndarray: (mean, std) if return_std=True, else just mean
        """

        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
            
        batch_size = x.shape[0]
        forecast_x = torch.arange(
            x.shape[-1], 
            x.shape[-1] + self.forecast_horizon, 
            dtype=torch.float32
        )
        
        means = np.zeros((batch_size, self.input_dim, self.forecast_horizon))
        if return_std:
            stds = np.zeros((batch_size, self.input_dim, self.forecast_horizon))
            
        # Generate predictions for each dimension
        for dim in range(self.input_dim):
            model = self.models[dim]
            model.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = model(forecast_x)
                means[:, dim, :] = pred.mean.numpy()
                if return_std:
                    stds[:, dim, :] = pred.stddev.numpy()
                    
        if return_std:
            return means, stds
        return means
    
    def compute_loss(self, pred, target):
        """
        Compute MSE loss
        """
        return np.mean((pred - target) ** 2)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        """
        Initialize GP model with training data
        
        Args:
            train_x (torch.Tensor): Training input timestamps
            train_y (torch.Tensor): Training target values
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel()
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)