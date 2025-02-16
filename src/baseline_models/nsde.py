import torch
import torch.nn as nn
from torchsde import sdeint

class TimeSeriesSDE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, forecast_horizon=10, noise_type="diagonal"):
        """
        Neural SDE for time series forecasting
        
        Args:
            input_dim (int): Number of features/dimensions in the time series (N)
            hidden_dim (int): Size of hidden layers
            forecast_horizon (int): Number of future time steps to forecast (H)
            noise_type (str): Type of noise ('diagonal' or 'general')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.noise_type = noise_type
        self.sde_type = "ito"  # Using Itô SDE formulation
        
        # Drift network (deterministic part)
        self.drift_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Diffusion network (stochastic part)
        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  # Ensures positive diffusion
        )
    
    def f(self, t, y):
        """
        Drift function (deterministic part of the SDE)
        
        Args:
            t (torch.Tensor): Time point
            y (torch.Tensor): Current state (B, N)
            
        Returns:
            torch.Tensor: Drift term
        """
        return self.drift_net(y)
    
    def g(self, t, y):
        """
        Diffusion function (stochastic part of the SDE)
        
        Args:
            t (torch.Tensor): Time point
            y (torch.Tensor): Current state (B, N)
            
        Returns:
            torch.Tensor: Diffusion term
        """
        return self.diffusion_net(y)
    
    def forward(self, x, n_samples=1):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input time series of shape (B, N, T)
            n_samples (int): Number of Monte Carlo samples for prediction
            
        Returns:
            torch.Tensor: Forecasted values of shape (B, N, H) or (n_samples, B, N, H)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get the last state from the input sequence
        initial_state = x[:, :, -1]  # Shape: (B, N)
        
        # Create time points for forecasting
        ts = torch.linspace(0, self.forecast_horizon, self.forecast_horizon).to(device)
        
        # Initialize list to store multiple samples
        all_predictions = []
        
        for _ in range(n_samples):
            # Solve SDE
            # sdeint returns a tensor of shape (H, B, N)
            predictions = sdeint(
                sde=self,
                y0=initial_state,
                ts=ts,
                method='srk',  # Stochastic Runge-Kutta
                dt=1.0/self.forecast_horizon,  # Time step size
                adaptive=False,
                rtol=1e-3,
                atol=1e-3,
            )
            
            # Reshape predictions to (B, N, H)
            predictions = predictions.permute(1, 2, 0)
            all_predictions.append(predictions)
        
        if n_samples == 1:
            return all_predictions[0]
        else:
            # Stack all samples: (n_samples, B, N, H)
            return torch.stack(all_predictions)
    
    def compute_loss(self, pred, target, std=None, add_sin_cos=False):
        """
        Compute MSE loss between predictions and targets
        """
        if pred.dim() == 4:  # Multiple samples case
            pred = pred.mean(dim=0)
        
        if add_sin_cos:
            # Exclude last two dimensions (sin/cos)
            pred = pred[:, :-2]
            target = target[:, :-2]
            if std is not None:
                std = std[:-2]
        
        if std is not None:
            weights = 1.0 / (std ** 2)
            weights = weights.to(pred.device)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)