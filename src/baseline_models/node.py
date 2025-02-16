import torch
import torch.nn as nn
from torchdiffeq import odeint

class TimeSeriesNODE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, forecast_horizon=10):
        """
        Neural ODE for time series forecasting
        
        Args:
            input_dim (int): Number of features/dimensions in the time series (N)
            hidden_dim (int): Size of hidden layers in the ODE function
            forecast_horizon (int): Number of future time steps to forecast (H)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # ODE function network
        self.ode_func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input time series of shape (B, N, T)
            
        Returns:
            torch.Tensor: Forecasted values of shape (B, N, H)
        """
        batch_size = x.shape[0]
        
        # Get the last state from the input sequence
        initial_state = x[:, :, -1]  # Shape: (B, N)
        
        # Create time points for forecasting
        t = torch.linspace(0, self.forecast_horizon, self.forecast_horizon).to(x.device)
        
        # Define the ODE function
        def ode_func(t, state):
            return self.ode_func(state)
        
        # Solve ODE to get forecasts
        # odeint returns a tensor of shape (H, B, N)
        predictions = odeint(
            ode_func,
            initial_state,
            t,
            method='rk4'
        )
        
        # Reshape predictions to (B, N, H)
        predictions = predictions.permute(1, 2, 0)
        
        return predictions
    
    def compute_loss(self, pred, target, std=None, add_sin_cos=False):
        """
        Compute MSE loss between predictions and targets
        
        Args:
            pred (torch.Tensor): Predicted values of shape (B, N, H)
            target (torch.Tensor): Target values of shape (B, N, H)
            std (torch.Tensor, optional): Standard deviations for weighted MSE
            add_sin_cos (bool): Whether sinusoidal features are present
        """
        if add_sin_cos:
            # Exclude last two dimensions (sin/cos)
            pred = pred[:, :-2]
            target = target[:, :-2]
            if std is not None:
                std = std[:-2]
        
        if std is not None:
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)