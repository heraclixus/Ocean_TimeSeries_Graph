import torch
import torch.nn as nn
from torchdiffeq import odeint
from baseline_models.utils import PeriodicActivation

# Time Series Neural ODE this model is only using the last time step of the input sequence
class TimeSeriesNODE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, forecast_horizon=10, use_periodic_activation=False):
        """
        Neural ODE for time series forecasting
        
        Args:
            input_dim (int): Number of features/dimensions in the time series (N)
            hidden_dim (int): Size of hidden layers in the ODE function
            forecast_horizon (int): Number of future time steps to forecast (H)
            use_periodic_activation (bool): Whether to use periodic activation
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.use_periodic_activation = use_periodic_activation
        # ODE function network
        if use_periodic_activation:
            self.ode_func = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                PeriodicActivation(),
                nn.Linear(hidden_dim, hidden_dim),
                PeriodicActivation(),
                nn.Linear(hidden_dim, input_dim)
            )
        else:
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
                std = std[:-2]
        if std is not None:
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)


# Define the ODE function
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, use_periodic_activation=False):
        super(ODEFunc, self).__init__()
        self.use_periodic_activation = use_periodic_activation
        if use_periodic_activation:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                PeriodicActivation(),
                nn.Linear(hidden_dim, hidden_dim),
                PeriodicActivation(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(self, t, x):
        return self.net(x)

# Define the full model with Encoder-Decoder structure
class NeuralODEForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_series_length, forecast_length, use_periodic_activation=False):
        super(NeuralODEForecaster, self).__init__()
        self.input_dim = input_dim
        self.time_series_length = time_series_length
        self.forecast_length = forecast_length
        self.hidden_dim = hidden_dim
        self.use_periodic_activation = use_periodic_activation
        # Encoder (GRU)
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # ODE Function
        self.ode_func = ODEFunc(hidden_dim, use_periodic_activation)

        # Decoder (GRU)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Encoder: Process the input sequence with a GRU
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, time_series_length, input_dim)
        _, hidden = self.encoder_gru(x)  # Get last hidden state (shape: [1, batch_size, hidden_dim])
        
        # Define time points for ODE evolution
        t_eval = torch.linspace(0, self.forecast_length, self.forecast_length).to(x.device)

        # Neural ODE evolution of hidden state
        evolved_hidden = odeint(self.ode_func, hidden.squeeze(0), t_eval, method='rk4')  # (forecast_length, batch_size, hidden_dim)

        evolved_hidden = evolved_hidden.permute(1, 0, 2)  # Reshape to (batch_size, forecast_length, hidden_dim)

        # Decoder GRU expects (batch_size, forecast_length, hidden_dim), so we initialize it with the last hidden state
        decoder_hidden = evolved_hidden[:, -1, :].unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)

        # Prepare decoder input (zero tensor to start)
        decoder_input = torch.zeros(batch_size, self.forecast_length, self.hidden_dim).to(x.device)

        # Decoder: Process evolved hidden states to generate output sequence
        decoder_output, _ = self.decoder_gru(decoder_input, decoder_hidden)  # (batch_size, forecast_length, hidden_dim)

        # Final output transformation
        output = self.output_layer(decoder_output)  # Shape: (batch_size, forecast_length, input_dim)

        return output.permute(0, 2, 1)  # Reshape to (batch_size, input_dim, forecast_length)

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
                std = std[:-2]
        if std is not None:
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)