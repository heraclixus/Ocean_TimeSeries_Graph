import torch
import torch.nn as nn
from torchsde import sdeint
import torchsde
from baseline_models.utils import PeriodicActivation

class TimeSeriesSDE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, forecast_horizon=10, noise_type="diagonal", use_periodic_activation=False):
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
        self.use_periodic_activation = use_periodic_activation
        self.sde_type = "ito"  # Using Itô SDE formulation
        
        # Drift network (deterministic part)
        if use_periodic_activation:
            self.drift_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                PeriodicActivation(),
                nn.Linear(hidden_dim, input_dim)
            )
        else:
            self.drift_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim)
            )
        

        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()
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
            std = std[:-2]
        
        if std is not None:
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)




# Define the Neural SDE Function
class SDEFunc(torchsde.SDEIto):  # Inherits from torchsde's Ito SDE class
    def __init__(self, hidden_dim, use_periodic_activation=False):
        super(SDEFunc, self).__init__(noise_type="diagonal")
        self.hidden_dim = hidden_dim
        self.use_periodic_activation = use_periodic_activation
        if use_periodic_activation:
            self.f_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                PeriodicActivation(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.f_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Diffusion function g (stochastic part)
        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )

    # Drift term (dx/dt = f(x, t))
    def f(self, t, x):
        return self.f_net(x)

    # Diffusion term (dW_t = g(x, t) * dB_t, where B_t is Brownian motion)
    def g(self, t, x):
        return self.g_net(x)

# Define the full model with Encoder-SDE-Decoder structure
class NeuralSDEForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_series_length, forecast_length, use_periodic_activation=False):
        super(NeuralSDEForecaster, self).__init__()
        self.input_dim = input_dim
        self.time_series_length = time_series_length
        self.forecast_length = forecast_length
        self.hidden_dim = hidden_dim
        self.use_periodic_activation = use_periodic_activation
        # Encoder (GRU)
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # SDE Function
        self.sde_func = SDEFunc(hidden_dim, use_periodic_activation)

        # Decoder (GRU)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, n_samples=10):
        batch_size = x.shape[0]

        # Encoder: Process the input sequence with a GRU
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, time_series_length, input_dim)
        _, hidden = self.encoder_gru(x)  # Get last hidden state (shape: [1, batch_size, hidden_dim])

        # Define time points for SDE evolution
        t_eval = torch.linspace(0, self.forecast_length, self.forecast_length).to(x.device)

        # Initialize an array to store multiple sample paths
        all_forecasts = []
        for _ in range(n_samples):
            # Brownian motion for stochasticity
            bm = torchsde.BrownianInterval(
                t0=0, t1=self.forecast_length, size=(batch_size, self.hidden_dim), device=x.device
            )

            # Neural SDE evolution of hidden state
            evolved_hidden = torchsde.sdeint(self.sde_func, hidden.squeeze(0), t_eval, bm, method='euler')  
            # Shape: (forecast_length, batch_size, hidden_dim)

            evolved_hidden = evolved_hidden.permute(1, 0, 2)  # Reshape to (batch_size, forecast_length, hidden_dim)

            # Decoder GRU expects (batch_size, forecast_length, hidden_dim), so we initialize it with the last hidden state
            decoder_hidden = evolved_hidden[:, -1, :].unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)

            # Prepare decoder input (zero tensor to start)
            decoder_input = torch.zeros(batch_size, self.forecast_length, self.hidden_dim).to(x.device)

            # Decoder: Process evolved hidden states to generate output sequence
            decoder_output, _ = self.decoder_gru(decoder_input, decoder_hidden)  # (batch_size, forecast_length, hidden_dim)

            # Final output transformation
            output = self.output_layer(decoder_output)  # Shape: (batch_size, forecast_length, input_dim)
            all_forecasts.append(output.permute(0, 2, 1))  # Reshape to (batch_size, input_dim, forecast_length)
        # Stack samples to get (num_samples, batch_size, input_dim, forecast_length)
        return torch.stack(all_forecasts)
    
    def compute_loss(self, pred, target, std=None, add_sin_cos=False):
        """
        Compute MSE loss between predictions and targets
        """
        if pred.dim() == 4:  # Multiple samples case
            pred = pred.mean(dim=0)
                
        if add_sin_cos:
            std = std[:-2]
        
        if std is not None:
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)