import torch
import torch.nn as nn
import torchcde

class CubicSplineInterpolation(nn.Module):
    """Cubic spline interpolation for creating a continuous path from discrete observations."""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
    
    def forward(self, x, t=None):
        """
        Args:
            x (torch.Tensor): Input of shape (B, N, T)
            t (torch.Tensor, optional): Timestamps
        Returns:
            torchcde.CubicSpline: Interpolated path
        """
        batch_size, channels, length = x.shape
        
        # If no timestamps provided, use equally spaced points
        if t is None:
            t = torch.linspace(0, length - 1, length).to(x.device)
        
        # Reshape for torchcde
        x = x.permute(0, 2, 1)  # (B, T, N)
        
        # Add time dimension to X if not present
        coeffs = torchcde.natural_cubic_spline_coeffs(x, t)
        interpolation = torchcde.CubicSpline(coeffs, t)
        
        return interpolation

class CDEFunc(nn.Module):
    """Vector field for the CDE."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Neural network for the vector field
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * input_dim)
        )
    
    def forward(self, t, h):
        """
        Args:
            t (torch.Tensor): Current time
            h (torch.Tensor): Current hidden state (B, hidden_dim)
        Returns:
            torch.Tensor: Matrix of shape (B, hidden_dim, input_dim)
        """
        # Compute vector field
        vector_field = self.net(h)
        vector_field = vector_field.view(-1, self.hidden_dim, self.input_dim)
        return vector_field

class TimeSeriesCDE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, forecast_horizon=10):
        """
        Neural CDE for time series forecasting
        
        Args:
            input_dim (int): Number of features/dimensions in the time series (N)
            hidden_dim (int): Size of hidden layers
            forecast_horizon (int): Number of future time steps to forecast (H)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Initial hidden state projection
        self.initial_proj = nn.Linear(input_dim, hidden_dim)
        
        # CDE function (vector field)
        self.func = CDEFunc(input_dim, hidden_dim)
        
        # Path interpolation
        self.interpolation = CubicSplineInterpolation(input_dim)
        
        # Decoder for forecasting
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
    
    def forward(self, x, t=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input time series of shape (B, N, T)
            t (torch.Tensor, optional): Timestamps for the input series
            
        Returns:
            torch.Tensor: Forecasted values of shape (B, N, H)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Create interpolated path
        path = self.interpolation(x, t)
        
        # Get initial hidden state
        h0 = self.initial_proj(x[:, :, 0])
        
        # Solve the CDE
        solution = torchcde.cdeint(
            func=self.func,
            z=path,
            t=path.interval,
            h0=h0,
            method="rk4",
            rtol=1e-3,
            atol=1e-3,
        )
        
        # Get final hidden state
        final_h = solution[-1]
        
        # Generate forecasts using decoder
        forecasts = []
        current_h = final_h
        
        # Autoregressive generation of forecasts
        for _ in range(self.forecast_horizon):
            next_value = self.decoder(current_h)
            forecasts.append(next_value)
            
            # Update hidden state using the CDE function
            # This step helps maintain temporal dependencies
            vector_field = self.func(None, current_h)
            current_h = current_h + torch.bmm(
                vector_field, 
                next_value.unsqueeze(-1)
            ).squeeze(-1)
        
        # Stack forecasts
        forecasts = torch.stack(forecasts, dim=-1)  # (B, N, H)
        
        return forecasts
    
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



# Define the Neural CDE Function
class CDEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(CDEFunc, self).__init__()
        self.hidden_dim = hidden_dim

        # CDE function (f) that maps hidden state to derivative
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, z):
        return self.f_net(z)

# Define the full model with Encoder-CDE-Decoder structure
class NeuralCDEForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_series_length, forecast_length):
        super(NeuralCDEForecaster, self).__init__()
        self.input_dim = input_dim
        self.time_series_length = time_series_length
        self.forecast_length = forecast_length
        self.hidden_dim = hidden_dim

        # Encoder (GRU)
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # CDE Function
        self.cde_func = CDEFunc(hidden_dim)

        # Decoder (GRU)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, num_samples=10):
        batch_size = x.shape[0]

        # Encode the input sequence
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, time_series_length, input_dim)
        _, hidden = self.encoder_gru(x)  # Get last hidden state (shape: [1, batch_size, hidden_dim])

        # Create CDE interpolation
        t = torch.linspace(0, self.time_series_length - 1, self.time_series_length).to(x.device)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)

        # Define time points for CDE evolution
        t_eval = torch.linspace(0, self.forecast_length, self.forecast_length).to(x.device)

        # Solve CDE
        evolved_hidden = torchcde.cdeint(X, self.cde_func, hidden.squeeze(0), t_eval, method='rk4')
        evolved_hidden = evolved_hidden.permute(1, 0, 2)  # Shape: (batch_size, forecast_length, hidden_dim)

        # Decoder GRU expects (batch_size, forecast_length, hidden_dim), initialize with last evolved hidden state
        decoder_hidden = evolved_hidden[:, -1, :].unsqueeze(0)  # Shape: (1, batch_size, hidden_dim)

        # Prepare decoder input (zero tensor to start)
        decoder_input = torch.zeros(batch_size, self.forecast_length, self.hidden_dim).to(x.device)

        # Decoder GRU processes evolved hidden states to generate output sequence
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