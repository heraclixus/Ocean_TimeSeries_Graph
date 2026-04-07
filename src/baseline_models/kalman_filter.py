import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class KalmanFilter(nn.Module):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        process_variance: float = 1e-4,
        measurement_variance: float = 1e-2
    ):
        """
        Linear Kalman Filter for time series forecasting
        
        Args:
            state_dim: Dimension of state vector
            observation_dim: Dimension of observation vector
            process_variance: Process noise variance
            measurement_variance: Measurement noise variance
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        
        # State transition matrix (F)
        self.F = nn.Parameter(torch.eye(state_dim), requires_grad=True)
        
        # Observation matrix (H)
        self.H = nn.Parameter(torch.randn(observation_dim, state_dim), requires_grad=True)
        
        # Process noise covariance (Q)
        self.Q = process_variance * torch.eye(state_dim)
        
        # Measurement noise covariance (R)
        self.R = measurement_variance * torch.eye(observation_dim)
        
        # Initial state covariance (P)
        self.P = torch.eye(state_dim)
    
    def predict(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction step
        
        Args:
            state: Current state estimate (B, state_dim)
            covariance: Current state covariance (B, state_dim, state_dim)
            
        Returns:
            Predicted state and covariance
        """
        # Predict state
        predicted_state = torch.bmm(self.F.expand(state.size(0), -1, -1), state.unsqueeze(-1))
        predicted_state = predicted_state.squeeze(-1)
        
        # Predict covariance
        predicted_covariance = torch.bmm(
            torch.bmm(self.F.expand(state.size(0), -1, -1), covariance),
            self.F.transpose(0, 1).expand(state.size(0), -1, -1)
        ) + self.Q.expand(state.size(0), -1, -1)
        
        return predicted_state, predicted_covariance
    
    def update(
        self,
        predicted_state: torch.Tensor,
        predicted_covariance: torch.Tensor,
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update step
        
        Args:
            predicted_state: Predicted state (B, state_dim)
            predicted_covariance: Predicted covariance (B, state_dim, state_dim)
            measurement: Observation (B, observation_dim)
            
        Returns:
            Updated state and covariance
        """
        # Innovation (measurement residual)
        predicted_measurement = torch.bmm(
            self.H.expand(predicted_state.size(0), -1, -1),
            predicted_state.unsqueeze(-1)
        ).squeeze(-1)
        innovation = measurement - predicted_measurement
        
        # Innovation covariance
        S = torch.bmm(
            torch.bmm(self.H.expand(predicted_state.size(0), -1, -1), predicted_covariance),
            self.H.transpose(0, 1).expand(predicted_state.size(0), -1, -1)
        ) + self.R.expand(predicted_state.size(0), -1, -1)
        
        # Kalman gain
        K = torch.bmm(
            torch.bmm(predicted_covariance, self.H.transpose(0, 1).expand(predicted_state.size(0), -1, -1)),
            torch.inverse(S)
        )
        
        # Update state
        updated_state = predicted_state + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        # Update covariance
        I = torch.eye(self.state_dim).expand(predicted_state.size(0), -1, -1)
        updated_covariance = torch.bmm(
            I - torch.bmm(K, self.H.expand(predicted_state.size(0), -1, -1)),
            predicted_covariance
        )
        
        return updated_state, updated_covariance

class ExtendedKalmanFilter(KalmanFilter):
    """Extended Kalman Filter with neural network state transition and observation models"""
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: int = 64,
        process_variance: float = 1e-4,
        measurement_variance: float = 1e-2
    ):
        super().__init__(state_dim, observation_dim, process_variance, measurement_variance)
        
        # Replace linear matrices with neural networks
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.observation_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, observation_dim)
        )
    
    def predict(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction step with nonlinear state transition"""
        # Predict state using neural network
        predicted_state = self.state_transition(state)
        
        # Compute Jacobian of state transition
        jacobian = torch.autograd.functional.jacobian(
            self.state_transition,
            state,
            create_graph=True
        )
        
        # Predict covariance using Jacobian
        predicted_covariance = torch.bmm(
            torch.bmm(jacobian, covariance),
            jacobian.transpose(1, 2)
        ) + self.Q.expand(state.size(0), -1, -1)
        
        return predicted_state, predicted_covariance
    
    def update(
        self,
        predicted_state: torch.Tensor,
        predicted_covariance: torch.Tensor,
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update step with nonlinear observation model"""
        # Predict measurement using neural network
        predicted_measurement = self.observation_model(predicted_state)
        
        # Compute Jacobian of observation model
        jacobian = torch.autograd.functional.jacobian(
            self.observation_model,
            predicted_state,
            create_graph=True
        )
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # Innovation covariance
        S = torch.bmm(
            torch.bmm(jacobian, predicted_covariance),
            jacobian.transpose(1, 2)
        ) + self.R.expand(predicted_state.size(0), -1, -1)
        
        # Kalman gain
        K = torch.bmm(
            torch.bmm(predicted_covariance, jacobian.transpose(1, 2)),
            torch.inverse(S)
        )
        
        # Update state
        updated_state = predicted_state + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        # Update covariance
        I = torch.eye(self.state_dim).expand(predicted_state.size(0), -1, -1)
        updated_covariance = torch.bmm(
            I - torch.bmm(K, jacobian),
            predicted_covariance
        )
        
        return updated_state, updated_covariance

class KalmanForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: Optional[int] = None,
        filter_type: str = 'linear',
        **kwargs
    ):
        """
        Kalman Filter based forecaster
        
        Args:
            input_dim: Number of input features
            state_dim: Dimension of state vector (default: 2 * input_dim)
            filter_type: 'linear' or 'extended'
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim or 2 * input_dim
        
        # Initialize Kalman filter
        if filter_type == 'linear':
            self.kf = KalmanFilter(self.state_dim, input_dim, **kwargs)
        else:
            self.kf = ExtendedKalmanFilter(self.state_dim, input_dim, **kwargs)
    
    def forward(
        self,
        x: torch.Tensor,
        horizon: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate forecasts
        
        Args:
            x: Input tensor (B, N, T)
            horizon: Number of steps to forecast
            
        Returns:
            Tuple of (forecasts, forecast_covs)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize state estimate
        state = torch.zeros(batch_size, self.state_dim).to(device)
        covariance = torch.eye(self.state_dim).expand(batch_size, -1, -1).to(device)
        
        # Process historical data
        for t in range(x.shape[2]):
            # Predict
            state, covariance = self.kf.predict(state, covariance)
            # Update
            state, covariance = self.kf.update(state, covariance, x[:, :, t])
        
        # Generate forecasts
        forecasts = []
        forecast_covs = []
        
        for _ in range(horizon):
            # Predict next state
            state, covariance = self.kf.predict(state, covariance)
            
            # Get observation estimate
            if isinstance(self.kf, ExtendedKalmanFilter):
                forecast = self.kf.observation_model(state)
            else:
                forecast = torch.bmm(
                    self.kf.H.expand(batch_size, -1, -1),
                    state.unsqueeze(-1)
                ).squeeze(-1)
            
            forecasts.append(forecast)
            forecast_covs.append(covariance)
        
        forecasts = torch.stack(forecasts, dim=2)  # (B, N, H)
        forecast_covs = torch.stack(forecast_covs, dim=1)  # (B, H, state_dim, state_dim)
        
        return forecasts, forecast_covs
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_cov: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        add_sin_cos: bool = False
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets
        """
        if add_sin_cos:
            # Exclude last two dimensions (sin/cos)
            pred = pred[:, :-2]
            target = target[:, :-2]
            if std is not None:
                std = std[:-2]
            if pred_cov is not None:
                pred_cov = pred_cov[:, :-2, :-2]
        
        if pred_cov is not None:
            # Use negative log likelihood if covariance is available
            error = (pred - target).unsqueeze(-1)
            nll = torch.bmm(
                torch.bmm(error.transpose(2, 3), torch.inverse(pred_cov)),
                error
            ).squeeze()
            nll = nll + torch.log(torch.det(pred_cov))
            return torch.mean(nll)
        elif std is not None:
            weights = 1.0 / (std ** 2)
            weights = weights.to(pred.device)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)