import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class KoopmanEncoder(nn.Module):
    """Neural network for learning Koopman observables"""
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [64, 128]):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N)
        Returns:
            Encoded observables of shape (B, latent_dim)
        """
        return self.encoder(x)

class KoopmanDecoder(nn.Module):
    """Neural network for reconstructing state from observables"""
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (B, latent_dim)
        Returns:
            Reconstructed state of shape (B, output_dim)
        """
        return self.decoder(z)

class KoopmanOperator(nn.Module):
    """Linear Koopman operator"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (B, latent_dim)
        Returns:
            Evolved latent state of shape (B, latent_dim)
        """
        return self.K(z)

class DeepKoopman(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [64, 128],
        lambda_id: float = 1.0,
        lambda_pred: float = 1.0,
        lambda_rec: float = 1.0
    ):
        """
        Deep Koopman Operator for time series forecasting
        
        Args:
            input_dim: Number of features/dimensions
            latent_dim: Dimension of Koopman observables
            hidden_dims: Hidden dimensions for encoder/decoder
            lambda_id: Weight for identity loss
            lambda_pred: Weight for prediction loss
            lambda_rec: Weight for reconstruction loss
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_id = lambda_id
        self.lambda_pred = lambda_pred
        self.lambda_rec = lambda_rec
        
        # Networks
        self.encoder = KoopmanEncoder(input_dim, latent_dim, hidden_dims)
        self.decoder = KoopmanDecoder(latent_dim, input_dim, hidden_dims[::-1])
        self.koopman = KoopmanOperator(latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, N, T)
        Returns:
            Tuple of (reconstructions, predictions, latent_states)
        """
        batch_size, n_dims, seq_len = x.shape
        
        # Encode all states to observables
        x_flat = x.transpose(1, 2).reshape(-1, n_dims)  # (B*T, N)
        z = self.encoder(x_flat)  # (B*T, latent_dim)
        z = z.view(batch_size, seq_len, self.latent_dim)  # (B, T, latent_dim)
        
        # Reconstruct states
        x_rec = self.decoder(z.reshape(-1, self.latent_dim))  # (B*T, N)
        x_rec = x_rec.view(batch_size, seq_len, n_dims)  # (B, T, N)
        
        # Evolve system using Koopman operator
        z_next = self.koopman(z[:, :-1])  # (B, T-1, latent_dim)
        
        return x_rec.transpose(1, 2), z_next, z
    
    def predict(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Generate forecasts
        
        Args:
            x: Input tensor of shape (B, N, T)
            horizon: Number of steps to forecast
        Returns:
            Forecasts of shape (B, N, H)
        """
        batch_size = x.shape[0]
        
        # Get last state
        z_last = self.encoder(x[:, :, -1])  # (B, latent_dim)
        
        # Generate forecasts in latent space
        predictions = []
        z_current = z_last
        
        for _ in range(horizon):
            z_next = self.koopman(z_current)
            pred = self.decoder(z_next)
            predictions.append(pred)
            z_current = z_next
        
        predictions = torch.stack(predictions, dim=2)  # (B, N, H)
        return predictions
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_rec: torch.Tensor,
        z_next: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total loss
        
        Args:
            x: Input tensor (B, N, T)
            x_rec: Reconstructed states (B, N, T)
            z_next: Predicted next latent states (B, T-1, latent_dim)
            z: Current latent states (B, T, latent_dim)
        """
        # Reconstruction loss
        loss_rec = F.mse_loss(x_rec, x)
        
        # Prediction loss (in latent space)
        loss_pred = F.mse_loss(z_next, z[:, 1:])
        
        # Identity loss (linear dynamics in latent space)
        loss_id = torch.mean(torch.abs(self.koopman.K.weight - torch.eye(self.latent_dim).to(x.device)))
        
        # Total loss
        total_loss = (
            self.lambda_rec * loss_rec +
            self.lambda_pred * loss_pred +
            self.lambda_id * loss_id
        )
        
        return total_loss

# Dictionary-based Koopman (alternative approach)
class DictionaryKoopman:
    def __init__(self, input_dim: int, n_observables: int):
        """
        Dictionary-based Koopman approach
        
        Args:
            input_dim: Number of features/dimensions
            n_observables: Number of dictionary elements
        """
        self.input_dim = input_dim
        self.n_observables = n_observables
        self.K = None
    
    def dictionary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dictionary of observables
        
        Args:
            x: Input tensor (B, N)
        Returns:
            Observables (B, n_observables)
        """
        # Example dictionary functions
        observables = [
            x,  # Identity
            x**2,  # Quadratic
            torch.sin(x),  # Sine
            torch.cos(x),  # Cosine
            torch.exp(x)  # Exponential
        ]
        return torch.cat(observables, dim=1)
    
    def fit(self, x: torch.Tensor):
        """
        Fit Koopman operator using least squares
        
        Args:
            x: Input tensor (B, N, T)
        """
        # Convert to observables
        psi = self.dictionary(x.transpose(1, 2).reshape(-1, self.input_dim))
        psi = psi.view(-1, x.shape[2], self.n_observables)
        
        # Solve for K using least squares
        X = psi[:, :-1].reshape(-1, self.n_observables)
        Y = psi[:, 1:].reshape(-1, self.n_observables)
        
        self.K = torch.linalg.lstsq(X, Y).solution
    
    def predict(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Generate forecasts
        
        Args:
            x: Input tensor (B, N, T)
            horizon: Forecast horizon
        Returns:
            Forecasts (B, N, H)
        """
        # Get last state in observable space
        psi_last = self.dictionary(x[:, :, -1])
        
        # Generate forecasts
        predictions = []
        psi_current = psi_last
        
        for _ in range(horizon):
            psi_next = psi_current @ self.K.T
            # Extract original state variables (first input_dim components)
            pred = psi_next[:, :self.input_dim]
            predictions.append(pred)
            psi_current = psi_next
        
        return torch.stack(predictions, dim=2)