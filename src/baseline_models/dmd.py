import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd, pinv

class DMDForecast:
    def __init__(self, input_dim, rank=None, mode='exact', svd_rank=0.99):
        """
        Dynamic Mode Decomposition for time series forecasting
        
        Args:
            input_dim (int): Number of features/dimensions
            rank (int, optional): Truncation rank for DMD modes
            mode (str): 'exact' or 'standard' DMD
            svd_rank (float): Cumulative energy threshold for rank truncation
        """
        self.input_dim = input_dim
        self.rank = rank
        self.mode = mode
        self.svd_rank = svd_rank
        
        # DMD attributes
        self.eigenvalues = None
        self.modes = None
        self.amplitudes = None
        self.Atilde = None
        self.dt = 1.0  # Time step
        
    def _compute_rank(self, singular_values):
        """
        Compute rank based on cumulative energy threshold
        """
        if self.rank is not None:
            return min(self.rank, len(singular_values))
            
        cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        rank = np.searchsorted(cumulative_energy, self.svd_rank) + 1
        return rank
    
    def fit(self, X):
        """
        Fit DMD model
        
        Args:
            X (torch.Tensor): Input tensor of shape (B, N, T)
                B: batch size
                N: number of features
                T: time steps
        """
        # Convert to numpy for computation
        X = X.cpu().numpy()
        
        # Handle batched data by averaging
        if X.ndim == 3:
            X = X.mean(axis=0)  # Average across batch dimension
        
        # Split data into snapshot matrices
        X1 = X[:, :-1]  # (N, T-1)
        X2 = X[:, 1:]   # (N, T-1)
        
        # Compute SVD of X1
        U, Sigma, Vh = svd(X1, full_matrices=False)
        
        # Determine rank for truncation
        rank = self._compute_rank(Sigma)
        
        # Truncate
        U_r = U[:, :rank]
        Sigma_r = Sigma[:rank]
        Vh_r = Vh[:rank, :]
        
        if self.mode == 'standard':
            # Standard DMD
            self.Atilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1/Sigma_r)
            self.modes = U_r
            
        elif self.mode == 'exact':
            # Exact DMD
            self.Atilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1/Sigma_r)
            eigenvalues, eigenvectors = np.linalg.eig(self.Atilde)
            
            # Compute DMD modes
            self.modes = X2 @ Vh_r.T @ np.diag(1/Sigma_r) @ eigenvectors
            self.eigenvalues = eigenvalues
            
            # Compute amplitudes
            b = np.linalg.lstsq(self.modes, X1[:, 0], rcond=None)[0]
            self.amplitudes = b
            
        return self
    
    def predict(self, x, horizon):
        """
        Generate forecasts
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
            horizon (int): Number of steps to forecast
            
        Returns:
            torch.Tensor: Forecasts of shape (B, N, H)
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Use last time point for prediction
        x_last = x[:, :, -1].cpu().numpy()  # (B, N)
        
        if self.mode == 'standard':
            # Standard DMD prediction
            predictions = []
            for step in range(horizon):
                if step == 0:
                    pred = x_last
                else:
                    pred = (self.modes @ self.Atilde @ np.linalg.pinv(self.modes) @ pred.T).T
                predictions.append(pred)
            
            predictions = np.stack(predictions, axis=-1)  # (B, N, H)
            
        elif self.mode == 'exact':
            # Exact DMD prediction
            predictions = np.zeros((batch_size, self.input_dim, horizon))
            
            for b in range(batch_size):
                # Compute initial amplitudes for this batch
                b_init = np.linalg.lstsq(self.modes, x_last[b], rcond=None)[0]
                
                # Time dynamics
                time_dynamics = np.zeros((len(self.eigenvalues), horizon), dtype=complex)
                for i, eig in enumerate(self.eigenvalues):
                    time_dynamics[i, :] = np.power(eig, np.arange(horizon))
                
                # Compute predictions
                predictions[b] = np.real(self.modes @ np.diag(b_init) @ time_dynamics)
        
        return torch.tensor(predictions, device=device, dtype=torch.float32)
    
    def compute_loss(self, pred, target):
        """
        Compute MSE loss between predictions and targets
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: MSE loss
        """
        return torch.mean((pred - target) ** 2)


class OnlineDMD(nn.Module):
    """Online/Mini-batch DMD implementation"""
    def __init__(self, input_dim, rank=None, forgetting_factor=0.98):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.forgetting_factor = forgetting_factor
        
        # Initialize online DMD matrices
        self.A = nn.Parameter(torch.eye(input_dim), requires_grad=False)
        self.P = nn.Parameter(torch.eye(input_dim), requires_grad=False)
    
    def update(self, x_prev, x_next):
        """
        Update DMD matrices with new data
        
        Args:
            x_prev (torch.Tensor): Previous state (B, N)
            x_next (torch.Tensor): Next state (B, N)
        """
        for i in range(x_prev.shape[0]):
            xt = x_prev[i:i+1].T  # (N, 1)
            yt = x_next[i:i+1].T  # (N, 1)
            
            # Update P matrix
            self.P.data = (1/self.forgetting_factor) * (
                self.P - (self.P @ xt @ xt.T @ self.P) / 
                (self.forgetting_factor + xt.T @ self.P @ xt)
            )
            
            # Update A matrix
            self.A.data = self.A + (yt - self.A @ xt) @ xt.T @ self.P
    
    def predict(self, x, horizon):
        """
        Generate forecasts using online DMD
        
        Args:
            x (torch.Tensor): Input tensor (B, N, T)
            horizon (int): Forecast horizon
            
        Returns:
            torch.Tensor: Forecasts (B, N, H)
        """
        batch_size = x.shape[0]
        predictions = []
        
        # Use last state for prediction
        current_state = x[:, :, -1]  # (B, N)
        
        for _ in range(horizon):
            next_state = current_state @ self.A.T
            predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(predictions, dim=-1)  # (B, N, H)