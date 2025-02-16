import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
import numpy as np

class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.b = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): Node features (B, N, F)
            adj (torch.Tensor): Adjacency matrix (B, N, N)
        """
        # Normalize adjacency matrix
        deg = torch.sum(adj, dim=-1, keepdim=True)  # (B, N, 1)
        deg = torch.clamp(deg, min=1.0)  # Avoid division by zero
        norm_adj = adj / deg
        
        # Graph convolution
        x = torch.bmm(norm_adj, x)  # (B, N, F)
        x = self.W(x) + self.b
        return x

class GraphODEFunc(nn.Module):
    """ODE function using GNN"""
    def __init__(self, node_features, hidden_dim):
        super().__init__()
        
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(node_features, hidden_dim),
            GraphConvLayer(hidden_dim, hidden_dim),
            GraphConvLayer(hidden_dim, node_features)
        ])
        
    def forward(self, t, x, adj):
        """
        Args:
            t (torch.Tensor): Current time point
            x (torch.Tensor): Current state (B, N, F)
            adj (torch.Tensor): Adjacency matrix (B, N, N)
        """
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, adj)
            if i < len(self.graph_layers) - 1:
                x = F.tanh(x)
        return x

class GraphNeuralODE(nn.Module):
    def __init__(self, node_features, hidden_dim=64, forecast_horizon=10):
        """
        Graph Neural ODE for time series forecasting
        
        Args:
            node_features (int): Number of features per node
            hidden_dim (int): Size of hidden layers
            forecast_horizon (int): Number of future time steps to forecast
        """
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # ODE function with GNN
        self.ode_func = GraphODEFunc(node_features, hidden_dim)
        
        # Optional: learnable adjacency matrix
        self.learn_adj = True
        if self.learn_adj:
            self.adj_weights = nn.Parameter(torch.randn(1, node_features, node_features))
        
    def get_adjacency(self, x):
        """
        Get or compute adjacency matrix
        
        Args:
            x (torch.Tensor): Input tensor (B, N, F)
            
        Returns:
            torch.Tensor: Adjacency matrix (B, N, N)
        """
        batch_size = x.shape[0]
        if self.learn_adj:
            # Use learnable adjacency matrix
            adj = torch.sigmoid(self.adj_weights)  # (1, N, N)
            adj = adj.expand(batch_size, -1, -1)  # (B, N, N)
        else:
            # Compute adjacency based on feature similarity
            x_norm = F.normalize(x, p=2, dim=-1)
            adj = torch.bmm(x_norm, x_norm.transpose(1, 2))
            adj = F.softmax(adj / np.sqrt(self.node_features), dim=-1)
        
        return adj
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input time series of shape (B, N, T, F)
                B: batch size
                N: number of nodes
                T: time steps
                F: features per node
            
        Returns:
            torch.Tensor: Forecasted values of shape (B, N, H, F)
        """
        batch_size, num_nodes, time_steps, features = x.shape
        device = x.device
        
        # Get the last state from the input sequence
        initial_state = x[:, :, -1, :]  # Shape: (B, N, F)
        
        # Get or compute adjacency matrix
        adj = self.get_adjacency(initial_state)
        
        # Create time points for forecasting
        t = torch.linspace(0, self.forecast_horizon, self.forecast_horizon).to(device)
        
        # Define the ODE function wrapper to include adjacency
        def ode_func(t, state):
            return self.ode_func(t, state, adj)
        
        # Solve ODE to get forecasts
        predictions = odeint(
            ode_func,
            initial_state,
            t,
            method='rk4',
            rtol=1e-3,
            atol=1e-3
        )
        
        # Reshape predictions to (B, N, H, F)
        predictions = predictions.permute(1, 2, 0, 3)
        
        return predictions
    
    def compute_loss(self, pred, target, std=None, add_sin_cos=False):
        """
        Compute MSE loss between predictions and targets
        
        Args:
            pred (torch.Tensor): Predicted values of shape (B, N, H, F)
            target (torch.Tensor): Target values of shape (B, N, H, F)
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
            weights = 1.0 / (std ** 2)
            weights = weights.to(pred.device)
            weights = weights.view(1, -1, 1, 1)  # Shape: (1, N, 1, 1) for GraphODE
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)