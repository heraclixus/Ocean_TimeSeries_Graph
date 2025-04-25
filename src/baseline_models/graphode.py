import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from baseline_models.utils import PeriodicActivation
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

class GraphODEFunc_GNODE(nn.Module):
    """ODE function using GNN"""
    def __init__(self, node_features, hidden_dim, use_periodic_activation=False, graph_encoder="gcn"):
        super().__init__()
        self.use_periodic_activation = use_periodic_activation
        self.graph_encoder = graph_encoder

        self.periodic_activation = PeriodicActivation()
        self.tanh_activation = nn.Tanh()
        
        # Initialize graph layers based on encoder type
        self.graph_layers = nn.ModuleList()
        if graph_encoder == "gcn":
            self.graph_layers.extend([
                GCNConv(node_features, 64),
                GCNConv(64, hidden_dim),   
                GCNConv(hidden_dim, hidden_dim),
                GCNConv(hidden_dim, 64),
                GCNConv(64, node_features)
            ])
        elif graph_encoder == "gat":
            # GAT with fewer attention heads and smaller dimensions to save memory
            self.graph_layers.extend([
                GATConv(node_features, 32, heads=2, dropout=0.2),  # Reduced heads and dimension
                GATConv(32 * 2, hidden_dim // 2, heads=2, dropout=0.2),   
                GATConv(hidden_dim // 2, hidden_dim // 2, heads=2, dropout=0.2),
                GATConv(hidden_dim // 2, 32, heads=2, dropout=0.2),
                GATConv(32, node_features, heads=2, dropout=0.2)
            ])
        
    def forward(self, t, x, edge_index, batch=None):
        """
        Args:
            t (torch.Tensor): Current time point
            x (torch.Tensor): Current state (B*N, F)
            edge_index (torch.Tensor): Graph connectivity (2, E)
            batch (torch.Tensor, optional): Batch assignments for each node
        """
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, edge_index)
            if i < len(self.graph_layers) - 1:
                if self.use_periodic_activation:
                    x = self.periodic_activation(x)
                else:
                    x = self.tanh_activation(x)
        return x

class GraphNeuralODE(nn.Module):
    def __init__(self, node_features=1, hidden_dim=64, forecast_horizon=10, 
                 use_periodic_activation=False, graph_encoder="gcn"):
        """
        Graph Neural ODE for time series forecasting
        
        Args:
            node_features (int): Number of features per node (default=1 for scalar time series)
            hidden_dim (int): Size of hidden layers
            forecast_horizon (int): Number of future time steps to forecast
            use_periodic_activation (bool): Whether to use periodic activation
            graph_encoder (str): Type of graph encoder to use ("gcn" or "gat")
        """
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.use_periodic_activation = use_periodic_activation
        self.graph_encoder = graph_encoder
        
        # ODE function with GNN
        self.ode_func = GraphODEFunc_GNODE(node_features, hidden_dim, 
                                          self.use_periodic_activation, 
                                          self.graph_encoder)
    
    def forward(self, x, edge_index=None):
        """
        Optimized forward pass that processes batches efficiently
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, num_nodes, timesteps)
            edge_index (torch.Tensor): Graph connectivity of shape (2, num_edges)
            
        Returns:
            torch.Tensor: Forecasted values of shape (batch_size, num_nodes, forecast_horizon)
        """
        batch_size, num_nodes, timesteps = x.shape
        device = x.device
        
        # Take the last timestep as initial condition
        x_init = x[:, :, -1].reshape(batch_size * num_nodes, 1)  # (B*N, 1)
        
        # Handle batch indexing for batch graphs
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        
        # If edge_index is not provided, create one (but don't create entire graph for each batch item)
        if edge_index is None:
            # Create a base edge_index for a single graph
            base_edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Exclude self-loops
                        base_edges.append([i, j])
            base_edge_index = torch.tensor(base_edges, device=device).t()
            
            # Repeat edge_index for each batch with proper offsets
            edge_index = []
            for b in range(batch_size):
                offset = b * num_nodes
                batch_edge_index = base_edge_index.clone()
                batch_edge_index[0] += offset
                batch_edge_index[1] += offset
                edge_index.append(batch_edge_index)
            edge_index = torch.cat(edge_index, dim=1)
        
        # Create time points for forecasting
        t = torch.linspace(0, self.forecast_horizon, self.forecast_horizon, device=device)
        
        # Define the ODE function wrapper to include edge_index and batch
        def ode_func(t, state):
            return self.ode_func(t, state, edge_index, batch_idx)
        
        # Choose a more memory-efficient solver for GAT
        method = 'euler' if self.graph_encoder == 'gat' else 'rk4'
        rtol = 1e-2 if self.graph_encoder == 'gat' else 1e-3
        atol = 1e-2 if self.graph_encoder == 'gat' else 1e-3
        
        # Solve ODE to get forecasts
        predictions = odeint(
            ode_func,
            x_init,
            t,
            method=method,
            rtol=rtol,
            atol=atol,
            options={'step_size': 1.0 if self.graph_encoder == 'gat' else None}
        )  # Shape: (T, B*N, 1)
        
        # Reshape predictions to (B, N, H)
        predictions = predictions.view(self.forecast_horizon, batch_size, num_nodes, self.node_features)
        predictions = predictions.permute(1, 2, 0, 3).squeeze(-1)  # (B, N, H)
        
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
        if std is not None:
            if add_sin_cos:
                std = std[:-2]
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)  # Shape: (1, N, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)

class GraphODEFunc(nn.Module):
    """ODE function for graph evolution"""
    def __init__(self, hidden_dim, edge_index, use_periodic_activation, graph_encoder="gcn"):
        super(GraphODEFunc, self).__init__()
        self.edge_index = edge_index
        self.graph_encoder = graph_encoder
        
        # Graph layers for state evolution
        self.graph_layers = nn.ModuleList()
        if graph_encoder == "gcn":
            self.graph_layers.extend([
                GCNConv(hidden_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim)
            ])
        elif graph_encoder == "gat":
            self.graph_layers.extend([
                GATConv(hidden_dim, hidden_dim, heads=2, dropout=0.3),  # Single head attention
                GATConv(hidden_dim, hidden_dim, heads=2, dropout=0.3)   # Single head attention
            ])
        
        self.use_periodic_activation = use_periodic_activation
        self.periodic_activation = PeriodicActivation()

    def forward(self, t, h):
        """
        Compute the derivative at current time point
        
        Args:
            t (torch.Tensor): Current time point
            h (torch.Tensor): Current hidden state (batch, nodes, hidden)
            
        Returns:
            torch.Tensor: Derivative of hidden state
        """
        # Ensure edge_index is on the same device as h
        if self.edge_index.device != h.device:
            self.edge_index = self.edge_index.to(h.device)
            
        # Process through graph layers
        for gcn in self.graph_layers:
            h = gcn(h, self.edge_index)
            if self.use_periodic_activation:
                h = self.periodic_activation(h)
            else:
                h = F.tanh(h)
    
        return h

class NeuralGDEForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_series_length, forecast_length, 
                 num_nodes, use_periodic_activation, graph_encoder="gcn"):
        """
        Neural GDE with encoder-decoder structure for forecasting
        
        Args:
            input_dim (int): Number of input features per node
            hidden_dim (int): Size of hidden dimensions
            time_series_length (int): Length of input time series
            forecast_length (int): Number of steps to forecast
            num_nodes (int): Number of nodes in the graph
            use_periodic_activation (bool): Whether to use periodic activation function
            graph_encoder (str): Type of graph encoder to use ("gcn" or "gat")
        """
        super(NeuralGDEForecaster, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_series_length = time_series_length
        self.forecast_length = forecast_length
        self.num_nodes = num_nodes
        self.use_periodic_activation = use_periodic_activation
        self.graph_encoder = graph_encoder

        # Graph layers for spatial relationships - input is 1 since we have scalar time series
        self.spatial_gcn = nn.ModuleList()
        if graph_encoder == "gcn":
            self.spatial_gcn.extend([
                GCNConv(1, 64),  # Changed from input_dim to 1
                GCNConv(64, hidden_dim),
                GCNConv(hidden_dim, hidden_dim)
            ])
        elif graph_encoder == "gat":
            self.spatial_gcn.extend([
                GATConv(1, 32, heads=2, dropout=0.3),  # Single head, reduced dimension
                GATConv(32, 64, heads=2, dropout=0.3),
                GATConv(64, hidden_dim, heads=2, dropout=0.3)
            ])
        
        # Temporal attention
        if self.use_periodic_activation:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                PeriodicActivation(), 
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # GRU for temporal dependencies
        self.encoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, 1)  # Changed from input_dim to 1

    def forward(self, x, edge_index=None):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, time)
            edge_index (torch.Tensor): Graph connectivity of shape (2, num_edges)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_nodes, horizon)
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device
        
        # Process each time step with GCN independently
        h_seq = []
        for t in range(self.time_series_length):
            # Extract features for current timestep (B, N, 1)
            x_t = x[:, :, t].reshape(batch_size * num_nodes, 1)
            
            # Process through GCN layers
            h = x_t
            
            # Create batch indexing for each node
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
            
            # If no edge_index provided, create a simple one for this batch
            if edge_index is None:
                # Create edges for a single graph first
                base_edges = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:  # Exclude self-loops
                            base_edges.append([i, j])
                base_edge_index = torch.tensor(base_edges, device=device).t()
                
                # Create batched edge_index with proper offsets
                edge_index_batched = []
                for b in range(batch_size):
                    offset = b * num_nodes
                    batch_edge = base_edge_index.clone()
                    batch_edge[0] += offset
                    batch_edge[1] += offset
                    edge_index_batched.append(batch_edge)
                edge_index_batched = torch.cat(edge_index_batched, dim=1)
            else:
                edge_index_batched = edge_index
            
            # Process through GCN with batched graph
            for gcn in self.spatial_gcn:
                h = gcn(h, edge_index_batched)
                h = F.relu(h)
            
            # Reshape to (batch_size, num_nodes, hidden_dim)
            h = h.view(batch_size, num_nodes, -1)
            h_seq.append(h)
        
        # Stack sequence along time dimension (B, T, N, H)
        h_seq = torch.stack(h_seq, dim=1)
        
        # Apply temporal attention
        attention = self.temporal_attention(h_seq)
        attention_weights = F.softmax(attention, dim=1)
        
        # Weighted combination of features (B, N, H)
        node_features = torch.sum(h_seq * attention_weights, dim=1)
        
        # Process through GRU for each node independently
        node_features_flat = node_features.view(batch_size * num_nodes, -1)
        _, hidden = self.encoder_gru(node_features_flat.unsqueeze(1))
        hidden = hidden.squeeze(0)  # (B*N, H)
        
        # Define GDE function
        ode_func = GraphODEFunc(
            hidden_dim=self.hidden_dim,
            edge_index=edge_index_batched,
            use_periodic_activation=self.use_periodic_activation,
            graph_encoder=self.graph_encoder
        ).to(device)
        
        # Create time points for forecasting
        t = torch.linspace(0, self.forecast_length, self.forecast_length, device=device)
        
        # Use more memory-efficient solver for GAT
        method = 'euler' if self.graph_encoder == 'gat' else 'rk4'
        rtol = 1e-2 if self.graph_encoder == 'gat' else 1e-3
        atol = 1e-2 if self.graph_encoder == 'gat' else 1e-3
        
        # Solve GDE
        evolved_hidden = odeint(
            ode_func,
            hidden,
            t,
            method=method,
            rtol=rtol,
            atol=atol,
            options={'step_size': 1.0 if self.graph_encoder == 'gat' else None}
        )  # (T, B*N, H)
        
        # Reshape evolved states to (T, B, N, H)
        evolved_hidden = evolved_hidden.view(
            self.forecast_length, 
            batch_size, 
            num_nodes, 
            -1
        )
        
        # Project to output
        evolved_hidden = evolved_hidden.permute(1, 2, 0, 3)  # (B, N, T, H)
        predictions = self.output_layer(evolved_hidden)  # (B, N, T, 1)
        
        return predictions.squeeze(-1)  # (B, N, T)

    def compute_loss(self, pred, target, std=None, add_sin_cos=False):
        """
        Compute MSE loss between predictions and targets
        """
        if std is not None:
            if add_sin_cos:
                std = std[:-2]
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)
