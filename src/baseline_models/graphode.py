import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from baseline_models.utils import PeriodicActivation

class GraphODEFunc_GNODE(nn.Module):
    """ODE function using GNN"""
    def __init__(self, node_features, hidden_dim, use_periodic_activation=False):
        super().__init__()
        self.use_periodic_activation = use_periodic_activation

        self.periodic_activation = PeriodicActivation()
        self.tanh_activation = nn.Tanh()
        self.graph_layers = nn.ModuleList([
            GCNConv(node_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, node_features)
        ])
        
    def forward(self, t, x, edge_index):
        """
        Args:
            t (torch.Tensor): Current time point
            x (torch.Tensor): Current state (B*N, F)
            edge_index (torch.Tensor): Graph connectivity (2, E)
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
    def __init__(self, node_features, hidden_dim=64, forecast_horizon=10, use_periodic_activation=False):
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
        self.use_periodic_activation = use_periodic_activation
        
        # ODE function with GNN
        self.ode_func = GraphODEFunc_GNODE(node_features, hidden_dim, self.use_periodic_activation)
        
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
    
    def create_fully_connected_edges(self, num_nodes, device):
        """Create edges for fully connected graph"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops as GCNConv adds them automatically
                    edges.append([i, j])
        return torch.tensor(edges, device=device).t()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input time series of shape (B, N, T)
                B: batch size
                N: number of nodes
                T: time steps
                F: features per node
            
        Returns:
            torch.Tensor: Forecasted values of shape (B, N, F)
        """
        x = x.unsqueeze(-1)
        batch_size, num_nodes, _, features = x.shape
        device = x.device
        # Create edge_index for fully connected graph
        edge_index = self.create_fully_connected_edges(num_nodes, device)
        
        # Get the last state from the input sequence
        initial_state = x[:, :, -1, :]  # Shape: (B, N, F)
        
        # Reshape for GCNConv: (B, N, F) -> (B*N, F)
        initial_state = initial_state.reshape(-1, features)
        
        # Create time points for forecasting
        t = torch.linspace(0, self.forecast_horizon, self.forecast_horizon).to(device)
        
        # Define the ODE function wrapper to include edge_index
        def ode_func(t, state):
            return self.ode_func(t, state, edge_index)
        
        # Solve ODE to get forecasts
        predictions = odeint(
            ode_func,
            initial_state,
            t,
            method='rk4',
            rtol=1e-3,
            atol=1e-3
        )  # Shape: (T, B*N, F)
        # Reshape predictions to (B, F, N, H)
        predictions = predictions.view(self.forecast_horizon, batch_size, num_nodes, features)
        predictions = predictions.permute(1, 3, 2, 0).squeeze(1)
        return predictions
    
    def compute_loss(self, pred, target, std=None, add_sin_cos=False):
        """
        Compute MSE loss between predictions and targets
        
        Args:
            pred (torch.Tensor): Predicted values of shape (B, F, N)
            target (torch.Tensor): Target values of shape (B, F, N)
            std (torch.Tensor, optional): Standard deviations for weighted MSE
            add_sin_cos (bool): Whether sinusoidal features are present
        """
        if std is not None:
            if add_sin_cos:
                std = std[:-2]
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)  # Shape: (1, N, 1) for GraphODE
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)

class GraphODEFunc(nn.Module):
    """ODE function for graph evolution"""
    def __init__(self, hidden_dim, edge_index, use_periodic_activation):
        super(GraphODEFunc, self).__init__()
        self.edge_index = edge_index
        
        # GCN layers for state evolution
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
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
            
        # Process through GCN layers
        for gcn in self.gcn_layers:
            h = gcn(h, self.edge_index)
            if self.use_periodic_activation:
                h = self.periodic_activation(h)
            else:
                h = F.tanh(h)
    
        return h

class NeuralGDEForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_series_length, forecast_length, 
                 num_nodes, use_periodic_activation):
        """
        Neural GDE with encoder-decoder structure for forecasting
        
        Args:
            input_dim (int): Number of input features per node
            hidden_dim (int): Size of hidden dimensions
            time_series_length (int): Length of input time series
            forecast_length (int): Number of steps to forecast
            num_nodes (int): Number of nodes in the graph
        """
        super(NeuralGDEForecaster, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_series_length = time_series_length
        self.forecast_length = forecast_length
        self.num_nodes = num_nodes
        self.use_periodic_activation = use_periodic_activation

        # GCN layers for spatial relationships - input is 1 since we have scalar time series
        self.spatial_gcn = nn.ModuleList([
            GCNConv(1, hidden_dim),  # Changed from input_dim to 1
            GCNConv(hidden_dim, hidden_dim)
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

    def create_spatiotemporal_graph(self, x):
        """
        Create spatial-temporal graph from input sequence
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, time)
            
        Returns:
            tuple: (node_features, edge_index)
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Create spatial edges (fully connected at each timestamp)
        spatial_edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    spatial_edges.append([i, j])
        spatial_edges = torch.tensor(spatial_edges, device=device).t()
        
        # Process features through GCN layers
        node_features = []
        for t in range(self.time_series_length):
            # Get features at time t: (batch_size, num_nodes)
            h = x[:, :, t]
            
            # Process each batch separately
            batch_features = []
            for b in range(batch_size):
                h_b = h[b].unsqueeze(-1)  # Shape: (num_nodes, 1)
                
                # Apply GCN layers
                for gcn in self.spatial_gcn:
                    h_b = gcn(h_b, spatial_edges)  # Input: (num_nodes, 1), Output: (num_nodes, hidden_dim)
                    h_b = F.relu(h_b)
                
                batch_features.append(h_b)
            
            # Stack batch dimension
            h = torch.stack(batch_features, dim=0)  # Shape: (batch_size, num_nodes, hidden_dim)
            node_features.append(h)
        
        # Stack temporal sequence
        node_features = torch.stack(node_features, dim=1)  # (batch_size, time, num_nodes, hidden_dim)
        
        # Apply temporal attention
        attention = self.temporal_attention(node_features)
        attention_weights = F.softmax(attention, dim=1)
        
        # Weighted combination of features
        node_features = torch.sum(node_features * attention_weights, dim=1)  # (batch_size, num_nodes, hidden_dim)
        
        return node_features, spatial_edges  # Return spatial_edges instead of full edge_index

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, time)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_nodes, horizon)
        """
        batch_size = x.shape[0]
        device = x.device        
        # Create spatiotemporal graph and get node features
        node_features, edge_index = self.create_spatiotemporal_graph(x)
        
        # Process temporal sequence with GRU
        node_features = node_features.reshape(batch_size * self.num_nodes, -1)  # Flatten for GRU
        _, hidden = self.encoder_gru(node_features.unsqueeze(1))
        hidden = hidden.view(batch_size, self.num_nodes, -1)
        
        # Create time points for GDE evolution
        t = torch.linspace(0, self.forecast_length, self.forecast_length).to(device)
        
        # Define GDE function
        ode_func = GraphODEFunc(
            hidden_dim=self.hidden_dim,
            edge_index=edge_index.to(device),  # Ensure edge_index is on correct device
            use_periodic_activation=self.use_periodic_activation
        ).to(device)  # Move entire ODE function to correct device
        
        # Solve GDE
        evolved_hidden = odeint(
            ode_func,
            hidden,
            t,
            method='rk4'
        )  # (time, batch, nodes, hidden)
        
        # Project to output space
        predictions = self.output_layer(evolved_hidden.permute(1, 2, 0, 3))
        
        return predictions.permute(0, 1, 3, 2).squeeze(2)  # (batch, nodes, time)

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