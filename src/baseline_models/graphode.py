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

class NeuralGDEForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_series_length, forecast_length, num_nodes):
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
        
        # Spatial graph learning layer
        self.spatial_graph_learner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal attention for connecting same nodes across time
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # GRU for encoding temporal dependencies
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Neural ODE function for evolving hidden states
        self.gde_func = GraphODEFunc(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes
        )
        
        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def create_spatiotemporal_graph(self, x):
        """
        Create spatial-temporal graph from input sequence
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, features, time)
            
        Returns:
            tuple: (node_features, adjacency_matrix)
        """
        batch_size = x.shape[0]
        
        # Reshape input for processing
        x = x.permute(0, 3, 1, 2)  # (batch_size, time, nodes, features)
        
        # Learn spatial relationships
        spatial_features = self.spatial_graph_learner(x)  # (batch_size, time, nodes, hidden)
        
        # Create spatial adjacency matrix using dot product similarity
        spatial_adj = torch.matmul(
            spatial_features, 
            spatial_features.transpose(-2, -1)
        )  # (batch_size, time, nodes, nodes)
        spatial_adj = torch.softmax(spatial_adj / np.sqrt(self.hidden_dim), dim=-1)
        
        # Create temporal connections using attention
        temporal_weights = self.temporal_attention(spatial_features)  # (batch_size, time, nodes, 1)
        temporal_weights = torch.softmax(temporal_weights, dim=1)  # Normalize across time
        
        # Combine spatial and temporal relationships
        spatiotemporal_features = torch.sum(
            spatial_features * temporal_weights, 
            dim=1
        )  # (batch_size, nodes, hidden)
        
        return spatiotemporal_features, spatial_adj[:, -1]  # Use last spatial adjacency

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, features, time)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_nodes, features, horizon)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Create spatial-temporal graph
        node_features, adjacency = self.create_spatiotemporal_graph(x)
        
        # Encode temporal sequence
        x_temporal = x.permute(0, 1, 3, 2)  # (batch_size, nodes, time, features)
        x_temporal = x_temporal.reshape(batch_size * self.num_nodes, -1, self.input_dim)
        _, hidden = self.encoder_gru(x_temporal)
        hidden = hidden.view(batch_size, self.num_nodes, -1)  # (batch_size, nodes, hidden)
        
        # Combine with graph features
        hidden = hidden + node_features
        
        # Create time points for GDE evolution
        t = torch.linspace(0, self.forecast_length, self.forecast_length).to(device)
        
        # Define GDE function with current adjacency
        def ode_func(t, h):
            return self.gde_func(t, h, adjacency)
        
        # Solve GDE
        evolved_hidden = odeint(
            ode_func,
            hidden,
            t,
            method='rk4'
        )  # (time, batch, nodes, hidden)
        
        # Reshape for decoder
        evolved_hidden = evolved_hidden.permute(1, 2, 0, 3)  # (batch, nodes, time, hidden)
        evolved_hidden = evolved_hidden.reshape(batch_size * self.num_nodes, -1, self.hidden_dim)
        
        # Initialize decoder with last hidden state
        decoder_hidden = evolved_hidden[:, -1:, :]
        
        # Prepare decoder input (zeros)
        decoder_input = torch.zeros(
            batch_size * self.num_nodes,
            self.forecast_length,
            self.hidden_dim
        ).to(device)
        
        # Decode sequence
        decoder_output, _ = self.decoder_gru(decoder_input, decoder_hidden.transpose(0, 1))
        decoder_output = decoder_output.view(batch_size, self.num_nodes, -1, self.hidden_dim)
        
        # Project to output space
        predictions = self.output_layer(decoder_output)  # (batch, nodes, time, features)
        
        return predictions.permute(0, 1, 3, 2)  # (batch, nodes, features, time)

    def compute_loss(self, pred, target, std=None, add_sin_cos=False):
        """
        Compute MSE loss between predictions and targets
        """
        if add_sin_cos:
            std = std[:-2]
        
        if std is not None:
            std = torch.tensor(std, device=pred.device)
            weights = 1.0 / (std ** 2)
            weights = weights.view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return torch.mean((pred - target) ** 2)

class GraphODEFunc(nn.Module):
    """ODE function for graph evolution"""
    def __init__(self, hidden_dim, num_nodes):
        super(GraphODEFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Node update function
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge update function
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, h, adjacency):
        """
        Compute the derivative at current time point
        
        Args:
            t (torch.Tensor): Current time point
            h (torch.Tensor): Current hidden state (batch, nodes, hidden)
            adjacency (torch.Tensor): Adjacency matrix (batch, nodes, nodes)
            
        Returns:
            torch.Tensor: Derivative of hidden state
        """
        batch_size = h.shape[0]
        
        # Compute messages using adjacency matrix
        messages = torch.bmm(adjacency, h)  # (batch, nodes, hidden)
        
        # Concatenate node states with messages
        node_inputs = torch.cat([h, messages], dim=-1)  # (batch, nodes, hidden*2)
        
        # Update node states
        dh = self.node_update(node_inputs)
        
        # Update edge weights (adjacency)
        edge_features = torch.cat([
            h.unsqueeze(2).expand(-1, -1, self.num_nodes, -1),
            h.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        ], dim=-1)  # (batch, nodes, nodes, hidden*2)
        
        edge_updates = self.edge_update(edge_features).squeeze(-1)  # (batch, nodes, nodes)
        adjacency = adjacency + edge_updates  # Update adjacency matrix
        
        return dh