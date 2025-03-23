import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from baseline_models.utils import PeriodicActivation
from torch_geometric.data import Data, Batch

class GraphODEFunc_GNODE(nn.Module):
    """ODE function using GNN"""
    def __init__(self, node_features, hidden_dim, use_periodic_activation=False):
        super().__init__()
        self.use_periodic_activation = use_periodic_activation

        self.periodic_activation = PeriodicActivation()
        self.tanh_activation = nn.Tanh()
        self.graph_layers = nn.ModuleList([
            GCNConv(node_features, 64),
            GCNConv(64, hidden_dim),   
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, 64),
            GCNConv(64, node_features)
        ])
        
    def forward(self, t, x, edge_index):
        """
        Args:
            t (torch.Tensor): Current time point
            x (torch.Tensor): Current state (B*N, F)
            edge_index (torch.Tensor): Graph connectivity (2, E*B)
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
    def __init__(self, node_features=1, hidden_dim=64, forecast_horizon=10, use_periodic_activation=False):
        """
        Graph Neural ODE for time series forecasting
        
        Args:
            node_features (int): Number of features per node (default=1 for scalar time series)
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
    
    def create_graph_batch(self, x, device, edge_index=None):
        """
        Create PyG Data objects for batch processing
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, timesteps)
            device (torch.device): Device to create tensors on
            
        Returns:
            Batch: PyG Batch object containing all graphs
        """
        batch_size, num_nodes, _ = x.shape
        data_list = []
        
        for b in range(batch_size):
            edges = []
            # Create fully connected graph (excluding self-loops)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Exclude self-loops as GCNConv adds them automatically
                        edges.append([i, j])
            if edge_index == None:
               edge_index = torch.tensor(edges, device=device).t()
            else:
                edge_index = edge_index.to(device)
            
            # Take the last timestep as initial condition
            x_last = x[b, :, -1].unsqueeze(-1)  # (num_nodes, 1)            
            # Create Data object for this batch item
            data = Data(
                x=x_last,  # (num_nodes, 1)
                edge_index=edge_index,
                num_nodes=num_nodes
            )
            data_list.append(data)
        
        return Batch.from_data_list(data_list)
    
    def forward(self, x, edge_index=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, num_nodes, timesteps)
            
        Returns:
            torch.Tensor: Forecasted values of shape (batch_size, num_nodes, forecast_horizon)
        """
        batch_size, num_nodes, timesteps = x.shape
        device = x.device
        
        # Create batch of graphs
        batch = self.create_graph_batch(x, device, edge_index)
        
        # Create time points for forecasting
        t = torch.linspace(0, self.forecast_horizon, self.forecast_horizon).to(device)
        
        # Flatten batch dimension for ODE solver
        x_flat = batch.x.view(-1, self.node_features)  # (B*N, 1)
        
        # Define the ODE function wrapper to include edge_index
        def ode_func(t, state):
            return self.ode_func(t, state, batch.edge_index)
        
        # Solve ODE to get forecasts
        predictions = odeint(
            ode_func,
            x_flat,  # (B*N, 1)
            t,
            method='rk4',
            rtol=1e-3,
            atol=1e-3
        )  # Shape: (T, B*N, 1)
        
        # Reshape predictions to (B, N, H)
        predictions = predictions.view(self.forecast_horizon, batch_size, num_nodes)
        predictions = predictions.permute(1, 2, 0)  # (B, N, H)
        
        return predictions.squeeze(-1)  # Remove feature dimension since it's 1
    
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
            GCNConv(1, 64),  # Changed from input_dim to 1
            GCNConv(64, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
        ])
        
        # Temporal attention
        if self.use_periodic_activation:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                PeriodicActivation(), 
                nn.Linear(hidden_dim, hidden_dim),
                PeriodicActivation(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
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

    def create_spatiotemporal_graph(self, batch_size, time_steps, device, edge_index=None):
        """
        Create spatial-temporal graph structure and return PyG Data objects
        
        Args:
            batch_size (int): Number of graphs in the batch
            time_steps (int): Number of time steps
            device (torch.device): Device to create tensors on
            edge_index (torch.Tensor, optional): Static spatial graph edges of shape (2, num_edges)
            
        Returns:
            list[Data]: List of PyG Data objects for each batch
        """
        data_list = []
        
        # Create edges for each batch
        for b in range(batch_size):
            all_edges = []
            temporal_edges = []
            
            # For each timestep
            for t in range(time_steps):
                # Calculate the base offset for current timestep
                current_offset = t * self.num_nodes
                
                # Add spatial edges if provided, otherwise create fully connected graph
                if edge_index is not None:
                    # Add the provided spatial edges with appropriate time offset
                    src = edge_index[0] + current_offset
                    dst = edge_index[1] + current_offset
                    all_edges.extend(torch.stack([src, dst], dim=0).t().tolist())
                else:
                    # Create fully connected spatial edges within timestep
                    for i in range(self.num_nodes):
                        for j in range(self.num_nodes):
                            if i != j:  # Exclude self-loops
                                all_edges.append([current_offset + i, current_offset + j])
                
                # Add temporal edges (connecting to next timestep)
                if t < time_steps - 1:
                    next_offset = (t + 1) * self.num_nodes
                    for i in range(self.num_nodes):
                        # Forward temporal edge
                        temporal_edges.append([current_offset + i, next_offset + i])
                        # Backward temporal edge
                        temporal_edges.append([next_offset + i, current_offset + i])
            
            # Combine all edges and convert to tensor
            all_edges.extend(temporal_edges)
            edge_index = torch.tensor(all_edges, device=device).t()
            
            # Create Data object for this batch
            data = Data(
                edge_index=edge_index,
                num_nodes=time_steps * self.num_nodes
            )
            data_list.append(data)
        
        return data_list

    def forward(self, x, edge_index=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, time)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_nodes, horizon)
        """
        batch_size = x.shape[0]
        device = x.device

        # Create spatiotemporal graphs as PyG Data objects
        data_list = self.create_spatiotemporal_graph(
            batch_size=batch_size,
            time_steps=self.time_series_length,
            device=device,
            edge_index=edge_index
        )
        
        # Create batch of graphs
        batch = Batch.from_data_list(data_list)
        
        # Reshape input for spatiotemporal processing
        x_reshaped = x.permute(0, 2, 1).reshape(-1, 1)  # (batch_size * time_steps * num_nodes, 1)
        batch.x = x_reshaped
        
        # Process through GCN layers
        h = batch.x
        for gcn in self.spatial_gcn:
            h = gcn(h, batch.edge_index)
            h = F.relu(h)
        
        # Reshape back: (batch_size, time_steps, num_nodes, hidden_dim)
        h = h.view(batch_size, self.time_series_length, self.num_nodes, -1)
        
        # Apply temporal attention
        attention = self.temporal_attention(h)
        attention_weights = F.softmax(attention, dim=1)
        
        # Weighted combination of features
        node_features = torch.sum(h * attention_weights, dim=1)  # (batch_size, num_nodes, hidden_dim)
        
        # Process temporal sequence with GRU
        node_features = node_features.reshape(batch_size * self.num_nodes, -1)
        _, hidden = self.encoder_gru(node_features.unsqueeze(1))
        
        # Reshape hidden to match spatiotemporal graph structure
        hidden = hidden.squeeze(0)
        hidden = hidden.unsqueeze(1).expand(-1, self.time_series_length, -1)
        hidden = hidden.reshape(batch_size * self.time_series_length * self.num_nodes, -1)
        batch.x = hidden
        
        # Create time points for GDE evolution
        t = torch.linspace(0, self.forecast_length, self.forecast_length).to(device)
        
        # Define GDE function
        ode_func = GraphODEFunc(
            hidden_dim=self.hidden_dim,
            edge_index=batch.edge_index,
            use_periodic_activation=self.use_periodic_activation
        ).to(device)
        
        # Solve GDE
        evolved_hidden = odeint(
            ode_func,
            batch.x,
            t,
            method='rk4'
        )
        
        # Reshape evolved hidden states
        evolved_hidden = evolved_hidden.view(
            self.forecast_length, 
            batch_size, 
            self.time_series_length,
            self.num_nodes, 
            -1
        )
        
        # Take the last timestep's predictions
        evolved_hidden = evolved_hidden[:, :, -1, :, :]
        
        # Project to output space
        predictions = self.output_layer(evolved_hidden.permute(1, 2, 0, 3))
        
        return predictions.squeeze(-1)  # (batch, nodes, forecast_length)

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
