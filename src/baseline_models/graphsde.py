import torch
import torch.nn as nn
import torchsde
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from baseline_models.utils import PeriodicActivation

class GraphSDEFunc(nn.Module):
    """SDE function using GNN for both drift and diffusion"""
    def __init__(self, node_features, hidden_dim, use_periodic_activation=False):
        super().__init__()
        self.use_periodic_activation = use_periodic_activation
        self.node_features = node_features
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
        self.periodic_activation = PeriodicActivation()
        self.tanh_activation = nn.Tanh()
        
        # Drift GNN layers (f)
        self.drift_layers = nn.ModuleList([
            GCNConv(node_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, node_features)
        ])
        
        # Diffusion GNN layers (g)
        self.diffusion_layers = nn.ModuleList([
            GCNConv(node_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, node_features)
        ])
        
        # Store edge_index for forward pass
        self.edge_index = None
    
    def set_edge_index(self, edge_index):
        """Set the edge_index for the forward pass"""
        self.edge_index = edge_index
    
    def f(self, t, x):
        """Drift function"""
        h = x
        for i, layer in enumerate(self.drift_layers):
            h = layer(h, self.edge_index)
            if i < len(self.drift_layers) - 1:
                if self.use_periodic_activation:
                    h = self.periodic_activation(h)
                else:
                    h = self.tanh_activation(h)
        return h
    
    def g(self, t, x):
        """Diffusion function"""
        h = x
        for i, layer in enumerate(self.diffusion_layers):
            h = layer(h, self.edge_index)
            if i < len(self.diffusion_layers) - 1:
                if self.use_periodic_activation:
                    h = self.periodic_activation(h)
                else:
                    h = self.tanh_activation(h)
        # Ensure diffusion is positive
        return torch.abs(h)

class GraphNeuralSDE(nn.Module):
    def __init__(self, node_features=1, hidden_dim=64, forecast_horizon=10, use_periodic_activation=False):
        """
        Graph Neural SDE for time series forecasting
        
        Args:
            node_features (int): Number of features per node (default=1 for scalar time series)
            hidden_dim (int): Size of hidden layers
            forecast_horizon (int): Number of future time steps to forecast
            use_periodic_activation (bool): Whether to use periodic activation functions
        """
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.use_periodic_activation = use_periodic_activation
        
        # SDE function with GNN
        self.sde_func = GraphSDEFunc(node_features, hidden_dim, self.use_periodic_activation)
        
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
            edge_index (torch.Tensor, optional): Optional predefined edge indices
            
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
            if edge_index is None:
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
    
    def forward(self, x, edge_index=None, n_samples=1):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, num_nodes, timesteps)
            edge_index (torch.Tensor, optional): Optional predefined edge indices
            n_samples (int): Number of Monte Carlo samples for SDE
            
        Returns:
            torch.Tensor: Forecasted values of shape (batch_size, num_nodes, forecast_horizon)
        """
        batch_size, num_nodes, timesteps = x.shape
        device = x.device
        
        # Create batch of graphs
        batch = self.create_graph_batch(x, device, edge_index)
        
        # Create time points for forecasting
        ts = torch.linspace(0, self.forecast_horizon, self.forecast_horizon, device=device)
        
        # Flatten batch dimension for SDE solver
        x_flat = batch.x.view(-1, self.node_features)  # (B*N, 1)
        
        # Set edge_index in SDE function
        self.sde_func.set_edge_index(batch.edge_index)
        
        # We'll collect multiple samples for the stochastic process
        all_samples = []
        for _ in range(n_samples):
            # Solve SDE
            predictions = torchsde.sdeint(
                self.sde_func,
                x_flat,
                ts,
                method='srk',
                dt=1.0 / self.forecast_horizon,
                adaptive=False,
                rtol=1e-3,
                atol=1e-3
            )  # Shape: (T, B*N, 1)
            
            # Reshape predictions to (B, N, H)
            pred_sample = predictions.view(self.forecast_horizon, batch_size, num_nodes, self.node_features)
            pred_sample = pred_sample.permute(1, 2, 0, 3)  # (B, N, H, F)
            all_samples.append(pred_sample)
        
        # Stack and average the samples
        all_samples = torch.stack(all_samples, dim=0)  # (n_samples, B, N, H, F)
        predictions = all_samples.mean(dim=0)  # (B, N, H, F)
        
        return predictions.squeeze(-1)  # (B, N, H) Remove feature dimension since it's 1
    
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