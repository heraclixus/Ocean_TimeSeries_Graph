import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def normalize(series, original_max, original_min):
    """
    Normalize a series based on given max and min values
    
    Args:
        series (np.ndarray): Series to normalize with shape (timesteps, nodes)
        original_max (np.ndarray): Maximum values with shape (nodes,)
        original_min (np.ndarray): Minimum values with shape (nodes,)
        
    Returns:
        np.ndarray: Normalized series
    """
    # Create a mask for dimensions where max and min are both zero
    zero_range_mask = (original_max == 0) & (original_min == 0)
    
    # Initialize normalized series as a copy of the input
    normalized = series.copy()
    
    # Only normalize dimensions where the range is non-zero
    non_zero_range = ~zero_range_mask
    if np.any(non_zero_range):
        normalized[:, non_zero_range] = (series[:, non_zero_range] - original_min[non_zero_range]) / (original_max[non_zero_range] - original_min[non_zero_range])
    
    return normalized

def inverse_normalize(scaled_series, original_max, original_min):
    """
    Inverse normalize a series based on given max and min values
    
    Args:
        scaled_series (np.ndarray): Normalized series with shape (batch, nodes, horizon) or (nodes, horizon)
        original_max (np.ndarray): Maximum values with shape (nodes,)
        original_min (np.ndarray): Minimum values with shape (nodes,)
        
    Returns:
        np.ndarray: Original scale series
    """
    # Create a mask for dimensions where max and min are both zero
    zero_range_mask = (original_max == 0) & (original_min == 0)
    non_zero_range = ~zero_range_mask
    
    # Handle different input shapes
    input_shape = scaled_series.shape
    
    # Initialize denormalized series as a copy of the input
    denormalized = scaled_series.copy()
    
    # Handle 3D input (batch, nodes, horizon)
    if len(input_shape) == 3:
        batch_size, num_nodes, horizon = input_shape
        # Check if the mask dimensions match
        if len(non_zero_range) != num_nodes:
            # If the mask is too large, truncate it
            if len(non_zero_range) > num_nodes:
                non_zero_range = non_zero_range[:num_nodes]
                original_max = original_max[:num_nodes]
                original_min = original_min[:num_nodes]
            # If the mask is too small, extend it
            else:
                temp_mask = np.zeros(num_nodes, dtype=bool)
                temp_mask[:len(non_zero_range)] = non_zero_range
                non_zero_range = temp_mask
                
                temp_max = np.zeros(num_nodes)
                temp_max[:len(original_max)] = original_max
                original_max = temp_max
                
                temp_min = np.zeros(num_nodes)
                temp_min[:len(original_min)] = original_min
                original_min = temp_min
        
        # Apply denormalization
        for b in range(batch_size):
            for n in range(num_nodes):
                if non_zero_range[n]:
                    denormalized[b, n, :] = scaled_series[b, n, :] * (original_max[n] - original_min[n]) + original_min[n]
    
    # Handle 2D input (nodes, horizon)
    elif len(input_shape) == 2:
        num_nodes, horizon = input_shape
        # Check if the mask dimensions match
        if len(non_zero_range) != num_nodes:
            # If the mask is too large, truncate it
            if len(non_zero_range) > num_nodes:
                non_zero_range = non_zero_range[:num_nodes]
                original_max = original_max[:num_nodes]
                original_min = original_min[:num_nodes]
            # If the mask is too small, extend it
            else:
                temp_mask = np.zeros(num_nodes, dtype=bool)
                temp_mask[:len(non_zero_range)] = non_zero_range
                non_zero_range = temp_mask
                
                temp_max = np.zeros(num_nodes)
                temp_max[:len(original_max)] = original_max
                original_max = temp_max
                
                temp_min = np.zeros(num_nodes)
                temp_min[:len(original_min)] = original_min
                original_min = temp_min
        
        # Apply denormalization
        for n in range(num_nodes):
            if non_zero_range[n]:
                denormalized[n, :] = scaled_series[n, :] * (original_max[n] - original_min[n]) + original_min[n]
    
    return denormalized

class OceanGraphDataset:
    def __init__(self, graph_file="../../data/wrapped_grid_graph.pt", 
                 use_normalization=True,
                 use_region_only=False,
                 train_length=700):
        """
        Initialize Ocean Graph Dataset that loads from a PyTorch Geometric data object
        
        Args:
            graph_file (str): Path to PyTorch Geometric data file 
                              containing graph with time series as node features
            use_normalization (bool): Whether to normalize data
            use_region_only (bool): Whether to use only nodes in the ENSO region
            train_length (int): Number of time steps to use for training
        """
        self.graph_file = graph_file
        self.use_normalization = use_normalization
        self.use_region_only = use_region_only
        self.train_length = train_length
        
        self._read_data()
        
    def _read_data(self):
        """
        Read and preprocess data from PyTorch Geometric data file
        """
        # Load graph data
        print(f"Loading graph data from {self.graph_file}...")

        if torch.__version__ >= "2.6.0":
            self.graph_data = torch.load(self.graph_file, weights_only=False)
        else:
            self.graph_data = torch.load(self.graph_file)
        
        # Extract node features, edge indices, and edge attributes
        self.node_features = self.graph_data.x
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr
        
        # The last column in node features indicates ENSO region (binary mask)
        self.enso_mask = self.node_features[:, -1].bool()
        
        # The rest of the columns are time series (900 time steps)
        self.time_series = self.node_features[:, :-1]
        
        print(f"Graph data loaded with {self.time_series.shape[0]} nodes and {self.time_series.shape[1]} time steps")
        print(f"ENSO region contains {self.enso_mask.sum().item()} nodes")
        
        # If use_region_only is True, keep only nodes in the ENSO region
        if self.use_region_only:
            self._filter_enso_region()
        
        # Convert time series to numpy for easier processing
        node_data = self.time_series.numpy()
        
        # Transpose to get (time_steps, nodes) instead of (nodes, time_steps)
        self._dataset = node_data.T
        
        # Split data into train and test sets
        self._train_dataset = self._dataset[:self.train_length]
        self._test_dataset = self._dataset[self.train_length:]
        
        # Calculate statistics for normalization
        self._max = np.max(self._train_dataset, axis=0)
        self._min = np.min(self._train_dataset, axis=0)
        self._std = np.std(self._train_dataset, axis=0)
        
        # Store original data before normalization
        self._train_dataset_orig = self._train_dataset.copy()
        self._test_dataset_orig = self._test_dataset.copy()
        
        # Normalize if specified
        if self.use_normalization:
            self._train_dataset = normalize(self._train_dataset, self._max, self._min)
            self._test_dataset = normalize(self._test_dataset, self._max, self._min)
            
    def _filter_enso_region(self):
        """
        Filter graph to keep only nodes in the ENSO region
        """
        print("Filtering graph to keep only ENSO region nodes...")
        
        # Get indices of nodes in ENSO region
        enso_indices = torch.where(self.enso_mask)[0]
        self._enso_indices = enso_indices
        
        # Create mapping from old indices to new indices
        old_to_new = torch.full((self.node_features.shape[0],), -1, dtype=torch.long)
        for new_idx, old_idx in enumerate(enso_indices):
            old_to_new[old_idx] = new_idx
        
        # Filter time series to keep only ENSO region nodes
        self.time_series = self.time_series[enso_indices]
        
        # Filter edges to keep only those where both endpoints are in ENSO region
        mask = self.enso_mask[self.edge_index[0]] & self.enso_mask[self.edge_index[1]]
        self.edge_index = self.edge_index[:, mask]
        self.edge_attr = self.edge_attr[mask]
        
        # Remap edge indices to new node indices
        self.edge_index = old_to_new[self.edge_index]
        
        print(f"Filtered graph now has {self.time_series.shape[0]} nodes and {self.edge_index.shape[1]} edges")
        
        # Update enso_mask
        self.enso_mask = torch.ones(len(enso_indices), dtype=torch.bool)
    
    def _get_targets_and_features(self):
        """Create sliding window features and targets"""
        stacked_target = np.array(self._train_dataset)
        self.train_features = [
            stacked_target[i : i + self.window, :].T
            for i in range(stacked_target.shape[0] - self.horizon - self.window)
        ]
        self.train_targets = [
            stacked_target[i + self.window : i + self.window + self.horizon, :].T
            for i in range(stacked_target.shape[0] - self.horizon - self.window)
        ]
        
        stacked_target_test = np.array(self._test_dataset)
        self.test_features = [
            stacked_target_test[i : i + self.window, :].T
            for i in range(stacked_target_test.shape[0] - self.horizon - self.window)
        ]
        self.test_targets = [
            stacked_target_test[i + self.window : i + self.window + self.horizon, :].T
            for i in range(stacked_target_test.shape[0] - self.horizon - self.window)
        ]
    
    def get_dataset(self, window, horizon) -> StaticGraphTemporalSignal:
        """
        Return the Ocean Graph dataset iterator
        
        Args:
            window (int): The number of time steps in input window
            horizon (int): The number of time steps to predict
            
        Returns:
            tuple: (train_dataset, test_dataset) as StaticGraphTemporalSignal
        """
        self.window = window
        self.horizon = horizon
        self._get_targets_and_features()
        
        # Convert edge_index to numpy array
        edge_index_np = self.edge_index.numpy()
        
        # Extract edge weights (assuming edge_attr is the weight)
        if hasattr(self, 'edge_attr') and self.edge_attr is not None and self.edge_attr.shape[1] == 1:
            edge_weights = self.edge_attr.squeeze().numpy()
        else:
            # Default to all ones if no weights
            edge_weights = np.ones(edge_index_np.shape[1])
        
        train_dataset = StaticGraphTemporalSignal(
            edge_index_np, edge_weights, self.train_features, self.train_targets
        )
        test_dataset = StaticGraphTemporalSignal(
            edge_index_np, edge_weights, self.test_features, self.test_targets
        )
        
        return train_dataset, test_dataset
    
    def get_enso_mask(self):
        """
        Return binary mask indicating which nodes are in the ENSO region
        
        Returns:
            torch.Tensor: Boolean mask (True for nodes in ENSO region)
        """
        return self.enso_mask
    
    def get_enso_indices(self):
        """
        Return indices of nodes in the ENSO region
        
        Returns:
            torch.Tensor: Indices of nodes in the ENSO region
        """
        if hasattr(self, '_enso_indices'):
            return self._enso_indices
        else:
            return torch.where(self.enso_mask)[0]

    def visualize_graph(self, save_path=None):
        """
        Visualize the graph with ENSO region highlighted
        
        Args:
            save_path (str, optional): Path to save the visualization. If None, display only.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from torch_geometric.utils import to_networkx
            
            # Convert to networkx for visualization
            G = to_networkx(Data(x=self.time_series, edge_index=self.edge_index), to_undirected=True)
            
            plt.figure(figsize=(12, 10))
            
            # Get node positions - this might need adjustment based on your specific dataset
           
            # Use spring layout if positions not available
            pos = nx.spring_layout(G, seed=42)
            
            # Draw non-ENSO nodes
            if not self.use_region_only:
                non_enso_nodes = [i for i, is_enso in enumerate(self.enso_mask) if not is_enso]
                nx.draw_networkx_nodes(G, pos, nodelist=non_enso_nodes, 
                                    node_color='blue', alpha=0.3, node_size=20)
            
            # Draw ENSO nodes
            enso_nodes = [i for i, is_enso in enumerate(self.enso_mask) if is_enso]
            nx.draw_networkx_nodes(G, pos, nodelist=enso_nodes, 
                                node_color='red', alpha=0.7, node_size=30)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
            
            plt.title(f'Ocean Graph with ENSO Region\n{self.time_series.shape[0]} nodes, {self.edge_index.shape[1]} edges')
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path)
                print(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib and/or NetworkX not available for visualization")
            

# Test the dataset
if __name__ == "__main__":
    print("Testing OceanGraphDataset...")
    
    # Test with full graph
    print("\n=== Testing with full graph ===")
    dataset_full = OceanGraphDataset(use_region_only=False)
    
    # Print dataset information
    print(f"Full graph: {dataset_full.time_series.shape[0]} nodes")
    print(f"ENSO region: {dataset_full.enso_mask.sum().item()} nodes")
    print(f"Training dataset shape: {dataset_full._train_dataset.shape}")
    print(f"Testing dataset shape: {dataset_full._test_dataset.shape}")
    
    # Get full graph datasets
    train_dataset_full, test_dataset_full = dataset_full.get_dataset(window=6, horizon=12)
    
    # Print dataset statistics
    print("\nFull Graph Dataset Statistics:")
    print(f"Number of nodes: {dataset_full.time_series.shape[0]}")
    print(f"Number of edges: {dataset_full.edge_index.shape[1]}")
    print(f"Train features shape: {len(dataset_full.train_features)} x {dataset_full.train_features[0].shape}")
    print(f"Train targets shape: {len(dataset_full.train_targets)} x {dataset_full.train_targets[0].shape}")
    
    # Test with ENSO region only
    print("\n=== Testing with ENSO region only ===")
    dataset_enso = OceanGraphDataset(use_region_only=True)
    
    # Print dataset information
    print(f"ENSO region graph: {dataset_enso.time_series.shape[0]} nodes")
    print(f"Training dataset shape: {dataset_enso._train_dataset.shape}")
    print(f"Testing dataset shape: {dataset_enso._test_dataset.shape}")
    
    # Get ENSO region datasets
    train_dataset_enso, test_dataset_enso = dataset_enso.get_dataset(window=12, horizon=24)
    
    # Print dataset statistics
    print("\nENSO Region Dataset Statistics:")
    print(f"Number of nodes: {dataset_enso.time_series.shape[0]}")
    print(f"Number of edges: {dataset_enso.edge_index.shape[1]}")
    print(f"Train features shape: {len(dataset_enso.train_features)} x {dataset_enso.train_features[0].shape}")
    print(f"Train targets shape: {len(dataset_enso.train_targets)} x {dataset_enso.train_targets[0].shape}")
    
    # Verify data can be used in a model
    print("\nVerifying data can be used in a PyG model:")
    batch = next(iter(train_dataset_enso))
    print(f"Batch x shape: {batch.x.shape}")
    print(f"Batch y shape: {batch.y.shape}")
    print(f"Batch edge_index shape: {batch.edge_index.shape}")
    
    # Try visualization
    try:
        dataset_full.visualize_graph("ocean_graph_full.png")
        dataset_enso.visualize_graph("ocean_graph_enso.png")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\nSetup complete!")
