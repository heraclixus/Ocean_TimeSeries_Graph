import networkx as nx
import torch
import numpy as np
import scipy.io
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric.utils import from_networkx

def create_graph_from_mat(first_grid):
    mask = first_grid != 0 
    sea_coords = torch.nonzero(mask, as_tuple=False)
    G = nx.Graph()
    print(mask.shape)
    rows, cols = mask.shape

    for coord in sea_coords:
        x, y = coord.tolist()
        G.add_node((x, y))  # add node

        # Possible 4 neighbors: left, right, up, down (no diagonals)
        neighbors = [
            ((x - 1) % rows, y),  # up
            ((x + 1) % rows, y),  # down
            (x, (y - 1) % cols),  # left (wrap around)
            (x, (y + 1) % cols),  # right (wrap around)
        ]

        for nx_, ny_ in neighbors:
            if mask[nx_, ny_]:
                G.add_edge((x, y), (nx_, ny_))
    
    return G, sea_coords

def normalize(series, original_max, original_min):
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
    # Create a mask for dimensions where max and min are both zero
    zero_range_mask = (original_max == 0) & (original_min == 0)
    
    # Initialize denormalized series as a copy of the input
    denormalized = scaled_series.copy()
    
    # Only denormalize dimensions where the range is non-zero
    non_zero_range = ~zero_range_mask
    if np.any(non_zero_range):
        denormalized[..., non_zero_range] = scaled_series[..., non_zero_range] * (original_max[non_zero_range] - original_min[non_zero_range]) + original_min[non_zero_range]
    
    return denormalized

class OceanGraphDataset:
    def __init__(self, raw_file="../../data/raw_file.mat", 
                 indsst_file="../../data/indsst.mat", 
                 tdata_file="../../data/tdata.mat",
                 use_normalization=True,
                 train_length=700):
        """
        Initialize Ocean Graph Dataset
        
        Args:
            raw_file (str): Path to raw data file with shape (900, 8100) that will be reshaped to (900, 180, 45)
            indsst_file (str): Path to mask file for region of interest
            tdata_file (str): Path to data in the region of interest
            use_normalization (bool): Whether to normalize data
            train_length (int): Number of time steps to use for training
        """
        self.raw_file = raw_file
        self.indsst_file = indsst_file
        self.tdata_file = tdata_file
        self.use_normalization = use_normalization
        self.train_length = train_length
        
        self._read_data()
        
    def _read_data(self):
        """
        Read and preprocess data from files
        """
        # Load raw data (900, 8100)
        raw_data = scipy.io.loadmat(self.raw_file)["rawdata"]
        
        # Reshape raw data from (900, 8100) to (900, 180, 45)
        time_steps, flattened_size = raw_data.shape
        assert flattened_size == 180 * 45, f"Expected flattened size to be 8100 (180*45), but got {flattened_size}"
        self._raw_data = raw_data.reshape(time_steps, 180, 45)
        self._lat_dim, self._lon_dim = 180, 45
        print(f"Raw data reshaped from {raw_data.shape} to {self._raw_data.shape}")
        
        # Load mask for region of interest
        self._indsst = scipy.io.loadmat(self.indsst_file)["indsst"].astype(bool)
        
        # Load data in region of interest
        self._tdata = scipy.io.loadmat(self.tdata_file)["tdata"]
        
        # Create graph from first time step of raw data
        self.first_grid = torch.tensor(self._raw_data[0])
        print(f"self.first_grid.shape: {self.first_grid.shape}")
        self.G, self.sea_coords = create_graph_from_mat(self.first_grid)
        
        # Number of nodes in the graph (sea points)
        self._n_nodes = len(self.G.nodes())
        print(f"Created graph with {self._n_nodes} nodes (sea points) and {len(self.G.edges())} edges")
        
        # Extract time series for sea points more efficiently
        sea_coords_np = self.sea_coords.numpy()
        self._dataset = np.zeros((time_steps, self._n_nodes))
        
        # Use vectorized approach for better performance
        for t in range(time_steps):
            grid_t = self._raw_data[t]
            for i, (x, y) in enumerate(sea_coords_np):
                self._dataset[t, i] = grid_t[x, y]
        
        print(f"Extracted time series with shape {self._dataset.shape}")
        
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
            
    def _convert_graph_to_pyg(self):
        """Convert NetworkX graph to PyG format"""
        # Convert NetworkX graph to PyG format
        pyg_graph = from_networkx(self.G)
        
        # Extract edge indices
        self._edges = pyg_graph.edge_index
        
        # Initialize edge weights to ones
        self._edge_weights = np.ones(self._edges.shape[1])
    
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
        self._convert_graph_to_pyg()
        self._get_targets_and_features()
        
        train_dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.train_features, self.train_targets
        )
        test_dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.test_features, self.test_targets
        )
        
        return train_dataset, test_dataset
    
    def get_node_mapping(self):
        """
        Return mapping of graph nodes to grid coordinates
        
        Returns:
            dict: Mapping of node indices to grid coordinates
        """
        node_mapping = {}
        for i, coord in enumerate(self.sea_coords):
            x, y = coord.tolist()
            node_mapping[i] = (x, y)
        
        return node_mapping

# Test the dataset
if __name__ == "__main__":
    print("Testing OceanGraphDataset...")
    
    # Initialize dataset
    dataset = OceanGraphDataset()
    
    # Print dataset information
    print(f"Raw data shape: {dataset._raw_data.shape}")
    print(f"Sea points (nodes): {dataset._n_nodes}")
    print(f"Dataset shape before splitting: {dataset._dataset.shape}")
    print(f"Training dataset shape: {dataset._train_dataset.shape}")
    print(f"Testing dataset shape: {dataset._test_dataset.shape}")
    
    # Get graph datasets
    train_dataset, test_dataset = dataset.get_dataset(window=12, horizon=24)
    
    # Print dataset statistics
    print("\nGraph Dataset Statistics:")
    print(f"Number of nodes: {dataset._n_nodes}")
    print(f"Number of edges: {dataset._edges.shape[1]}")
    print(f"Train features shape: {len(dataset.train_features)} x {dataset.train_features[0].shape}")
    print(f"Train targets shape: {len(dataset.train_targets)} x {dataset.train_targets[0].shape}")
    print(f"Test features shape: {len(dataset.test_features)} x {dataset.test_features[0].shape}")
    print(f"Test targets shape: {len(dataset.test_targets)} x {dataset.test_targets[0].shape}")
    
    # Verify data can be used in a model
    print("\nVerifying data can be used in a PyG model:")
    batch = next(iter(train_dataset))
    print(f"Batch x shape: {batch.x.shape}")
    print(f"Batch y shape: {batch.y.shape}")
    print(f"Batch edge_index shape: {batch.edge_index.shape}")
    
    print("\nSetup complete!")