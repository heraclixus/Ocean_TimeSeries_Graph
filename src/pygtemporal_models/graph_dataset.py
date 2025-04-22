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
        self._indsst_original = scipy.io.loadmat(self.indsst_file)["indsst"].astype(bool)
        print(f"Original indsst mask shape: {self._indsst_original.shape}")
        
        # Load data in region of interest
        self._tdata = scipy.io.loadmat(self.tdata_file)["tdata"]
        
        # Create graph from first time step of raw data
        self.first_grid = torch.tensor(self._raw_data[0])
        print(f"self.first_grid.shape: {self.first_grid.shape}")
        self.G, self.sea_coords = create_graph_from_mat(self.first_grid)
        
        # Number of nodes in the graph (sea points)
        self._n_nodes = len(self.G.nodes())
        print(f"Created graph with {self._n_nodes} nodes (sea points) and {len(self.G.edges())} edges")
        
        # Create a mapping from original grid coordinates to node indices
        self._create_node_mapping()
        
        # Create indsst mapping for nodes of interest
        self._create_indsst_node_mapping()
        
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
            
    def _create_node_mapping(self):
        """
        Create a mapping from original grid coordinates to node indices
        """
        self.coord_to_node = {}
        for i, coord in enumerate(self.sea_coords):
            x, y = coord.tolist()
            self.coord_to_node[(x, y)] = i
        
        # Also create reverse mapping (node index to coordinates)
        self.node_to_coord = {}
        for i, coord in enumerate(self.sea_coords):
            x, y = coord.tolist()
            self.node_to_coord[i] = (x, y)
    
    def _create_indsst_node_mapping(self):
        """
        Create a mapping from the original indsst mask to node indices in the graph
        
        This converts the binary mask in the original grid space to indices in the graph space
        """
        # Find coordinates in the original indsst mask that are True
        indsst_coords = np.argwhere(self._indsst_original)
        
        # For each of these coordinates, find the corresponding node index in our graph
        indsst_node_indices = []
        
        for coord in indsst_coords:
            x, y = coord
            if (x, y) in self.coord_to_node:  # Check if this point is in our graph (is ocean)
                node_idx = self.coord_to_node[(x, y)]
                indsst_node_indices.append(node_idx)
        
        # Convert to numpy array for easier indexing
        self._indsst = np.array(indsst_node_indices, dtype=np.int64)
        print(f"Created indsst node mapping with {len(self._indsst)} nodes of interest")
        
        # Keep track of indices in the original grid that are both in indsst and are sea points
        self._indsst_sea_coords = [self.node_to_coord[idx] for idx in indsst_node_indices]
        print(f"Number of sea coordinates in region of interest: {len(self._indsst_sea_coords)}")
        
        # Create a binary mask for nodes that are in indsst
        self._indsst_mask = np.zeros(self._n_nodes, dtype=bool)
        self._indsst_mask[self._indsst] = True
    
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
        return self.node_to_coord
    
    def get_coord_to_node_mapping(self):
        """
        Return mapping of grid coordinates to node indices
        
        Returns:
            dict: Mapping of grid coordinates to node indices
        """
        return self.coord_to_node
    
    def get_indsst_nodes(self):
        """
        Return array of node indices that correspond to the region of interest
        
        Returns:
            numpy.ndarray: Array of node indices in the region of interest
        """
        return self._indsst
    
    def get_indsst_mask(self):
        """
        Return boolean mask indicating which nodes are in the region of interest
        
        Returns:
            numpy.ndarray: Boolean mask of shape (n_nodes,)
        """
        return self._indsst_mask
    
    def visualize_graph_with_region(self, save_path=None):
        """
        Visualize the graph with region of interest highlighted
        
        Args:
            save_path (str, optional): Path to save the visualization. If None, display only.
        """
        import matplotlib.pyplot as plt
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Get all node coordinates
        node_coords = np.array([self.node_to_coord[i] for i in range(self._n_nodes)])
        
        # Get region of interest coordinates
        roi_coords = np.array(self._indsst_sea_coords)
        
        # Plot all nodes
        plt.scatter(node_coords[:, 1], node_coords[:, 0], s=5, c='blue', alpha=0.5, label='Sea Points')
        
        # Highlight nodes in region of interest
        plt.scatter(roi_coords[:, 1], roi_coords[:, 0], s=10, c='red', label='Region of Interest')
        
        # Draw edges - this can be slow for large graphs
        edge_list = list(self.G.edges())
        if len(edge_list) < 10000:  # Only draw edges if there aren't too many
            for edge in edge_list:
                x1, y1 = edge[0]
                x2, y2 = edge[1]
                plt.plot([y1, y2], [x1, x2], 'k-', alpha=0.1)
        
        plt.title(f'Ocean Graph with Region of Interest\n{self._n_nodes} nodes, {len(self._indsst)} in ROI')
        plt.legend()
        plt.gca().invert_yaxis()  # Invert y-axis to match matrix coordinates
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

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
    
    # Print information about the region of interest mapping
    print("\nRegion of Interest Mapping:")
    print(f"Original indsst shape: {dataset._indsst_original.shape}")
    print(f"Number of points in original indsst: {np.sum(dataset._indsst_original)}")
    print(f"Number of nodes in graph: {dataset._n_nodes}")
    print(f"Number of nodes in indsst mapping: {len(dataset._indsst)}")
    print(f"First 10 indsst node indices: {dataset._indsst[:10]}")
    
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
    
    # Demonstrate how to extract data for region of interest
    print("\nExtract data for region of interest:")
    roi_indices = dataset.get_indsst_nodes()
    x_batch = batch.x.numpy()
    y_batch = batch.y.numpy()
    
    # Extract features and targets for the region of interest
    x_roi = x_batch[roi_indices, :]
    y_roi = y_batch[roi_indices, :]
    print(f"ROI x shape: {x_roi.shape}")
    print(f"ROI y shape: {y_roi.shape}")
    
    # Visualize the graph with region of interest
    try:
        dataset.visualize_graph_with_region(save_path="ocean_graph_with_roi.png")
        print("Visualization saved to ocean_graph_with_roi.png")
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\nSetup complete!")