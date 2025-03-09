from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import numpy as np
import scipy.io 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from utils_visualization_forecast import create_comparison_animation_data

# series in this case has dimension of (B, N)
# max and min has dimension of (N,)
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

# inverse normalize takes in data of size: (B, N, T)
# the max and min are of shape (N,)
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



def stochastic_batch_data_to_timeseries(batched_ts):

    if batched_ts.shape[1] == 1: 
        # (n_samples, n, t)
        return batched_ts.squeeze().transpose(0, 2, 1)
    n_timeseries = []
    print(f"batched_ts = {batched_ts.shape}")
    for i in range(batched_ts.shape[0]):
        batched_ts_i = batched_ts[i].copy()
        # (b, 1, n, t) -> (n, t)
        print(f"batched_ts_i = {batched_ts_i.shape}")
        final_series = batch_data_to_timeseries(batched_ts_i)
        n_timeseries.append(final_series)
    # (n_samples, n, t)
    return np.array(n_timeseries)


def batch_data_to_timeseries(batched_ts):
    if isinstance(batched_ts, torch.Tensor): 
        batched_ts = batched_ts.detach().cpu().numpy()

    if len(batched_ts.shape) == 4:
        batched_ts = batched_ts.squeeze(1)

    if len(batched_ts) == 1:
        return batched_ts.squeeze().T 
    # (1, t, n)
    time_series = batched_ts[0, :, :].T.copy()  # Start with the first window
    # Add one new point from each subsequent window
    for i in range(1, len(batched_ts)):
        # each appending is: (1, n)
        time_series = np.concatenate((time_series, np.expand_dims(batched_ts[i, :, -1],axis=0)),axis=0)
    # (T,n)
    return time_series

class SSTDatasetLoader():

    def __init__(self, filepath, use_normalization, n_pcs, add_sin_cos=False, train_length=700):
        """
        Initialize dataset loader
        
        Args:
            filepath (str): Path to data file
            use_normalization (bool): Whether to normalize data
            n_pcs (int): Number of principal components to use
            add_sin_cos (bool): Whether to add sinusoidal features
            train_length (int): Number of time steps to use for training
        """
        self.use_normalization = use_normalization
        self.n_pcs = n_pcs
        self.add_sin_cos = add_sin_cos
        self.train_length = train_length
        self.num_nodes = n_pcs + 2 if add_sin_cos else n_pcs
        self._read_data(filepath)
    
    def _read_data(self, filepath):
        """
        Read and preprocess data
        
        Args:
            filepath (str): Path to data file
        """
        self._dataset =  np.load(filepath)[:, :self.n_pcs]
        
        # Add sinusoidal features if requested
        if self.add_sin_cos:
            time_steps = np.arange(len(self._dataset))
            period = 12
            sin_wave = np.sin(2 * np.pi * time_steps / period)
            cos_wave = np.cos(2 * np.pi * time_steps / period)
            
            # Add as new columns
            sin_wave = sin_wave.reshape(-1, 1)
            cos_wave = cos_wave.reshape(-1, 1)
            self._dataset = np.concatenate([self._dataset, sin_wave, cos_wave], axis=1)
            
            self._n_actual_pcs = self.n_pcs
            self.n_pcs += 2
        else:
            self._n_actual_pcs = self.n_pcs
        
        self._n_nodes = self._dataset.shape[-1]
        
        # Split using train_length instead of ratio
        self._train_dataset = self._dataset[:self.train_length]
        self._test_dataset = self._dataset[self.train_length:]
        
        # Only compute statistics on actual PCs (not sin/cos)
        if self.add_sin_cos:
            self._max = np.concatenate([
                np.max(self._train_dataset[:, :self._n_actual_pcs], axis=0),
                np.ones(2)  # For sin/cos features
            ])
            self._min = np.concatenate([
                np.min(self._train_dataset[:, :self._n_actual_pcs], axis=0),
                -np.ones(2)  # For sin/cos features
            ])
            self._std = np.concatenate([
                np.std(self._train_dataset[:, :self._n_actual_pcs], axis=0),
                np.ones(2)  # For sin/cos features
            ])
        else:
            self._max = np.max(self._train_dataset, axis=0)
            self._min = np.min(self._train_dataset, axis=0)
            self._std = np.std(self._train_dataset, axis=0)

        self._train_dataset_orig = self._train_dataset
        self._test_dataset_orig = self._test_dataset
        
        if self.use_normalization:
            self._train_dataset = normalize(self._train_dataset, self._max, self._min)
            self._test_dataset = normalize(self._test_dataset, self._max, self._min)

    def _get_edges(self, self_loop=True):
        # fully connected graph with self-loop
        edges = []
        for i in range(self._n_nodes):
            for j in range(self._n_nodes):
                if self_loop == False and i == j:
                    continue
                edges.append([i,j])
        self._edges = np.array(edges).T

        adjacency = np.zeros((self._n_nodes, self._n_nodes), dtype=int)
        # Vectorized assignment: set adjacency[src, dst] = 1 for all edges
        adjacency[self._edges[0], self._edges[1]] = 1
        self._adj_mat = adjacency
    

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
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
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **window** *(int)* - The number of time window.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.window = window
        self.horizon = horizon
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        train_dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.train_features, self.train_targets
        )
        test_dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.test_features, self.test_targets
        )
        return train_dataset, test_dataset
    

    def plot_std(self):
        sns.barplot(x=np.arange(len(self._std)) ,y=self._std)
        plt.xlabel("pcs")
        plt.ylabel("standard deviation") 
        plt.savefig(f"std_for_pcs_top{self.n_pcs}.png")



# class for grid data
class SSTGridDataLoader():
    def __init__(self, filepath, use_normalization, add_sin_cos=False, train_length=700, coarse_grain_factor=5):
        """
        Initialize dataset loader for grid SST data
        
        Args:
            filepath (str): Path to data file
            use_normalization (bool): Whether to normalize data
            add_sin_cos (bool): Whether to add sinusoidal features
            train_length (int): Number of time steps to use for training
            coarse_grain_factor (int): Factor by which to reduce spatial dimensions
        """
        self.use_normalization = use_normalization
        self.add_sin_cos = add_sin_cos
        self.train_length = train_length
        self.coarse_grain_factor = coarse_grain_factor
        self._read_data(filepath)
    
    def _coarse_grain(self, data):
        """
        Coarse grain the spatial dimensions by averaging over sub-grids
        
        Args:
            data (np.ndarray): Input data of shape (time, lat, lon)
        Returns:
            np.ndarray: Coarse-grained data
        """
        time_steps, lat, lon = data.shape
        
        # Calculate new dimensions
        new_lat = lat // self.coarse_grain_factor
        new_lon = lon // self.coarse_grain_factor

        self.lat = new_lat 
        self.lon = new_lon
        
        # Reshape to create sub-grids
        reshaped = data.reshape(
            time_steps, 
            new_lat, self.coarse_grain_factor,
            new_lon, self.coarse_grain_factor
        )
        
        # Average over sub-grids
        coarse_grained = reshaped.mean(axis=(2, 4))
        
        print(f"Original shape: {data.shape}")
        print(f"Coarse-grained shape: {coarse_grained.shape}")
        
        return coarse_grained
    
    def _read_data(self, filepath):
        """Read and preprocess grid data"""
        # Load grid data of shape (time, lat, lon)
        grid_data_orig = np.load(filepath)
        time_steps, lat, lon = grid_data_orig.shape
        
        # Apply coarse graining
        grid_data = self._coarse_grain(grid_data_orig)
        time_steps, self.lat_dim, self.lon_dim = grid_data.shape

        # show animation from the result of coarse-graining
        if not os.path.exists(f"grid_comparison_coarse_grain={self.coarse_grain_factor}.mp4"):
            create_comparison_animation_data(original_data=grid_data_orig, 
                                            coarse_data=grid_data, 
                                            output_path=f"grid_comparison_coarse_grain={self.coarse_grain_factor}.mp4")
        
        # Reshape to (time, nodes) where nodes = lat * lon
        self._dataset = grid_data.reshape(time_steps, self.lat_dim * self.lon_dim)        
        print(f"self._dataset = {self._dataset.shape}")
        
        # Add sinusoidal features if requested
        # if self.add_sin_cos:
        #     time_steps = np.arange(len(self._dataset))
        #     period = 12  # Annual cycle
        #     sin_wave = np.sin(2 * np.pi * time_steps / period)
        #     cos_wave = np.cos(2 * np.pi * time_steps / period)
            
        #     # Add sin/cos as new nodes
        #     sin_wave = sin_wave.reshape(-1, 1)
        #     cos_wave = cos_wave.reshape(-1, 1)
            
        #     self._dataset = np.concatenate([
        #         self._dataset,
        #         np.tile(sin_wave, (1, self.lat_dim * self.lon_dim)),
        #         np.tile(cos_wave, (1, self.lat_dim * self.lon_dim))
        #     ], axis=1)
        
        # print(f"self._dataset with features = {self._dataset.shape}")
        self._n_nodes = self._dataset.shape[1]

        # Rest of the method remains the same
        self._train_dataset = self._dataset[:self.train_length]
        self._test_dataset = self._dataset[self.train_length:]
        
        self._max = np.max(self._train_dataset, axis=0)
        self._min = np.min(self._train_dataset, axis=0)
        self._std = np.std(self._train_dataset, axis=0)
        
        self._train_dataset_orig = self._train_dataset.copy()
        self._test_dataset_orig = self._test_dataset.copy()
        
        if self.use_normalization:
            self._train_dataset = normalize(self._train_dataset, self._max, self._min)
            self._test_dataset = normalize(self._test_dataset, self._max, self._min)

    def _get_edges(self, self_loop=True):
        """Create edges for grid structure with 8-neighborhood connectivity"""
        edges = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(self.lat_dim):
            for j in range(self.lon_dim):
                node = i * self.lon_dim + j
                
                # Add self-loop if requested
                if self_loop:
                    edges.append([node, node])
                
                # Add edges to 8 neighbors
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.lat_dim and 0 <= nj < self.lon_dim:
                        neighbor = ni * self.lon_dim + nj
                        edges.append([node, neighbor])
        
        # Convert to torch tensor of type Long
        self._edges = torch.tensor(edges, dtype=torch.long).T
        
        # Create adjacency matrix (keep as numpy for compatibility)
        self._adj_mat = np.zeros((self._n_nodes, self._n_nodes), dtype=int)
        self._adj_mat[self._edges.numpy()[0], self._edges.numpy()[1]] = 1

    def _get_edge_weights(self):
        """Initialize edge weights to ones"""
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        """Create sliding window features and targets"""
        # Use the same implementation as SSTDatasetLoader
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
        Return the SST grid dataset iterator
        
        Args:
            window (int): The number of time steps in input window
            horizon (int): The number of time steps to predict
        Returns:
            tuple: (train_dataset, test_dataset) as StaticGraphTemporalSignal
        """
        self.window = window
        self.horizon = horizon
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        train_dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.train_features, self.train_targets
        )
        test_dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.test_features, self.test_targets
        )
        return train_dataset, test_dataset
    

# test
if __name__ == "__main__":
    input_file = "../../data/ersst_anomaly.npy"

    sst_dataloader = SSTGridDataLoader(filepath=input_file, use_normalization=True)
    
    train_dataset_orig, test_dataset_orig = sst_dataloader._train_dataset_orig, sst_dataloader._test_dataset_orig

    train_dataset, test_dataset = sst_dataloader.get_dataset(window=12, horizon=24)
    train_input = np.array(train_dataset.features)
    train_target = np.array(train_dataset.targets) 
    test_input = np.array(test_dataset.features)
    test_target = np.array(test_dataset.targets) 
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).unsqueeze(1)  # (B, F, N, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).unsqueeze(1)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=64, shuffle=False, drop_last=True)

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).unsqueeze(1) # (B, F, N, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).unsqueeze(1)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=1, shuffle=False,drop_last=True)


    print(f"train_x_tensor = {train_x_tensor.shape}")
    print(f"train_target_tensor = {train_target_tensor.shape}")
    print(f"test_x_tensor = {test_x_tensor.shape}")
    print(f"test_target_tensor = {test_target_tensor.shape}")
    print(f"train_x_tensor_max = {train_x_tensor.max()}")
    print(f"train_x_tensor_min = {train_x_tensor.min()}")
    print(f"train_target_tensor_max = {train_target_tensor.max()}")
    print(f"train_target_tensor_min = {train_target_tensor.min()}")

    # make sure that loader de normalize gets back the actual data
    _, label = next(iter(train_loader))
    ts = batch_data_to_timeseries(label.numpy())
    label_transformed = inverse_normalize(ts, sst_dataloader._max, sst_dataloader._min)
    print(f"label_transformed = {label_transformed.shape}")
    print(label_transformed.max())
    print(label_transformed.min())
    print(train_dataset_orig.max())
    print(train_dataset_orig.min())
    print(test_dataset_orig.max())
    print(test_dataset_orig.min())
    