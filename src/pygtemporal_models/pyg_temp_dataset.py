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
    # print(f"batched_ts = {batched_ts.shape}")
    for i in range(batched_ts.shape[0]):
        batched_ts_i = batched_ts[i].copy()
        # (b, 1, n, t) -> (n, t)
        # print(f"batched_ts_i = {batched_ts_i.shape}")
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
    def __init__(self, filepath, use_normalization, 
                 use_region_data=False,
                 add_sin_cos=False, train_length=700, coarse_grain_factor=5):
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
        self.use_region_data = use_region_data
        self._read_data(filepath)

    def find_region_indices(self, mask):
        # Find all True positions (rows and columns where mask is True)
        rows, cols = np.where(mask)
        
        # Find the bounding box
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        
        # Return useful information
        return min_row, max_row, min_col, max_col
    

    def _coarse_grain_outside_region(self, grid_data, coarse_factor, min_row=None, max_row=None, min_col=None, max_col=None, region_mask=None):
        """
        Coarse grain grid data outside of a specified region of interest.
        
        Args:
            grid_data (np.ndarray): Original grid data of shape (rows, cols) or (time, rows, cols)
            coarse_factor (int): Factor by which to coarsen the grid (e.g., 2 means every 2x2 cells become 1 cell)
            min_row (int, optional): Minimum row index of region to preserve
            max_row (int, optional): Maximum row index of region to preserve
            min_col (int, optional): Minimum column index of region to preserve
            max_col (int, optional): Maximum column index of region to preserve
            region_mask (np.ndarray, optional): Boolean mask of shape (rows, cols) where True indicates cells to preserve
                                            (takes precedence over min/max indices if provided)
        
        Returns:
            np.ndarray: Coarse-grained grid with preserved region
        """
        # Check if input has time dimension
        has_time_dim = len(grid_data.shape) == 3
        
        if has_time_dim:
            time_steps, rows, cols = grid_data.shape
            
            # Process each time step
            results = []
            for t in range(time_steps):
                coarse_t = self._coarse_grain_outside_region(
                    grid_data[t], coarse_factor, min_row, max_row, min_col, max_col, region_mask
                )
                results.append(coarse_t)
            return np.stack(results, axis=0)
        
        # No time dimension
        rows, cols = grid_data.shape
        
        # Create region mask if not provided
        if region_mask is None:
            region_mask = np.zeros((rows, cols), dtype=bool)
            
            # Use the provided boundaries to define the region to preserve
            if min_row is not None and max_row is not None and min_col is not None and max_col is not None:
                region_mask[min_row:max_row+1, min_col:max_col+1] = True
            else:
                raise ValueError("Either region_mask or all of min_row, max_row, min_col, max_col must be provided")
        
        # Calculate dimensions for coarse-grained grid
        region_height = max_row - min_row + 1 if min_row is not None else 0
        region_width = max_col - min_col + 1 if min_col is not None else 0
        
        # Calculate the number of rows/cols that will be in the coarse-grained areas
        coarse_rows_before = min_row // coarse_factor if min_row is not None else 0
        coarse_rows_after = (rows - (max_row + 1)) // coarse_factor if max_row is not None else 0
        coarse_cols_before = min_col // coarse_factor if min_col is not None else 0
        coarse_cols_after = (cols - (max_col + 1)) // coarse_factor if max_col is not None else 0
        
        # Calculate new dimensions
        new_rows = coarse_rows_before + region_height + coarse_rows_after
        new_cols = coarse_cols_before + region_width + coarse_cols_after
        
        # Create the new coarse-grained grid
        result = np.zeros((new_rows, new_cols), dtype=grid_data.dtype)
        
        # Fill in the preserved region (direct copy)
        if min_row is not None and max_row is not None and min_col is not None and max_col is not None:
            result[coarse_rows_before:coarse_rows_before+region_height, 
                coarse_cols_before:coarse_cols_before+region_width] = grid_data[min_row:max_row+1, min_col:max_col+1]
        
        # Handle the area before the preserved region (rows)
        for i in range(coarse_rows_before):
            row_start = i * coarse_factor
            row_end = min((i + 1) * coarse_factor, min_row)
            
            # For each column in the result grid
            for j in range(new_cols):
                if j < coarse_cols_before:
                    # Coarse-grain the area before preserved region (columns)
                    col_start = j * coarse_factor
                    col_end = min((j + 1) * coarse_factor, min_col)
                    result[i, j] = np.mean(grid_data[row_start:row_end, col_start:col_end])
                elif j < coarse_cols_before + region_width:
                    # Mix of coarse-grained rows and preserved columns
                    col_idx = min_col + (j - coarse_cols_before)
                    result[i, j] = np.mean(grid_data[row_start:row_end, col_idx])
                else:
                    # Coarse-grain the area after preserved region (columns)
                    col_start = max_col + 1 + (j - coarse_cols_before - region_width) * coarse_factor
                    col_end = min(col_start + coarse_factor, cols)
                    result[i, j] = np.mean(grid_data[row_start:row_end, col_start:col_end])
        
        # Handle the area after the preserved region (rows)
        for i in range(coarse_rows_after):
            row_start = max_row + 1 + i * coarse_factor
            row_end = min(row_start + coarse_factor, rows)
            
            # For each column in the result grid
            for j in range(new_cols):
                if j < coarse_cols_before:
                    # Mix of coarse-grained rows and coarse-grained columns (before)
                    col_start = j * coarse_factor
                    col_end = min((j + 1) * coarse_factor, min_col)
                    result[coarse_rows_before + region_height + i, j] = np.mean(grid_data[row_start:row_end, col_start:col_end])
                elif j < coarse_cols_before + region_width:
                    # Mix of coarse-grained rows and preserved columns
                    col_idx = min_col + (j - coarse_cols_before)
                    result[coarse_rows_before + region_height + i, j] = np.mean(grid_data[row_start:row_end, col_idx])
                else:
                    # Coarse-grain both rows and columns (after)
                    col_start = max_col + 1 + (j - coarse_cols_before - region_width) * coarse_factor
                    col_end = min(col_start + coarse_factor, cols)
                    result[coarse_rows_before + region_height + i, j] = np.mean(grid_data[row_start:row_end, col_start:col_end])
        
        # Handle the preserved region rows with coarse-grained columns before
        for i in range(region_height):
            row_idx = min_row + i
            
            # Coarse-grain columns before the preserved region
            for j in range(coarse_cols_before):
                col_start = j * coarse_factor
                col_end = min((j + 1) * coarse_factor, min_col)
                result[coarse_rows_before + i, j] = np.mean(grid_data[row_idx, col_start:col_end])
            
            # Coarse-grain columns after the preserved region
            for j in range(coarse_cols_after):
                col_start = max_col + 1 + j * coarse_factor
                col_end = min(col_start + coarse_factor, cols)
                result[coarse_rows_before + i, coarse_cols_before + region_width + j] = np.mean(grid_data[row_idx, col_start:col_end])
        
        return result
    
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
        grid_data_orig = scipy.io.loadmat(filepath)["nino34_data"][0][0]["subdata3d"]
        region_mask = scipy.io.loadmat(filepath)["nino34_data"][0][0]["region_mask"]
        region_data = scipy.io.loadmat(filepath)["nino34_data"][0][0]["region_data"]
        min_row, max_row, min_col, max_col = self.find_region_indices(region_mask)
        region_data = region_data[:, min_row:max_row, min_col:max_col]
        print(f"region_data.shape = {region_data.shape}")
        indsst = scipy.io.loadmat(filepath)["nino34_data"][0][0]["indsst"].astype(np.int32).flatten()
        print(f"Original indsst shape: {indsst.shape}")
        time_steps, lat, lon = grid_data_orig.shape
        
        # Apply coarse graining
        if self.coarse_grain_factor > 1:
            grid_data = self._coarse_grain_outside_region(grid_data_orig, 
                                                        self.coarse_grain_factor, 
                                                        min_row, max_row,
                                                        min_col, max_col, 
                                                        region_mask)
            
            # Create new region mask for coarse-grained grid
            # Calculate new positions after coarse-graining
            coarse_rows_before = min_row // self.coarse_grain_factor
            coarse_cols_before = min_col // self.coarse_grain_factor
            region_height = max_row - min_row + 1
            region_width = max_col - min_col + 1
            
            # Get new dimensions
            _, new_lat, new_lon = grid_data.shape
            new_region_mask = np.zeros((new_lat, new_lon), dtype=bool)
            new_min_row = coarse_rows_before
            new_max_row = coarse_rows_before + region_height - 1
            new_min_col = coarse_cols_before
            new_max_col = coarse_cols_before + region_width - 1
            new_region_mask[new_min_row:new_max_row+1, new_min_col:new_max_col+1] = True
            
            # Update indsst for the new grid structure
            # Approach: Since we preserved the region exactly, indices in the region
            # maintain their relative positions, just shifted to the new location
            
            # Each index in indsst corresponds to a position in the flattened original grid
            # We need to convert it to (row, col) in the original grid, check if it's in
            # the region, then convert it to the corresponding index in the new grid
            
            # First, map each indsst value to its new position
            new_indsst = []
            for idx in indsst:
                # Convert flat index to 2D coordinates in original grid
                orig_row = idx // lon
                orig_col = idx % lon
                
                # Check if this point is in the region of interest
                if region_mask[orig_row, orig_col]:
                    # Get this point's position relative to region's top-left corner
                    rel_row = orig_row - min_row
                    rel_col = orig_col - min_col
                    
                    # Calculate new position in coarse-grained grid
                    new_row = new_min_row + rel_row
                    new_col = new_min_col + rel_col
                    
                    # Convert to flat index in new grid
                    new_idx = new_row * new_lon + new_col
                    new_indsst.append(new_idx)
            
            # Just to be safe, directly add any points that are in the region mask 
            # but might have been missed in the mapping above
            if len(new_indsst) == 0:  # Only do this if we haven't found any indices
                for i in range(min_row, max_row + 1):
                    for j in range(min_col, max_col + 1):
                        if region_mask[i, j]:
                            # Get position relative to region start
                            rel_row = i - min_row
                            rel_col = j - min_col
                            
                            # Calculate new position
                            new_row = new_min_row + rel_row
                            new_col = new_min_col + rel_col
                            
                            # Add to new indices
                            new_idx = new_row * new_lon + new_col
                            new_indsst.append(new_idx)
            
            new_indsst = np.array(new_indsst, dtype=np.int32)
            
            # If we still have zero-length indsst, it's safer to use all indices in the region
            if len(new_indsst) == 0:
                print("WARNING: Failed to map indsst indices. Using all points in region as fallback.")
                new_indsst = np.where(new_region_mask.flatten())[0]
            
            # Debug output
            print(f"Original region dimensions: {min_row}-{max_row}, {min_col}-{max_col}")
            print(f"New region position: {new_min_row}-{new_max_row}, {new_min_col}-{new_max_col}")
            print(f"Original grid: {lat}x{lon}, New grid: {new_lat}x{new_lon}")
            print(f"Coarse rows/cols before: {coarse_rows_before}, {coarse_cols_before}")
            print(f"Original indsst length: {len(indsst)}, New indsst length: {len(new_indsst)}")
            
            # Update instance variables
            self._region_mask = new_region_mask
            self._indsst = new_indsst
        else:
            grid_data = grid_data_orig
            self._region_mask = region_mask
            self._indsst = indsst
        
        print(f"grid_data.shape = {grid_data.shape}, coarse_grain_factor = {self.coarse_grain_factor}")
        
        if self.use_region_data:
            time_steps, self.lat_dim, self.lon_dim = region_data.shape
        else:
            time_steps, self.lat_dim, self.lon_dim = grid_data.shape

        # show animation from the result of coarse-graining
        # if not os.path.exists(f"grid_comparison_coarse_grain={self.coarse_grain_factor}.mp4"):
        #     create_comparison_animation_data(original_data=grid_data_orig, 
        #                                     coarse_data=grid_data, 
        #                                     output_path=f"grid_comparison_coarse_grain={self.coarse_grain_factor}.mp4")
        
        # Reshape to (time, nodes) where nodes = lat * lon
        self._region_data = region_data
        self._grid_data = grid_data
        if self.use_region_data:
            self._dataset = region_data.reshape(time_steps, -1)
        else:
            self._dataset = grid_data.reshape(time_steps, -1)
        print(f"use region dataset = {self.use_region_data}")
        print(f"lat = {self.lat_dim}")
        print(f"lon = {self.lon_dim}")
        print(f"self._dataset = {self._dataset.shape}")
        print(f"Region mask shape: {self._region_mask.shape}")
        print(f"Number of indsst indices: {len(self._indsst)}")
        
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
    