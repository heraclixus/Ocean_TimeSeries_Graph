from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import numpy as np
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# series in this case has dimension of (B, N)
# max and min has dimension of (N,)
def normalize(series, original_max, original_min):
    return (series - original_min) / (original_max - original_min)

# inverse normalize takes in data of size: (B, N, T)
# the max and min are of shape (N,)
def inverse_normalize(scaled_series, original_max, original_min):
    return scaled_series * (original_max-original_min) + original_min



def stochastic_batch_data_to_timeseries(batched_ts, n_pcs=20, sin_cos=False):

    if batched_ts.shape[1] == 1: 
        # (n_samples, n, t)
        return batched_ts.squeeze().transpose(0, 2, 1)

    batched_ts = np.expand_dims(batched_ts, axis=2) # (n_samples, b, 1, n, t)
    if sin_cos:
        batched_ts = batched_ts[:,:,:,:-2, :]
    n_timeseries = []
    for i in range(batched_ts.shape[0]):
        batched_ts_i = batched_ts[i].copy()
        time_series = batched_ts_i[0,0,0,:].copy()
        for j in range(1, len(batched_ts_i)):
            time_series = np.append(time_series, batched_ts_i[j,0,0,-1])
        T = len(time_series) - (n_pcs-1)  # Total number of possible windows of size 20
        final_series = np.array([time_series[i:i+n_pcs] for i in range(T)])
        n_timeseries.append(final_series)
    return np.array(n_timeseries)


def batch_data_to_timeseries(batched_ts, n_pcs=20, sin_cos=False):

    if len(batched_ts) == 1:
        return batched_ts.squeeze().T 

    # The first window gives us the first 24 points
    if len(batched_ts.shape) == 3:
        # (b, n, t) -> (b, 1, n, t)
        batched_ts = np.expand_dims(batched_ts, axis=1)

    if sin_cos:
        batched_ts = batched_ts[:, :, :-2, :]
   
    time_series = batched_ts[0, 0, 0, :].copy()  # Start with the first window
    # Add one new point from each subsequent window
    for i in range(1, len(batched_ts)):
        time_series = np.vstack((time_series, np.expand_dims(batched_ts[i, 0, :, -1], axis=0)))
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
        self._read_data(filepath)
    
    def _read_data(self, filepath):
        """
        Read and preprocess data
        
        Args:
            filepath (str): Path to data file
        """
        self._dataset = scipy.io.loadmat(filepath)['pcs'][:, :self.n_pcs]
        
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
    

# test
if __name__ == "__main__":
    input_file = "../../data/sst_pcs.mat"
    n_pcs=20
    sst_dataloader = SSTDatasetLoader(filepath=input_file, use_normalization=True, n_pcs=20)
    
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

    # make sure that loader de normalize gets back the actual data
    _, label = next(iter(train_loader))
    np.save("../label.npy", label.numpy())
    print(f"label = {label.shape}")
    print(label.max())
    print(label.min())
    label_transformed = inverse_normalize(label.numpy(), sst_dataloader._max, sst_dataloader._min)
    print(label_transformed.shape)
    print(label_transformed.max())
    print(label_transformed.min())
    print(train_dataset_orig.max())
    print(test_dataset_orig.min())
    np.save("../train_split.npy", train_dataset_orig)
    np.save("../test_split.npy", test_dataset_orig)

    print(train_target.shape)
    