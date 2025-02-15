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


def batch_data_to_timeseries(batched_ts):
    # The first window gives us the first 24 points
    if len(batched_ts.shape) == 3:
        # (b, n, t) -> (b, 1, n, t)
        batched_ts = np.expand_dims(batched_ts, axis=1)
    time_series = batched_ts[0, 0, :, :].copy().T  # Start with the first window
    # Add one new point from each subsequent window
    for i in range(1, len(batched_ts)):
        time_series = np.vstack((time_series, np.expand_dims(batched_ts[i, 0, :, -1], axis=0)))
    return time_series

class SSTDatasetLoader():

    def __init__(self, filepath, use_normalization, n_pcs):
        self.use_normalization = use_normalization
        self.n_pcs = n_pcs
        self._read_data(filepath)
    
    def _read_data(self, filepath):
        self._dataset = np.load(filepath)[:, :self.n_pcs]
        self._n_nodes = int(self._dataset.shape[-1])

        self._train_dataset = self._dataset[:round(self._dataset.shape[0] * 0.9)]
        self._test_dataset = self._dataset[round(self._dataset.shape[0] * 0.9):]
        self._max = np.max(self._train_dataset, axis=0)
        self._min = np.min(self._train_dataset, axis=0)
        # std to be used for loss function weights
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
    