from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import scipy.io
import torch



class SSTDatasetLoader():

    def __init__(self, filepath):
        self._read_data(filepath)
    
    
    def _read_data(self, filepath):
        self._dataset = scipy.io.loadmat(filepath)['pcs']
        self._n_nodes = self._dataset.shape[-1]

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
        stacked_target = np.array(self._dataset)
        print(stacked_target.shape)
        self.features = [
            stacked_target[i : i + self.window, :].T
            for i in range(stacked_target.shape[0] - self.horizon - self.window)
        ]
        self.targets = [
            stacked_target[i + self.window : i + self.window + self.horizon, :].T
            for i in range(stacked_target.shape[0] - self.horizon - self.window)
        ]

        print(np.array(self.features).shape)
        print(np.array(self.targets).shape)
    
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
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset




if __name__ == "__main__":
    sst_dataloader = SSTDatasetLoader(filepath="../../data/sst_pcs.mat")

    dataset = sst_dataloader.get_dataset(window=12, horizon=24)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    # print("Number of train buckets: ", len(set(train_dataset)))
    # print("Number of test buckets: ", len(set(test_dataset)))
    train_input = np.array(train_dataset.features) # (27399, 207, 2, 12)
    train_target = np.array(train_dataset.targets) # (27399, 207, 12)
    print(f"train_input = {train_input.shape}")
    print(f"train_target = {train_target.shape}")
    test_input = np.array(test_dataset.features) # (, 207, 2, 12)
    test_target = np.array(test_dataset.targets) # (, 207, 12)  
    print(f"test_input = {test_input.shape}")
    print(f"test_target = {test_target.shape}")

    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).unsqueeze(1)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).unsqueeze(1)  # (B, N, T)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=32, shuffle=False, drop_last=True)

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).unsqueeze(1) # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).unsqueeze(1) # (B, N, T)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=1, shuffle=False,drop_last=True)

    print(f"train_x_tensor = {train_x_tensor.shape}")
    print(f'train_target_tensor = {train_target_tensor.shape}')
    print(f"test_x_tensor = {test_x_tensor.shape}")
    print(f"test_target_tensor = {test_target_tensor.shape}")

    from torch_geometric_temporal.nn.attention.mtgnn import MTGNN

    model = MTGNN(gcn_true=True, build_adj=False, gcn_depth=3, num_nodes=sst_dataloader._n_nodes, 
                  kernel_set=[1,1], kernel_size=1, dropout=0.5, subgraph_size=12, node_dim=1, dilation_exponential=1,
                  conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=1, out_dim=24, 
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True )

    encoder_input, label = next(iter(train_loader))
    print(f"encoder_input = {encoder_input.shape}")
    print(f'label = {label.shape}') 

    adj_mat = torch.from_numpy(sst_dataloader._adj_mat).type(torch.FloatTensor)
    print(f"adj_mat = {adj_mat.shape}")
    output = model(encoder_input, A_tilde=adj_mat).permute(0,3,2,1)
    print(f"output = {output.shape}")