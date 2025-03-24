import numpy as np
import scipy.io

class SSTResidualDatasetLoader():

    def __init__(self, 
                 data_file_path="../../data/fcst.mat", 
                 residuals_file_path="../../data/residual.mat", 
                 n_pcs=20, 
                 train_length=700):
        """
        Initialize dataset loader
        
        Args:
            data_file_path (str): Path to data file
            residuals_file_path (str): Path to residuals file
            use_normalization (bool): Whether to normalize data
            n_pcs (int): Number of principal components to use
            train_length (int): Number of time steps to use for training
        """
        self.n_pcs = n_pcs
        self.train_length = train_length
        self.num_nodes = n_pcs
        self._read_data(data_file_path, residuals_file_path)
    
    def _read_data(self, data_file_path, residuals_file_path):
        """
        Read and preprocess data
        
        Args:
            data_file_path (str): Path to data file
            residuals_file_path (str): Path to residuals file
        """
        self.X =  scipy.io.loadmat(data_file_path)['fcst'] # (887, 12, 20)
        self.y = scipy.io.loadmat(residuals_file_path)['residual'] # (887, 12, 20)
        assert self.X.shape == self.y.shape
        self.X_train = self.X[:self.train_length]
        self.X_test = self.X[self.train_length:]
        self.y_train = self.y[:self.train_length]
        self.y_test = self.y[self.train_length:]