import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error

class ARIMAX:
    def __init__(self, order=(1,1), trend='c'):
        """
        ARIMAX model for multivariate time series
        
        Args:
            order (tuple): ARIMAX order (p,q)
                p: autoregressive order
                q: moving average order
            trend (str): Trend term specification
        """
        self.order = order
        self.trend = trend
        self.model = None
        self.is_fitted = False
        
    def fit(self, x):
        """
        Fit ARIMAX model
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
        """
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
            
        # Only use actual PCs (excluding sin/cos) for VARMAX
        n_dims = x.shape[1] - 2 if hasattr(self, 'add_sin_cos') and self.add_sin_cos else x.shape[1]
        x_fit = x[0, :n_dims].T  # Use first batch, shape: (T, N)
        
        # Fit model
        self.model = VARMAX(x_fit, order=self.order, trend=self.trend)
        self.model = self.model.fit(disp=False)
        self.is_fitted = True
        
    def predict(self, x, horizon):
        """
        Generate forecasts
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
            horizon (int): Forecast horizon
            
        Returns:
            np.ndarray: Forecasts of shape (B, N, H)
        """
        if not self.is_fitted:
            raise RuntimeError("Model needs to be fitted first!")
            
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
            
        batch_size, n_dims, _ = x.shape
        n_actual_dims = n_dims - 2 if hasattr(self, 'add_sin_cos') and self.add_sin_cos else n_dims
        forecasts = np.zeros((batch_size, n_dims, horizon))
        
        for b in range(batch_size):
            # Forecast actual PCs
            x_batch = x[b, :n_actual_dims].T  # Shape: (T, N)
            model = VARMAX(x_batch, order=self.order, trend=self.trend)
            model = model.fit(disp=False)
            forecast = model.forecast(steps=horizon)  # Shape: (H, N)
            forecasts[b, :n_actual_dims] = forecast.T
            
            # If using sin/cos features, generate future values
            if hasattr(self, 'add_sin_cos') and self.add_sin_cos:
                last_t = len(x[b, 0, :])
                t = np.arange(last_t, last_t + horizon)
                period = 12
                forecasts[b, -2, :] = np.sin(2 * np.pi * t / period)  # sin wave
                forecasts[b, -1, :] = np.cos(2 * np.pi * t / period)  # cos wave
            
        return forecasts
    
    def compute_loss(self, pred, target):
        """
        Compute MSE loss
        """
        return mean_squared_error(target, pred)
