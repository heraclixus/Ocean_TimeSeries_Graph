import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class MultiARIMA:
    def __init__(self, order=(1,1,1)):
        """
        Multivariate ARIMA model (implemented as multiple univariate ARIMA models)
        
        Args:
            order (tuple): ARIMA order (p,d,q)
                p: autoregressive order
                d: differencing order
                q: moving average order
        """
        self.order = order
        self.models = []
        self.is_fitted = False
        
    def fit(self, x):
        """
        Fit ARIMA model for each dimension
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
        """
        # Convert to numpy if tensor
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
            
        # Fit separate ARIMA model for each dimension
        # Only fit on actual PCs (excluding sin/cos)
        n_dims = x.shape[1] - 2 if hasattr(self, 'add_sin_cos') and self.add_sin_cos else x.shape[1]
        self.models = []
        
        for i in range(n_dims):
            model = ARIMA(x[0, i, :], order=self.order)
            fitted = model.fit()
            self.models.append(fitted)
        
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
            for i in range(n_actual_dims):
                model = ARIMA(x[b, i, :], order=self.order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
                forecasts[b, i, :] = forecast
            
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
