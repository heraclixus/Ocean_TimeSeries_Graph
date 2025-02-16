import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error

class MultiGARCH:
    def __init__(self, p=1, q=1, vol='GARCH'):
        """
        Multivariate GARCH model (implemented as multiple univariate GARCH models)
        
        Args:
            p (int): GARCH lag order
            q (int): ARCH lag order
            vol (str): Volatility model type
        """
        self.p = p
        self.q = q
        self.vol = vol
        self.models = []
        self.is_fitted = False
        
    def fit(self, x):
        """
        Fit GARCH model for each dimension
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
        """
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
            
        # Fit separate GARCH model for each dimension (excluding sin/cos)
        n_dims = x.shape[1] - 2 if hasattr(self, 'add_sin_cos') and self.add_sin_cos else x.shape[1]
        self.models = []
        
        for i in range(n_dims):
            model = arch_model(x[0, i, :], p=self.p, q=self.q, vol=self.vol)
            fitted = model.fit(disp='off')
            self.models.append(fitted)
            
        self.is_fitted = True
        
    def predict(self, x, horizon):
        """
        Generate forecasts
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T)
            horizon (int): Forecast horizon
            
        Returns:
            tuple: (forecasts, volatility) of shape (B, N, H)
        """
        if not self.is_fitted:
            raise RuntimeError("Model needs to be fitted first!")
            
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
            
        batch_size, n_dims, _ = x.shape
        n_actual_dims = n_dims - 2 if hasattr(self, 'add_sin_cos') and self.add_sin_cos else n_dims
        forecasts = np.zeros((batch_size, n_dims, horizon))
        volatility = np.zeros((batch_size, n_actual_dims, horizon))  # Only for actual PCs
        
        for b in range(batch_size):
            # Forecast actual PCs
            for i in range(n_actual_dims):
                model = arch_model(x[b, i, :], p=self.p, q=self.q, vol=self.vol)
                fitted = model.fit(disp='off')
                forecast = fitted.forecast(horizon=horizon)
                forecasts[b, i, :] = forecast.mean.values[-horizon:]
                volatility[b, i, :] = np.sqrt(forecast.variance.values[-horizon:])
            
            # If using sin/cos features, generate future values
            if hasattr(self, 'add_sin_cos') and self.add_sin_cos:
                last_t = len(x[b, 0, :])
                t = np.arange(last_t, last_t + horizon)
                period = 12
                forecasts[b, -2, :] = np.sin(2 * np.pi * t / period)  # sin wave
                forecasts[b, -1, :] = np.cos(2 * np.pi * t / period)  # cos wave
            
        return forecasts, volatility
    
    def compute_loss(self, pred, target):
        """
        Compute MSE loss
        """
        return mean_squared_error(target, pred)
