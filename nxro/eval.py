from typing import Tuple

import torch
import numpy as np


@torch.no_grad()
def compute_rmse(model, dl, device: str = 'cpu') -> float:
    model.eval()
    mse_total, n_total = 0.0, 0
    for x_t, t_y, x_next in dl:
        x_t = x_t.to(device)
        x_next = x_next.to(device)
        t_y = t_y.to(device)
        dt = 1.0 / 12.0
        dxdt = model(x_t, t_y)
        x_hat = x_t + dxdt * dt
        mse = torch.mean((x_hat - x_next) ** 2, dim=(1, 0)).item() if x_hat.ndim == 2 else torch.mean((x_hat - x_next) ** 2).item()
        mse_total += mse * x_t.size(0)
        n_total += x_t.size(0)
    return float(np.sqrt(mse_total / max(n_total, 1)))


