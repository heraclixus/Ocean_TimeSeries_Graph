from typing import Tuple

import torch
import numpy as np


@torch.no_grad()
def compute_rmse(model, dl, device: str = 'cpu') -> float:
    model.eval()
    mse_total, n_total = 0.0, 0
    memory_depth = int(getattr(model, 'memory_depth', 0) or 0)
    for batch in dl:
        if len(batch) == 4:
            x_in, t_in, x_next, _ = batch
        else:
            x_in, t_in, x_next = batch
        x_in = x_in.to(device)
        x_next = x_next.to(device)
        t_in = t_in.to(device)
        dt = 1.0 / 12.0
        dxdt = model(x_in, t_in)
        if memory_depth > 0:
            x_hat = x_in[:, -1, :] + dxdt * dt
        else:
            x_hat = x_in + dxdt * dt
        mse = torch.mean((x_hat - x_next) ** 2, dim=(1, 0)).item() if x_hat.ndim == 2 else torch.mean((x_hat - x_next) ** 2).item()
        mse_total += mse * x_in.size(0)
        n_total += x_in.size(0)
    return float(np.sqrt(mse_total / max(n_total, 1)))

