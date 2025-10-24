import torch


@torch.no_grad()
def euler_step(model, x_t: torch.Tensor, t_years: torch.Tensor, dt: float = 1.0/12.0) -> torch.Tensor:
    """One Euler step: X_{t+1} = X_t + f(X_t, t) * dt.

    Args:
        model: drift function f(x,t)
        x_t: [B, n_vars]
        t_years: [B]
        dt: month step in years (1/12)
    Returns:
        x_{t+1}: [B, n_vars]
    """
    dxdt = model(x_t, t_years)
    return x_t + dxdt * dt


