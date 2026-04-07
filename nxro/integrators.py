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


@torch.no_grad()
def euler_step_memory(model, x_history: torch.Tensor, t_history: torch.Tensor,
                      dt: float = 1.0 / 12.0):
    """One Euler step for memory-aware models.

    Args:
        model: drift function f(history, time_history)
        x_history: [B, P+1, n_vars] ordered from oldest to current
        t_history: [B, P+1] ordered from oldest to current
        dt: month step in years (1/12)
    Returns:
        x_next: [B, n_vars]
        x_history_next: [B, P+1, n_vars]
        t_history_next: [B, P+1]
    """
    dxdt = model(x_history, t_history)
    x_next = x_history[:, -1, :] + dxdt * dt
    x_history_next = torch.cat([x_history[:, 1:, :], x_next.unsqueeze(1)], dim=1)
    t_next = t_history[:, -1:] + dt
    t_history_next = torch.cat([t_history[:, 1:], t_next], dim=1)
    return x_next, x_history_next, t_history_next

