# utils/returns.py
import torch

def lambda_return(
    reward: torch.Tensor,         # [B,H,1] or [H,B,1]
    value: torch.Tensor,          # same shape as reward
    discount: torch.Tensor,       # same shape as reward, e.g. gamma * cont_prob
    lam: float = 0.95,
    bootstrap: torch.Tensor | None = None,  # [B,1] or [H?] depending on layout
    time_major: bool = False,
) -> torch.Tensor:
    """
    Dreamer-style lambda return.

    If time_major=False (default): tensors are [B,H,1]
    If time_major=True: tensors are [H,B,1]

    Returns:
      returns: same shape as reward/value
    """
    if not time_major:
        # convert to [H,B,1] for easy backward recursion
        reward = reward.transpose(0, 1)
        value = value.transpose(0, 1)
        discount = discount.transpose(0, 1)

    H = reward.shape[0]

    if bootstrap is None:
        bootstrap = value[-1]  # [B,1]
    # ensure bootstrap shape [B,1]
    if bootstrap.ndim == 1:
        bootstrap = bootstrap.unsqueeze(-1)

    returns = torch.zeros_like(value)
    next_return = bootstrap  # G_{H} base

    # Backward recursion
    for t in reversed(range(H)):
        next_value = value[t + 1] if (t + 1) < H else bootstrap
        inp = reward[t] + discount[t] * ((1.0 - lam) * next_value + lam * next_return)
        returns[t] = inp
        next_return = inp

    if not time_major:
        returns = returns.transpose(0, 1)
    return returns
