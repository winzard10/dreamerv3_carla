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
    # force [B,1]
    bootstrap = bootstrap.view(bootstrap.shape[0], 1)
    
    assert reward.shape == value.shape == discount.shape
    assert reward.ndim == 3 and reward.shape[-1] == 1

    returns = torch.zeros_like(value)
    value_tp1 = torch.cat([value[1:], bootstrap.unsqueeze(0)], dim=0)  # [H,B,1]
    next_return = bootstrap
    for t in reversed(range(H)):
        next_value = value_tp1[t]
        next_return = reward[t] + discount[t] * ((1-lam)*next_value + lam*next_return)
        returns[t] = next_return

    if not time_major:
        returns = returns.transpose(0, 1)
    return returns
