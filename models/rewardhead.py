# models/rewardhead.py
import torch
import torch.nn as nn


class RewardHead(nn.Module):
    """
    Predicts reward distribution logits over a fixed TwoHot support.

    Training loss and mean extraction are handled externally by TwoHotDist:
      - loss:  twohot.ce_loss(reward_head(deter, stoch, goal), symlog(reward))
      - mean:  symexp(twohot.mean(reward_head(deter, stoch, goal)))
    """
    def __init__(
        self,
        deter_dim: int = 512,
        stoch_dim: int = 1024,
        goal_dim: int = 0,
        hidden_dim: int = 512,
        bins: int = 255,
        vmin: float = -20.0,
        vmax: float = 20.0,
    ):
        super().__init__()
        self.goal_dim = goal_dim
        in_dim = deter_dim + stoch_dim + goal_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.out = nn.Linear(hidden_dim, bins)

    def forward(self, deter, stoch_flat, goal=None):
        if self.goal_dim > 0:
            assert goal is not None, "RewardHead expects goal when goal_dim > 0"
            x = torch.cat([deter, stoch_flat, goal], dim=-1)
        else:
            x = torch.cat([deter, stoch_flat], dim=-1)
        return self.out(self.trunk(x))  # [..., bins]