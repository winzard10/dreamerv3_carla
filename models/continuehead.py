import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinueHead(nn.Module):
    """
    Predicts continuation probability c_t = P(not_done_t) as a Bernoulli(logits).
    DreamerV3 uses this as 'discount' = gamma * c_t inside lambda-returns.
    """
    def __init__(
        self,
        deter_dim=512,
        stoch_dim=1024,
        goal_dim=0,
        hidden_dim=512,
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
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, deter, stoch_flat, goal=None):
        if self.goal_dim > 0:
            assert goal is not None, "ContinueHead expects goal when goal_dim > 0"
            x = torch.cat([deter, stoch_flat, goal], dim=-1)
        else:
            x = torch.cat([deter, stoch_flat], dim=-1)

        h = self.trunk(x)
        return self.out(h)  # [..., 1] logits

    def prob(self, logits):
        return torch.sigmoid(logits)  # [..., 1]

    def loss(self, logits, not_done_target):
        """
        not_done_target: [..., 1] float in {0,1} or [0,1]
        Use BCE with logits (more stable than sigmoid + BCE).
        """
        return F.binary_cross_entropy_with_logits(logits, not_done_target, reduction="mean")
