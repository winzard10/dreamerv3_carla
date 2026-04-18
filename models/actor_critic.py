import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Continuous actor with prev_action input.

    Low-dim signals (goal + prev_action = 4 dims) go through a small
    embedding layer before concat with the 1536-dim state — otherwise
    they get drowned out. Same design as RSSM's action_embed.

    forward() signature:
        action, log_prob, entropy, mean_action =
            actor(deter, stoch_flat, goal, prev_action, sample=True)
    """
    def __init__(
        self,
        deter_dim=512,
        stoch_dim=1024,
        goal_dim=2,
        action_dim=2,
        hidden_dim=512,
        min_std=0.1,
        init_std=1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std    = min_std
        self.goal_dim   = goal_dim

        # Embed low-dim signals (goal + prev_action) to 64 dims
        low_dim_in = goal_dim + action_dim
        self.low_dim_embed = nn.Sequential(
            nn.Linear(low_dim_in, 64),
            nn.ELU(),
        )

        trunk_in_dim = deter_dim + stoch_dim + 64

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head  = nn.Linear(hidden_dim, action_dim)

        # Initialize std bias so softplus(bias) + min_std ≈ init_std
        target = max(init_std - min_std, 1e-3)
        bias   = math.log(math.exp(target) - 1.0)
        nn.init.constant_(self.std_head.bias, bias)

    def forward(self, deter, stoch_flat, goal, prev_action, sample=True):
        # Embed low-dim signals together, then concat with state
        low_cat   = torch.cat([goal, prev_action], dim=-1)
        low_embed = self.low_dim_embed(low_cat)  # [B, 64]

        x = torch.cat([deter, stoch_flat, low_embed], dim=-1)
        h = self.trunk(x)

        mean = self.mean_head(h)
        std  = F.softplus(self.std_head(h)) + self.min_std
        dist = torch.distributions.Normal(mean, std)

        raw = dist.rsample() if sample else mean

        action  = torch.tanh(raw)
        log_det = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        log_prob = (dist.log_prob(raw) - log_det).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        mean_action = torch.tanh(mean)

        return action, log_prob, entropy, mean_action


class Critic(nn.Module):
    """
    Critic — estimates V(state). Does NOT take prev_action explicitly
    because the GRU already encodes action history through deter.
    Keeping critic simpler reduces overparameterization.
    """
    def __init__(
        self,
        deter_dim=512,
        stoch_dim=1024,
        goal_dim=2,
        hidden_dim=512,
        bins=255,
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

    def forward(self, deter, stoch_flat, goal):
        x = torch.cat([deter, stoch_flat, goal], dim=-1)
        return self.out(self.trunk(x))