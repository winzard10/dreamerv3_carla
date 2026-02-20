import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.twohot import TwoHotDist, symlog, symexp

class Actor(nn.Module):
    """
    DreamerV3-ish continuous actor:
      - outputs mean and std of a Normal
      - samples with rsample()
      - squashes with tanh
      - returns action, log_prob, entropy, mean_action
    """
    def __init__(
        self,
        deter_dim=512,
        stoch_dim=1024,
        goal_dim=0,          # set 0 for faithful; >0 if you want goal-conditioned
        action_dim=2,
        hidden_dim=512,
        min_std=0.1,
        init_std=1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
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
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head  = nn.Linear(hidden_dim, action_dim)

        # initialize std near init_std (in "pre-softplus" space)
        # softplus(bias) + min_std ≈ init_std  =>  softplus(bias) ≈ init_std - min_std
        target = max(init_std - min_std, 1e-3)
        # inverse softplus approx: x = log(exp(y)-1)
        bias = math.log(math.exp(target) - 1.0)
        nn.init.constant_(self.std_head.bias, bias)

    @staticmethod
    def _tanh_squash(raw, eps=1e-6):
        action = torch.tanh(raw)
        # stable log(1 - tanh(raw)^2)
        log_det = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        return action, log_det

    def forward(self, deter, stoch_flat, goal=None, sample=True):    
        if self.goal_dim > 0:
            assert goal is not None, "Actor expects goal when goal_dim > 0"
            x = torch.cat([deter, stoch_flat, goal], dim=-1)
        else:
            x = torch.cat([deter, stoch_flat], dim=-1)

        h = self.trunk(x)
        mean = self.mean_head(h)

        # V3-ish: std = softplus(std_head) + min_std
        std = F.softplus(self.std_head(h)) + self.min_std

        dist = torch.distributions.Normal(mean, std)

        if sample:
            raw = dist.rsample()
        else:
            raw = mean  # deterministic

        action = torch.tanh(raw)
        log_det = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        log_prob = (dist.log_prob(raw) - log_det).sum(dim=-1)
        base_ent = dist.entropy().sum(dim=-1)
        entropy = base_ent  # cheap & stable approximation
        mean_action = torch.tanh(mean)
        return action, log_prob, entropy, mean_action

class Critic(nn.Module):
    def __init__(
        self,
        deter_dim=512,
        stoch_dim=1024,
        goal_dim=0,
        hidden_dim=512,
        bins=255,
        vmin=-5.0,
        vmax=5.0,
        eps=1e-8,
        support_type="uniform",  # "uniform" or "log"
    ):
        super().__init__()
        assert bins >= 2
        assert vmax > vmin
        self.goal_dim = goal_dim
        self.bins = int(bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.eps = float(eps)
        self.support_type = support_type

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

        # --- register support as buffer so it moves with .to(device) ---
        if support_type == "uniform":
            support = torch.linspace(vmin, vmax, bins)
        elif support_type == "log":
            u = torch.linspace(-1.0, 1.0, bins)
            support = u.sign() * (u.abs() ** 3)  # denser near 0
            support = (support - support.min()) / (support.max() - support.min() + eps)
            support = vmin + support * (vmax - vmin)
            support, _ = torch.sort(support)
        else:
            raise ValueError(f"Unknown support_type: {support_type}")

        self.register_buffer("support", support)  # [bins]

    def forward(self, deter, stoch_flat, goal=None):
        if self.goal_dim > 0:
            assert goal is not None, "Critic expects goal when goal_dim > 0"
            x = torch.cat([deter, stoch_flat, goal], dim=-1)
        else:
            x = torch.cat([deter, stoch_flat], dim=-1)

        h = self.trunk(x)
        return self.out(h)  # [..., bins]

    # ----- distribution helpers -----

    def _clamp_target(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.vmin, self.vmax)

    def log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(logits, dim=-1)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def mean_symlog(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mean in symlog-space: [...]
        """
        p = self.probs(logits)
        return (p * self.support.view(*([1] * (p.ndim - 1)), -1)).sum(dim=-1)

    def mean_value(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mean in raw value space: [...]
        """
        return symexp(self.mean_symlog(logits))

    def loss(self, logits: torch.Tensor, target_value: torch.Tensor, reduction="mean") -> torch.Tensor:
        """
        Cross-entropy loss against two-hot(symlog(target_value)).
        target_value: [...] in raw space
        """
        target_symlog = symlog(target_value)
        target_probs = self.two_hot(target_symlog)   # [..., bins]
        logp = self.log_probs(logits)                # [..., bins]
        loss = -(target_probs * logp).sum(dim=-1)    # [...]

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(reduction)