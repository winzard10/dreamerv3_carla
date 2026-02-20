import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.twohot import symlog, symexp

class RewardHead(nn.Module):
    def __init__(
        self,
        deter_dim=512,
        stoch_dim=1024,
        goal_dim=0,
        hidden_dim=512,
        bins=255,
        vmin=-20.0,
        vmax=20.0,
        eps=1e-8,
        support_type="uniform",
    ):
        super().__init__()
        assert bins >= 2 and vmax > vmin
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

        # register support buffer
        if support_type == "uniform":
            support = torch.linspace(vmin, vmax, bins)
        elif support_type == "log":
            u = torch.linspace(-1.0, 1.0, bins)
            support = u.sign() * (u.abs() ** 3)
            support = (support - support.min()) / (support.max() - support.min() + eps)
            support = vmin + support * (vmax - vmin)
            support, _ = torch.sort(support)
        else:
            raise ValueError(f"Unknown support_type: {support_type}")

        self.register_buffer("support", support)  # [bins]

    def forward(self, deter, stoch_flat, goal=None):
        if self.goal_dim > 0:
            assert goal is not None
            x = torch.cat([deter, stoch_flat, goal], dim=-1)
        else:
            x = torch.cat([deter, stoch_flat], dim=-1)
        return self.out(self.trunk(x))  # [..., bins]

    def _clamp_target(self, x):
        return torch.clamp(x, self.vmin, self.vmax)

    def two_hot(self, target_symlog: torch.Tensor) -> torch.Tensor:
        t = self._clamp_target(target_symlog.to(dtype=self.support.dtype, device=self.support.device))

        idx_high = torch.searchsorted(self.support, t).clamp(0, self.bins - 1)
        idx_low = (idx_high - 1).clamp(0, self.bins - 1)

        v_low = self.support[idx_low]
        v_high = self.support[idx_high]
        denom = (v_high - v_low).clamp(min=self.eps)

        w_high = ((t - v_low) / denom).clamp(0.0, 1.0)
        w_low = 1.0 - w_high

        same = (idx_low == idx_high).to(w_low.dtype)
        w_low = w_low + same * w_high
        w_high = w_high * (1.0 - same)

        probs = torch.zeros(*t.shape, self.bins, device=t.device, dtype=t.dtype)
        probs.scatter_add_(-1, idx_low.unsqueeze(-1), w_low.unsqueeze(-1))
        probs.scatter_add_(-1, idx_high.unsqueeze(-1), w_high.unsqueeze(-1))
        return probs / (probs.sum(dim=-1, keepdim=True) + self.eps)

    def mean_symlog(self, logits):
        p = F.softmax(logits, dim=-1)
        return (p * self.support.view(*([1] * (p.ndim - 1)), -1)).sum(dim=-1)

    def mean_reward(self, logits):
        return symexp(self.mean_symlog(logits))

    def loss(self, logits, reward, reduction="mean"):
        target_symlog = symlog(reward)
        target_probs = self.two_hot(target_symlog)
        logp = F.log_softmax(logits, dim=-1)
        loss = -(target_probs * logp).sum(dim=-1)
        if reduction == "mean": return loss.mean()
        if reduction == "sum": return loss.sum()
        if reduction == "none": return loss
        raise ValueError(reduction)