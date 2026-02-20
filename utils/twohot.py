import math
import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    # symlog used by DreamerV3
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)

class TwoHotDist:
    """
    Two-hot distribution over a fixed support for (symlog) scalars.

    - The model outputs logits over `num_bins`.
    - Targets are scalars (typically symlog(reward) or symlog(return)).
    - We convert target scalars into a "two-hot" probability distribution
      by linearly interpolating between the two nearest bins.

    Default support is uniform in symlog-space on [vmin, vmax].

    Usage:
      dist = TwoHotDist(num_bins=255, vmin=-20.0, vmax=20.0)
      target_probs = dist.two_hot(target_symlog)          # [..., num_bins]
      loss = dist.ce_loss(logits, target_symlog)          # scalar
      value = dist.mean(logits)                           # [...]
    """
    def __init__(
        self,
        num_bins: int = 255,
        vmin: float = -20.0,
        vmax: float = 20.0,
        device=None,
        dtype=torch.float32,
        eps: float = 1e-8,
        support_type: str = "uniform",  # or "log"
    ):
        assert num_bins >= 2
        assert vmax > vmin
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.eps = float(eps)

        self.device = device
        self.dtype = dtype

        # support: [num_bins]
        self.support_type = support_type

        if support_type == "uniform":
            self.support = torch.linspace(vmin, vmax, num_bins, device=device, dtype=dtype)
        elif support_type == "log":
            # log support in symlog-space, denser near 0
            # map u in [-1,1] -> symlog-space via sinh-like curve
            u = torch.linspace(-1.0, 1.0, num_bins, device=device, dtype=dtype)
            # scale to [vmin,vmax] in a smooth way
            # (this is a simple, stable version; we can tweak later)
            self.support = (u.sign() * (u.abs() ** 3))  # denser near 0
            self.support = (self.support - self.support.min()) / (self.support.max() - self.support.min() + eps)
            self.support = vmin + self.support * (vmax - vmin)
            self.support, _ = torch.sort(self.support)
        else:
            raise ValueError(f"Unknown support_type: {support_type}")

        # IMPORTANT: delta only valid for uniform support
        if support_type == "uniform":
            self.delta = (vmax - vmin) / (num_bins - 1)
        else:
            self.delta = None


    def to(self, device=None, dtype=None):
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        self.device = device
        self.dtype = dtype
        self.support = self.support.to(device=device, dtype=dtype)
        return self

    def _clamp_target(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.vmin, self.vmax)

    def two_hot(self, target: torch.Tensor) -> torch.Tensor:
        target = target.to(device=self.support.device, dtype=self.support.dtype)
        t = self._clamp_target(target)

        # find insertion index in sorted support
        # idx_high in [0..num_bins-1], idx_low = idx_high-1
        idx_high = torch.searchsorted(self.support, t)
        idx_high = idx_high.clamp(0, self.num_bins - 1)
        idx_low = (idx_high - 1).clamp(0, self.num_bins - 1)

        v_low = self.support[idx_low]
        v_high = self.support[idx_high]
        denom = (v_high - v_low).clamp(min=self.eps)

        w_high = ((t - v_low) / denom).clamp(0.0, 1.0)
        w_low = 1.0 - w_high

        same = (idx_low == idx_high).to(w_low.dtype)
        w_low = w_low + same * w_high
        w_high = w_high * (1.0 - same)

        probs = torch.zeros(*t.shape, self.num_bins, device=t.device, dtype=t.dtype)
        probs.scatter_add_(-1, idx_low.unsqueeze(-1), w_low.unsqueeze(-1))
        probs.scatter_add_(-1, idx_high.unsqueeze(-1), w_high.unsqueeze(-1))
        probs = probs / (probs.sum(dim=-1, keepdim=True) + self.eps)
        return probs


    def log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [..., num_bins]
        returns: [..., num_bins] log-probabilities
        """
        return F.log_softmax(logits, dim=-1)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        returns scalar mean in symlog-space: [...]
        """
        p = self.probs(logits)
        # support broadcast to logits shape
        return (p * self.support.view(*([1] * (p.ndim - 1)), -1)).sum(dim=-1)
    
    def log_prob(self, logits: torch.Tensor, target_symlog: torch.Tensor) -> torch.Tensor:
        target_probs = self.two_hot(target_symlog)
        logp = self.log_probs(logits)
        return (target_probs * logp).sum(dim=-1)   # [...], already log-prob

    def nll(self, logits: torch.Tensor, target_symlog: torch.Tensor, reduction="mean") -> torch.Tensor:
        nll = -self.log_prob(logits, target_symlog)
        if reduction == "mean": return nll.mean()
        if reduction == "sum": return nll.sum()
        if reduction == "none": return nll
        raise ValueError(reduction)

    def ce_loss(
        self,
        logits: torch.Tensor,
        target_symlog: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Cross-entropy between two-hot(target) and predicted logits.
        logits: [..., num_bins]
        target_symlog: [...] (same leading shape as logits without last dim)
        """
        target_probs = self.two_hot(target_symlog)  # [..., num_bins]
        logp = self.log_probs(logits)               # [..., num_bins]
        loss = -(target_probs * logp).sum(dim=-1)   # [...]

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {reduction}")
