# models/rssm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


class RSSM(nn.Module):
    """
    DreamerV3-style discrete RSSM:
      - stochastic state: C categoricals with K classes (onehot / probs)
      - deterministic state: GRU
      - prior: p(z_t | h_t)
      - posterior: q(z_t | h_t, embed_t, goal_t)

    Features:
      - unimix on categorical probabilities
      - straight-through onehot sampling (no gumbel)
      - KL balancing
      - free nats
    """
    def __init__(
        self,
        deter_dim: int = 512,
        act_dim: int = 2,
        embed_dim: int = 1024,
        goal_dim: int = 2,
        stoch_categoricals: int = 32,  # C
        stoch_classes: int = 32,       # K
        unimix_ratio: float = 0.01,
        kl_balance: float = 0.8,       # alpha
        free_nats: float = 1.0,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.goal_dim = goal_dim

        self.C = int(stoch_categoricals)
        self.K = int(stoch_classes)
        self.stoch_dim = self.C * self.K

        self.unimix_ratio = float(unimix_ratio)
        self.kl_balance = float(kl_balance)
        self.free_nats = float(free_nats)

        # GRU input: prev stochastic (flattened) + action
        self.gru = nn.GRUCell(self.stoch_dim + self.act_dim, self.deter_dim)

        # Prior logits from deter
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, self.deter_dim),
            nn.ELU(),
            nn.Linear(self.deter_dim, self.stoch_dim),
        )

        # Posterior logits from (deter + embed + goal)
        self.post_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim + self.goal_dim, self.deter_dim),
            nn.ELU(),
            nn.Linear(self.deter_dim, self.stoch_dim),
        )

    # ----------------------------
    # Helpers: shapes + unimix
    # ----------------------------
    def _reshape_logits(self, logits_flat: torch.Tensor) -> torch.Tensor:
        # [..., C*K] -> [..., C, K]
        return logits_flat.view(*logits_flat.shape[:-1], self.C, self.K)

    def flatten_stoch(self, stoch: torch.Tensor) -> torch.Tensor:
        # [..., C, K] -> [..., C*K]
        return stoch.reshape(*stoch.shape[:-2], self.stoch_dim)

    def _unimix_probs(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: [..., C, K]
        if self.unimix_ratio <= 0.0:
            return probs
        uni = torch.full_like(probs, 1.0 / self.K)
        probs = (1.0 - self.unimix_ratio) * probs + self.unimix_ratio * uni
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    def dist_from_logits_flat(self, logits_flat: torch.Tensor):
        """
        logits_flat: [..., C*K]
        returns:
          logits: [..., C, K]
          probs:  [..., C, K] (after unimix)
          logp:   [..., C, K] (log probs after unimix)
        """
        logits = self._reshape_logits(logits_flat)
        probs = F.softmax(logits, dim=-1)
        probs = self._unimix_probs(probs)
        logp = torch.log(probs + 1e-8)
        return logits, probs, logp

    def straight_through_onehot_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [..., C, K]
        Returns ST onehot sample with probabilities defined by unimixed softmax(logits).
        """
        probs = F.softmax(logits, dim=-1)
        probs = self._unimix_probs(probs)

        # sample from the *actual* probs (after unimix)
        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()                              # [..., C]
        onehot = F.one_hot(idx, self.K).to(logits.dtype) # [..., C, K]

        # straight-through trick
        return onehot + probs - probs.detach()

    # ----------------------------
    # State init
    # ----------------------------
    def initial(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        # uniform logits => uniform probs; ST sample is fine
        init_logits = torch.zeros(batch_size, self.C, self.K, device=device)
        stoch = self.straight_through_onehot_from_logits(init_logits)
        return deter, stoch

    # ----------------------------
    # One-step transitions
    # ----------------------------
    def img_step(self, prev_deter: torch.Tensor, prev_stoch: torch.Tensor, action: torch.Tensor):
        """
        Prior transition:
          h_t = GRU([prev_stoch_flat, action], prev_deter)
          z_t ~ p(z_t | h_t)
        Returns:
          deter: [B, D]
          stoch: [B, C, K]
          prior_logits_flat: [B, C*K]
        """
        prev_stoch_flat = self.flatten_stoch(prev_stoch)
        deter = self.gru(torch.cat([prev_stoch_flat, action], dim=-1), prev_deter)

        prior_logits_flat = self.prior_net(deter)              # [B, C*K]
        prior_logits = self._reshape_logits(prior_logits_flat) # [B, C, K]
        prior_stoch = self.straight_through_onehot_from_logits(prior_logits)

        return deter, prior_stoch, prior_logits_flat

    def obs_step(
        self,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor,
        action: torch.Tensor,
        embed: torch.Tensor,
        goal: torch.Tensor,
    ):
        """
        Posterior update:
          deter = GRU transition
          prior_logits_flat = prior(deter)
          post_logits_flat  = post(deter, embed, goal)
          z_post ~ q(z | ...)
        Returns:
          deter, post_stoch, post_logits_flat, prior_logits_flat
        """
        prev_stoch_flat = self.flatten_stoch(prev_stoch)
        deter = self.gru(torch.cat([prev_stoch_flat, action], dim=-1), prev_deter)

        prior_logits_flat = self.prior_net(deter)
        post_in = torch.cat([deter, embed, goal], dim=-1)
        post_logits_flat = self.post_net(post_in)

        post_logits = self._reshape_logits(post_logits_flat)
        post_stoch = self.straight_through_onehot_from_logits(post_logits)

        return deter, post_stoch, post_logits_flat, prior_logits_flat

    # ----------------------------
    # Rollouts over data
    # ----------------------------
    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor, resets=None):
        """
        embeds:  [B, T, E]
        actions: [B, T, A]
        goals:   [B, T, G]
        resets:  [B, T] bool (optional): reset before step t
        """
        B, T, _ = embeds.shape
        device = embeds.device

        deter, stoch = self.initial(B, device=device)

        deters, stochs, post_logits, prior_logits = [], [], [], []

        for t in range(T):
            if resets is not None:
                r = resets[:, t].to(device=device).float().unsqueeze(-1)  # [B,1]
                keep = 1.0 - r
                init_d, init_s = self.initial(B, device=device)
                deter = deter * keep + init_d * r
                stoch = stoch * keep[:, None, None] + init_s * r[:, None, None]

            deter, stoch, post_l, prior_l = self.obs_step(
                deter, stoch, actions[:, t], embeds[:, t], goals[:, t]
            )
            deters.append(deter)
            stochs.append(stoch)
            post_logits.append(post_l)
            prior_logits.append(prior_l)

        return {
            "deter": torch.stack(deters, dim=1),         # [B,T,D]
            "stoch": torch.stack(stochs, dim=1),         # [B,T,C,K]
            "post_logits": torch.stack(post_logits, dim=1),   # [B,T,C*K]
            "prior_logits": torch.stack(prior_logits, dim=1), # [B,T,C*K]
        }

    def imagine(self, start_deter: torch.Tensor, start_stoch: torch.Tensor, actor, goal: torch.Tensor, horizon: int):
        """
        Dream rollout using prior dynamics only.
        Returns:
          deter: [B, H, D]
          stoch: [B, H, C, K]
          ent:   [B, H]
        """
        deter = start_deter
        stoch = start_stoch

        deters, stochs, ents = [], [], []

        for _ in range(horizon):
            stoch_flat = self.flatten_stoch(stoch)
            action, logp, ent, _ = actor(deter, stoch_flat, goal, sample=True)

            deter, stoch, _ = self.img_step(deter, stoch, action)

            deters.append(deter)
            stochs.append(stoch)
            ents.append(ent)

        return {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
            "ent": torch.stack(ents, dim=1),
        }

    # ----------------------------
    # KL loss (balanced + free nats)
    # ----------------------------
    def kl_loss(self, post_logits_flat: torch.Tensor, prior_logits_flat: torch.Tensor) -> torch.Tensor:
        """
        post_logits_flat:  [B, T, C*K]
        prior_logits_flat: [B, T, C*K]
        Returns scalar.
        """
        _, post_probs, post_logp = self.dist_from_logits_flat(post_logits_flat)   # [B,T,C,K]
        _, prior_probs, prior_logp = self.dist_from_logits_flat(prior_logits_flat)

        # KL(q||p) per categorical then sum over C
        # Balanced KL:
        #   alpha * KL(stopgrad(q) || p) + (1-alpha) * KL(q || stopgrad(p))
        kl_rep = (stopgrad(post_probs) * (stopgrad(post_logp) - prior_logp)).sum(dim=-1).sum(dim=-1)  # [B,T]
        kl_dyn = (post_probs * (post_logp - stopgrad(prior_logp))).sum(dim=-1).sum(dim=-1)            # [B,T]

        kl = self.kl_balance * kl_rep + (1.0 - self.kl_balance) * kl_dyn

        # free nats (applied per timestep)
        kl = torch.clamp(kl, min=self.free_nats)
        return kl.mean()