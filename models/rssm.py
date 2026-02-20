# models/rssm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


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
        # stoch can be [B, T, C, K] or [B, C, K]
        # This keeps everything except the last two dims and merges them
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
        returns: [..., C, K] straight-through onehot sample (grad through probs)
        """
        # soft probs (with unimix)
        probs = self._unimix_probs(F.softmax(logits, -1))
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        dist = D.Categorical(probs=probs)
        idx = dist.sample()  # [..., C]
        onehot = F.one_hot(idx, self.K).to(logits.dtype)  # [..., C, K]

        # straight-through: forward hard, backward soft
        return onehot + probs - probs.detach()


    def straight_through_onehot(self, probs: torch.Tensor) -> torch.Tensor:
        """
        probs: [..., C, K] (assumed normalized)
        returns: [..., C, K] straight-through onehot sample (grad through probs)
        """
        probs = self._unimix_probs(probs)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()  # [..., C]
        onehot = F.one_hot(idx, self.K).to(probs.dtype)

        return onehot + probs - probs.detach()

    # ----------------------------
    # State init
    # ----------------------------
    def initial(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        probs = torch.full((batch_size, self.C, self.K), 1.0 / self.K, device=device)
        stoch = self.straight_through_onehot(probs)          # [B,C,K]
        return deter, stoch

    # ----------------------------
    # One-step transitions
    # ----------------------------

    def obs_step(self, prev_deter, prev_stoch, action, embed, goal):
        # # --- DEBUG PRINTS ---
        # # Only print on the first step of the first batch to avoid flooding
        # if not hasattr(self, '_debug_done'):
        #     print("\n" + "="*50)
        #     print("SHAPE DEBUGGER (Inside obs_step)")
        #     print(f"prev_deter shape: {prev_deter.shape}")
        #     print(f"prev_stoch shape: {prev_stoch.shape}")
        #     print(f"action shape:     {action.shape}")
        #     print(f"embed shape:      {embed.shape}")
        #     print(f"goal shape:       {goal.shape}")
            
        #     prev_stoch_flat_test = self.flatten_stoch(prev_stoch)
        #     print(f"prev_stoch_flat:  {prev_stoch_flat_test.shape}")
        #     print("="*50 + "\n")
        #     self._debug_done = True
        # # --------------------
        # 1. Capture the current batch size explicitly
        B = prev_deter.shape[0]

        # 2. Flatten stoch while FORCING it to keep the batch dimension
        # This prevents [16, 32, 32] from becoming [16384]
        prev_stoch_flat = self.flatten_stoch(prev_stoch).view(B, -1)
        
        # 3. Ensure action, embed, and goal also respect the batch dimension
        action_in = action.view(B, -1)
        embed_in = embed.view(B, -1)
        goal_in = goal.view(B, -1)
        deter_in = prev_deter.view(B, -1)
            
        # 4. Now cat is safe: [16, 1024] + [16, 2] -> [16, 1026]
        x = torch.cat([prev_stoch_flat, action_in], dim=-1)
        
        # 5. Update GRU
        deter = self.gru(x, deter_in)

        prior_logits_flat = self.prior_net(deter) 

        # 6. Cat for posterior: [16, 512] + [16, 1024] + [16, 2]
        post_in = torch.cat([deter, embed_in, goal_in], dim=-1)
        post_logits_flat = self.post_net(post_in)
        post_logits = self._reshape_logits(post_logits_flat)
        post_stoch = self.straight_through_onehot_from_logits(post_logits)

        return deter, post_stoch, post_logits_flat, prior_logits_flat

    def img_step(self, prev_deter, prev_stoch, action, temp=0.5, relaxed=True):
        B = prev_deter.shape[0]
        prev_stoch_flat = self.flatten_stoch(prev_stoch).view(B, -1)
        
        # Ensure batch dimension is preserved
        deter_in = prev_deter.view(B, -1)
        action_in = action.view(B, -1)

        x = torch.cat([prev_stoch_flat, action_in], dim=-1)
        deter = self.gru(x, deter_in)

        prior_logits_flat = self.prior_net(deter)
        prior_logits = self._reshape_logits(prior_logits_flat)
        prior_stoch = self.straight_through_onehot_from_logits(prior_logits)
        return deter, prior_stoch, prior_logits_flat

    
    def imagine(self, start_deter, start_stoch, actor, goal, horizon: int):
        deter = start_deter
        stoch = start_stoch

        deters = [deter]   # include s0
        stochs = [stoch]
        ents = []
        actions = []

        for _ in range(horizon):
            stoch_flat = self.flatten_stoch(stoch)
            action, logp, entropy, mean = actor(deter, stoch_flat, goal, sample=True)

            deter, stoch, _ = self.img_step(deter, stoch, action)

            actions.append(action)
            ents.append(entropy)
            deters.append(deter)   # now includes s1..sH
            stochs.append(stoch)

        return {
            "deter": torch.stack(deters, dim=1),  # [B, H+1, D]
            "stoch": torch.stack(stochs, dim=1),  # [B, H+1, C, K]
            "action": torch.stack(actions, dim=1),# [B, H, act]
            "ent": torch.stack(ents, dim=1),      # [B, H]
        }

    # ----------------------------
    # Rollouts over data
    # ----------------------------
    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor, resets=None):
        B, T, _ = embeds.shape
        device = embeds.device

        # Initialize hidden states [B, D] and [B, C, K]
        deter, stoch = self.initial(B, device=device)

        deters, stochs, post_logits, prior_logits = [], [], [], []

        for t in range(T):
            if resets is not None:
                # resets is [B, T]. Indexing [:, t] gives [B].
                # We MUST ensure r is [B, 1] for broadcasting.
                r = resets[:, t].to(device=device).float().view(B, 1) 
                keep = 1.0 - r
                
                # Get fresh initial states for just this batch size
                init_d, init_s = self.initial(B, device=device)
                
                # Apply reset: keep old state if r=0, use new state if r=1
                deter = deter * keep + init_d * r
                # stoch is [B, C, K], so r needs to be [B, 1, 1]
                stoch = stoch * keep.view(B, 1, 1) + init_s * r.view(B, 1, 1)

            # Ensure inputs to obs_step are strictly 2D [B, Features]
            # This prevents the 16386 error
            cur_action = actions[:, t].view(B, -1)
            cur_embed = embeds[:, t].view(B, -1)
            cur_goal = goals[:, t].view(B, -1)

            deter, stoch, post_l, prior_l = self.obs_step(
                deter, stoch, cur_action, cur_embed, cur_goal
            )
            deters.append(deter)
            stochs.append(stoch)
            post_logits.append(post_l)
            prior_logits.append(prior_l)
            # print(f"DEBUG Loop t={t}: deter={deter.shape}, stoch={stoch.shape}, action={actions[:,t].shape}")

        return {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
            "post_logits": torch.stack(post_logits, dim=1),
            "prior_logits": torch.stack(prior_logits, dim=1),
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