# models/rssm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


class RSSM(nn.Module):
    def __init__(
        self,
        deter_dim: int = 512,
        act_dim: int = 2,
        embed_dim: int = 1024,
        goal_dim: int = 2,
        stoch_categoricals: int = 32,
        stoch_classes: int = 32,
        unimix_ratio: float = 0.01,
        kl_balance: float = 0.8,
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

        self.num_blocks = 8
        assert self.deter_dim % self.num_blocks == 0
        self.block_dim = self.deter_dim // self.num_blocks

        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_dim + self.act_dim + self.goal_dim, self.deter_dim),
            nn.ELU(),
            nn.Linear(self.deter_dim, self.deter_dim),
            nn.ELU(),
        )

        self.gru_blocks = nn.ModuleList([
            nn.GRUCell(self.block_dim, self.block_dim)
            for _ in range(self.num_blocks)
        ])

        hidden = self.deter_dim * 2

        # self.prior_net = nn.Sequential(
        #     nn.Linear(self.deter_dim, hidden),
        #     nn.ELU(),
        #     nn.Linear(hidden, hidden),
        #     nn.ELU(),
        #     nn.Linear(hidden, self.stoch_dim),
        # )
        
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.stoch_dim),
        )

        self.post_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim + self.goal_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.stoch_dim),
        )
    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _reshape_logits(self, logits_flat: torch.Tensor) -> torch.Tensor:
        return logits_flat.view(*logits_flat.shape[:-1], self.C, self.K)

    def flatten_stoch(self, stoch: torch.Tensor) -> torch.Tensor:
        return stoch.reshape(*stoch.shape[:-2], self.stoch_dim)

    def _unimix_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if self.unimix_ratio <= 0.0:
            return probs
        uni = torch.full_like(probs, 1.0 / self.K)
        probs = (1.0 - self.unimix_ratio) * probs + self.unimix_ratio * uni
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    def dist_from_logits_flat(self, logits_flat: torch.Tensor):
        logits = self._reshape_logits(logits_flat)
        probs = F.softmax(logits, dim=-1)
        probs = self._unimix_probs(probs)
        logp = torch.log(probs + 1e-8)
        return logits, probs, logp

    def straight_through_onehot_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self._unimix_probs(F.softmax(logits, -1))
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        idx = D.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.K).to(logits.dtype)
        return onehot + probs - probs.detach()

    def straight_through_onehot(self, probs: torch.Tensor) -> torch.Tensor:
        probs = self._unimix_probs(probs)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        idx = D.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.K).to(probs.dtype)
        return onehot + probs - probs.detach()

    # def _gru_step(self, prev_stoch_flat, action_in, goal_in, deter_in):
    #     x = torch.cat([prev_stoch_flat, action_in, goal_in], dim=-1)
    #     x = self.pre_gru(x)
    #     return self.gru(x, deter_in)
    
    def _gru_step(self, prev_stoch_flat, action_in, goal_in, deter_in):
        x = torch.cat([prev_stoch_flat, action_in, goal_in], dim=-1)
        x = self.pre_gru(x)  # [B, deter_dim]

        x_blocks = torch.chunk(x, self.num_blocks, dim=-1)
        h_blocks = torch.chunk(deter_in, self.num_blocks, dim=-1)

        next_blocks = [
            gru(xb, hb)
            for gru, xb, hb in zip(self.gru_blocks, x_blocks, h_blocks)
        ]

        return torch.cat(next_blocks, dim=-1)
    
    def initial(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        probs = torch.full((batch_size, self.C, self.K), 1.0 / self.K, device=device)
        stoch = self.straight_through_onehot(probs)
        return deter, stoch

    def obs_step(self, prev_deter, prev_stoch, action, embed, goal):
        B = prev_deter.shape[0]
        prev_stoch_flat = self.flatten_stoch(prev_stoch).view(B, -1)
        action_in = action.view(B, -1)
        embed_in = embed.view(B, -1)
        goal_in = goal.view(B, -1)
        deter_in = prev_deter.view(B, -1)

        deter = self._gru_step(prev_stoch_flat, action_in, goal_in, deter_in)
        prior_logits_flat = self.prior_net(deter)

        post_in = torch.cat([deter, embed_in, goal_in], dim=-1)
        post_logits_flat = self.post_net(post_in)
        post_logits = self._reshape_logits(post_logits_flat)
        post_stoch = self.straight_through_onehot_from_logits(post_logits)

        return deter, post_stoch, post_logits_flat, prior_logits_flat

    def img_step(self, prev_deter, prev_stoch, action, goal):
        B = prev_deter.shape[0]
        prev_stoch_flat = self.flatten_stoch(prev_stoch).view(B, -1)
        deter_in = prev_deter.view(B, -1)
        action_in = action.view(B, -1)
        goal_in = goal.view(B, -1)

        deter = self._gru_step(prev_stoch_flat, action_in, goal_in, deter_in)

        prior_logits_flat = self.prior_net(deter)
        prior_logits = self._reshape_logits(prior_logits_flat)
        prior_stoch = self.straight_through_onehot_from_logits(prior_logits)

        return deter, prior_stoch, prior_logits_flat

    # -----------------------------------------------------------------------
    # Rollouts
    # -----------------------------------------------------------------------

    def observe(self, embeds, actions, goals, resets=None):
        B, T, _ = embeds.shape
        device = embeds.device
        deter, stoch = self.initial(B, device=device)
        deters, stochs, post_logits_list, prior_logits_list = [], [], [], []

        for t in range(T):
            if resets is not None:
                r = resets[:, t].to(device=device).float().view(B, 1)
                keep = 1.0 - r
                init_d, init_s = self.initial(B, device=device)
                deter = deter * keep + init_d * r
                stoch = stoch * keep.view(B, 1, 1) + init_s * r.view(B, 1, 1)

            cur_action = actions[:, t].view(B, -1)
            cur_embed = embeds[:, t].view(B, -1)
            cur_goal = goals[:, t].view(B, -1)

            deter, stoch, post_l, prior_l = self.obs_step(
                deter, stoch, cur_action, cur_embed, cur_goal
            )
            deters.append(deter)
            stochs.append(stoch)
            post_logits_list.append(post_l)
            prior_logits_list.append(prior_l)

        return {
            "deter":        torch.stack(deters, dim=1),
            "stoch":        torch.stack(stochs, dim=1),
            "post_logits":  torch.stack(post_logits_list, dim=1),
            "prior_logits": torch.stack(prior_logits_list, dim=1),
        }

    def imagine(self, start_deter, start_stoch, actor, goal, horizon: int):
        deter = start_deter
        stoch = start_stoch
        deters = [deter]
        stochs = [stoch]
        ents = []
        actions = []

        for _ in range(horizon):
            stoch_flat = self.flatten_stoch(stoch)
            action, logp, entropy, mean = actor(deter, stoch_flat, goal, sample=True)
            deter, stoch, _ = self.img_step(deter, stoch, action, goal)
            actions.append(action)
            ents.append(entropy)
            deters.append(deter)
            stochs.append(stoch)

        return {
            "deter":  torch.stack(deters, dim=1),   # [B, H+1, D]
            "stoch":  torch.stack(stochs, dim=1),   # [B, H+1, C, K]
            "action": torch.stack(actions, dim=1),  # [B, H, A]
            "ent":    torch.stack(ents, dim=1),     # [B, H]
        }

    # -----------------------------------------------------------------------
    # Losses
    # -----------------------------------------------------------------------

    def kl_loss(self, post_logits_flat, prior_logits_flat):
        _, post_probs, post_logp   = self.dist_from_logits_flat(post_logits_flat)
        _, prior_probs, prior_logp = self.dist_from_logits_flat(prior_logits_flat)

        # Balanced KL: alpha trains prior (rep), (1-alpha) trains posterior (dyn)
        kl_rep = (stopgrad(post_probs) * (stopgrad(post_logp) - prior_logp)).sum(-1).sum(-1)
        kl_dyn = (post_probs * (post_logp - stopgrad(prior_logp))).sum(-1).sum(-1)

        kl = self.kl_balance * kl_rep + (1.0 - self.kl_balance) * kl_dyn
        kl = torch.clamp(kl, min=self.free_nats)
        return kl.mean()

    def overshooting_loss(self, deter_seq, stoch_seq, actions_seq, goals_seq,
                          post_logits_bt, k=3):
        B, T, _ = deter_seq.shape
        device = deter_seq.device
        losses = []
        max_k = min(k, T - 1)
        if max_k <= 0:
            return torch.zeros((), device=device)

        for t in range(T - 1):
            deter = deter_seq[:, t]
            stoch = stoch_seq[:, t]
            rollout_limit = min(max_k, T - 1 - t)
            for j in range(1, rollout_limit + 1):
                deter, stoch, prior_logits_flat = self.img_step(
                    deter, stoch,
                    actions_seq[:, t + j],
                    goals_seq[:, t + j],
                )
                target = post_logits_bt[:, t + j]
                losses.append(self.kl_loss(
                    target.unsqueeze(1),
                    prior_logits_flat.unsqueeze(1),
                ))

        if not losses:
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()