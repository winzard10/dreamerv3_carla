# utils/train_utils.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from params import (
    DEVICE, H, W, NUM_CLASSES,
    GAMMA, LAMBDA, IMAG_HORIZON,
    SEM_SCALE, REWARD_SCALE, CONT_SCALE, KL_SCALE, ENT_SCALE,
    GOAL_SCALE, VEC_SCALE,
    OVERSHOOT_K, OVERSHOOT_SCALE,
    TARGET_EMA,
)
from utils.lambda_returns import lambda_return
from utils.twohot import symlog, symexp


# =============================================================================
# General utilities
# =============================================================================

def ema_update(target: torch.nn.Module, online: torch.nn.Module, tau: float):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))


def gaussian_nll(x, mean, std=0.1, eps=1e-6):
    var = std ** 2 + eps
    return 0.5 * ((x - mean) ** 2) / var + torch.log(torch.tensor(std + eps, device=x.device))


def make_resets_from_dones(dones_seq: torch.Tensor) -> torch.Tensor:
    resets = torch.zeros_like(dones_seq, dtype=torch.bool)
    resets[:, 1:] = dones_seq[:, :-1]
    return resets


def preprocess_batch(depths, sems, vectors, goals, actions, rewards, dones):
    B, T, C, H_, W_ = depths.shape
    depth_in    = depths.reshape(B * T, 1, H, W).float() / 255.0
    sem_ids     = sems.reshape(B * T, H, W).long().clamp(0, NUM_CLASSES - 1)
    vec_in      = vectors.reshape(B * T, -1).float()
    goal_in     = goals.reshape(B * T, -1).float()
    actions_seq = actions.float()
    rewards_seq = rewards.float()
    dones_seq   = dones.bool()
    goals_seq   = goals.float()
    return depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq


def clone_batch_to_cpu(batch):
    return tuple(x.detach().cpu().clone() for x in batch)


def make_strip(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        images = images.squeeze(1)
    return torch.cat(list(images), dim=-1).unsqueeze(0)


def semantic_to_vis(sem_ids: torch.Tensor) -> torch.Tensor:
    return sem_ids.float() / float(NUM_CLASSES - 1)


def save_checkpoint(path, models, opts, global_step, episode=None):
    d = {k: v.state_dict() for k, v in models.items()}
    d.update({k + "_opt": v.state_dict() for k, v in opts.items()})
    d["global_step"] = global_step
    if episode is not None:
        d["episode"] = episode
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(d, path)


# =============================================================================
# Visualization helpers
# =============================================================================

@torch.no_grad()
def rollout_seeded_prior(rssm, post_deter_seq, post_stoch_seq,
                          actions_seq, goals_seq, resets=None):
    """
    1-step prior prediction at each timestep.
    For each t, seeds from posterior at t-1 and does a single img_step.
    Avoids compounding errors from chained prior rollouts.

    t=0: no predecessor — use prior_net(post_deter_t0) as stand-in.
    t>0: seed from posterior at t-1, one img_step → prior at t.
    """
    B, T, D = post_deter_seq.shape
    device  = post_deter_seq.device

    prior_deters = []
    prior_stochs = []
    prior_logits = []

    # t=0: no predecessor — use posterior deter as stand-in
    prior_deters.append(post_deter_seq[:, 0])
    prior_stochs.append(post_stoch_seq[:, 0])
    prior_logits.append(rssm.prior_net(post_deter_seq[:, 0]))

    for t in range(1, T):
        # Always seed fresh from posterior at t-1 — no error accumulation
        deter = post_deter_seq[:, t - 1]
        stoch = post_stoch_seq[:, t - 1]

        # Re-seed from posterior at episode resets
        if resets is not None:
            r = resets[:, t].to(device).bool()
            if r.any():
                deter = torch.where(r.unsqueeze(-1), post_deter_seq[:, t], deter)
                stoch = torch.where(r.view(B, 1, 1), post_stoch_seq[:, t], stoch)

        # One img_step from t-1 posterior → prior prediction for t
        deter, stoch, logits_t = rssm.img_step(
            deter, stoch, actions_seq[:, t], goals_seq[:, t]
        )
        prior_deters.append(deter)
        prior_stochs.append(stoch)
        prior_logits.append(logits_t)

    return (
        torch.stack(prior_deters, dim=1),   # [B, T, D]
        torch.stack(prior_stochs, dim=1),   # [B, T, C, K]
        torch.stack(prior_logits, dim=1),   # [B, T, C*K]
    )


@torch.no_grad()
def decode_post_and_seeded_prior(rssm, decoder, post_deter_seq, post_stoch_seq,
                                  post_logits_bt, actions_seq, goals_seq, resets=None):
    B, T, D = post_deter_seq.shape

    post_deter_flat  = post_deter_seq.reshape(B * T, D)
    _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_bt.reshape(B * T, -1))
    post_recon_depth, post_sem_logits, _, _ = decoder(
        post_deter_flat, post_probs.reshape(B * T, -1)
    )

    prior_deter_seq, _, prior_logits_bt = rollout_seeded_prior(
        rssm, post_deter_seq, post_stoch_seq, actions_seq, goals_seq, resets
    )
    _, prior_probs, _ = rssm.dist_from_logits_flat(prior_logits_bt.reshape(B * T, -1))
    prior_recon_depth, prior_sem_logits, _, _ = decoder(
        prior_deter_seq.reshape(B * T, D), prior_probs.reshape(B * T, -1)
    )

    return post_recon_depth, post_sem_logits, prior_recon_depth, prior_sem_logits


@torch.no_grad()
def compute_rssm_out(batch, encoder, rssm):
    """
    Forward pass only — no gradient update.
    Used for computing rssm_out on the fixed validation batch.
    Call once after capturing the fixed batch and reuse the result each log step.
    """
    depths, sems, vectors, goals, actions, rewards, dones = batch
    depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = \
        preprocess_batch(depths, sems, vectors, goals, actions, rewards, dones)
    B, T = actions_seq.shape[:2]

    resets           = make_resets_from_dones(dones_seq)
    prev_actions_seq = torch.zeros_like(actions_seq)
    prev_actions_seq[:, 1:] = actions_seq[:, :-1]
    prev_actions_seq *= (1.0 - resets.float().unsqueeze(-1))

    embeds = encoder(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in).view(B, T, -1)
    post   = rssm.observe(embeds, prev_actions_seq, goals_seq, resets=resets)

    return dict(
        deter_seq=post["deter"], stoch_seq=post["stoch"],
        post_logits_bt=post["post_logits"], prior_logits_bt=post["prior_logits"],
        prev_actions_seq=prev_actions_seq, goals_seq=goals_seq,
        depth_in=depth_in, sem_ids=sem_ids, goal_in=goal_in,
        resets=resets, B=B, T=T,
    )
    

@torch.no_grad()
def log_recon_panels(writer, global_step, tag_prefix,
                     depth_in, sem_ids,
                     post_recon_depth, post_sem_logits,
                     prior_recon_depth, prior_sem_logits,
                     T=10):  # pass T so we can index the last timestep
    # Last timestep of first batch item — most warmed-up GRU state
    idx = T - 1  # b=0, t=T-1

    vis_depth = torch.cat([depth_in[idx:idx+1], post_recon_depth[idx:idx+1], prior_recon_depth[idx:idx+1]], dim=-1)
    writer.add_image(f"{tag_prefix}/Depth_GT_Post_Prior", vis_depth.squeeze(0), global_step)

    t_vis  = (sem_ids[idx:idx+1].float() / (NUM_CLASSES - 1)).unsqueeze(0)
    po_vis = (torch.argmax(post_sem_logits[idx:idx+1], dim=1).float() / (NUM_CLASSES - 1)).unsqueeze(0)
    pr_vis = (torch.argmax(prior_sem_logits[idx:idx+1], dim=1).float() / (NUM_CLASSES - 1)).unsqueeze(0)
    writer.add_image(f"{tag_prefix}/Semantic_GT_Post_Prior",
                     torch.cat([t_vis, po_vis, pr_vis], dim=-1).squeeze(0), global_step)


@torch.no_grad()
def log_dataset_action_rollout(writer, global_step, rssm, decoder,
                                start_deter, start_stoch, start_post_logits,
                                future_goals, future_actions,
                                horizon=5, tag_prefix="Visuals", num_examples=4):
    rssm.eval(); decoder.eval()
    try:
        B      = start_deter.shape[0]
        H_roll = min(horizon, future_actions.shape[1])
        deter, stoch = start_deter, start_stoch

        deters, logits_seq = [deter], []
        for j in range(H_roll):
            deter, stoch, logits_flat = rssm.img_step(
                deter, stoch, future_actions[:, j], future_goals[:, j]
            )
            deters.append(deter)
            logits_seq.append(logits_flat)

        deters          = torch.stack(deters, dim=1)
        prior_logits_bt = torch.stack(logits_seq, dim=1)
        deter_flat      = deters.reshape(B * (H_roll + 1), -1)

        _, seed_probs, _    = rssm.dist_from_logits_flat(start_post_logits)
        _, rollout_probs, _ = rssm.dist_from_logits_flat(
            prior_logits_bt.reshape(B * H_roll, -1)
        )

        all_stoch = torch.cat([
            seed_probs.reshape(B, -1).unsqueeze(1),
            rollout_probs.reshape(B, H_roll, -1),
        ], dim=1).reshape(B * (H_roll + 1), -1)

        recon_depth, sem_logits, _, _ = decoder(deter_flat, all_stoch)
        recon_depth = recon_depth.view(B, H_roll + 1, 1, H, W)
        sem_pred    = torch.argmax(sem_logits, dim=1).view(B, H_roll + 1, H, W)

        n = min(B, num_examples)
        depth_panel = torch.cat([make_strip(recon_depth[i]) for i in range(n)], dim=1)
        sem_panel   = torch.cat([make_strip(semantic_to_vis(sem_pred[i])) for i in range(n)], dim=1)

        writer.add_image(f"{tag_prefix}/DatasetAction_Depth",    depth_panel, global_step)
        writer.add_image(f"{tag_prefix}/DatasetAction_Semantic", sem_panel,   global_step)
    finally:
        rssm.train(); decoder.train()


@torch.no_grad()
def log_imagination_rollout(writer, global_step, rssm, decoder, actor,
                             start_deter, start_stoch, goal0,
                             horizon=5, tag_prefix="Visuals", num_examples=4):
    rssm.eval(); decoder.eval(); actor.eval()
    try:
        B = start_deter.shape[0]
        deter, stoch = start_deter, start_stoch
        deters, logits_seq = [deter], []

        for _ in range(horizon):
            action, _, _, _ = actor(deter, rssm.flatten_stoch(stoch), goal0, sample=True)
            deter, stoch, logits_flat = rssm.img_step(deter, stoch, action, goal0)
            deters.append(deter)
            logits_seq.append(logits_flat)

        deters          = torch.stack(deters, dim=1)
        prior_logits_bt = torch.stack(logits_seq, dim=1)

        _, rollout_probs, _ = rssm.dist_from_logits_flat(
            prior_logits_bt.reshape(B * horizon, -1)
        )
        all_stoch = torch.cat([
            rssm.flatten_stoch(start_stoch).unsqueeze(1),
            rollout_probs.reshape(B, horizon, -1),
        ], dim=1).reshape(B * (horizon + 1), -1)

        recon_depth, sem_logits, recon_goal, recon_vector = decoder(
            deters.reshape(B * (horizon + 1), -1), all_stoch
        )
        # make images for recon depth & sem
        recon_depth = recon_depth.view(B, horizon + 1, 1, H, W)
        sem_pred    = torch.argmax(sem_logits, dim=1).view(B, horizon + 1, H, W)

        n = min(B, num_examples)
        depth_panel = torch.cat([make_strip(recon_depth[i]) for i in range(n)], dim=1)
        sem_panel   = torch.cat([make_strip(semantic_to_vis(sem_pred[i])) for i in range(n)], dim=1)

        # make figures for recon goal
        # print("actual goal")
        # print(goal0.detach().cpu().numpy())
        # print("imagined goal")
        # print(recon_goal.detach().cpu().numpy())
        recon_goal = recon_goal.view(B, horizon + 1 , -1)
        # print(recon_goal.shape)

        _H = horizon + 1
        goal_fig, axs = plt.subplots(B, _H, figsize=(11, 4), sharex=True, sharey=True)
        for b in range(B):
            # draw actual goal at begining of imagination
            axs[b,0].scatter(goal0[b,0].item(), goal0[b,0].item(), c='blue', marker='o')
            # format axis
            axs[b,0].set_xlim(0, 0.4)
            axs[b,0].set_xticks(np.arange(0, 0.4+0.01, 0.1))
            axs[b,0].set_ylim(-0.2, 0.2)
            axs[b,0].set_yticks(np.arange(-0.2, 0.2+0.01, 0.1))
            for t in range(_H):
                # draw recon goal
                axs[b,t].scatter(recon_goal[b,t,0].item(), recon_goal[b,t,1].item(), c='red', marker='x')
                

        writer.add_image(f"{tag_prefix}/Imagined_Depth",    depth_panel, global_step)
        writer.add_image(f"{tag_prefix}/Imagined_Semantic", sem_panel,   global_step)
        writer.add_figure(f"{tag_prefix}/Imagined_Goal",    goal_fig,    global_step)
    finally:
        rssm.train(); decoder.train(); actor.train()


@torch.no_grad()
def log_visuals(writer, global_step, rssm_out, rssm, decoder,
                tag_prefix, fixed_rssm_out=None):
    """
    Log GT/Post/Prior reconstruction panels.
    fixed_rssm_out: pre-computed rssm_out for the fixed validation batch.
                    Compute once with compute_rssm_out() and reuse each log step.
                    Pass None to skip fixed batch logging.
    """
    def _log_one(out, tag):
        post_rd, post_sl, prior_rd, prior_sl = decode_post_and_seeded_prior(
            rssm, decoder,
            out["deter_seq"], out["stoch_seq"], out["post_logits_bt"],
            out["prev_actions_seq"], out["goals_seq"], out["resets"],
        )
        log_recon_panels(writer, global_step, tag,
                        out["depth_in"], out["sem_ids"],
                        post_rd, post_sl, prior_rd, prior_sl,
                        T=out["T"])

    _log_one(rssm_out, tag_prefix)

    if fixed_rssm_out is not None:
        _log_one(fixed_rssm_out, tag_prefix + "_Fixed")


# =============================================================================
# Training steps
# =============================================================================

def world_model_step(batch, encoder, rssm, decoder, reward_head, cont_head,
                     wm_opt, twohot, wm_params):
    """Single world model update. Returns (losses dict, rssm_out dict)."""
    depths, sems, vectors, goals, actions, rewards, dones = batch
    depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = \
        preprocess_batch(depths, sems, vectors, goals, actions, rewards, dones)
    B, T = actions_seq.shape[:2]

    resets           = make_resets_from_dones(dones_seq)
    prev_actions_seq = torch.zeros_like(actions_seq)
    prev_actions_seq[:, 1:] = actions_seq[:, :-1]
    prev_actions_seq *= (1.0 - resets.float().unsqueeze(-1))

    prev_rewards_seq = torch.zeros_like(rewards_seq)
    prev_rewards_seq[:, 1:] = rewards_seq[:, :-1]
    prev_rewards_seq *= (1.0 - resets.float())

    prev_cont_seq = torch.ones_like(dones_seq, dtype=torch.float32)
    prev_cont_seq[:, 1:] = 1.0 - dones_seq[:, :-1].float()
    prev_cont_seq = torch.where(resets, torch.ones_like(prev_cont_seq), prev_cont_seq)

    embeds = encoder(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in).view(B, T, -1)
    post   = rssm.observe(embeds, prev_actions_seq, goals_seq, resets=resets)

    deter_seq       = post["deter"]
    stoch_seq       = post["stoch"]
    post_logits_bt  = post["post_logits"]
    prior_logits_bt = post["prior_logits"]

    deter_flat      = deter_seq.reshape(B * T, -1)
    stoch_flat_hard = rssm.flatten_stoch(stoch_seq.reshape(B * T, rssm.C, rssm.K))

    _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_bt.reshape(B * T, -1))
    post_recon_depth, post_sem_logits, post_goal_pred, post_vec_pred = \
        decoder(deter_flat, post_probs.reshape(B * T, -1))

    post_sem_loss  = F.cross_entropy(post_sem_logits, sem_ids)
    post_depth_nll = gaussian_nll(depth_in, post_recon_depth).mean()
    post_goal_loss = F.mse_loss(post_goal_pred, goal_in)
    post_vec_loss  = F.mse_loss(post_vec_pred, vec_in)

    kl_loss        = rssm.kl_loss(post_logits_bt, prior_logits_bt)
    overshoot_loss = rssm.overshooting_loss(
        deter_seq, stoch_seq, prev_actions_seq,
        goals_seq, post_logits_bt, k=OVERSHOOT_K
    )
    reward_loss = twohot.ce_loss(
        reward_head(deter_flat, stoch_flat_hard, goal_in),
        symlog(prev_rewards_seq.reshape(-1))
    )
    cont_loss = F.binary_cross_entropy_with_logits(
        cont_head(deter_flat, stoch_flat_hard, goal_in),
        prev_cont_seq.reshape(-1, 1)
    )

    wm_loss = (
        post_depth_nll
        + SEM_SCALE       * post_sem_loss
        + GOAL_SCALE      * post_goal_loss
        + VEC_SCALE       * post_vec_loss
        + KL_SCALE        * kl_loss
        + OVERSHOOT_SCALE * overshoot_loss
        + REWARD_SCALE    * reward_loss
        + CONT_SCALE      * cont_loss
    )

    wm_opt.zero_grad(set_to_none=True)
    wm_loss.backward()
    torch.nn.utils.clip_grad_norm_(wm_params, max_norm=100.0)
    wm_opt.step()

    losses = dict(
        wm=wm_loss, depth_nll=post_depth_nll, sem=post_sem_loss,
        goal=post_goal_loss, vec=post_vec_loss,
        kl=kl_loss, overshoot=overshoot_loss,
        reward=reward_loss, cont=cont_loss,
    )
    rssm_out = dict(
        deter_seq=deter_seq, stoch_seq=stoch_seq,
        post_logits_bt=post_logits_bt, prior_logits_bt=prior_logits_bt,
        prev_actions_seq=prev_actions_seq, goals_seq=goals_seq,
        depth_in=depth_in, sem_ids=sem_ids, goal_in=goal_in,
        resets=resets, B=B, T=T,
    )
    return losses, rssm_out


def actor_critic_step(rssm_out, rssm, reward_head, cont_head, actor, critic,
                      target_critic, actor_opt, critic_opt, twohot, wm_params):
    """Single actor + critic update. Returns losses dict."""
    deter_seq = rssm_out["deter_seq"]
    stoch_seq = rssm_out["stoch_seq"]
    goals_seq = rssm_out["goals_seq"]
    B         = rssm_out["B"]

    start_deter = deter_seq[:, -1].detach()
    start_stoch = stoch_seq[:, -1].detach()
    goal0       = goals_seq[:, -1].detach()

    for p in wm_params:
        p.requires_grad_(False)

    try:
        for p in critic.parameters():
            p.requires_grad_(False)

        imag = rssm.imagine(start_deter, start_stoch, actor, goal0, horizon=IMAG_HORIZON)

        Bh      = B * IMAG_HORIZON
        Bh1     = B * (IMAG_HORIZON + 1)
        goal_h  = goal0.unsqueeze(1).expand(B, IMAG_HORIZON, 2).reshape(Bh, 2)
        goal_h1 = goal0.unsqueeze(1).expand(B, IMAG_HORIZON + 1, 2).reshape(Bh1, 2)

        next_deter_f = imag["deter"][:, 1:].reshape(Bh, -1)
        next_stoch_f = rssm.flatten_stoch(imag["stoch"][:, 1:].reshape(Bh, rssm.C, rssm.K))
        all_deter_f  = imag["deter"].reshape(Bh1, -1)
        all_stoch_f  = rssm.flatten_stoch(imag["stoch"].reshape(Bh1, rssm.C, rssm.K))

        imag_reward = symexp(
            twohot.mean(reward_head(next_deter_f, next_stoch_f, goal_h))
        ).view(B, IMAG_HORIZON, 1)
        discounts = (
            GAMMA * torch.sigmoid(cont_head(next_deter_f, next_stoch_f, goal_h))
        ).view(B, IMAG_HORIZON, 1).clamp(0, 1)

        with torch.no_grad():
            target_v = symexp(
                twohot.mean(target_critic(all_deter_f, all_stoch_f, goal_h1))
            ).view(B, IMAG_HORIZON + 1, 1)

        v = symexp(
            twohot.mean(critic(all_deter_f, all_stoch_f, goal_h1))
        ).view(B, IMAG_HORIZON + 1, 1)

        returns = lambda_return(
            reward=imag_reward, value=target_v[:, :-1], discount=discounts,
            lam=LAMBDA, bootstrap=target_v[:, -1], time_major=False,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones(B, 1, 1, device=DEVICE), discounts[:, :-1]], dim=1), dim=1
        )
        adv = returns - v[:, :-1].detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_opt.zero_grad(set_to_none=True)
        actor_loss = (
            -(weights * adv).mean()
            - ENT_SCALE * (weights * imag["ent"].unsqueeze(-1)).mean()
        )
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
        actor_opt.step()

    finally:
        for p in critic.parameters():
            p.requires_grad_(True)
        for p in wm_params:
            p.requires_grad_(True)

    prev_deter_f = imag["deter"][:, :-1].reshape(Bh, -1).detach()
    prev_stoch_f = rssm.flatten_stoch(
        imag["stoch"][:, :-1].reshape(Bh, rssm.C, rssm.K)
    ).detach()

    critic_opt.zero_grad(set_to_none=True)
    critic_loss = twohot.ce_loss(
        critic(prev_deter_f, prev_stoch_f, goal_h.detach()),
        symlog(returns.detach().view(-1))
    )
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
    critic_opt.step()
    ema_update(target_critic, critic, TARGET_EMA)

    return dict(actor=actor_loss, critic=critic_loss, returns=returns.mean())


def log_scalars(writer, global_step, losses, rssm_out, rssm, phase="Pretrain"):
    for k, v in losses.items():
        tag = f"{phase}/wm_loss" if k == "wm" else f"{phase}/{k}_loss"
        writer.add_scalar(tag, v.item(), global_step)

    with torch.no_grad():
        B, T = rssm_out["B"], rssm_out["T"]
        _, prior_probs, _ = rssm.dist_from_logits_flat(
            rssm_out["prior_logits_bt"].reshape(B * T, -1)
        )
        entropy = -(prior_probs * (prior_probs + 1e-8).log()).sum(-1).mean()
        writer.add_scalar(f"{phase}/prior_entropy", entropy.item(), global_step)