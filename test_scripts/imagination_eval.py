import os
import math
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.buffer import SequenceBuffer
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.decoder import MultiModalDecoder
from models.rewardhead import RewardHead
from models.continuehead import ContinueHead
from models.actor_critic import Actor


# -----------------------
# Config defaults
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 10
BATCH_SIZE = 8
NUM_CLASSES = 28
H, W = 128, 128

_TUNE = 1
DETER_DIM = 512 * _TUNE * _TUNE
EMBED_DIM = 1024 * _TUNE * _TUNE

BUFFER_PATH = "./data/expert_sequences"
CKPT_PATH = "checkpoints/world_model/world_model_pretrained.pth"
LOG_DIR = "./runs/imagination_eval"

IMAG_HORIZON = 10
NUM_EXAMPLES = 4


# -----------------------
# Helpers
# -----------------------
def preprocess_batch(depths, sems, vectors, goals, actions, rewards, dones):
    assert depths.ndim == 5 and sems.ndim == 5
    B, T, C, H_, W_ = depths.shape
    assert H_ == H and W_ == W and C == 1

    depth_in = depths.reshape(B * T, 1, H, W).to(dtype=torch.float32) / 255.0

    sem_ids = sems.reshape(B * T, H, W).to(dtype=torch.long)
    sem_ids = torch.clamp(sem_ids, 0, NUM_CLASSES - 1)

    vec_in = vectors.reshape(B * T, -1).to(dtype=torch.float32)
    goal_in = goals.reshape(B * T, -1).to(dtype=torch.float32)

    actions_seq = actions.to(dtype=torch.float32)
    rewards_seq = rewards.to(dtype=torch.float32)
    dones_seq = dones.to(dtype=torch.bool)
    goals_seq = goals.to(dtype=torch.float32)

    return depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq


def make_resets_from_dones(dones_seq: torch.Tensor) -> torch.Tensor:
    resets = torch.zeros_like(dones_seq, dtype=torch.bool)
    resets[:, 1:] = dones_seq[:, :-1]
    return resets


def make_image_strip(images: torch.Tensor) -> torch.Tensor:
    """
    images: [T, 1, H, W] or [T, H, W]
    returns: [1, H, T*W]
    """
    if images.ndim == 4:
        images = images.squeeze(1)
    assert images.ndim == 3
    strip = torch.cat([img for img in images], dim=-1)  # [H, T*W]
    return strip.unsqueeze(0)  # [1, H, T*W]


def make_batch_panel(strips) -> torch.Tensor:
    """
    strips: list of [1, H, W_total]
    returns: [1, B*H, W_total]
    """
    return torch.cat(strips, dim=1)


def semantic_to_vis(sem_ids: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    sem_ids: [T, H, W] or [B, T, H, W]
    returns float in [0,1] with same leading dims
    """
    return sem_ids.float() / float(max(1, num_classes - 1))


def build_models():
    encoder = MultiModalEncoder(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        sem_embed_dim=16,
    ).to(DEVICE)

    rssm = RSSM(
        deter_dim=DETER_DIM,
        act_dim=2,
        embed_dim=EMBED_DIM,
        goal_dim=2,
        stoch_categoricals=32 * _TUNE,
        stoch_classes=32 * _TUNE,
        unimix_ratio=0.01,
        kl_balance=0.8,
        free_nats=0.1,
    ).to(DEVICE)

    decoder = MultiModalDecoder(
        deter_dim=DETER_DIM,
        stoch_dim=rssm.stoch_dim,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    reward_head = RewardHead(
        deter_dim=DETER_DIM,
        stoch_dim=rssm.stoch_dim,
        goal_dim=2,
        hidden_dim=512,
        bins=255,
        vmin=-20.0,
        vmax=20.0,
    ).to(DEVICE)

    cont_head = ContinueHead(
        deter_dim=DETER_DIM,
        stoch_dim=rssm.stoch_dim,
        goal_dim=2,
        hidden_dim=512,
    ).to(DEVICE)

    actor = Actor(
        deter_dim=DETER_DIM,
        stoch_dim=rssm.stoch_dim,
        goal_dim=2,
        action_dim=2,
        hidden_dim=512,
        min_std=0.1,
        init_std=1.0,
    ).to(DEVICE)

    return encoder, rssm, decoder, reward_head, cont_head, actor


def load_checkpoint(
    ckpt_path: str,
    encoder,
    rssm,
    decoder,
    reward_head=None,
    cont_head=None,
    actor=None,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if "encoder" not in ckpt or "rssm" not in ckpt or "decoder" not in ckpt:
        raise KeyError("Checkpoint must contain encoder, rssm, and decoder weights.")

    encoder.load_state_dict(ckpt["encoder"])
    rssm.load_state_dict(ckpt["rssm"])
    decoder.load_state_dict(ckpt["decoder"])

    if reward_head is not None and "reward_head" in ckpt:
        reward_head.load_state_dict(ckpt["reward_head"])
    if cont_head is not None and "cont_head" in ckpt:
        cont_head.load_state_dict(ckpt["cont_head"])
    if actor is not None and "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])

    return ckpt


@torch.no_grad()
def posterior_seed_and_imagine(
    encoder,
    rssm,
    decoder,
    reward_head,
    cont_head,
    actor,
    batch,
    imag_horizon: int,
):
    depths, sems, vectors, goals, actions, rewards, dones = batch
    depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = preprocess_batch(
        depths, sems, vectors, goals, actions, rewards, dones
    )

    B, T = actions_seq.shape[:2]

    # Encode sequence
    embeds_flat = encoder(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in)
    embeds = embeds_flat.view(B, T, -1)

    resets = make_resets_from_dones(dones_seq)

    prev_actions_seq = torch.zeros_like(actions_seq)
    prev_actions_seq[:, 1:] = actions_seq[:, :-1]
    prev_actions_seq = prev_actions_seq * (1.0 - resets.float().unsqueeze(-1))

    post = rssm.observe(embeds, prev_actions_seq, goals_seq, resets=resets)

    deter_seq = post["deter"]         # [B, T, D]
    stoch_seq = post["stoch"]         # [B, T, C, K]
    post_logits_bt = post["post_logits"]

    # Posterior recon over observed sequence
    deter_flat = deter_seq.reshape(B * T, -1)
    post_logits_flat = post_logits_bt.reshape(B * T, -1)
    _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_flat)
    stoch_flat_soft = post_probs.reshape(B * T, -1)

    recon_depth_post, sem_logits_post = decoder(deter_flat, stoch_flat_soft)
    recon_depth_post = recon_depth_post.view(B, T, 1, H, W)
    sem_pred_post = torch.argmax(sem_logits_post, dim=1).view(B, T, H, W)

    # Seed from final posterior state
    start_deter = deter_seq[:, -1]
    start_stoch = stoch_seq[:, -1]
    goal0 = goals_seq[:, -1]

    imag = rssm.imagine(start_deter, start_stoch, actor, goal0, horizon=imag_horizon)

    # Decode imagined states
    B2, H_imag_plus_1, D = imag["deter"].shape
    assert B2 == B
    imag_deter_flat = imag["deter"].reshape(B * H_imag_plus_1, -1)

    imag_stoch_flat = rssm.flatten_stoch(
        imag["stoch"].reshape(B * H_imag_plus_1, rssm.C, rssm.K)
    )

    recon_depth_imag, sem_logits_imag = decoder(imag_deter_flat, imag_stoch_flat)
    recon_depth_imag = recon_depth_imag.view(B, H_imag_plus_1, 1, H, W)
    sem_pred_imag = torch.argmax(sem_logits_imag, dim=1).view(B, H_imag_plus_1, H, W)

    # Reward / continue diagnostics on imagined next states only
    next_deter = imag["deter"][:, 1:]    # [B, H, D]
    next_stoch = imag["stoch"][:, 1:]    # [B, H, C, K]

    Bh = B * imag_horizon
    next_deter_f = next_deter.reshape(Bh, -1)
    next_stoch_f = rssm.flatten_stoch(next_stoch.reshape(Bh, rssm.C, rssm.K))
    goal_h = goal0.unsqueeze(1).expand(B, imag_horizon, 2).reshape(Bh, 2)

    reward_logits = reward_head(next_deter_f, next_stoch_f, goal_h)
    cont_logits = cont_head(next_deter_f, next_stoch_f, goal_h)

    imag_reward_mean = reward_logits.mean().item()
    imag_reward_std = reward_logits.std().item()
    imag_cont_prob = torch.sigmoid(cont_logits).view(B, imag_horizon, 1)

    return {
        "depth_gt": depth_in.view(B, T, 1, H, W),
        "sem_gt": sem_ids.view(B, T, H, W),
        "depth_post": recon_depth_post,
        "sem_post": sem_pred_post,
        "depth_imag": recon_depth_imag,
        "sem_imag": sem_pred_imag,
        "imag_cont_prob": imag_cont_prob,
        "imag_reward_mean": imag_reward_mean,
        "imag_reward_std": imag_reward_std,
    }


def log_panels(writer: SummaryWriter, out: dict, global_step: int, num_examples: int):
    B = out["depth_gt"].shape[0]
    n = min(B, num_examples)

    # Observed sequence panels: GT vs posterior recon
    depth_obs_rows = []
    sem_obs_rows = []

    for i in range(n):
        gt_depth_strip = make_image_strip(out["depth_gt"][i])        # [1,H,TW]
        pr_depth_strip = make_image_strip(out["depth_post"][i])      # [1,H,TW]
        depth_obs_rows.append(torch.cat([gt_depth_strip, pr_depth_strip], dim=1))  # [1,2H,TW]

        gt_sem_vis = semantic_to_vis(out["sem_gt"][i])
        pr_sem_vis = semantic_to_vis(out["sem_post"][i])
        gt_sem_strip = make_image_strip(gt_sem_vis)
        pr_sem_strip = make_image_strip(pr_sem_vis)
        sem_obs_rows.append(torch.cat([gt_sem_strip, pr_sem_strip], dim=1))

    depth_obs_panel = make_batch_panel(depth_obs_rows)
    sem_obs_panel = make_batch_panel(sem_obs_rows)

    writer.add_image("ImaginationEval/Observed_GT_vs_Posterior_Depth", depth_obs_panel, global_step)
    writer.add_image("ImaginationEval/Observed_GT_vs_Posterior_Semantic", sem_obs_panel, global_step)

    # Imagination panels
    depth_imag_rows = []
    sem_imag_rows = []

    for i in range(n):
        # includes seed state at index 0, then imagined rollout
        imag_depth_strip = make_image_strip(out["depth_imag"][i])
        imag_sem_strip = make_image_strip(semantic_to_vis(out["sem_imag"][i]))

        depth_imag_rows.append(imag_depth_strip)
        sem_imag_rows.append(imag_sem_strip)

    depth_imag_panel = make_batch_panel(depth_imag_rows)
    sem_imag_panel = make_batch_panel(sem_imag_rows)

    writer.add_image("ImaginationEval/Imagined_Depth_Rollout", depth_imag_panel, global_step)
    writer.add_image("ImaginationEval/Imagined_Semantic_Rollout", sem_imag_panel, global_step)

    # Continue diagnostics
    writer.add_scalar("ImaginationEval/imag_cont_prob_mean", out["imag_cont_prob"].mean().item(), global_step)
    writer.add_scalar("ImaginationEval/imag_cont_prob_std", out["imag_cont_prob"].std().item(), global_step)
    writer.add_scalar("ImaginationEval/reward_logits_mean", out["imag_reward_mean"], global_step)
    writer.add_scalar("ImaginationEval/reward_logits_std", out["imag_reward_std"], global_step)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RSSM imagination rollout quality.")
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH, help="Path to checkpoint.")
    parser.add_argument("--buffer", type=str, default=BUFFER_PATH, help="Path to expert buffer.")
    parser.add_argument("--logdir", type=str, default=LOG_DIR, help="TensorBoard log directory.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length (buffer setting).")
    parser.add_argument("--imag-horizon", type=int, default=IMAG_HORIZON, help="Imagination horizon.")
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES, help="How many rows to visualize.")
    parser.add_argument("--step-tag", type=int, default=0, help="Manual global step tag for TensorBoard.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Buffer
    buffer = SequenceBuffer(capacity=100000, seq_len=args.seq_len, device=DEVICE)
    buffer.load_from_disk(args.buffer)

    # Models
    encoder, rssm, decoder, reward_head, cont_head, actor = build_models()
    ckpt = load_checkpoint(args.ckpt, encoder, rssm, decoder, reward_head, cont_head, actor)

    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_head.eval()
    cont_head.eval()
    actor.eval()

    batch = buffer.sample(args.batch_size)
    if batch is None:
        raise RuntimeError("Buffer returned None. Check data path or buffer contents.")

    global_step = int(ckpt.get("global_step", args.step_tag))
    out = posterior_seed_and_imagine(
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_head=reward_head,
        cont_head=cont_head,
        actor=actor,
        batch=batch,
        imag_horizon=args.imag_horizon,
    )

    log_panels(writer, out, global_step=global_step, num_examples=args.num_examples)

    print("Imagination evaluation complete.")
    print(f"TensorBoard logdir: {args.logdir}")
    print(f"Checkpoint step tag: {global_step}")
    print(f"Imagined continue mean: {out['imag_cont_prob'].mean().item():.4f}")
    print(f"Imagined continue std:  {out['imag_cont_prob'].std().item():.4f}")
    print(f"Reward logits mean:     {out['imag_reward_mean']:.4f}")
    print(f"Reward logits std:      {out['imag_reward_std']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()