# train.py
import os
import copy
import numpy as np
import torch
import carla
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from torchmetrics.functional.segmentation import dice_score

from models import rssm
from utils.buffer import SequenceBuffer
from utils.lambda_returns import lambda_return
from utils.twohot import TwoHotDist, symlog, symexp

from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.decoder import MultiModalDecoder
from models.rewardhead import RewardHead
from models.continuehead import ContinueHead
from models.actor_critic import Actor, Critic
from env.carla_wrapper import CarlaEnv

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda")

# data
SEQ_LEN = 10 # 50
BATCH_SIZE = 4
NUM_CLASSES = 28   # semantic ids [0..27] (ASSUMES your semantic actually is ids)
H, W = 128, 128

_TUNE = 1   # NOTE: int: scaling factor for latent dimensions; only for testing; set to 1 for original setup

DETER_DIM = 512 *_TUNE*_TUNE
EMBED_DIM = 1024 *_TUNE*_TUNE

PHASE_A_STEPS = 20000 # 20k, 40k
PHASE_A_PATH = "checkpoints/world_model/world_model_pretrained.pth"

# training
WM_LR = 8e-5
ACTOR_LR = 3e-5
CRITIC_LR = 3e-5

PHASE_B_STEPS = 2000
PART_B_EPISODE = 5000 # 5000
TRAIN_EVERY = 5   # Update the model every 5 steps
IMAG_HORIZON = 15
GAMMA = 0.99
LAMBDA = 0.95

# loss scales
SEM_SCALE = 10.0
REWARD_SCALE = 1.0
CONT_SCALE = 1.0
KL_SCALE = 1.0
ENT_SCALE = 1e-3
OVERSHOOT_K = 3
OVERSHOOT_SCALE = 0.5

# twohot support
BINS = 255
VMIN = -20.0
VMAX = 20.0

# target critic EMA
TARGET_EMA = 0.99

# checkpoints
LOAD_PRETRAINED = False
CKPT_DIR = "checkpoints/dreamerv3"
CKPT_PATH = os.path.join(CKPT_DIR, "dreamerv3_latest.pth")

IMAG_LOG_EVERY = 100
IMAG_LOG_HORIZON = 10
IMAG_LOG_EXAMPLES = 4

FIXED_VAL_ENABLED = True
LOG_ACTOR_IMAG_IN_PHASE_A = False


def ema_update(target: torch.nn.Module, online: torch.nn.Module, tau: float):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))
            
def gaussian_nll(x, mean, std=0.1, eps=1e-6):
    var = (std ** 2) + eps
    return 0.5 * ((x - mean) ** 2) / var + torch.log(torch.tensor(std + eps, device=x.device))


def preprocess_batch(depths, sems, vectors, goals, actions, rewards, dones):
    """
    Inputs from SequenceBuffer.sample():
      depths:  [B,T,1,H,W] float (from buffer.sample()) but originally uint8
      sems:    [B,T,1,H,W] float (from buffer.sample()) but originally uint8
      vectors: [B,T,3]
      goals:   [B,T,2]
      actions: [B,T,2]
      rewards: [B,T]
      dones:   [B,T] (float/bool)
    """
    assert depths.ndim == 5 and sems.ndim == 5
    B, T, C, H_, W_ = depths.shape
    assert H_ == H and W_ == W and C == 1

    # Depth input to encoder + target for decoder in [0,1]
    depth_in = depths.reshape(B * T, 1, H, W).to(dtype=torch.float32) / 255.0

    # Semantic
    sem_ids = sems.reshape(B * T, H, W).to(dtype=torch.long)
    sem_ids = torch.clamp(sem_ids, 0, NUM_CLASSES - 1)

    # Vector/goal for encoder (flatten time)
    vec_in = vectors.reshape(B * T, -1).to(dtype=torch.float32)
    goal_in = goals.reshape(B * T, -1).to(dtype=torch.float32)

    # Sequence tensors (keep time)
    actions_seq = actions.to(dtype=torch.float32)      # [B,T,2]
    rewards_seq = rewards.to(dtype=torch.float32)      # [B,T]
    dones_seq = dones.to(dtype=torch.bool)             # [B,T]
    goals_seq = goals.to(dtype=torch.float32)          # [B,T,2]
    
    # print(f"\n[DEBUG Preprocess] Batch Statistics:")
    # print(f"  - Depth (0-1): Min: {depth_in.min():.4f}, Max: {depth_in.max():.4f}, Mean: {depth_in.mean():.4f}")
    # print(f"  - Sem ID (0-27): Unique IDs present: {torch.unique(sem_ids).tolist()}")
    # print(f"  - Reward: Min: {rewards_seq.min():.2f}, Max: {rewards_seq.max():.2f}")
    # print(f"  - Goal:   Min: {goal_in.min():.2f}, Max: {goal_in.max():.2f}")

    return depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq


def make_resets_from_dones(dones_seq: torch.Tensor) -> torch.Tensor:
    """
    resets[:, t] == True means reset state BEFORE step t.
    If your dones mark terminal after transition at t, then resets should be shifted:
      resets[:, 0] = False
      resets[:, t] = dones[:, t-1]
    """
    resets = torch.zeros_like(dones_seq, dtype=torch.bool)
    resets[:, 1:] = dones_seq[:, :-1]
    return resets

def make_strip(images: torch.Tensor) -> torch.Tensor:
    """
    images: [T, 1, H, W] or [T, H, W]
    returns: [1, H, T*W]
    """
    if images.ndim == 4:
        images = images.squeeze(1)
    assert images.ndim == 3
    strip = torch.cat([img for img in images], dim=-1)   # [H, T*W]
    return strip.unsqueeze(0)                            # [1, H, T*W]


def semantic_to_vis(sem_ids: torch.Tensor) -> torch.Tensor:
    """
    sem_ids: [T, H, W]
    returns float [T, H, W] in [0, 1]
    """
    return sem_ids.float() / float(NUM_CLASSES - 1)


@torch.no_grad()
def log_imagination_rollout(
    writer,
    global_step: int,
    rssm,
    decoder,
    actor,
    start_deter: torch.Tensor,   # [B, D]
    start_stoch: torch.Tensor,   # [B, C, K]
    goal0: torch.Tensor,         # [B, 2]
    horizon: int = 5,
    tag_prefix: str = "Visuals",
    num_examples: int = 4,
):
    """
    Logs decoded imagination rollout panels:
      - Imagined depth rollout
      - Imagined semantic rollout

    Each row is:
      seed | imag1 | imag2 | ... | imagH
    """
    rssm_was_training = rssm.training
    decoder_was_training = decoder.training
    actor_was_training = actor.training
    
    rssm.eval()
    decoder.eval()
    actor.eval()
    
    try:
        B = start_deter.shape[0]
        deter = start_deter
        stoch = start_stoch

        deters = [deter]
        logits_seq = []
        ents = []
        actions = []

        for _ in range(horizon):
            stoch_flat = rssm.flatten_stoch(stoch)
            action, logp, entropy, mean = actor(deter, stoch_flat, goal0, sample=True)

            deter, stoch, prior_logits_flat = rssm.img_step(deter, stoch, action, goal0)

            actions.append(action)
            ents.append(entropy)
            deters.append(deter)
            logits_seq.append(prior_logits_flat)

        deters = torch.stack(deters, dim=1)              # [B, H+1, D]
        prior_logits_bt = torch.stack(logits_seq, dim=1) # [B, H, C*K]

        n = min(B, num_examples)

        Bh = B * (horizon + 1)
        imag_deter_flat = deters.reshape(Bh, -1)

        # Interleave seed into the sequence properly
        seed_stoch_flat = rssm.flatten_stoch(start_stoch)  # [B, C*K]
        _, prior_probs, _ = rssm.dist_from_logits_flat(prior_logits_bt.reshape(B * horizon, -1))
        rollout_stoch_flat = prior_probs.reshape(B, horizon, -1)  # [B, H, C*K]

        # Stack seed + rollout along time dim, then flatten
        all_stoch = torch.cat([
            seed_stoch_flat.unsqueeze(1),   # [B, 1, C*K]
            rollout_stoch_flat,              # [B, H, C*K]
        ], dim=1)                            # [B, H+1, C*K]
        decode_stoch_flat = all_stoch.reshape(B * (horizon + 1), -1)

        recon_depth, sem_logits = decoder(imag_deter_flat, decode_stoch_flat)
        recon_depth = recon_depth.view(B, horizon + 1, 1, H, W)
        sem_pred = torch.argmax(sem_logits, dim=1).view(B, horizon + 1, H, W)

        depth_rows = []
        sem_rows = []

        for i in range(n):
            depth_strip = make_strip(recon_depth[i])                 # [1, H, (H+1)*W]
            sem_strip = make_strip(semantic_to_vis(sem_pred[i]))    # [1, H, (H+1)*W]
            depth_rows.append(depth_strip)
            sem_rows.append(sem_strip)

        depth_panel = torch.cat(depth_rows, dim=1)  # [1, n*H, total_W]
        sem_panel = torch.cat(sem_rows, dim=1)      # [1, n*H, total_W]

        writer.add_image(f"{tag_prefix}/Imagined_Depth", depth_panel, global_step)
        writer.add_image(f"{tag_prefix}/Imagined_Semantic", sem_panel, global_step)
    finally:
        rssm.train(rssm_was_training)
        decoder.train(decoder_was_training)
        actor.train(actor_was_training)


@torch.no_grad()
def log_dataset_action_rollout(
    writer,
    global_step: int,
    rssm,
    decoder,
    start_deter: torch.Tensor,        # [B, D]
    start_stoch: torch.Tensor,        # [B, C, K] used only for hard recurrence
    start_post_logits: torch.Tensor,  # [B, C*K] used for soft seed decode
    future_goals: torch.Tensor,       # [B, H, 2]
    future_actions: torch.Tensor,     # [B, H, A]
    horizon: int = 5,
    tag_prefix: str = "Visuals",
    num_examples: int = 4,
):
    """
    Roll out the prior using REAL dataset actions, not actor actions.

    Each row is:
      seed | pred1 | pred2 | ... | predH
    """
    rssm_was_training = rssm.training
    decoder_was_training = decoder.training

    rssm.eval()
    decoder.eval()

    try:
        B = start_deter.shape[0]
        H_roll = min(horizon, future_actions.shape[1])

        deter = start_deter
        stoch = start_stoch

        deters = [deter]
        logits_seq = []

        for j in range(H_roll):
            action_j = future_actions[:, j]
            goal_j = future_goals[:, j]
            deter, stoch, prior_logits_flat = rssm.img_step(deter, stoch, action_j, goal_j)
            deters.append(deter)
            logits_seq.append(prior_logits_flat)

        deters = torch.stack(deters, dim=1)              # [B, H+1, D]
        prior_logits_bt = torch.stack(logits_seq, dim=1) # [B, H, C*K]

        Bh = B * (H_roll + 1)
        deter_flat = deters.reshape(Bh, -1)

        # seed frame uses soft posterior probs for fair comparison
        _, seed_probs, _ = rssm.dist_from_logits_flat(start_post_logits)   # [B, C, K]
        seed_stoch_flat = seed_probs.reshape(B, -1)                        # [B, C*K]

        # rollout steps use soft probs from prior logits
        _, prior_probs, _ = rssm.dist_from_logits_flat(prior_logits_bt.reshape(B * H_roll, -1))
        rollout_stoch_flat = prior_probs.reshape(B, H_roll, -1)            # [B, H, C*K]

        # interleave along time then flatten — matches deter_flat ordering
        all_stoch = torch.cat([
            seed_stoch_flat.unsqueeze(1),   # [B, 1, C*K]
            rollout_stoch_flat,              # [B, H, C*K]
        ], dim=1)                            # [B, H+1, C*K]
        decode_stoch_flat = all_stoch.reshape(B * (H_roll + 1), -1)

        recon_depth, sem_logits = decoder(deter_flat, decode_stoch_flat)
        recon_depth = recon_depth.view(B, H_roll + 1, 1, H, W)
        sem_pred = torch.argmax(sem_logits, dim=1).view(B, H_roll + 1, H, W)

        n = min(B, num_examples)
        depth_rows = []
        sem_rows = []

        for i in range(n):
            depth_strip = make_strip(recon_depth[i])
            sem_strip = make_strip(semantic_to_vis(sem_pred[i]))
            depth_rows.append(depth_strip)
            sem_rows.append(sem_strip)

        depth_panel = torch.cat(depth_rows, dim=1)
        sem_panel = torch.cat(sem_rows, dim=1)

        writer.add_image(f"{tag_prefix}/DatasetAction_Depth", depth_panel, global_step)
        writer.add_image(f"{tag_prefix}/DatasetAction_Semantic", sem_panel, global_step)

    finally:
        rssm.train(rssm_was_training)
        decoder.train(decoder_was_training)

def clone_batch_to_cpu(batch):
    return tuple(x.detach().cpu().clone() for x in batch)


# @torch.no_grad()
# def rollout_free_prior(
#     rssm,
#     actions_seq,
#     goals_seq,
#     resets=None,
# ):
#     """
#     Pure free-running prior rollout over time.
#     This does NOT use posterior correction from observations.

#     Returns:
#       prior_deter_seq:  [B, T, D]
#       prior_stoch_seq:  [B, T, C, K]   hard sampled states
#       prior_logits_seq: [B, T, C*K]    raw prior logits
#     """
#     B, T, _ = actions_seq.shape
#     device = actions_seq.device

#     deter, stoch = rssm.initial(B, device=device)

#     prior_deters = []
#     prior_stochs = []
#     prior_logits = []

#     for t in range(T):
#         if resets is not None:
#             r = resets[:, t].to(device=device).float().view(B, 1)
#             keep = 1.0 - r
#             init_d, init_s = rssm.initial(B, device=device)

#             deter = deter * keep + init_d * r
#             stoch = stoch * keep.view(B, 1, 1) + init_s * r.view(B, 1, 1)

#         deter, stoch, prior_logits_flat = rssm.img_step(
#             deter, stoch, actions_seq[:, t], goals_seq[:, t]
#         )
#         prior_deters.append(deter)
#         prior_stochs.append(stoch)
#         prior_logits.append(prior_logits_flat)

#     return (
#         torch.stack(prior_deters, dim=1),   # [B,T,D]
#         torch.stack(prior_stochs, dim=1),   # [B,T,C,K]
#         torch.stack(prior_logits, dim=1),   # [B,T,C*K]
#     )


# @torch.no_grad()
# def decode_post_and_free_prior(
#     rssm,
#     decoder,
#     post_deter_seq,
#     post_logits_bt,
#     actions_seq,
#     goals_seq,
#     resets=None,
# ):
#     """
#     Decode:
#       - posterior reconstruction from observe()
#       - free-running prior reconstruction from img_step rollout only
#     """
#     B, T, D = post_deter_seq.shape

#     # Posterior soft state decode
#     post_deter_flat = post_deter_seq.reshape(B * T, D)
#     post_logits_flat = post_logits_bt.reshape(B * T, -1)
#     _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_flat)
#     post_stoch_flat_soft = post_probs.reshape(B * T, -1)

#     post_recon_depth, post_sem_logits = decoder(post_deter_flat, post_stoch_flat_soft)

#     # Free-running prior rollout
#     prior_deter_seq, prior_stoch_seq, prior_logits_bt = rollout_free_prior(
#         rssm=rssm,
#         actions_seq=actions_seq,
#         goals_seq=goals_seq,
#         resets=resets,
#     )

#     prior_deter_flat = prior_deter_seq.reshape(B * T, D)
#     prior_logits_flat = prior_logits_bt.reshape(B * T, -1)

#     _, prior_probs, _ = rssm.dist_from_logits_flat(prior_logits_flat)
#     prior_stoch_flat_soft = prior_probs.reshape(B * T, -1)

#     prior_recon_depth, prior_sem_logits = decoder(prior_deter_flat, prior_stoch_flat_soft)

#     return (
#         post_recon_depth, post_sem_logits,
#         prior_recon_depth, prior_sem_logits,
#     )

@torch.no_grad()
def rollout_seeded_prior(
    rssm,
    post_deter_seq,   # [B, T, D]
    post_stoch_seq,   # [B, T, C, K]
    actions_seq,      # [B, T, A]  — prev_actions_seq convention (shifted)
    goals_seq,        # [B, T, 2]
    resets=None,      # [B, T] bool or None
):
    """
    Posterior-seeded free prior rollout.

    At t=0 (and at any reset), seeds deter/stoch from the posterior.
    For all subsequent steps, rolls forward purely with img_step — no
    posterior correction.

    The prior prediction for timestep t is produced by stepping from the
    state at t-1, so it represents what the model predicted BEFORE seeing
    the observation at t. This aligns with how prior_logits are produced
    inside observe().

    t=0 has no prior prediction in the strict sense (nothing preceded it),
    so we use prior_net(post_deter_t0) as a stand-in — it will look similar
    to the posterior at that frame, which is expected.

    Returns:
      prior_deter_seq:  [B, T, D]
      prior_stoch_seq:  [B, T, C, K]
      prior_logits_seq: [B, T, C*K]
    """
    B, T, D = post_deter_seq.shape
    device = post_deter_seq.device

    prior_deters = []
    prior_stochs = []
    prior_logits = []

    # Seed from posterior at t=0
    deter = post_deter_seq[:, 0]
    stoch = post_stoch_seq[:, 0]

    # t=0: no prior step preceded this — use posterior deter as stand-in
    prior_deters.append(deter)
    prior_stochs.append(stoch)
    prior_logits.append(rssm.prior_net(deter))

    for t in range(1, T):
        # Re-seed from posterior at episode resets
        if resets is not None:
            r = resets[:, t].to(device=device).bool()
            if r.any():
                deter = torch.where(r.unsqueeze(-1), post_deter_seq[:, t], deter)
                stoch = torch.where(r.view(B, 1, 1), post_stoch_seq[:, t], stoch)

        # Prior step: state at t-1 + action at t → prior prediction for t
        deter, stoch, prior_logits_t = rssm.img_step(
            deter, stoch, actions_seq[:, t], goals_seq[:, t]
        )

        prior_deters.append(deter)
        prior_stochs.append(stoch)
        prior_logits.append(prior_logits_t)

    return (
        torch.stack(prior_deters, dim=1),   # [B, T, D]
        torch.stack(prior_stochs, dim=1),   # [B, T, C, K]
        torch.stack(prior_logits, dim=1),   # [B, T, C*K]
    )
    
@torch.no_grad()
def decode_post_and_seeded_prior(
    rssm,
    decoder,
    post_deter_seq,
    post_stoch_seq,
    post_logits_bt,
    actions_seq,
    goals_seq,
    resets=None,
):
    """
    Decode:
      - posterior reconstruction from observe()
      - posterior-seeded free prior reconstruction
    """
    B, T, D = post_deter_seq.shape

    # Posterior soft state decode
    post_deter_flat = post_deter_seq.reshape(B * T, D)
    post_logits_flat = post_logits_bt.reshape(B * T, -1)
    _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_flat)
    post_stoch_flat_soft = post_probs.reshape(B * T, -1)

    post_recon_depth, post_sem_logits = decoder(post_deter_flat, post_stoch_flat_soft)

    # Posterior-seeded free prior rollout
    prior_deter_seq, prior_stoch_seq, prior_logits_bt = rollout_seeded_prior(
        rssm=rssm,
        post_deter_seq=post_deter_seq,
        post_stoch_seq=post_stoch_seq,
        actions_seq=actions_seq,
        goals_seq=goals_seq,
        resets=resets,
    )

    prior_deter_flat = prior_deter_seq.reshape(B * T, D)
    prior_logits_flat = prior_logits_bt.reshape(B * T, -1)

    _, prior_probs, _ = rssm.dist_from_logits_flat(prior_logits_flat)
    prior_stoch_flat_soft = prior_probs.reshape(B * T, -1)

    prior_recon_depth, prior_sem_logits = decoder(prior_deter_flat, prior_stoch_flat_soft)

    return (
        post_recon_depth, post_sem_logits,
        prior_recon_depth, prior_sem_logits,
    )


@torch.no_grad()
def log_recon_panels(
    writer,
    global_step,
    tag_prefix,
    depth_in,                # [B*T,1,H,W]
    sem_ids,                 # [B*T,H,W]
    post_recon_depth,        # [B*T,1,H,W]
    post_sem_logits,         # [B*T,C,H,W]
    prior_recon_depth,       # [B*T,1,H,W]
    prior_sem_logits,        # [B*T,C,H,W]
):
    # Depth: GT | Post | Prior
    t_depth = depth_in[0:1]
    post_depth = post_recon_depth[0:1]
    prior_depth = prior_recon_depth[0:1]
    vis_depth = torch.cat([t_depth, post_depth, prior_depth], dim=-1)
    writer.add_image(f"{tag_prefix}/Depth_GT_Post_Prior", vis_depth.squeeze(0), global_step)

    # Semantic: GT | Post | Prior
    t_sem_ids = sem_ids[0:1]
    post_sem_ids = torch.argmax(post_sem_logits[0:1], dim=1)
    prior_sem_ids = torch.argmax(prior_sem_logits[0:1], dim=1)

    t_sem_vis = (t_sem_ids.float() / float(NUM_CLASSES - 1)).unsqueeze(0)
    post_sem_vis = (post_sem_ids.float() / float(NUM_CLASSES - 1)).unsqueeze(0)
    prior_sem_vis = (prior_sem_ids.float() / float(NUM_CLASSES - 1)).unsqueeze(0)

    vis_sem = torch.cat([t_sem_vis, post_sem_vis, prior_sem_vis], dim=-1)
    writer.add_image(f"{tag_prefix}/Semantic_GT_Post_Prior", vis_sem.squeeze(0), global_step)

def main():
    print("Device:", DEVICE)

    # -----------------------
    # Buffer (offline training only)
    # -----------------------
    buffer = SequenceBuffer(capacity=100000, seq_len=SEQ_LEN, device=DEVICE)
    buffer.load_from_disk("./data/expert_sequences")
    
    fixed_val_batch = None
    if FIXED_VAL_ENABLED:
        tmp = buffer.sample(BATCH_SIZE)
        if tmp is not None:
            fixed_val_batch = clone_batch_to_cpu(tmp)
            print("[Info] Fixed validation batch captured for visual logging.")
        else:
            print("[Warning] Could not capture fixed validation batch.")

    writer = SummaryWriter(log_dir="./runs/dreamerv3_carla")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # -----------------------
    # Models
    # -----------------------
    encoder = MultiModalEncoder(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, sem_embed_dim=16).to(DEVICE)

    rssm = RSSM(
        deter_dim=DETER_DIM,
        act_dim=2,
        embed_dim=EMBED_DIM,
        goal_dim=2,
        stoch_categoricals=32 * _TUNE,
        stoch_classes=32 * _TUNE,
        unimix_ratio=0.01,
        kl_balance=0.8,
        free_nats=0.0,
    ).to(DEVICE)

    Z_DIM = rssm.stoch_dim  # C*K (default 1024)

    decoder = MultiModalDecoder(deter_dim=DETER_DIM, stoch_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    reward_head = RewardHead(
        deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2,
        hidden_dim=512, bins=BINS, vmin=VMIN, vmax=VMAX
    ).to(DEVICE)

    cont_head = ContinueHead(
        deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2, hidden_dim=512
    ).to(DEVICE)

    actor = Actor(
        deter_dim=DETER_DIM,
        stoch_dim=Z_DIM,
        goal_dim=2,      # keep if you want goal-conditioned behavior
        action_dim=2,
        hidden_dim=512,
        min_std=0.1,
        init_std=1.0,
    ).to(DEVICE)

    critic = Critic(
        deter_dim=DETER_DIM,
        stoch_dim=Z_DIM,
        goal_dim=2,
        hidden_dim=512,
        bins=BINS,
    ).to(DEVICE)

    target_critic = copy.deepcopy(critic).to(DEVICE)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # TwoHot helper (shared)
    twohot = TwoHotDist(num_bins=BINS, vmin=VMIN, vmax=VMAX, device=DEVICE).to(DEVICE)

    # -----------------------
    # Optims
    # -----------------------
    wm_params = (
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
        + list(reward_head.parameters())
        + list(cont_head.parameters())
    )
    wm_opt = torch.optim.Adam(wm_params, lr=WM_LR)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR, weight_decay=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, weight_decay=1e-4)

    # -----------------------
    # Phase A: WM pretrain
    # -----------------------
    global_step = 0
    print("\n[Phase A] Starting World Model Pretraining...")

    if LOAD_PRETRAINED and os.path.exists(PHASE_A_PATH):
        ckptA = torch.load(PHASE_A_PATH, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckptA["encoder"])
        rssm.load_state_dict(ckptA["rssm"])
        decoder.load_state_dict(ckptA["decoder"])
        reward_head.load_state_dict(ckptA["reward_head"])
        cont_head.load_state_dict(ckptA["cont_head"])
        wm_opt.load_state_dict(ckptA["wm_opt"])
        global_step = ckptA.get("global_step", 0)
        print(f"Loaded Phase A world model @ step {global_step}")
    else:
        encoder.train()
        rssm.train()
        decoder.train()
        reward_head.train()
        cont_head.train()

        pbar = tqdm(range(PHASE_A_STEPS), desc="[Phase A] WM pretrain")
        for step_A in pbar:
            batch = buffer.sample(BATCH_SIZE)
            if batch is None:
                continue

            depths, sems, vectors, goals, actions, rewards, dones = batch
            depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = preprocess_batch(
                depths, sems, vectors, goals, actions, rewards, dones
            )
            B, T = actions_seq.shape[:2]

            wm_opt.zero_grad(set_to_none=True)

            # Encode observations
            embeds_flat = encoder(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in)   # [B*T, E]
            embeds = embeds_flat.view(B, T, -1)                                      # [B, T, E]

            # Episode resets + previous actions
            resets = make_resets_from_dones(dones_seq)
            prev_actions_seq = torch.zeros_like(actions_seq)
            prev_actions_seq[:, 1:] = actions_seq[:, :-1]
            prev_actions_seq = prev_actions_seq * (1.0 - resets.float().unsqueeze(-1))

            # Observe sequence through RSSM
            post = rssm.observe(embeds, prev_actions_seq, goals_seq, resets=resets)

            # Previous reward / continue targets
            prev_rewards_seq = torch.zeros_like(rewards_seq)
            prev_rewards_seq[:, 1:] = rewards_seq[:, :-1]
            prev_rewards_seq = prev_rewards_seq * (1.0 - resets.float())

            prev_cont_seq = torch.ones_like(dones_seq, dtype=torch.float32)
            prev_cont_seq[:, 1:] = 1.0 - dones_seq[:, :-1].float()
            prev_cont_seq = torch.where(resets, torch.ones_like(prev_cont_seq), prev_cont_seq)

            # RSSM outputs
            deter_seq = post["deter"]                    # [B, T, D]
            stoch_seq = post["stoch"]                    # [B, T, C, K]
            post_logits_bt = post["post_logits"]         # [B, T, C*K]
            prior_logits_bt = post["prior_logits"]       # [B, T, C*K]

            # Flatten for heads / decoder
            deter_flat = deter_seq.reshape(B * T, -1)

            # Hard stochastic state for reward/continue heads
            stoch_flat_hard = rssm.flatten_stoch(
                stoch_seq.reshape(B * T, rssm.C, rssm.K)
            )

            # -----------------------------
            # Posterior reconstruction
            # -----------------------------
            post_logits_flat = post_logits_bt.reshape(B * T, -1)
            _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_flat)
            post_stoch_flat_soft = post_probs.reshape(B * T, -1)

            post_recon_depth, post_sem_logits = decoder(deter_flat, post_stoch_flat_soft)

            post_sem_loss = F.cross_entropy(post_sem_logits, sem_ids)
            post_depth_nll = gaussian_nll(depth_in, post_recon_depth, std=0.1).mean()

            # -----------------------------
            # KL + Overshooting
            # -----------------------------
            kl_loss = rssm.kl_loss(post_logits_bt, prior_logits_bt)

            overshoot_loss = rssm.overshooting_loss(
                deter_seq=deter_seq,
                stoch_seq=stoch_seq,
                actions_seq=prev_actions_seq,
                goals_seq=goals_seq,
                post_logits_bt=post_logits_bt,
                k=OVERSHOOT_K,
            )

            reward_logits = reward_head(deter_flat, stoch_flat_hard, goal_in)
            reward_target_symlog = symlog(prev_rewards_seq.reshape(-1))
            reward_loss = twohot.ce_loss(reward_logits, reward_target_symlog)

            cont_logits = cont_head(deter_flat, stoch_flat_hard, goal_in)
            cont_target = prev_cont_seq.reshape(-1, 1)
            cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

            post_recon_loss = post_depth_nll + SEM_SCALE * post_sem_loss

            wm_loss = (
                post_recon_loss
                + KL_SCALE * kl_loss
                + OVERSHOOT_SCALE * overshoot_loss
                + REWARD_SCALE * reward_loss
                + CONT_SCALE * cont_loss
            )

            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(rssm.parameters())
                + list(decoder.parameters())
                + list(reward_head.parameters())
                + list(cont_head.parameters()),
                max_norm=100.0,
            )
            wm_opt.step()

            global_step += 1
            
            if global_step % 10 == 0:
                writer.add_scalar("Pretrain/wm_loss", wm_loss.item(), global_step)
                writer.add_scalar("Pretrain/post_depth_nll_loss", post_depth_nll.item(), global_step)
                writer.add_scalar("Pretrain/post_sem_loss", post_sem_loss.item(), global_step)
                writer.add_scalar("Pretrain/kl_loss", kl_loss.item(), global_step)
                writer.add_scalar("Pretrain/overshoot_loss", overshoot_loss.item(), global_step)
                writer.add_scalar("Pretrain/reward_loss", reward_loss.item(), global_step)
                writer.add_scalar("Pretrain/cont_loss", cont_loss.item(), global_step)
                
                with torch.no_grad():
                    _, prior_probs_dbg, _ = rssm.dist_from_logits_flat(
                        prior_logits_bt.reshape(B * T, -1)
                    )
                    prior_entropy = -(
                        prior_probs_dbg * (prior_probs_dbg + 1e-8).log()
                    ).sum(dim=-1).mean()

                    writer.add_scalar("Pretrain/prior_entropy", prior_entropy.item(), global_step)

            # if global_step % 100 == 0:
            #     with torch.no_grad():
            #         # Depth GT | Recon
            #         t_depth = depth_in[0:1]
            #         r_depth = post_recon_depth[0:1]
            #         vis_depth = torch.cat([t_depth, r_depth], dim=-1)
            #         writer.add_image("Visuals_A/Depth_Recon", vis_depth.squeeze(0), global_step)

            #         # Semantic GT | Recon
            #         r_sem_ids = torch.argmax(post_sem_logits[0:1], dim=1)
            #         t_sem_ids = sem_ids[0:1]

            #         t_sem_vis = t_sem_ids.float() / float(NUM_CLASSES - 1)
            #         r_sem_vis = r_sem_ids.float() / float(NUM_CLASSES - 1)
            #         t_sem_vis = t_sem_vis.unsqueeze(0)
            #         r_sem_vis = r_sem_vis.unsqueeze(0)

            #         vis_sem = torch.cat([t_sem_vis, r_sem_vis], dim=-1)
            #         writer.add_image("Visuals_A/Semantic_Recon", vis_sem.squeeze(0), global_step)
            
            if global_step % 100 == 0:
                with torch.no_grad():
                    if fixed_val_batch is not None:
                        v_depths, v_sems, v_vectors, v_goals, v_actions, v_rewards, v_dones = fixed_val_batch
                        v_depth_in, v_sem_ids, v_vec_in, v_goal_in, v_actions_seq, v_rewards_seq, v_dones_seq, v_goals_seq = preprocess_batch(
                            v_depths.to(DEVICE),
                            v_sems.to(DEVICE),
                            v_vectors.to(DEVICE),
                            v_goals.to(DEVICE),
                            v_actions.to(DEVICE),
                            v_rewards.to(DEVICE),
                            v_dones.to(DEVICE),
                        )

                        vB, vT = v_actions_seq.shape[:2]

                        v_embeds_flat = encoder(v_depth_in, v_sem_ids.unsqueeze(1), v_vec_in, v_goal_in)
                        v_embeds = v_embeds_flat.view(vB, vT, -1)

                        v_resets = make_resets_from_dones(v_dones_seq)
                        v_prev_actions_seq = torch.zeros_like(v_actions_seq)
                        v_prev_actions_seq[:, 1:] = v_actions_seq[:, :-1]
                        v_prev_actions_seq = v_prev_actions_seq * (1.0 - v_resets.float().unsqueeze(-1))

                        v_post = rssm.observe(v_embeds, v_prev_actions_seq, v_goals_seq, resets=v_resets)

                        (
                            v_post_recon_depth, v_post_sem_logits,
                            v_prior_recon_depth, v_prior_sem_logits,
                        ) = decode_post_and_seeded_prior(
                            rssm=rssm,
                            decoder=decoder,
                            post_deter_seq=v_post["deter"],
                            post_stoch_seq=v_post["stoch"],
                            post_logits_bt=v_post["post_logits"],
                            actions_seq=v_prev_actions_seq,
                            goals_seq=v_goals_seq,
                            resets=v_resets,
                        )   

                        log_recon_panels(
                            writer=writer,
                            global_step=global_step,
                            tag_prefix="Visuals_A_Fixed",
                            depth_in=v_depth_in,
                            sem_ids=v_sem_ids,
                            post_recon_depth=v_post_recon_depth,
                            post_sem_logits=v_post_sem_logits,
                            prior_recon_depth=v_prior_recon_depth,
                            prior_sem_logits=v_prior_sem_logits,
                        )
                    (
                        post_recon_depth_dbg, post_sem_logits_dbg,
                        prior_recon_depth_dbg, prior_sem_logits_dbg,
                    ) = decode_post_and_seeded_prior(
                        rssm=rssm,
                        decoder=decoder,
                        post_deter_seq=deter_seq,
                        post_stoch_seq=stoch_seq,
                        post_logits_bt=post_logits_bt,
                        actions_seq=prev_actions_seq,
                        goals_seq=goals_seq,
                        resets=resets,
                    )

                    log_recon_panels(
                        writer=writer,
                        global_step=global_step,
                        tag_prefix="Visuals_A",
                        depth_in=depth_in,
                        sem_ids=sem_ids,
                        post_recon_depth=post_recon_depth_dbg,
                        post_sem_logits=post_sem_logits_dbg,
                        prior_recon_depth=prior_recon_depth_dbg,
                        prior_sem_logits=prior_sem_logits_dbg,
                    )
            
            if global_step % 100 == 0:
                os.makedirs(os.path.dirname(PHASE_A_PATH), exist_ok=True)
                torch.save({
                    "encoder": encoder.state_dict(),
                    "rssm": rssm.state_dict(),
                    "decoder": decoder.state_dict(),
                    "reward_head": reward_head.state_dict(),
                    "cont_head": cont_head.state_dict(),
                    "wm_opt": wm_opt.state_dict(),
                    "global_step": global_step,
                }, PHASE_A_PATH)
            
            if LOG_ACTOR_IMAG_IN_PHASE_A and global_step % IMAG_LOG_EVERY == 0:
                with torch.no_grad():
                    start_deter = deter_seq[:, -1].detach()
                    start_stoch = stoch_seq[:, -1].detach()
                    goal0 = goals_seq[:, -1].detach()

                    log_imagination_rollout(
                        writer=writer,
                        global_step=global_step,
                        rssm=rssm,
                        decoder=decoder,
                        actor=actor,
                        start_deter=start_deter,
                        start_stoch=start_stoch,
                        goal0=goal0,
                        horizon=IMAG_LOG_HORIZON,
                        tag_prefix="Visuals_A",
                        num_examples=IMAG_LOG_EXAMPLES,
                    )
            
            if global_step % IMAG_LOG_EVERY == 0:
                with torch.no_grad():
                    seed_t = max(0, T - 1 - IMAG_LOG_HORIZON)

                    start_deter = deter_seq[:, seed_t].detach()          # [B, D]
                    start_stoch = stoch_seq[:, seed_t].detach()          # [B, C, K]
                    start_post_logits = post_logits_bt[:, seed_t].detach()  # [B, C*K]

                    future_actions = prev_actions_seq[:, seed_t + 1 : seed_t + 1 + IMAG_LOG_HORIZON].detach()
                    future_goals = goals_seq[:, seed_t + 1 : seed_t + 1 + IMAG_LOG_HORIZON].detach()

                    log_dataset_action_rollout(
                        writer=writer,
                        global_step=global_step,
                        rssm=rssm,
                        decoder=decoder,
                        start_deter=start_deter,
                        start_stoch=start_stoch,
                        start_post_logits=start_post_logits,
                        future_actions=future_actions,
                        future_goals=future_goals,
                        horizon=IMAG_LOG_HORIZON,
                        tag_prefix="Visuals_A",
                        num_examples=IMAG_LOG_EXAMPLES,
                    )
            
            writer.flush()
            
            pbar.set_postfix({
                "wm": f"{wm_loss.item():.3f}",
                "kl": f"{kl_loss.item():.3f}",
            })

    # -----------------------
    # Load full checkpoint (optional)
    # -----------------------
    if LOAD_PRETRAINED and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        rssm.load_state_dict(ckpt["rssm"])
        decoder.load_state_dict(ckpt["decoder"])
        reward_head.load_state_dict(ckpt["reward_head"])
        cont_head.load_state_dict(ckpt["cont_head"])
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        target_critic.load_state_dict(ckpt["target_critic"])
        wm_opt.load_state_dict(ckpt["wm_opt"])
        actor_opt.load_state_dict(ckpt["actor_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        global_step = ckpt.get("global_step", global_step)
        print(f"Loaded checkpoint @ step {global_step}")

    # -----------------------
    # PHASE B: Online Training (Driving & Dreaming)
    # -----------------------
    print("\n[Phase B] Starting Online Interaction with CARLA...")
    env = CarlaEnv()
    spectator = env.world.get_spectator()

    start_episode = ckpt.get("episode", 0) if (LOAD_PRETRAINED and os.path.exists(CKPT_PATH)) else 0

    for episode in range(start_episode + 1, PART_B_EPISODE + 1):
        obs, _ = env.reset()
        episode_reward = 0.0

        # Initialize RSSM state for the new episode
        prev_deter, prev_stoch = rssm.initial(1, device=DEVICE)
        prev_action = torch.zeros(1, 2, device=DEVICE)

        pbar_steps = tqdm(range(PHASE_B_STEPS), desc=f"Episode {episode}", leave=False)
        for step in pbar_steps:
            # -----------------------
            # 1) Environment interaction
            # -----------------------
            with torch.no_grad():
                depth_in = (
                    torch.as_tensor(obs["depth"].copy())
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(DEVICE) / 255.0
                )
                sem_ids = (
                    torch.as_tensor(obs["semantic"].copy())
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .long()
                    .to(DEVICE)
                    .clamp(0, NUM_CLASSES - 1)
                )
                vec_in = torch.as_tensor(
                    obs.get("vector", np.zeros(3)).copy()
                ).unsqueeze(0).float().to(DEVICE)
                goal_in = torch.as_tensor(
                    obs.get("goal", np.zeros(2)).copy()
                ).unsqueeze(0).float().to(DEVICE)

                embed = encoder(depth_in, sem_ids, vec_in, goal_in)

                prev_deter, prev_stoch, _, _ = rssm.obs_step(
                    prev_deter, prev_stoch, prev_action, embed, goal_in
                )

                stoch_flat = rssm.flatten_stoch(prev_stoch)
                action_th, _, _, _ = actor(prev_deter, stoch_flat, goal_in, sample=True)

            act_np = action_th.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(act_np)
            done = terminated or truncated

            if torch.isnan(prev_deter).any():
                print("[CRITICAL] RSSM Deterministic state contains NaNs!")

            # Spectator camera
            v_transform = env.vehicle.get_transform()
            back_pos = (
                v_transform.location
                - (v_transform.get_forward_vector() * 8.0)
                + carla.Location(z=4.0)
            )
            spectator.set_transform(carla.Transform(back_pos, v_transform.rotation))

            # Store transition
            buffer.add(
                obs["depth"],
                obs["semantic"],
                obs.get("vector", np.zeros(3)),
                obs.get("goal", np.zeros(2)),
                act_np,
                reward,
                done,
            )

            episode_reward += reward
            obs = next_obs
            prev_action = action_th.detach()
            global_step += 1

            # -----------------------
            # 2) Model updates
            # -----------------------
            if global_step % TRAIN_EVERY == 0 and buffer.idx > BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                if batch is None:
                    continue

                depths, sems, vectors, goals, actions, rewards, dones = batch
                depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = preprocess_batch(
                    depths, sems, vectors, goals, actions, rewards, dones
                )
                B, T = actions_seq.shape[:2]

                # ----- World model update -----
                encoder.train()
                rssm.train()
                decoder.train()
                reward_head.train()
                cont_head.train()
                wm_opt.zero_grad(set_to_none=True)

                embeds_flat = encoder(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in)
                embeds = embeds_flat.view(B, T, -1)

                resets = make_resets_from_dones(dones_seq)

                prev_actions_seq = torch.zeros_like(actions_seq)
                prev_actions_seq[:, 1:] = actions_seq[:, :-1]
                prev_actions_seq = prev_actions_seq * (1.0 - resets.float().unsqueeze(-1))

                post = rssm.observe(embeds, prev_actions_seq, goals_seq, resets=resets)

                prev_rewards_seq = torch.zeros_like(rewards_seq)
                prev_rewards_seq[:, 1:] = rewards_seq[:, :-1]
                prev_rewards_seq = prev_rewards_seq * (1.0 - resets.float())

                prev_cont_seq = torch.ones_like(dones_seq, dtype=torch.float32)
                prev_cont_seq[:, 1:] = 1.0 - dones_seq[:, :-1].float()
                prev_cont_seq = torch.where(resets, torch.ones_like(prev_cont_seq), prev_cont_seq)

                # RSSM outputs
                deter_seq = post["deter"]                    # [B, T, D]
                stoch_seq = post["stoch"]                    # [B, T, C, K]
                post_logits_bt = post["post_logits"]         # [B, T, C*K]
                prior_logits_bt = post["prior_logits"]       # [B, T, C*K]

                deter_flat = deter_seq.reshape(B * T, -1)

                # Hard stochastic state for reward / continue / imagination seed
                stoch_flat_hard = rssm.flatten_stoch(
                    stoch_seq.reshape(B * T, rssm.C, rssm.K)
                )

                # -----------------------------
                # Posterior reconstruction
                # -----------------------------
                post_logits_flat = post_logits_bt.reshape(B * T, -1)
                _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_flat)
                post_stoch_flat_soft = post_probs.reshape(B * T, -1)

                post_recon_depth, post_sem_logits = decoder(deter_flat, post_stoch_flat_soft)

                post_sem_loss = F.cross_entropy(post_sem_logits, sem_ids)
                post_depth_nll = gaussian_nll(depth_in, post_recon_depth, std=0.1).mean()

                # -----------------------------
                # KL + Overshooting
                # -----------------------------
                kl_loss = rssm.kl_loss(post_logits_bt, prior_logits_bt)

                overshoot_loss = rssm.overshooting_loss(
                    deter_seq=deter_seq,
                    stoch_seq=stoch_seq,
                    actions_seq=prev_actions_seq,
                    goals_seq=goals_seq,
                    post_logits_bt=post_logits_bt,
                    k=OVERSHOOT_K,
                )

                reward_logits = reward_head(deter_flat, stoch_flat_hard, goal_in)
                reward_loss = twohot.ce_loss(
                    reward_logits,
                    symlog(prev_rewards_seq.reshape(-1)),
                )

                cont_logits = cont_head(deter_flat, stoch_flat_hard, goal_in)
                cont_target = prev_cont_seq.reshape(-1, 1)
                cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

                post_recon_loss = post_depth_nll + SEM_SCALE * post_sem_loss

                wm_loss = (
                    post_recon_loss
                    + KL_SCALE * kl_loss
                    + OVERSHOOT_SCALE * overshoot_loss
                    + REWARD_SCALE * reward_loss
                    + CONT_SCALE * cont_loss
                )

                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(wm_params, max_norm=100.0)
                wm_opt.step()

                # ----- Actor / Critic imagination -----
                actor.train()
                critic.train()

                start_deter = deter_seq[:, -1].detach()
                start_stoch = stoch_seq[:, -1].detach()
                goal0 = goals_seq[:, -1].detach()

                for p in wm_params:
                    p.requires_grad_(False)

                try:
                    for p in critic.parameters():
                        p.requires_grad_(False)

                    imag = rssm.imagine(start_deter, start_stoch, actor, goal0, horizon=IMAG_HORIZON)

                    prev_deter_imag = imag["deter"][:, :-1]
                    prev_stoch_imag = imag["stoch"][:, :-1]

                    next_deter = imag["deter"][:, 1:]
                    next_stoch = imag["stoch"][:, 1:]

                    Bh = B * IMAG_HORIZON

                    prev_deter_f = prev_deter_imag.reshape(Bh, -1)
                    prev_stoch_f = rssm.flatten_stoch(prev_stoch_imag.reshape(Bh, rssm.C, rssm.K))

                    next_deter_f = next_deter.reshape(Bh, -1)
                    next_stoch_f = rssm.flatten_stoch(next_stoch.reshape(Bh, rssm.C, rssm.K))

                    goal_h = goal0.unsqueeze(1).expand(B, IMAG_HORIZON, 2).reshape(Bh, 2)

                    imag_reward_symlog = twohot.mean(
                        reward_head(next_deter_f, next_stoch_f, goal_h)
                    )
                    imag_reward = symexp(imag_reward_symlog).view(B, IMAG_HORIZON, 1)

                    imag_cont_logits = cont_head(next_deter_f, next_stoch_f, goal_h)
                    imag_cont_prob = torch.sigmoid(imag_cont_logits).view(B, IMAG_HORIZON, 1)
                    discounts = (GAMMA * imag_cont_prob).clamp(0.0, 1.0)

                    Bh1 = B * (IMAG_HORIZON + 1)
                    all_deter_f = imag["deter"].reshape(Bh1, -1)
                    all_stoch_f = rssm.flatten_stoch(imag["stoch"].reshape(Bh1, rssm.C, rssm.K))
                    goal_h1 = goal0.unsqueeze(1).expand(B, IMAG_HORIZON + 1, 2).reshape(Bh1, 2)

                    with torch.no_grad():
                        target_v_symlog = twohot.mean(
                            target_critic(all_deter_f, all_stoch_f, goal_h1)
                        )
                        target_v = symexp(target_v_symlog).view(B, IMAG_HORIZON + 1, 1)

                    v_symlog = twohot.mean(critic(all_deter_f, all_stoch_f, goal_h1))
                    v = symexp(v_symlog).view(B, IMAG_HORIZON + 1, 1)

                    returns = lambda_return(
                        reward=imag_reward,
                        value=target_v[:, :-1],
                        discount=discounts,
                        lam=LAMBDA,
                        bootstrap=target_v[:, -1],
                        time_major=False,
                    )

                    weights = torch.cumprod(
                        torch.cat(
                            [torch.ones(B, 1, 1, device=DEVICE), discounts[:, :-1]],
                            dim=1,
                        ),
                        dim=1,
                    )

                    adv = returns - v[:, :-1].detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss = -(weights * adv).mean() - ENT_SCALE * (
                        weights * imag["ent"].unsqueeze(-1)
                    ).mean()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
                    actor_opt.step()

                finally:
                    for p in critic.parameters():
                        p.requires_grad_(True)
                    for p in wm_params:
                        p.requires_grad_(True)

                # ----- Critic update -----
                critic_opt.zero_grad(set_to_none=True)

                val_logits = critic(
                    prev_deter_f.detach(),
                    prev_stoch_f.detach(),
                    goal_h.detach(),
                )

                critic_loss = twohot.ce_loss(
                    val_logits,
                    symlog(returns.detach().view(-1)),
                )
                critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
                critic_opt.step()
                ema_update(target_critic, critic, TARGET_EMA)

                # ----- Logging -----
                if global_step % 20 == 0:
                    writer.add_scalar("Train/wm_loss", wm_loss.item(), global_step)
                    writer.add_scalar("Train/depth_nll_loss", post_depth_nll.item(), global_step)
                    writer.add_scalar("Train/sem_loss", post_sem_loss.item(), global_step)
                    writer.add_scalar("Train/kl_loss", kl_loss.item(), global_step)
                    writer.add_scalar("Train/overshoot_loss", overshoot_loss.item(), global_step)
                    writer.add_scalar("Train/critic_loss", critic_loss.item(), global_step)
                    writer.add_scalar("Train/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("Train/imag_return_mean", returns.mean().item(), global_step)
                    
                    with torch.no_grad():
                        _, prior_probs_dbg, _ = rssm.dist_from_logits_flat(
                            prior_logits_bt.reshape(B * T, -1)
                        )
                        prior_entropy = -(
                            prior_probs_dbg * (prior_probs_dbg + 1e-8).log()
                        ).sum(dim=-1).mean()

                        writer.add_scalar("Train/prior_entropy", prior_entropy.item(), global_step)

                if global_step % 100 == 0:
                    with torch.no_grad():
                        (
                            post_recon_depth_dbg, post_sem_logits_dbg,
                            prior_recon_depth_dbg, prior_sem_logits_dbg,
                        ) = decode_post_and_seeded_prior(
                            rssm=rssm,
                            decoder=decoder,
                            post_deter_seq=deter_seq,
                            post_stoch_seq=stoch_seq,
                            post_logits_bt=post_logits_bt,
                            actions_seq=prev_actions_seq,
                            goals_seq=goals_seq,
                            resets=resets,
                        )

                        log_recon_panels(
                            writer=writer,
                            global_step=global_step,
                            tag_prefix="Visuals_B",
                            depth_in=depth_in,
                            sem_ids=sem_ids,
                            post_recon_depth=post_recon_depth_dbg,
                            post_sem_logits=post_sem_logits_dbg,
                            prior_recon_depth=prior_recon_depth_dbg,
                            prior_sem_logits=prior_sem_logits_dbg,
                        )

                        if fixed_val_batch is not None:
                            v_depths, v_sems, v_vectors, v_goals, v_actions, v_rewards, v_dones = fixed_val_batch
                            v_depth_in, v_sem_ids, v_vec_in, v_goal_in, v_actions_seq, v_rewards_seq, v_dones_seq, v_goals_seq = preprocess_batch(
                                v_depths.to(DEVICE),
                                v_sems.to(DEVICE),
                                v_vectors.to(DEVICE),
                                v_goals.to(DEVICE),
                                v_actions.to(DEVICE),
                                v_rewards.to(DEVICE),
                                v_dones.to(DEVICE),
                            )

                            vB, vT = v_actions_seq.shape[:2]
                            v_embeds_flat = encoder(v_depth_in, v_sem_ids.unsqueeze(1), v_vec_in, v_goal_in)
                            v_embeds = v_embeds_flat.view(vB, vT, -1)

                            v_resets = make_resets_from_dones(v_dones_seq)
                            v_prev_actions_seq = torch.zeros_like(v_actions_seq)
                            v_prev_actions_seq[:, 1:] = v_actions_seq[:, :-1]
                            v_prev_actions_seq = v_prev_actions_seq * (1.0 - v_resets.float().unsqueeze(-1))

                            v_post = rssm.observe(v_embeds, v_prev_actions_seq, v_goals_seq, resets=v_resets)

                            (
                                v_post_recon_depth, v_post_sem_logits,
                                v_prior_recon_depth, v_prior_sem_logits,
                            ) = decode_post_and_seeded_prior(
                                rssm=rssm,
                                decoder=decoder,
                                post_deter_seq=v_post["deter"],
                                post_stoch_seq=v_post["stoch"],
                                post_logits_bt=v_post["post_logits"],
                                actions_seq=v_prev_actions_seq,
                                goals_seq=v_goals_seq,
                                resets=v_resets,
                            )

                            log_recon_panels(
                                writer=writer,
                                global_step=global_step,
                                tag_prefix="Visuals_B_Fixed",
                                depth_in=v_depth_in,
                                sem_ids=v_sem_ids,
                                post_recon_depth=v_post_recon_depth,
                                post_sem_logits=v_post_sem_logits,
                                prior_recon_depth=v_prior_recon_depth,
                                prior_sem_logits=v_prior_sem_logits,
                            )
                
                if global_step % IMAG_LOG_EVERY == 0:
                    with torch.no_grad():
                        start_deter = deter_seq[:, -1].detach()
                        start_stoch = stoch_seq[:, -1].detach()
                        goal0 = goals_seq[:, -1].detach()

                        log_imagination_rollout(
                            writer=writer,
                            global_step=global_step,
                            rssm=rssm,
                            decoder=decoder,
                            actor=actor,
                            start_deter=start_deter,
                            start_stoch=start_stoch,
                            goal0=goal0,
                            horizon=IMAG_LOG_HORIZON,
                            tag_prefix="Visuals_B",
                            num_examples=IMAG_LOG_EXAMPLES,
                        )

                if global_step % IMAG_LOG_EVERY == 0:
                    with torch.no_grad():
                        seed_t = max(0, T - 1 - IMAG_LOG_HORIZON)

                        start_deter = deter_seq[:, seed_t].detach()
                        start_stoch = stoch_seq[:, seed_t].detach()
                        start_post_logits = post_logits_bt[:, seed_t].detach()

                        future_actions = prev_actions_seq[:, seed_t + 1 : seed_t + 1 + IMAG_LOG_HORIZON].detach()
                        future_goals = goals_seq[:, seed_t + 1 : seed_t + 1 + IMAG_LOG_HORIZON].detach()

                        log_dataset_action_rollout(
                            writer=writer,
                            global_step=global_step,
                            rssm=rssm,
                            decoder=decoder,
                            start_deter=start_deter,
                            start_stoch=start_stoch,
                            start_post_logits=start_post_logits,
                            future_goals=future_goals,
                            future_actions=future_actions,
                            horizon=IMAG_LOG_HORIZON,
                            tag_prefix="Visuals_B",
                            num_examples=IMAG_LOG_EXAMPLES,
                        )
                
                pbar_steps.set_postfix({
                    "rew": f"{episode_reward:.1f}",
                    "act_L": f"{actor_loss.item():.2f}",
                })

            if done:
                break

        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.flush()
        print(f"Episode {episode} Complete | Reward: {episode_reward:.2f}")

        if episode % 50 == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "rssm": rssm.state_dict(),
                "decoder": decoder.state_dict(),
                "reward_head": reward_head.state_dict(),
                "cont_head": cont_head.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "wm_opt": wm_opt.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "global_step": global_step,
                "episode": episode,
            }, CKPT_PATH)
        
        if episode % 50 == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"dreamerv3_ep{episode}.pth")

            torch.save({
                "encoder": encoder.state_dict(),
                "rssm": rssm.state_dict(),
                "decoder": decoder.state_dict(),
                "reward_head": reward_head.state_dict(),
                "cont_head": cont_head.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "wm_opt": wm_opt.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "global_step": global_step,
                "episode": episode,
            }, ckpt_path)

if __name__ == "__main__":
    main()
