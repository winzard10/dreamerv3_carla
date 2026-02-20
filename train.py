# train.py
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.buffer import SequenceBuffer
from utils.lambda_returns import lambda_return
from utils.twohot import TwoHotDist, symlog, symexp

from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.decoder import MultiModalDecoder
from models.rewardhead import RewardHead
from models.continuehead import ContinueHead
from models.actor_critic import Actor, Critic

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
SEQ_LEN = 50
BATCH_SIZE = 16
NUM_CLASSES = 28   # semantic ids [0..27] (ASSUMES your semantic actually is ids)
H, W = 160, 160

# training
WM_LR = 8e-5
ACTOR_LR = 3e-5
CRITIC_LR = 3e-5

TRAIN_STEPS = 2000
IMAG_HORIZON = 15
GAMMA = 0.99
LAMBDA = 0.95

# loss scales
DEPTH_SCALE = 1000.0
SEM_SCALE = 10.0
REWARD_SCALE = 1.0
CONT_SCALE = 1.0
KL_SCALE = 1.0
ENT_SCALE = 1e-3

# twohot support
BINS = 255
VMIN = -20.0
VMAX = 20.0

# target critic EMA
TARGET_EMA = 0.99

# checkpoints
LOAD_PRETRAINED = True
CKPT_DIR = "checkpoints/dreamerv3"
CKPT_PATH = os.path.join(CKPT_DIR, "dreamerv3_latest.pth")


def ema_update(target: torch.nn.Module, online: torch.nn.Module, tau: float):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))


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

    # Semantic:
    # IMPORTANT: this assumes sems already contains class IDs [0..NUM_CLASSES-1] stored in that single channel.
    # If your CARLA wrapper is giving color/palette values 0..255, you MUST fix the wrapper; clamping here will not create correct labels.
    sem_ids = sems.reshape(B * T, H, W).to(dtype=torch.long)
    sem_ids = torch.clamp(sem_ids, 0, NUM_CLASSES - 1)

    # Encoder semantic input uses normalized IDs channel
    sem_in = sem_ids.unsqueeze(1).to(dtype=torch.float32) / float(NUM_CLASSES - 1)

    # Vector/goal for encoder (flatten time)
    vec_in = vectors.reshape(B * T, -1).to(dtype=torch.float32)
    goal_in = goals.reshape(B * T, -1).to(dtype=torch.float32)

    # Sequence tensors (keep time)
    actions_seq = actions.to(dtype=torch.float32)      # [B,T,2]
    rewards_seq = rewards.to(dtype=torch.float32)      # [B,T]
    dones_seq = dones.to(dtype=torch.bool)             # [B,T]
    goals_seq = goals.to(dtype=torch.float32)          # [B,T,2]

    return depth_in, sem_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq


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


def phase_a_pretrain_world_model(
    buffer: SequenceBuffer,
    encoder, rssm, decoder, reward_head, cont_head,
    wm_opt,
    twohot: TwoHotDist,
    writer: SummaryWriter,
    steps: int,
    global_step: int,
    save_path: str,
):
    encoder.train(); rssm.train(); decoder.train(); reward_head.train(); cont_head.train()

    pbar = tqdm(range(steps), desc="[Phase A] WM pretrain")
    for _ in pbar:
        batch = buffer.sample(BATCH_SIZE)
        if batch is None:
            continue

        depths, sems, vectors, goals, actions, rewards, dones = batch
        depth_in, sem_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = preprocess_batch(
            depths, sems, vectors, goals, actions, rewards, dones
        )
        B, T = actions_seq.shape[0], actions_seq.shape[1]

        wm_opt.zero_grad(set_to_none=True)

        embeds_flat = encoder(depth_in, sem_in, vec_in, goal_in)  # [B*T,E]
        embeds = embeds_flat.view(B, T, -1)

        resets = make_resets_from_dones(dones_seq)
        post = rssm.observe(embeds, actions_seq, goals_seq, resets=resets)

        deter_seq = post["deter"]              # [B,T,D]
        stoch_seq = post["stoch"]              # [B,T,C,K]
        post_logits = post["post_logits"]      # [B,T,C*K]
        prior_logits = post["prior_logits"]    # [B,T,C*K]

        deter_flat = deter_seq.reshape(B * T, -1)
        stoch_flat = rssm.flatten_stoch(stoch_seq.reshape(B * T, rssm.C, rssm.K))

        recon_depth, sem_logits = decoder(deter_flat, stoch_flat, out_hw=(H, W))

        depth_loss = F.mse_loss(recon_depth, depth_in)
        sem_loss = F.cross_entropy(sem_logits, sem_ids)

        kl_loss = rssm.kl_loss(post_logits, prior_logits)

        reward_logits = reward_head(deter_flat, stoch_flat)               # [B*T,BINS]
        reward_target_symlog = symlog(rewards_seq.reshape(-1))             # [B*T]
        reward_loss = twohot.ce_loss(reward_logits, reward_target_symlog)

        cont_logits = cont_head(deter_flat, stoch_flat)                    # [B*T,1]
        cont_target = (1.0 - dones_seq.float()).reshape(-1, 1)             # [B*T,1]
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        wm_loss = (
            DEPTH_SCALE * depth_loss
            + SEM_SCALE * sem_loss
            + KL_SCALE * kl_loss
            + REWARD_SCALE * reward_loss
            + CONT_SCALE * cont_loss
        )
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters())
            + list(reward_head.parameters()) + list(cont_head.parameters()),
            max_norm=100.0
        )
        wm_opt.step()

        global_step += 1
        if global_step % 10 == 0:
            writer.add_scalar("A/wm_loss", wm_loss.item(), global_step)
            writer.add_scalar("A/depth_loss", depth_loss.item(), global_step)
            writer.add_scalar("A/sem_loss", sem_loss.item(), global_step)
            writer.add_scalar("A/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("A/reward_loss", reward_loss.item(), global_step)
            writer.add_scalar("A/cont_loss", cont_loss.item(), global_step)

        pbar.set_postfix({"wm": f"{wm_loss.item():.3f}", "kl": f"{kl_loss.item():.3f}"})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "rssm": rssm.state_dict(),
        "decoder": decoder.state_dict(),
        "reward_head": reward_head.state_dict(),
        "cont_head": cont_head.state_dict(),
        "wm_opt": wm_opt.state_dict(),
        "global_step": global_step,
    }, save_path)

    return global_step


def main():
    print("Device:", DEVICE)

    # -----------------------
    # Buffer (offline training only)
    # -----------------------
    buffer = SequenceBuffer(capacity=100000, seq_len=SEQ_LEN, device=DEVICE)
    buffer.load_from_disk("./data/expert_sequences")

    writer = SummaryWriter(log_dir="./runs/dreamerv3_carla")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # -----------------------
    # Models
    # -----------------------
    encoder = MultiModalEncoder(latent_dim=1024).to(DEVICE)

    rssm = RSSM(
        deter_dim=512,
        act_dim=2,
        embed_dim=1024,   # must match encoder output
        goal_dim=2,
        stoch_categoricals=32,
        stoch_classes=32,
        unimix_ratio=0.01,
        kl_balance=0.8,
        free_nats=1.0,
    ).to(DEVICE)

    Z_DIM = rssm.stoch_dim  # C*K (default 1024)

    decoder = MultiModalDecoder(deter_dim=512, stoch_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    reward_head = RewardHead(
        deter_dim=512, stoch_dim=Z_DIM, goal_dim=0,
        hidden_dim=512, bins=BINS, vmin=VMIN, vmax=VMAX
    ).to(DEVICE)

    cont_head = ContinueHead(
        deter_dim=512, stoch_dim=Z_DIM, goal_dim=0, hidden_dim=512
    ).to(DEVICE)

    # ✅ FIXED actor init (no state_dim kwarg)
    actor = Actor(
        deter_dim=512,
        stoch_dim=Z_DIM,
        goal_dim=2,      # keep if you want goal-conditioned behavior
        action_dim=2,
        hidden_dim=512,
        min_std=0.1,
        init_std=1.0,
    ).to(DEVICE)

    critic = Critic(
        deter_dim=512, stoch_dim=Z_DIM, goal_dim=0,
        hidden_dim=512, bins=BINS, vmin=VMIN, vmax=VMAX
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
    PHASE_A_STEPS = 20000
    PHASE_A_PATH = "checkpoints/world_model/world_model_pretrained.pth"

    if LOAD_PRETRAINED and os.path.exists(PHASE_A_PATH):
        ckptA = torch.load(PHASE_A_PATH, map_location=DEVICE)
        encoder.load_state_dict(ckptA["encoder"])
        rssm.load_state_dict(ckptA["rssm"])
        decoder.load_state_dict(ckptA["decoder"])
        reward_head.load_state_dict(ckptA["reward_head"])
        cont_head.load_state_dict(ckptA["cont_head"])
        wm_opt.load_state_dict(ckptA["wm_opt"])
        global_step = ckptA.get("global_step", 0)
        print(f"Loaded Phase A world model @ step {global_step}")
    else:
        global_step = phase_a_pretrain_world_model(
            buffer=buffer,
            encoder=encoder, rssm=rssm, decoder=decoder,
            reward_head=reward_head, cont_head=cont_head,
            wm_opt=wm_opt,
            twohot=twohot,
            writer=writer,
            steps=PHASE_A_STEPS,
            global_step=global_step,
            save_path=PHASE_A_PATH,
        )

    # -----------------------
    # Load full checkpoint (optional)
    # -----------------------
    if LOAD_PRETRAINED and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
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
    # Train loop (offline)
    # -----------------------
    pbar = tqdm(range(TRAIN_STEPS), desc="DreamerV3 train")
    for _ in pbar:
        batch = buffer.sample(BATCH_SIZE)
        if batch is None:
            continue

        depths, sems, vectors, goals, actions, rewards, dones = batch
        depth_in, sem_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = preprocess_batch(
            depths, sems, vectors, goals, actions, rewards, dones
        )
        B, T = actions_seq.shape[0], actions_seq.shape[1]

        # -----------------------
        # World Model update
        # -----------------------
        encoder.train(); rssm.train(); decoder.train(); reward_head.train(); cont_head.train()
        wm_opt.zero_grad(set_to_none=True)

        embeds_flat = encoder(depth_in, sem_in, vec_in, goal_in)   # [B*T,E]
        embeds = embeds_flat.view(B, T, -1)

        resets = make_resets_from_dones(dones_seq)
        post = rssm.observe(embeds, actions_seq, goals_seq, resets=resets)

        deter_seq = post["deter"]
        stoch_seq = post["stoch"]
        post_logits = post["post_logits"]
        prior_logits = post["prior_logits"]

        deter_flat = deter_seq.reshape(B * T, -1)
        stoch_flat = rssm.flatten_stoch(stoch_seq.reshape(B * T, rssm.C, rssm.K))

        recon_depth, sem_logits = decoder(deter_flat, stoch_flat, out_hw=(H, W))

        depth_loss = F.mse_loss(recon_depth, depth_in)
        sem_loss = F.cross_entropy(sem_logits, sem_ids)
        kl_loss = rssm.kl_loss(post_logits, prior_logits)

        reward_logits = reward_head(deter_flat, stoch_flat)
        reward_target_symlog = symlog(rewards_seq.reshape(-1))
        reward_loss = twohot.ce_loss(reward_logits, reward_target_symlog)

        cont_logits = cont_head(deter_flat, stoch_flat)
        cont_target = (1.0 - dones_seq.float()).reshape(-1, 1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        wm_loss = (
            DEPTH_SCALE * depth_loss
            + SEM_SCALE * sem_loss
            + KL_SCALE * kl_loss
            + REWARD_SCALE * reward_loss
            + CONT_SCALE * cont_loss
        )
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm_params, max_norm=100.0)
        wm_opt.step()

        # -----------------------
        # Imagination (behavior learning)
        # -----------------------
        actor.train(); critic.train()

        start_deter = deter_seq[:, -1].detach()
        start_stoch = stoch_seq[:, -1].detach()
        goal0 = goals_seq[:, -1].detach()

        imag = rssm.imagine(start_deter, start_stoch, actor, goal0, horizon=IMAG_HORIZON)
        imag_deter = imag["deter"]    # [B,H,D]
        imag_stoch = imag["stoch"]    # [B,H,C,K]
        imag_ent = imag["ent"]        # [B,H]

        Bh = B * IMAG_HORIZON
        imag_deter_f = imag_deter.reshape(Bh, -1)
        imag_stoch_f = rssm.flatten_stoch(imag_stoch.reshape(Bh, rssm.C, rssm.K))

        # reward prediction
        imag_reward_logits = reward_head(imag_deter_f, imag_stoch_f)                # [Bh,BINS]
        imag_reward_symlog = twohot.mean(imag_reward_logits).reshape(B, IMAG_HORIZON, 1)
        imag_reward = symexp(imag_reward_symlog)                                    # [B,H,1]

        # continuation prediction
        imag_cont_logits = cont_head(imag_deter_f, imag_stoch_f).reshape(B, IMAG_HORIZON, 1)
        imag_cont_prob = torch.sigmoid(imag_cont_logits)
        discounts = (GAMMA * imag_cont_prob).clamp(0.0, 1.0)

        # target critic values
        with torch.no_grad():
            target_val_logits = target_critic(imag_deter_f, imag_stoch_f)
            target_val_symlog = twohot.mean(target_val_logits).reshape(B, IMAG_HORIZON, 1)
            target_val = symexp(target_val_symlog)

        bootstrap = target_val[:, -1, :]  # [B,1]
        returns = lambda_return(
            reward=imag_reward,
            value=target_val,
            discount=discounts,
            lam=LAMBDA,
            bootstrap=bootstrap,
            time_major=False,
        ).detach()

        # -----------------------
        # Critic update
        # -----------------------
        critic_opt.zero_grad(set_to_none=True)

        val_logits = critic(imag_deter_f.detach(), imag_stoch_f.detach())
        val_target_symlog = symlog(returns.reshape(-1))
        critic_loss = twohot.ce_loss(val_logits, val_target_symlog)

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
        critic_opt.step()

        ema_update(target_critic, critic, TARGET_EMA)

        # -----------------------
        # Actor update (simple Dreamer-style objective)
        # -----------------------
        actor_opt.zero_grad(set_to_none=True)

        val_logits_for_actor = critic(imag_deter_f, imag_stoch_f)
        val_symlog_for_actor = twohot.mean(val_logits_for_actor)
        val_for_actor = symexp(val_symlog_for_actor).view(B, IMAG_HORIZON, 1)

        actor_loss = -(val_for_actor.mean() + ENT_SCALE * imag_ent.mean())

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
        actor_opt.step()

        # -----------------------
        # Logging
        # -----------------------
        global_step += 1
        if global_step % 10 == 0:
            writer.add_scalar("wm/loss", wm_loss.item(), global_step)
            writer.add_scalar("wm/depth_loss", depth_loss.item(), global_step)
            writer.add_scalar("wm/sem_loss", sem_loss.item(), global_step)
            writer.add_scalar("wm/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("wm/reward_loss", reward_loss.item(), global_step)
            writer.add_scalar("wm/cont_loss", cont_loss.item(), global_step)
            writer.add_scalar("beh/critic_loss", critic_loss.item(), global_step)
            writer.add_scalar("beh/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("beh/imag_return_mean", returns.mean().item(), global_step)
            writer.add_scalar("beh/imag_entropy", imag_ent.mean().item(), global_step)

        pbar.set_postfix({"wm": f"{wm_loss.item():.3f}", "kl": f"{kl_loss.item():.3f}", "V": f"{returns.mean().item():.2f}"})

        # -----------------------
        # Save
        # -----------------------
        if global_step % 200 == 0:
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
            }, CKPT_PATH)

    print("Done.")


if __name__ == "__main__":
    main()