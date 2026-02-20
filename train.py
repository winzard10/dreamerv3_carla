# train.py
import os
import copy
import numpy as np
import torch
import carla
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
from env.carla_wrapper import CarlaEnv

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
SEQ_LEN = 10 # 50
BATCH_SIZE = 16
NUM_CLASSES = 28   # semantic ids [0..27] (ASSUMES your semantic actually is ids)
H, W = 160, 160

PHASE_A_STEPS = 20000 # 20000
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

    # Semantic:
    # IMPORTANT: this assumes sems already contains class IDs [0..NUM_CLASSES-1] stored in that single channel.
    # If your CARLA wrapper is giving color/palette values 0..255, you MUST fix the wrapper; clamping here will not create correct labels.
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
    encoder = MultiModalEncoder(latent_dim=1024, num_classes=NUM_CLASSES, sem_embed_dim=16).to(DEVICE)

    rssm = RSSM(
        deter_dim=512,
        act_dim=2,
        embed_dim=1024,   # must match encoder output
        goal_dim=2,
        stoch_categoricals=32,
        stoch_classes=32,
        unimix_ratio=0.01,
        kl_balance=0.8,
        free_nats=0.1,
    ).to(DEVICE)

    Z_DIM = rssm.stoch_dim  # C*K (default 1024)

    decoder = MultiModalDecoder(deter_dim=512, stoch_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    reward_head = RewardHead(
        deter_dim=512, stoch_dim=Z_DIM, goal_dim=2,
        hidden_dim=512, bins=BINS, vmin=VMIN, vmax=VMAX
    ).to(DEVICE)

    cont_head = ContinueHead(
        deter_dim=512, stoch_dim=Z_DIM, goal_dim=2, hidden_dim=512
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
        deter_dim=512, stoch_dim=Z_DIM, goal_dim=2,
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
        encoder.train(); rssm.train(); decoder.train(); reward_head.train(); cont_head.train()

        pbar = tqdm(range(PHASE_A_STEPS), desc="[Phase A] WM pretrain")
        for _ in pbar:
            batch = buffer.sample(BATCH_SIZE)
            if batch is None:
                continue

            depths, sems, vectors, goals, actions, rewards, dones = batch
            depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = preprocess_batch(
                depths, sems, vectors, goals, actions, rewards, dones
            )
            B, T = actions_seq.shape[0], actions_seq.shape[1]

            wm_opt.zero_grad(set_to_none=True)

            embeds_flat = encoder(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in)  # [B*T,E]
            embeds = embeds_flat.view(B, T, -1)

            resets = make_resets_from_dones(dones_seq)
            # --- NEW: Shift actions to align with causality (s_t = f(s_{t-1}, a_{t-1}, o_t)) ---
            prev_actions_seq = torch.zeros_like(actions_seq)
            prev_actions_seq[:, 1:] = actions_seq[:, :-1]
            
            # Zero out the previous action if a reset occurred (start of a new episode)
            prev_actions_seq = prev_actions_seq * (1.0 - resets.float().unsqueeze(-1))
            # -----------------------------------------------------------------------------------
            post = rssm.observe(embeds, prev_actions_seq, goals_seq, resets=resets)
            # print("post stoch shape:", post["stoch"].shape)

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

            reward_logits = reward_head(deter_flat, stoch_flat, goal_in)               # [B*T,BINS]
            reward_target_symlog = symlog(rewards_seq.reshape(-1))             # [B*T]
            reward_loss = twohot.ce_loss(reward_logits, reward_target_symlog)

            cont_logits   = cont_head(deter_flat, stoch_flat, goal_in)                    # [B*T,1]
            cont_target = (1.0 - dones_seq.float()).reshape(-1, 1)             # [B*T,1]
            cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
            
            depth_nll = gaussian_nll(depth_in, recon_depth, std=0.1).mean()

            wm_loss = (
                depth_nll
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
                writer.add_scalar("Pretrain/wm_loss", wm_loss.item(), global_step)
                writer.add_scalar("Pretrain/depth_loss", depth_loss.item(), global_step)
                writer.add_scalar("Pretrain/sem_loss", sem_loss.item(), global_step)
                writer.add_scalar("Pretrain/kl_loss", kl_loss.item(), global_step)
                writer.add_scalar("Pretrain/reward_loss", reward_loss.item(), global_step)
                writer.add_scalar("Pretrain/cont_loss", cont_loss.item(), global_step)

            pbar.set_postfix({"wm": f"{wm_loss.item():.3f}", "kl": f"{kl_loss.item():.3f}"})

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
        episode_reward = 0
        
        # Initialize RSSM state for the new episode
        prev_deter, prev_stoch = rssm.initial(1, device=DEVICE)
        prev_action = torch.zeros(1, 2, device=DEVICE)
        
        pbar_steps = tqdm(range(PHASE_B_STEPS), desc=f"Episode {episode}", leave=False) # Max 2000 steps per ep
        for step in pbar_steps:
            # --- 1. Environment Interaction (Perception & Action) ---
            with torch.no_grad():
                # Prepare observations (Batch Size 1)
                depth_in = torch.as_tensor(obs['depth'].copy()).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
                sem_ids = torch.as_tensor(obs['semantic'].copy()).permute(2,0,1).unsqueeze(0).long().to(DEVICE).clamp(0, NUM_CLASSES - 1)
                
                vec_in = torch.as_tensor(obs.get('vector', np.zeros(3)).copy()).unsqueeze(0).float().to(DEVICE)
                goal_in = torch.as_tensor(obs.get('goal', np.zeros(2)).copy()).unsqueeze(0).float().to(DEVICE)

                # Encode vision + state
                embed = encoder(depth_in, sem_ids, vec_in, goal_in)
                
                # Update RSSM state based on reality
                prev_deter, prev_stoch, _, _ = rssm.obs_step(prev_deter, prev_stoch, prev_action, embed, goal_in)
                # Get Action from Actor
                stoch_flat = rssm.flatten_stoch(prev_stoch)
                # sample=True applies the Actor's predicted std_dev for exploration noise!
                action_th, _, _, _ = actor(prev_deter, stoch_flat, goal_in, sample=True) 
                
            # Step CARLA
            act_np = action_th.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(act_np)
            done = terminated or truncated
            
            # DEBUG: Check if RSSM state is blowing up (NaN check)
            if torch.isnan(prev_deter).any():
                print("[CRITICAL] RSSM Deterministic state contains NaNs!")
            # DEBUG: Check Actor exploration
            # print(f"[DEBUG Online] Step {step} | Reward: {reward:.2f} | Action (Steer/Throt): {act_np[0]:.2f}/{act_np[1]:.2f}")
            
            # Update Spectator Camera (Optional, for viewing)
            v_transform = env.vehicle.get_transform()
            back_pos = v_transform.location - (v_transform.get_forward_vector() * 8.0) + carla.Location(z=4.0)
            spectator.set_transform(carla.Transform(back_pos, v_transform.rotation))

            # Store in Buffer
            buffer.add(obs['depth'], obs['semantic'], obs.get('vector', np.zeros(3)), 
                       obs.get('goal', np.zeros(2)), act_np, reward, done)
            
            episode_reward += reward
            obs = next_obs
            prev_action = action_th.detach()
            global_step += 1

            # --- 2. Model Updates (Learning from Buffer) ---
            if global_step % TRAIN_EVERY == 0 and buffer.idx > BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                if batch is None: continue
                depths, sems, vectors, goals, actions, rewards, dones = batch
                
                depth_in, sem_ids, vec_in, goal_in, actions_seq, rewards_seq, dones_seq, goals_seq = preprocess_batch(
                    depths, sems, vectors, goals, actions, rewards, dones
                )
                B, T = actions_seq.shape[0], actions_seq.shape[1]

                # ----- World Model Update -----
                encoder.train(); rssm.train(); decoder.train(); reward_head.train(); cont_head.train()
                wm_opt.zero_grad(set_to_none=True)

                embeds_flat = encoder(depth_in, sem_ids.unsqueeze(1), vec_in, goal_in)
                embeds = embeds_flat.view(B, T, -1)
                resets = make_resets_from_dones(dones_seq)
                # --- NEW: Shift actions to align with causality (s_t = f(s_{t-1}, a_{t-1}, o_t)) ---
                prev_actions_seq = torch.zeros_like(actions_seq)
                prev_actions_seq[:, 1:] = actions_seq[:, :-1]
                
                # Zero out the previous action if a reset occurred (start of a new episode)
                prev_actions_seq = prev_actions_seq * (1.0 - resets.float().unsqueeze(-1))
                # -----------------------------------------------------------------------------------
                post = rssm.observe(embeds, prev_actions_seq, goals_seq, resets=resets)

                deter_seq = post["deter"]
                stoch_seq = post["stoch"]
                deter_flat = deter_seq.reshape(B * T, -1)
                stoch_flat = rssm.flatten_stoch(stoch_seq.reshape(B * T, rssm.C, rssm.K))

                recon_depth, sem_logits = decoder(deter_flat, stoch_flat, out_hw=(H, W))
                depth_loss = F.mse_loss(recon_depth, depth_in)
                sem_loss = F.cross_entropy(sem_logits, sem_ids)
                kl_loss = rssm.kl_loss(post["post_logits"], post["prior_logits"])
                
                reward_logits = reward_head(deter_flat, stoch_flat, goal_in)          # [B*T, BINS]
                reward_loss = twohot.ce_loss(reward_logits, symlog(rewards_seq.reshape(-1)))

                cont_logits = cont_head(deter_flat, stoch_flat, goal_in)             # [B*T, 1]
                cont_target = (1.0 - dones_seq.float()).reshape(-1, 1)
                cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
                
                depth_nll = gaussian_nll(depth_in, recon_depth, std=0.1).mean()

                wm_loss = (depth_nll + SEM_SCALE * sem_loss + KL_SCALE * kl_loss + 
                           REWARD_SCALE * reward_loss + CONT_SCALE * cont_loss)
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(wm_params, max_norm=100.0)
                wm_opt.step()

                # ----- Actor/Critic Imagination -----
                actor.train(); critic.train()

                start_deter = deter_seq[:, -1].detach()
                start_stoch = stoch_seq[:, -1].detach()
                goal0 = goals_seq[:, -1].detach()

                # Freeze world model params
                for p in wm_params:
                    p.requires_grad_(False)

                try:
                    # Freeze critic params during actor update
                    for p in critic.parameters():
                        p.requires_grad_(False)

                    imag = rssm.imagine(start_deter, start_stoch, actor, goal0, horizon=IMAG_HORIZON)

                    # states
                    imag_deter = imag["deter"][:, :-1]   # s0..s_{H-1}  shape [B,H,D]
                    imag_stoch = imag["stoch"][:, :-1]   # s0..s_{H-1}
                    next_deter = imag["deter"][:, 1:]    # s1..s_H
                    next_stoch = imag["stoch"][:, 1:]

                    Bh = B * IMAG_HORIZON
                    imag_deter_f = imag_deter.reshape(Bh, -1)
                    imag_stoch_f = rssm.flatten_stoch(imag_stoch.reshape(Bh, rssm.C, rssm.K))

                    goal_h = goal0.unsqueeze(1).expand(B, IMAG_HORIZON, 2).reshape(Bh, 2)

                    # rewards/continue should be predicted at s_t (same length H)
                    imag_reward_symlog = twohot.mean(reward_head(imag_deter_f, imag_stoch_f, goal_h))
                    imag_reward = symexp(imag_reward_symlog).view(B, IMAG_HORIZON, 1)

                    imag_cont_logits = cont_head(imag_deter_f, imag_stoch_f, goal_h)
                    imag_cont_prob = torch.sigmoid(imag_cont_logits).view(B, IMAG_HORIZON, 1)
                    discounts = (GAMMA * imag_cont_prob).clamp(0.0, 1.0)

                    # values should be V(s_t) for t=0..H, so compute on imag["deter"] (H+1)
                    Bh1 = B * (IMAG_HORIZON + 1)
                    all_deter_f = imag["deter"].reshape(Bh1, -1)
                    all_stoch_f = rssm.flatten_stoch(imag["stoch"].reshape(Bh1, rssm.C, rssm.K))
                    goal_h1 = goal0.unsqueeze(1).expand(B, IMAG_HORIZON + 1, 2).reshape(Bh1, 2)

                    with torch.no_grad():
                        target_v_symlog = twohot.mean(target_critic(all_deter_f, all_stoch_f, goal_h1))
                        target_v = symexp(target_v_symlog).view(B, IMAG_HORIZON + 1, 1)

                    v_symlog = twohot.mean(critic(all_deter_f, all_stoch_f, goal_h1))
                    v = symexp(v_symlog).view(B, IMAG_HORIZON + 1, 1)

                    # lambda return uses V(s_{t+1}) inside recursion, but your function expects value indexed with reward
                    # easiest: pass value = target_v[:, :-1] (V(s_t)) and bootstrap = target_v[:, -1] (V(s_H))
                    returns = lambda_return(
                        reward=imag_reward,                   # [B,H,1] = r_t
                        value=target_v[:, :-1],               # [B,H,1] = V(s_t)
                        discount=discounts,                   # [B,H,1] = gamma_t
                        lam=LAMBDA,
                        bootstrap=target_v[:, -1],            # [B,1]   = V(s_H)
                        time_major=False
                    )  # [B,H,1]

                    # weight_t = prod_{i < t} discounts_i
                    # (prepend 1.0 for t=0)
                    weights = torch.cumprod(
                        torch.cat([torch.ones(B, 1, 1, device=DEVICE), discounts[:, :-1]], dim=1),
                        dim=1
                    )  # [B,H,1]
                    
                    # advantage baseline: compare returns vs V(s_t)
                    adv = returns - v[:, :-1].detach()

                    # V3-ish normalization (strongly recommended)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # ---------- actor loss: weighted advantage (baseline) + entropy bonus ----------
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss = -(weights * adv).mean() - ENT_SCALE * (weights * imag["ent"].unsqueeze(-1)).mean()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
                    actor_opt.step()
                    
                    assert imag_reward.shape == (B, IMAG_HORIZON, 1)
                    assert discounts.shape == (B, IMAG_HORIZON, 1)
                    assert target_v.shape == (B, IMAG_HORIZON + 1, 1)
                    assert v.shape == (B, IMAG_HORIZON + 1, 1)
                    assert returns.shape == (B, IMAG_HORIZON, 1)
                    assert imag["ent"].shape == (B, IMAG_HORIZON)

                finally:
                    for p in critic.parameters():
                        p.requires_grad_(True)
                    for p in wm_params:
                        p.requires_grad_(True)

                # Critic update (normal)
                critic_opt.zero_grad(set_to_none=True)

                # critic predicts logits in symlog-space distribution bins
                val_logits = critic(imag_deter_f.detach(), imag_stoch_f.detach(), goal_h.detach())   # [Bh, bins]

                # returns are in RAW space -> convert to symlog target inside ce_loss call
                critic_loss = twohot.ce_loss(val_logits, symlog(returns.detach().view(-1)))          # [Bh]
                critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
                critic_opt.step()
                ema_update(target_critic, critic, TARGET_EMA)

                # ----- Logging -----
                if global_step % 20 == 0:
                    writer.add_scalar("Train/wm_loss", wm_loss.item(), global_step)
                    writer.add_scalar("Train/depth_loss", depth_loss.item(), global_step)
                    writer.add_scalar("Train/kl_loss", kl_loss.item(), global_step)
                    writer.add_scalar("Train/critic_loss", critic_loss.item(), global_step)
                    writer.add_scalar("Train/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("Train/imag_return_mean", returns.mean().item(), global_step)
                    writer.add_histogram("Visuals/Imagined_Rewards", imag_reward, global_step)
                
                # # ----- DREAM VISUALIZATION WITH REWARD OVERLAY -----
                # if global_step % 100 == 0:
                #     with torch.no_grad():
                #         # 1. Prepare 15-step dream data for the first batch item
                #         seq_deter = imag["deter"][0]  
                #         seq_stoch = rssm.flatten_stoch(imag["stoch"][0]) 
                        
                #         # 2. Get the scores the Critic/Reward model gave this dream
                #         # imag_reward was [B, H, 1] -> we take [H]
                #         seq_rewards = imag_reward[0].squeeze(-1) 
                #         seq_conts = imag_cont_prob[0].squeeze(-1)
                        
                #         # 3. Decode into frames [H, 1, 160, 160]
                #         dream_depth, _ = decoder(seq_deter, seq_stoch, out_hw=(H, W))
                        
                #         # 4. Create a "HUD" (Heads-Up Display)
                #         # We'll darken a small square in the top-left and brighten it 
                #         # based on the reward value so you can "see" the score.
                #         hud_overlay = torch.ones_like(dream_depth)
                #         for t in range(IMAG_HORIZON):
                #             # Normalize reward for display (-3 to 1.1 -> 0 to 1)
                #             rew_norm = (seq_rewards[t] + 3.0) / 4.1
                #             # Draw a 20x20 pixel reward indicator in the corner
                #             hud_overlay[t, 0, :20, :20] = rew_norm
                #             # Draw a 20x20 pixel "life" indicator (Continue Prob)
                #             hud_overlay[t, 0, :20, 25:45] = seq_conts[t]
                        
                #         # Combine dream with the indicator
                #         vis_video = torch.clamp(dream_depth + hud_overlay * 0.5, 0, 1)
                #         writer.add_video("Visuals/Dream_with_Scores", vis_video.unsqueeze(0), global_step, fps=5)
                
                # # ----- NEW: Dream Visualization -----
                # # We do this slightly less often (every 100 steps) to save GPU memory
                # if global_step % 100 == 0:
                #     with torch.no_grad():
                #         # 1. Grab just the FIRST imaginary sequence in the batch to save compute
                #         # imag["deter"] shape: [Batch, Horizon, Dim]
                #         seq_deter = imag["deter"][0]  # [Horizon, Dim]
                #         seq_stoch = rssm.flatten_stoch(imag["stoch"][0]) # [Horizon, C*K]
                        
                #         # 2. Decode the imagined states into depth maps
                #         dream_depth, _ = decoder(seq_deter, seq_stoch, out_hw=(H, W)) # [Horizon, 1, 160, 160]
                        
                #         # 3. Format for TensorBoard Video: [Batch, Time, Channels, Height, Width]
                #         # We add a Batch dimension of 1
                #         video_tensor = dream_depth.unsqueeze(0) 
                        
                #         # 4. Add to TensorBoard at 5 frames per second
                #         writer.add_video("Visuals/Dreamed_Depth", video_tensor, global_step, fps=5)
                
                pbar_steps.set_postfix({"rew": f"{episode_reward:.1f}", "act_L": f"{actor_loss.item():.2f}"})

            if done: break
        
        # Episode end logging & saving
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        print(f"Episode {episode} Complete | Reward: {episode_reward:.2f}")
        
        if episode % 50 == 0:
            torch.save({
                "encoder": encoder.state_dict(), "rssm": rssm.state_dict(),
                "decoder": decoder.state_dict(), "reward_head": reward_head.state_dict(),
                "cont_head": cont_head.state_dict(), "actor": actor.state_dict(),
                "critic": critic.state_dict(), "target_critic": target_critic.state_dict(),
                "wm_opt": wm_opt.state_dict(), "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(), "global_step": global_step,
                "episode": episode
            }, CKPT_PATH)

if __name__ == "__main__":
    main()
