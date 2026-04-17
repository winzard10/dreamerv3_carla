# train.py
import os
import copy
import numpy as np
import torch
import carla

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from params import *
from utils.train_utils import (
    clone_batch_to_cpu, save_checkpoint,
    compute_rssm_out, world_model_step, actor_critic_step,
    log_scalars, log_visuals,
    log_dataset_action_rollout, log_imagination_rollout,
)
from utils.buffer import SequenceBuffer
from utils.twohot import TwoHotDist

from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.decoder import MultiModalDecoder
from models.rewardhead import RewardHead
from models.continuehead import ContinueHead
from models.actor_critic import Actor, Critic
from env.carla_wrapper import CarlaEnv


def main():
    print("Device:", DEVICE)

    # ------------------------------------------------------------------
    # Buffer + fixed validation batch
    # ------------------------------------------------------------------
    buffer = SequenceBuffer(capacity=100000, seq_len=SEQ_LEN, device=DEVICE)
    buffer.load_from_disk("./data/expert_sequences")

    fixed_val_batch = None
    if FIXED_VAL_ENABLED:
        tmp = buffer.sample(BATCH_SIZE)
        if tmp is not None:
            fixed_val_batch = clone_batch_to_cpu(tmp)
            print("[Info] Fixed validation batch captured.")

    writer = SummaryWriter(log_dir="./runs/dreamerv3_carla")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    encoder = MultiModalEncoder(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, sem_embed_dim=16).to(DEVICE)

    rssm = RSSM(
        deter_dim=DETER_DIM, act_dim=2, embed_dim=EMBED_DIM, goal_dim=2,
        stoch_categoricals=STOCH_CATEGORICALS, stoch_classes=STOCH_CLASSES,
        unimix_ratio=0.01, kl_balance=0.8, free_nats=0.5,
    ).to(DEVICE)

    Z_DIM   = rssm.stoch_dim
    decoder = MultiModalDecoder(deter_dim=DETER_DIM, stoch_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    reward_head = RewardHead(deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2,
                          hidden_dim=512, bins=BINS).to(DEVICE)
    cont_head     = ContinueHead(deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2, hidden_dim=512).to(DEVICE)
    actor         = Actor(deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2, action_dim=2,
                          hidden_dim=512, min_std=0.1, init_std=1.0).to(DEVICE)
    critic        = Critic(deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2, hidden_dim=512, bins=BINS).to(DEVICE)
    target_critic = copy.deepcopy(critic).to(DEVICE)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    twohot = TwoHotDist(num_bins=BINS, vmin=VMIN, vmax=VMAX, device=DEVICE).to(DEVICE)

    wm_params = (list(encoder.parameters()) + list(rssm.parameters()) +
                 list(decoder.parameters()) + list(reward_head.parameters()) +
                 list(cont_head.parameters()))

    wm_opt     = torch.optim.Adam(wm_params, lr=WM_LR)
    actor_opt  = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR, weight_decay=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, weight_decay=1e-4)

    all_models = dict(encoder=encoder, rssm=rssm, decoder=decoder,
                      reward_head=reward_head, cont_head=cont_head,
                      actor=actor, critic=critic, target_critic=target_critic)
    all_opts   = dict(wm=wm_opt, actor=actor_opt, critic=critic_opt)

    # ------------------------------------------------------------------
    # Phase A: World model pretraining
    # ------------------------------------------------------------------
    global_step = 0

    if LOAD_PRETRAINED and os.path.exists(PHASE_A_PATH):
        ckptA = torch.load(PHASE_A_PATH, map_location=DEVICE, weights_only=False)
        for name in ["encoder", "rssm", "decoder", "reward_head", "cont_head"]:
            all_models[name].load_state_dict(ckptA[name])
        wm_opt.load_state_dict(ckptA["wm_opt"])
        global_step = ckptA.get("global_step", 0)
        print(f"Loaded Phase A checkpoint @ step {global_step}")
    else:
        for m in [encoder, rssm, decoder, reward_head, cont_head]:
            m.train()
        
        fixed_rssm_out = None
        pbar = tqdm(range(PHASE_A_STEPS), desc="[Phase A] WM pretrain")
        for _ in pbar:
            batch = buffer.sample(BATCH_SIZE)
            if batch is None:
                continue

            losses, rssm_out = world_model_step(
                batch, encoder, rssm, decoder, reward_head, cont_head,
                wm_opt, twohot, wm_params
            )
            global_step += 1

            if global_step % 10 == 0:
                log_scalars(writer, global_step, losses, rssm_out, rssm, phase="Pretrain")

            if global_step % 100 == 0:
                with torch.no_grad():
                    if fixed_val_batch is not None:
                        fixed_rssm_out = compute_rssm_out(
                            [x.to(DEVICE) for x in fixed_val_batch], encoder, rssm
                        )
                    log_visuals(writer, global_step, rssm_out, rssm, decoder,
                                "Visuals_A", fixed_rssm_out)
                save_checkpoint(
                    PHASE_A_PATH,
                    {k: all_models[k] for k in ["encoder", "rssm", "decoder", "reward_head", "cont_head"]},
                    {"wm": wm_opt}, global_step
                )

            if global_step % IMAG_LOG_EVERY == 0:
                seed_t = max(0, rssm_out["T"] - 1 - IMAG_LOG_HORIZON)
                with torch.no_grad():
                    log_dataset_action_rollout(
                        writer, global_step, rssm, decoder,
                        start_deter=rssm_out["deter_seq"][:, seed_t].detach(),
                        start_stoch=rssm_out["stoch_seq"][:, seed_t].detach(),
                        start_post_logits=rssm_out["post_logits_bt"][:, seed_t].detach(),
                        future_goals=rssm_out["goals_seq"][:, seed_t+1:seed_t+1+IMAG_LOG_HORIZON].detach(),
                        future_actions=rssm_out["prev_actions_seq"][:, seed_t+1:seed_t+1+IMAG_LOG_HORIZON].detach(),
                        horizon=IMAG_LOG_HORIZON, tag_prefix="Visuals_A", num_examples=IMAG_LOG_EXAMPLES,
                    )

            pbar.set_postfix({"wm": f"{losses['wm'].item():.3f}", "kl": f"{losses['kl'].item():.3f}"})
            writer.flush()

    # ------------------------------------------------------------------
    # Phase B: Online training
    # ------------------------------------------------------------------
    ckpt = {}
    if LOAD_PRETRAINED and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        for name, model in all_models.items():
            model.load_state_dict(ckpt[name])
        wm_opt.load_state_dict(ckpt["wm_opt"])
        actor_opt.load_state_dict(ckpt["actor_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        global_step = ckpt.get("global_step", global_step)
        print(f"Loaded Phase B checkpoint @ step {global_step}")

    print("\n[Phase B] Starting Online Interaction with CARLA...")
    env       = CarlaEnv()
    spectator = env.world.get_spectator()

    fixed_rssm_out_B = None
    for episode in range(ckpt.get("episode", 0) + 1, PART_B_EPISODE + 1):
        obs, _         = env.reset()
        episode_reward = 0.0
        prev_deter, prev_stoch = rssm.initial(1, device=DEVICE)
        prev_action = torch.zeros(1, 2, device=DEVICE)

        for m in all_models.values():
            m.train()

        pbar_steps = tqdm(range(PHASE_B_STEPS), desc=f"Episode {episode}", leave=False)
        for step in pbar_steps:
            # Environment interaction
            with torch.no_grad():
                depth_in = torch.as_tensor(obs["depth"].copy()).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
                sem_ids  = torch.as_tensor(obs["semantic"].copy()).permute(2,0,1).unsqueeze(0).long().to(DEVICE).clamp(0, NUM_CLASSES-1)
                vec_in   = torch.as_tensor(obs.get("vector", np.zeros(3)).copy()).unsqueeze(0).float().to(DEVICE)
                goal_in  = torch.as_tensor(obs.get("goal", np.zeros(2)).copy()).unsqueeze(0).float().to(DEVICE)

                embed = encoder(depth_in, sem_ids, vec_in, goal_in)
                prev_deter, prev_stoch, _, _ = rssm.obs_step(prev_deter, prev_stoch, prev_action, embed, goal_in)
                action_th, _, _, _ = actor(prev_deter, rssm.flatten_stoch(prev_stoch), goal_in, sample=True)

            act_np = action_th.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            v_t = env.vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                v_t.location - v_t.get_forward_vector() * 8.0 + carla.Location(z=4.0),
                v_t.rotation
            ))

            buffer.add(obs["depth"], obs["semantic"],
                       obs.get("vector", np.zeros(3)), obs.get("goal", np.zeros(2)),
                       act_np, reward, done)

            episode_reward += reward
            obs         = next_obs
            prev_action = action_th.detach()
            global_step += 1

            # Model updates
            if global_step % TRAIN_EVERY == 0 and buffer.idx > BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                if batch is None:
                    continue

                losses, rssm_out = world_model_step(
                    batch, encoder, rssm, decoder, reward_head, cont_head,
                    wm_opt, twohot, wm_params
                )
                ac_losses = actor_critic_step(
                    rssm_out, rssm, reward_head, cont_head, actor, critic,
                    target_critic, actor_opt, critic_opt, twohot, wm_params
                )

                if global_step % 20 == 0:
                    log_scalars(writer, global_step, losses, rssm_out, rssm, phase="Train")
                    writer.add_scalar("Train/actor_loss",       ac_losses["actor"].item(),   global_step)
                    writer.add_scalar("Train/critic_loss",      ac_losses["critic"].item(),  global_step)
                    writer.add_scalar("Train/imag_return_mean", ac_losses["returns"].item(), global_step)

                if global_step % 100 == 0:
                    with torch.no_grad():
                        if fixed_val_batch is not None:
                            fixed_rssm_out_B = compute_rssm_out(
                                [x.to(DEVICE) for x in fixed_val_batch], encoder, rssm
                            )
                        log_visuals(writer, global_step, rssm_out, rssm, decoder,
                                    "Visuals_B", fixed_rssm_out_B)

                if global_step % IMAG_LOG_EVERY == 0:
                    with torch.no_grad():
                        log_imagination_rollout(
                            writer, global_step, rssm, decoder, actor,
                            rssm_out["deter_seq"][:, -1].detach(),
                            rssm_out["stoch_seq"][:, -1].detach(),
                            rssm_out["goals_seq"][:, -1].detach(),
                            horizon=IMAG_LOG_HORIZON, tag_prefix="Visuals_B",
                            num_examples=IMAG_LOG_EXAMPLES,
                        )

                pbar_steps.set_postfix({
                    "rew":   f"{episode_reward:.1f}",
                    "act_L": f"{ac_losses['actor'].item():.3f}",
                })

            if done:
                break

        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.flush()
        print(f"Episode {episode} | Reward: {episode_reward:.2f}")

        if episode % 50 == 0:
            save_checkpoint(CKPT_PATH, all_models, all_opts, global_step, episode)
            save_checkpoint(
                os.path.join(CKPT_DIR, f"dreamerv3_ep{episode}.pth"),
                all_models, all_opts, global_step, episode
            )


if __name__ == "__main__":
    main()