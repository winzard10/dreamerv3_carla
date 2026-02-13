import os
import carla
import torch
import torch.nn.functional as F
import numpy as np
from env.carla_wrapper import CarlaEnv
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor, Critic
from models.decoder import MultiModalDecoder
from utils.buffer import SequenceBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- Configuration & Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 50 
BATCH_SIZE = 16
HORIZON = 15
LEARNING_RATE = 3e-4
LAMBDA = 0.95 

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def compute_lambda_returns(rewards, values, discount=0.99, lambd=0.95):
    """Calculates targets for the Critic inside the 'dream'"""
    # values shape: [HORIZON, Batch, 1]
    returns = torch.zeros_like(values)
    
    # The return for the very last step is just the Critic's prediction (Bootstrap)
    last_v = values[-1]
    returns[-1] = last_v
    
    # Iterate backwards from the second-to-last step (HORIZON - 2) down to 0
    # This avoids the "index out of bounds" error
    for t in reversed(range(len(returns) - 1)):
        returns[t] = rewards[t] + discount * ((1 - lambd) * values[t+1] + lambd * last_v)
        last_v = returns[t]
        
    return returns

def train():
    # 1. Setup Environment & Buffer
    env = CarlaEnv()
    buffer = SequenceBuffer(capacity=20000, seq_len=SEQ_LEN, device=DEVICE)
    buffer.load_from_disk("./data/expert_sequences")
    writer = SummaryWriter(log_dir="./runs/dreamerv3_carla")
    global_step = 0

    # 2. Model Initialization
    encoder = MultiModalEncoder().to(DEVICE)
    rssm = RSSM(hidden_dim=512).to(DEVICE)
    # Ensure decoder is initialized with correct dims
    decoder = MultiModalDecoder(deter_dim=512, stoch_dim=32).to(DEVICE) 
    actor = Actor(hidden_dim=512).to(DEVICE)
    critic = Critic(hidden_dim=512).to(DEVICE)

    # 3. Optimizers
    wm_opt = torch.optim.Adam(list(encoder.parameters()) + 
                              list(rssm.parameters()) + 
                              list(decoder.parameters()), lr=LEARNING_RATE)
    ac_opt = torch.optim.Adam(list(actor.parameters()) + 
                              list(critic.parameters()), lr=8e-5)

    # --- PHASE A: Pre-training (Expert Knowledge) ---
    print("\n[Phase A] Pre-training on Expert PID Data...")
    pbar_pre = tqdm(range(500), desc="Pre-training")
    for step_pre in pbar_pre:
        # Sample 6 items
        depths, sems, vectors, actions, rewards, _ = buffer.sample(BATCH_SIZE)
        
        wm_opt.zero_grad()
        
        # Flatten Batch and Time for Encoder
        B, T, C, H, W = depths.shape
        flat_depth = depths.view(B * T, C, H, W)
        flat_sem = sems.view(B * T, C, H, W)
        flat_vec = vectors.view(B * T, -1) 

        flat_embeds = encoder(flat_depth, flat_sem, flat_vec)
        embeds = flat_embeds.view(B, T, -1)
        
        # RSSM Forward
        (post_h, post_z), (prior_h, prior_z) = rssm.observe_sequence(embeds, actions)
    
        # Decoder Forward
        recon_depth, recon_sem = decoder(post_h, post_z)
        
        # --- FIX: Flatten Targets to match Decoder Output [800, 1, 160, 160] ---
        target_depth = depths.view(-1, 1, 160, 160)
        target_sem = sems.view(-1, 1, 160, 160)
        
        # Loss Calculation (Using MSE for Semantics since output is 1 channel scalar)
        loss_wm = F.mse_loss(recon_depth, target_depth) + \
                  F.mse_loss(recon_sem, target_sem) + \
                  rssm.kl_loss(post_z, prior_z) 
        
        loss_wm.backward()
        wm_opt.step()
        
        if step_pre % 10 == 0:
            writer.add_scalar("Pretrain/WM_Loss", loss_wm.item(), step_pre)
            writer.add_scalar("Pretrain/KL_Loss", rssm.kl_loss(post_z, prior_z).item(), step_pre)
        
        pbar_pre.set_postfix({"WM_Loss": f"{loss_wm.item():.4f}"})

    # --- PHASE B: Online Training (Driving & Dreaming) ---
    print("\n[Phase B] Starting Online Interaction...")
    spectator = env.world.get_spectator()
    for episode in range(100):
        obs = env.reset()
        h = rssm.get_initial_state(1, DEVICE)
        episode_reward = 0
        
        pbar_steps = tqdm(range(500), desc=f"Episode {episode}", leave=False)
        for step in pbar_steps:
            # 1. Action Selection
            with torch.no_grad():
                # Add .copy() to depth, semantic, and vector
                depth_in = torch.as_tensor(obs['depth'].copy()).to(DEVICE).float().permute(2,0,1).unsqueeze(0) / 255.0
                sem_in = torch.as_tensor(obs['semantic'].copy()).to(DEVICE).float().permute(2,0,1).unsqueeze(0)
                # Ensure vector is also copied if it comes from numpy
                vec_val = obs.get('vector', [0,0,0])
                if isinstance(vec_val, np.ndarray):
                    vec_val = vec_val.copy()
                vec_in = torch.as_tensor(vec_val).to(DEVICE).float().unsqueeze(0)

                embed = encoder(depth_in, sem_in, vec_in)
                z = rssm.representation_model(torch.cat([h, embed], dim=-1)) 
                action = actor(h, z)
            
            act_np = action.cpu().numpy()[0]
            next_obs, reward, done, _ = env.step(act_np)
            
            # Access the vehicle from your env wrapper
            v_transform = env.vehicle.get_transform()
            forward_vec = v_transform.get_forward_vector()
            
            # Calculate position (8m back, 4m up)
            back_pos = v_transform.location - (forward_vec * 8.0) + carla.Location(z=4.0)
            
            # Update camera
            spectator.set_transform(carla.Transform(back_pos, v_transform.rotation))
            
            buffer.add(obs['depth'], obs['semantic'], obs.get('vector', np.zeros(3)), act_np, reward, done)
            episode_reward += reward
            
            h = rssm.gru(torch.cat([z, action], dim=-1), h)
            obs = next_obs
            
            global_step += 1

            # 2. Update Loop
            if step % 5 == 0 and buffer.idx > BATCH_SIZE:
                depths, sems, vectors, actions, rewards, _ = buffer.sample(BATCH_SIZE)
                
                wm_opt.zero_grad()
                
                B_dim, T_dim, C, H, W = depths.shape
                flat_depth = depths.view(B_dim * T_dim, C, H, W)
                flat_sem = sems.view(B_dim * T_dim, C, H, W)
                flat_vec = vectors.view(B_dim * T_dim, -1)
                
                flat_embeds = encoder(flat_depth, flat_sem, flat_vec)
                embeds = flat_embeds.view(B_dim, T_dim, -1)
                
                (post_h, post_z), (prior_h, prior_z) = rssm.observe_sequence(embeds, actions)
                
                recon_depth, recon_sem = decoder(post_h, post_z)
                
                # --- FIX: Flatten Targets ---
                target_depth = depths.view(-1, 1, 160, 160)
                target_sem = sems.view(-1, 1, 160, 160)
                
                loss_wm = F.mse_loss(recon_depth, target_depth) + \
                          F.mse_loss(recon_sem, target_sem) + \
                          rssm.kl_loss(post_z, prior_z)
                
                loss_wm.backward()
                wm_opt.step()

                # --- UPDATE IMAGINATION ---
                ac_opt.zero_grad()
                imag_states_h, imag_states_z = rssm.imagine(h.detach(), z.detach(), actor, horizon=HORIZON)
                
                imag_rewards = symlog(rssm.predict_reward(imag_states_h, imag_states_z))
                imag_values = critic(imag_states_h, imag_states_z)
                
                targets = compute_lambda_returns(imag_rewards, imag_values)
                loss_actor = -targets.mean()
                loss_critic = F.mse_loss(critic(imag_states_h, imag_states_z), targets.detach())
                
                (loss_actor + loss_critic).backward()
                ac_opt.step()
                
                writer.add_scalar("Train/WM_Loss", loss_wm.item(), global_step)
                writer.add_scalar("Train/Actor_Loss", loss_actor.item(), global_step)
                writer.add_scalar("Train/Critic_Loss", loss_critic.item(), global_step)
                
                if step % 100 == 0:
                    with torch.no_grad():
                        # Stack real and recon side-by-side
                        vis = torch.cat([target_depth[0:1], recon_depth[0:1]], dim=-1)
                        # Normalize to [0, 1] for display
                        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
                        writer.add_image("Visuals/Depth_Recon", vis.squeeze(0), global_step)
                
                pbar_steps.set_postfix({"Rew": f"{episode_reward:.1f}", "WM_Loss": f"{loss_wm.item():.3f}"})

            if done: break
        
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        
        print(f"Episode {episode} Complete | Reward: {episode_reward:.2f}")
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        
        checkpoint_data = {
            'encoder': encoder.state_dict(),
            'rssm': rssm.state_dict(),
            'actor': actor.state_dict(),
            'critic': critic.state_dict() # Optional, but good for resuming training
        }
        torch.save(checkpoint_data, f"checkpoints/dreamerv3_ep{episode}.pth")

if __name__ == "__main__":
    train()