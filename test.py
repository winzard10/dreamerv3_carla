import torch
import carla
import numpy as np
from env.carla_wrapper import CarlaEnv
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TOWN = 'Town02'
MODEL_PATH = "./checkpoints/dreamerv3/dreamerv3_ep899.pth" # './checkpoints/dreamer_v3_final.pth'

def test(num_episodes=5):
    env = CarlaEnv()
    client = env.client
    
    # 1. Load World and RE-APPLY Synchronous Settings
    # print(f"Loading {TOWN}...")
    # client.load_world(TOWN)
    # world = client.get_world()
    world = env.world
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    # 2. Weather Stress Test
    # world.set_weather(carla.WeatherParameters.HardRainNight)
    spectator = world.get_spectator()

    # 3. Load Models
    encoder = MultiModalEncoder().to(DEVICE)
    rssm = RSSM().to(DEVICE)
    actor = Actor().to(DEVICE)
    
    # Assuming the checkpoint is a dict with model states
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder'])
    rssm.load_state_dict(checkpoint['rssm'])
    actor.load_state_dict(checkpoint['actor'])
    
    encoder.eval()
    rssm.eval()
    actor.eval()

    total_velocities = []

    for ep in range(num_episodes):
        obs = env.reset()
        h = rssm.get_initial_state(1, DEVICE)
        # Initialize z with zeros for the first step
        z = torch.zeros(1, 32).to(DEVICE) 
        # CRITICAL: Track the previous action
        prev_action = torch.zeros(1, 2).to(DEVICE)
        
        ep_velocity = []
        done = False
        
        while not done:
            with torch.no_grad():
                d_in = torch.as_tensor(obs['depth'].copy()).to(DEVICE).float().permute(2,0,1).unsqueeze(0) / 255.0
                s_in = torch.as_tensor(obs['semantic'].copy()).to(DEVICE).float().permute(2,0,1).unsqueeze(0) / 28.0
                v_in = torch.as_tensor(obs['vector'].copy()).to(DEVICE).float().unsqueeze(0)
                g_in = torch.as_tensor(obs['goal'].copy()).to(DEVICE).float().unsqueeze(0) # NEW
                
                print(f"Goal Input: {g_in}") 

                h = rssm.gru(torch.cat([z, prev_action], dim=-1), h)
                
                # Encoder and Actor now take g_in (the Goal)
                embed = encoder(d_in, s_in, v_in, g_in)
                z = rssm.representation_model(torch.cat([h, embed, g_in], dim=-1))
                action = actor(h, z, g_in)
                print(f"Action: {action.cpu().numpy()[0]}")
                prev_action = action # Store for next loop
            
            # Step Environment
            obs, reward, done, _ = env.step(action.cpu().numpy()[0])
            
            # --- Spectator Chase Cam ---
            v_trans = env.vehicle.get_transform()
            cam_pos = v_trans.location - (v_trans.get_forward_vector() * 8.0) + carla.Location(z=4.0)
            spectator.set_transform(carla.Transform(cam_pos, v_trans.rotation))

            ep_velocity.append(obs['vector'][0])
            world.tick() # Ensure world moves in sync

        avg_ep_v = np.mean(ep_velocity)
        total_velocities.append(avg_ep_v)
        print(f"Episode {ep+1} | Avg Speed: {avg_ep_v:.2f} km/h")

    print(f"\nFinal Average Velocity: {np.mean(total_velocities):.2f} km/h")

if __name__ == "__main__":
    test()