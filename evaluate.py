import torch
import carla
import numpy as np
from env.carla_wrapper import CarlaEnv
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOWN = 'Town02'  # Out-of-Distribution testing map
MODEL_PATH = './models/checkpoints/dreamer_v3_final.pth'

def evaluate(num_episodes=5):
    # 1. Initialize Wrapper and Load Unseen Town
    env = CarlaEnv()
    client = env.client
    client.load_world(TOWN)
    
    # 2. Set Weather Stress Test: Hard Rain at Night
    weather = carla.WeatherParameters.HardRainNight
    env.world.set_weather(weather)
    print(f"--- Starting Evaluation in {TOWN} under Hard Rain Night ---")

    # 3. Load Trained Models
    encoder = MultiModalEncoder().to(DEVICE)
    rssm = RSSM().to(DEVICE)
    actor = Actor().to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH)
    encoder.load_state_dict(checkpoint['encoder'])
    rssm.load_state_dict(checkpoint['rssm'])
    actor.load_state_dict(checkpoint['actor'])
    
    encoder.eval()
    rssm.eval()
    actor.eval()

    # Metrics for UMich Project Report
    total_velocities = []
    success_count = 0

    for ep in range(num_episodes):
        obs = env.reset()
        state = rssm.get_initial_state(1, DEVICE)
        ep_velocity = []
        done = False
        
        print(f"Episode {ep+1} started...")
        
        while not done:
            with torch.no_grad():
                # Process Inputs for RTX 4070
                depth = torch.as_tensor(obs['depth']).to(DEVICE).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                semantic = torch.as_tensor(obs['semantic']).to(DEVICE).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                vector = torch.as_tensor(obs['vector']).to(DEVICE).unsqueeze(0)
                
                embed = encoder(depth, semantic, vector)
                _, state = rssm.observe(embed, torch.zeros(1, 2).to(DEVICE), state)
                action = actor(state[0], state[1])
            
            obs, reward, done, info = env.step(action.cpu().numpy()[0])
            
            # Track Velocity (km/h)
            kmh = obs['vector'][0]
            ep_velocity.append(kmh)

        avg_ep_v = np.mean(ep_velocity)
        total_velocities.append(avg_ep_v)
        print(f"Episode {ep+1} Finished. Avg Speed: {avg_ep_v:.2 km/h}")

    # 4. Final Comparison vs. Baseline
    final_avg_speed = np.mean(total_velocities)
    print("\n--- Final Evaluation Results ---")
    print(f"Average Velocity: {final_avg_speed:.2 km/h}")
    print(f"Baseline Velocity: 9.8 km/h")
    print(f"Performance Gain: {((final_avg_speed - 9.8) / 9.8) * 100:.1 f}%")

if __name__ == "__main__":
    evaluate()