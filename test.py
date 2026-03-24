import torch
import carla
import numpy as np
from env.carla_wrapper import CarlaEnv
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/dreamerv3/dreamerv3_latest.pth"
NUM_CLASSES = 28
TEST_TOWN = "Town10HD"  # Change this to test on different towns # Town01, Town02, Town10HD

def run_evaluation(town_name, num_episodes=5):
    print(f"\n>>> Starting Evaluation on {town_name} <<<")
    
    # Initialize env with specific town
    env = CarlaEnv(town=town_name) # Ensure your CarlaEnv wrapper supports town selection
    world = env.world
    carla_map = world.get_map()
    spectator = world.get_spectator()
    
    available_maps = env.client.get_available_maps()
    print("Available Maps:")
    for map_name in available_maps:
        print(f" - {map_name}")

    # Load Models (Same as your original test.py)
    encoder = MultiModalEncoder(latent_dim=1024, num_classes=NUM_CLASSES).to(DEVICE)
    rssm = RSSM().to(DEVICE)
    actor = Actor(goal_dim=2).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder"])
    rssm.load_state_dict(checkpoint["rssm"])
    actor.load_state_dict(checkpoint["actor"])
    
    encoder.eval(); rssm.eval(); actor.eval()

    town_results = {
        "speeds": [],
        "center_distances": [],
        "travel_distances": [],
        "rewards": []
    }

    for ep in range(num_episodes):
        obs, _ = env.reset()
        deter, stoch = rssm.initial(1, device=DEVICE)
        prev_action = torch.zeros(1, 2, device=DEVICE)
        
        # Metric Trackers
        ep_rewards = []
        ep_speeds = []
        ep_center_dist = []
        total_dist = 0.0
        prev_loc = env.vehicle.get_location()
        
        done = False
        while not done:
            with torch.no_grad():
                depth = torch.as_tensor(obs["depth"]).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
                sem = torch.as_tensor(obs["semantic"]).permute(2, 0, 1).unsqueeze(0).long().to(DEVICE).clamp(0, NUM_CLASSES - 1)
                vec = torch.as_tensor(obs["vector"]).unsqueeze(0).float().to(DEVICE)
                goal = torch.as_tensor(obs["goal"]).unsqueeze(0).float().to(DEVICE)

                embed = encoder(depth, sem, vec, goal)
                deter, stoch, _, _ = rssm.obs_step(deter, stoch, prev_action, embed, goal)
                stoch_flat = rssm.flatten_stoch(stoch)
                action, _, _, _ = actor(deter, stoch_flat, goal, sample=False)
                prev_action = action

            # Step Env
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated

            # --- METRIC CALCULATION ---
            # 1. Speed (km/h)
            ep_speeds.append(obs["vector"][0])
            
            # 2. Travel Distance
            curr_loc = env.vehicle.get_location()
            total_dist += curr_loc.distance(prev_loc)
            prev_loc = curr_loc
            
            # 3. Center Line Distance
            waypoint = carla_map.get_waypoint(curr_loc)
            ep_center_dist.append(curr_loc.distance(waypoint.transform.location))
            
            # 4. Reward
            ep_rewards.append(reward)

            # Camera logic...
            v_trans = env.vehicle.get_transform()
            spectator.set_transform(carla.Transform(v_trans.location - v_trans.get_forward_vector()*8 + carla.Location(z=4), v_trans.rotation))

        # Store Episode Results
        town_results["speeds"].append(np.mean(ep_speeds))
        town_results["center_distances"].append(np.mean(ep_center_dist))
        town_results["travel_distances"].append(total_dist)
        town_results["rewards"].append(np.mean(ep_rewards))
        
        print(f"  Ep {ep+1} | Dist: {total_dist:.1f}m | Avg Speed: {np.mean(ep_speeds):.1f}km/h | Avg Reward: {np.mean(ep_rewards):.2f}")

    return town_results

def test_all_towns(town):
    # towns = ["Town01", "Town02", "Town10HD"] # Town10 uses HD suffix in some CARLA versions
    final_report = {}

    final_report[town] = run_evaluation(town)

    print("\n" + "="*40)
    print("FINAL MULTI-TOWN PERFORMANCE REPORT")
    print("="*40)
    for town, metrics in final_report.items():
        print(f"[{town}]")
        print(f"  Avg Speed:           {np.mean(metrics['speeds']):.2f} km/h")
        print(f"  Avg Center Distance: {np.mean(metrics['center_distances']):.2f} m")
        print(f"  Total Travel Dist:   {np.mean(metrics['travel_distances']):.2f} m")
        print(f"  Avg Episode Reward:  {np.mean(metrics['rewards']):.2f}")
        print("-" * 20)

if __name__ == "__main__":
    test_all_towns(TEST_TOWN)