import os
from time import time
import carla
import numpy as np
import torch
from env.carla_wrapper import CarlaEnv

# Configuration
SAVE_DIR = "./data/expert_sequences"
TARGET_STEPS = 500 # 50000
SEQ_LEN = 50         

def run_collection():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Initialize Environment
    env = CarlaEnv()
    obs, _ = env.reset()
    
    current_seq = {
        "depth": [], "semantic": [], "goal": [], "vector": [],
        "action": [], "reward": []
    }

    print(f"Starting data collection for {TARGET_STEPS} steps...")

    for step in range(TARGET_STEPS):
        # --- 1. EXPERT LOGIC (Updated) ---
        v_transform = env.vehicle.get_transform()
        v_loc = v_transform.location
        
        # FIX: Look ahead in the ACTUAL route list, not just "next on map"
        # We aim at the current target + 1 to be smooth (Lookahead)
        target_idx = min(env.current_waypoint_index + 2, len(env.route_waypoints) - 1)
        target_wp = env.route_waypoints[target_idx]
        
        # Calculate Vector to Target
        v_vec = v_transform.get_forward_vector()
        w_vec = target_wp.transform.location - v_loc
        
        # Normalize (Handle div by zero)
        w_vec_norm = np.sqrt(w_vec.x**2 + w_vec.y**2) + 1e-5
        w_vec_x = w_vec.x / w_vec_norm
        w_vec_y = w_vec.y / w_vec_norm
        
        # Calculate Angle (Dot Product & Cross Product)
        # v_vec is already normalized by CARLA
        dot = v_vec.x * w_vec_x + v_vec.y * w_vec_y
        cross = v_vec.x * w_vec_y - v_vec.y * w_vec_x
        
        angle = np.arctan2(cross, dot)   # signed, stable
        steer = np.clip(0.85 * angle, -1.0, 1.0)
        
        # Throttle: -0.2 maps to 0.4 (40%) in your new wrapper
        action = np.array([steer, -0.2], dtype=np.float32)

        # --- 2. Step Environment ---
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update Spectator Camera (Optional, but good for watching)
        spectator = env.world.get_spectator()
        back_pos = v_loc - (v_vec * 8.0) + carla.Location(z=4.0)
        spectator.set_transform(carla.Transform(back_pos, v_transform.rotation))

        # --- 3. Save to Buffer ---
        current_seq["depth"].append(obs["depth"])
        current_seq["semantic"].append(obs["semantic"])
        current_seq["goal"].append(obs["goal"])
        current_seq["vector"].append(obs["vector"])
        current_seq["action"].append(action)
        current_seq["reward"].append(reward)
        current_seq["done"].append(done)

        # --- 4. Save to Disk ---
        if len(current_seq["action"]) == SEQ_LEN:
            seq_idx = int(time.time() * 1000)
            np.savez_compressed(
                f"{SAVE_DIR}/seq_{seq_idx}.npz",
                depth=np.array(current_seq["depth"]),
                semantic=np.array(current_seq["semantic"]),
                goal=np.array(current_seq["goal"]),
                vector=np.array(current_seq["vector"]),
                action=np.array(current_seq["action"]),
                reward=np.array(current_seq["reward"]),
                done=np.array(current_seq["done"])
            )
            for key in current_seq: current_seq[key] = []

        obs = next_obs
        
        # Logging & Reset
        if (step+1) % 100 == 0:
            print(f"Step {step+1}/{TARGET_STEPS} | Last Reward: {reward:.2f}")

        if done: 
            print(f"Episode Done at Step {step}. Resetting...")
            obs, _ = env.reset()
            # Clear buffer on reset to avoid mixing episodes in one sequence
            for key in current_seq: current_seq[key] = []

    print(f"Collection complete. Data saved to {SAVE_DIR}")
    env._cleanup()

if __name__ == "__main__":
    run_collection()