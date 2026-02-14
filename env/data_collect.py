import os
import carla
import numpy as np
import torch
from env.carla_wrapper import CarlaEnv

# Configuration
SAVE_DIR = "./data/expert_sequences"
TARGET_STEPS = 50000  # Start with a small batch to test
SEQ_LEN = 50         # DreamerV3 prefers sequences

def run_collection():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    env = CarlaEnv()
    obs = env.reset()
    
    # Storage for the current sequence
    current_seq = {
        "depth": [],
        "semantic": [],
        "goal": [],
        "vector": [],
        "action": [],
        "reward": []
    }

    print(f"Starting data collection for {TARGET_STEPS} steps...")

    for step in range(TARGET_STEPS):
        # 1. Simple PID-like logic using CARLA Waypoints
        vehicle_transform = env.vehicle.get_transform()
        waypoint = env.world.get_map().get_waypoint(vehicle_transform.location)
        next_waypoint = waypoint.next(2.0)[0] # Look 2 meters ahead
        
        # Calculate steering angle
        v_begin = vehicle_transform.location
        v_end = v_begin + vehicle_transform.get_forward_vector()
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y])
        
        w_vec = np.array([next_waypoint.transform.location.x - v_begin.x, 
                          next_waypoint.transform.location.y - v_begin.y])
        
        dot = np.dot(v_vec, w_vec) / (np.linalg.norm(v_vec) * np.linalg.norm(w_vec))
        cross = np.cross(v_vec, w_vec)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        if cross < 0: angle *= -1
        
        # PID Steering + Constant Throttle
        steer = np.clip(1.5 * angle, -1.0, 1.0)
        action = np.array([steer, 0.4], dtype=np.float32)

        # 2. Step Environment
        next_obs, reward, done, _ = env.step(action)
        
        spectator = env.world.get_spectator()
        v_transform = env.vehicle.get_transform()
        
        # Calculate position (8m back, 4m up)
        forward_vec = v_transform.get_forward_vector()
        back_pos = v_transform.location - (forward_vec * 8.0) + carla.Location(z=4.0)
        
        # Update camera
        spectator.set_transform(carla.Transform(back_pos, v_transform.rotation))

        # 3. Save to sequence buffer
        current_seq["depth"].append(obs["depth"])
        current_seq["semantic"].append(obs["semantic"])
        current_seq["goal"].append(obs["goal"])
        current_seq["vector"].append(obs["vector"])
        current_seq["action"].append(action)
        current_seq["reward"].append(reward)

        # 4. If sequence is full, save to disk
        if len(current_seq["action"]) == SEQ_LEN:
            seq_idx = len(os.listdir(SAVE_DIR))
            np.savez_compressed(
                f"{SAVE_DIR}/seq_{seq_idx}.npz",
                depth=np.array(current_seq["depth"]),
                semantic=np.array(current_seq["semantic"]),
                goal=np.array(current_seq["goal"]),
                vector=np.array(current_seq["vector"]),
                action=np.array(current_seq["action"]),
                reward=np.array(current_seq["reward"])
            )
            # Reset local sequence
            for key in current_seq: current_seq[key] = []

        obs = next_obs
        
        if done or step % 500 == 0:
            print(f"Step {step}/{TARGET_STEPS} | Last Reward: {reward:.2f}")
            if done: obs = env.reset()

    print(f"Collection complete. Data saved to {SAVE_DIR}")
    env._cleanup()

if __name__ == "__main__":
    run_collection()