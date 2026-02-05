import carla
import numpy as np
from env.carla_wrapper import CarlaEnv

def get_pid_control(vehicle, waypoint):
    # Calculate steering using a simple lateral PID
    v_vec = vehicle.get_transform().get_forward_vector()
    w_vec = waypoint.transform.location - vehicle.get_location()
    
    # Dot product and cross product for angle
    dot = v_vec.x * w_vec.x + v_vec.y * w_vec.y
    cross = v_vec.x * w_vec.y - v_vec.y * w_vec.x
    angle = np.arctan2(cross, dot)
    
    # Steering PID constants (K_p = 1.2 is a good starting point)
    steer = np.clip(1.2 * angle, -1.0, 1.0)
    return steer

def collect_data(steps=10000):
    env = CarlaEnv()
    obs = env.reset()
    
    for i in range(steps):
        # 1. Get current waypoint from CARLA Map
        waypoint = env.world.get_map().get_waypoint(env.vehicle.get_location())
        next_waypoint = waypoint.next(2.0)[0] # Look 2 meters ahead
        
        # 2. Calculate PID steering and constant throttle
        steer = get_pid_control(env.vehicle, next_waypoint)
        action = [steer, 0.5] # 50% throttle for steady speed
        
        # 3. Step and save to your buffer
        next_obs, reward, done, _ = env.step(action)
        # (Save code for buffer goes here)
        
        obs = next_obs
        if done: obs = env.reset()
        if i % 100 == 0: print(f"Collected {i} steps...")

if __name__ == "__main__":
    collect_data()