import os
import time
import carla
import numpy as np

from params import (
    COLLECT_SAVE_DIR, COLLECT_TARGET_STEPS, SEQ_LEN,
    COLLECT_LOOKAHEAD, COLLECT_STEER_GAIN, COLLECT_THROTTLE,
    COLLECT_LOG_EVERY,
)
from env.carla_wrapper import CarlaEnv


def compute_expert_action(env) -> np.ndarray:
    """
    Pure pursuit expert controller.
    Looks COLLECT_LOOKAHEAD waypoints ahead for smooth steering,
    uses fixed throttle for constant-speed driving.
    """
    v_transform = env.vehicle.get_transform()
    v_loc       = v_transform.location
    v_fwd       = v_transform.get_forward_vector()

    target_idx = min(env.current_waypoint_index + COLLECT_LOOKAHEAD,
                     len(env.route_waypoints) - 1)
    target_loc = env.route_waypoints[target_idx].transform.location

    w_vec      = target_loc - v_loc
    w_norm     = np.sqrt(w_vec.x ** 2 + w_vec.y ** 2) + 1e-5
    w_vec_x    = w_vec.x / w_norm
    w_vec_y    = w_vec.y / w_norm

    dot   = v_fwd.x * w_vec_x + v_fwd.y * w_vec_y
    cross = v_fwd.x * w_vec_y - v_fwd.y * w_vec_x
    angle = np.arctan2(cross, dot)
    steer = np.clip(COLLECT_STEER_GAIN * angle, -1.0, 1.0)
    throttle = 2 * COLLECT_THROTTLE - 1

    return np.array([steer, throttle], dtype=np.float32)


def save_sequence(seq: dict, save_dir: str):
    seq_idx = int(time.time() * 1000)
    np.savez_compressed(
        f"{save_dir}/seq_{seq_idx}.npz",
        depth    = np.array(seq["depth"]),
        semantic = np.array(seq["semantic"]),
        goal     = np.array(seq["goal"]),
        vector   = np.array(seq["vector"]),
        action   = np.array(seq["action"]),
        reward   = np.array(seq["reward"]),
        done     = np.array(seq["done"]),
    )


def empty_seq() -> dict:
    return {"depth": [], "semantic": [], "goal": [], "vector": [],
            "action": [], "reward": [], "done": []}


def run_collection():
    os.makedirs(COLLECT_SAVE_DIR, exist_ok=True)

    env      = CarlaEnv()
    obs, _   = env.reset()
    seq      = empty_seq()
    seq_saved = 0

    print(f"Starting data collection — target: {COLLECT_TARGET_STEPS} steps, "
          f"seq_len: {SEQ_LEN}")
    print(f"  depth shape:    {obs['depth'].shape}")
    print(f"  semantic shape: {obs['semantic'].shape}")

    for step in range(COLLECT_TARGET_STEPS):
        action = compute_expert_action(env)

        # Update spectator camera
        v_transform = env.vehicle.get_transform()
        v_loc       = v_transform.location
        v_fwd       = v_transform.get_forward_vector()
        env.world.get_spectator().set_transform(carla.Transform(
            v_loc - (v_fwd * 8.0) + carla.Location(z=4.0),
            v_transform.rotation,
        ))

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Accumulate transition
        seq["depth"].append(obs["depth"])
        seq["semantic"].append(obs["semantic"])
        seq["goal"].append(obs["goal"])
        seq["vector"].append(obs["vector"])
        seq["action"].append(action)
        seq["reward"].append(reward)
        seq["done"].append(done)

        # Flush complete sequence to disk
        if len(seq["action"]) == SEQ_LEN:
            save_sequence(seq, COLLECT_SAVE_DIR)
            seq_saved += 1
            seq = empty_seq()

        obs = next_obs

        if (step + 1) % COLLECT_LOG_EVERY == 0:
            print(f"  Step {step + 1:>6}/{COLLECT_TARGET_STEPS} | "
                  f"Seqs saved: {seq_saved} | "
                  f"Reward: {reward:.2f}")

        if done:
            print(f"  Episode ended at step {step + 1}. Resetting...")
            obs, _ = env.reset()
            seq    = empty_seq()   # discard incomplete sequence at episode boundary

    print(f"\nCollection complete — {seq_saved} sequences saved to {COLLECT_SAVE_DIR}")
    env._cleanup()


if __name__ == "__main__":
    run_collection()