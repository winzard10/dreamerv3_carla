import os
import time
import carla
import numpy as np
from params import (
    COLLECT_SAVE_DIR, COLLECT_TARGET_STEPS, SEQ_LEN,
    COLLECT_LOOKAHEAD, COLLECT_STEER_GAIN, COLLECT_THROTTLE,
    COLLECT_LOG_EVERY,
)
from env.carla_wrapper import CarlaEnv, GOAL_LOOKAHEAD

# Maximum steps per episode before forcing a reset.
# At 0.05s per step, 1000 steps = 50 seconds of driving.
# This ensures variety of spawn locations and avoids driving in circles.
MAX_EPISODE_STEPS = 1000


def compute_expert_action(env) -> np.ndarray:
    """
    Pure pursuit expert controller.
    Uses COLLECT_LOOKAHEAD waypoints ahead for smooth steering.
    COLLECT_LOOKAHEAD must match GOAL_LOOKAHEAD in carla_wrapper.py.
    """
    assert COLLECT_LOOKAHEAD == GOAL_LOOKAHEAD, (
        f"COLLECT_LOOKAHEAD ({COLLECT_LOOKAHEAD}) must match "
        f"GOAL_LOOKAHEAD ({GOAL_LOOKAHEAD}) in carla_wrapper.py — "
        f"otherwise goal vectors in collected data won't match what the env produces."
    )

    v_transform = env.vehicle.get_transform()
    v_loc       = v_transform.location
    v_fwd       = v_transform.get_forward_vector()

    target_idx = min(
        env.current_waypoint_index + COLLECT_LOOKAHEAD,
        len(env.route_waypoints) - 1
    )
    target_loc = env.route_waypoints[target_idx].transform.location

    w_vec   = target_loc - v_loc
    w_norm  = np.sqrt(w_vec.x ** 2 + w_vec.y ** 2) + 1e-5
    w_vec_x = w_vec.x / w_norm
    w_vec_y = w_vec.y / w_norm

    dot   = v_fwd.x * w_vec_x + v_fwd.y * w_vec_y
    cross = v_fwd.x * w_vec_y - v_fwd.y * w_vec_x
    angle = np.arctan2(cross, dot)
    steer = np.clip(COLLECT_STEER_GAIN * angle, -1.0, 1.0)

    # COLLECT_THROTTLE is in [0,1] space, convert to [-1,1] action space
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
    return {
        "depth": [], "semantic": [], "goal": [], "vector": [],
        "action": [], "reward": [], "done": [],
    }


def run_collection():
    os.makedirs(COLLECT_SAVE_DIR, exist_ok=True)

    env    = CarlaEnv()
    obs, _ = env.reset()
    seq    = empty_seq()
    seq_saved    = 0
    episode_step = 0
    episode_num  = 1

    print(f"Starting data collection — target: {COLLECT_TARGET_STEPS} steps, "
          f"seq_len: {SEQ_LEN}")
    print(f"  COLLECT_LOOKAHEAD: {COLLECT_LOOKAHEAD} waypoints "
          f"({COLLECT_LOOKAHEAD * 2.0:.1f}m ahead)")
    print(f"  MAX_EPISODE_STEPS: {MAX_EPISODE_STEPS} "
          f"({MAX_EPISODE_STEPS * 0.05:.0f}s per episode)")
    print(f"  depth shape:    {obs['depth'].shape}")
    print(f"  semantic shape: {obs['semantic'].shape}")
    print(f"  goal shape:     {obs['goal'].shape}  (normalized /10.0)")
    print(f"  vector shape:   {obs['vector'].shape} (normalized /10.0)")

    for step in range(COLLECT_TARGET_STEPS):
        action = compute_expert_action(env)

        # Spectator camera
        v_transform = env.vehicle.get_transform()
        v_loc       = v_transform.location
        v_fwd       = v_transform.get_forward_vector()
        env.world.get_spectator().set_transform(carla.Transform(
            v_loc - (v_fwd * 8.0) + carla.Location(z=4.0),
            v_transform.rotation,
        ))

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_step += 1

        # Determine if we should reset:
        # 1. Natural termination (collision, off-road, stuck) — mark done=True
        # 2. Time limit hit — mark done=False (not a real episode end,
        #    just a forced respawn for variety)
        natural_done  = terminated or truncated
        time_limit_hit = episode_step >= MAX_EPISODE_STEPS
        should_reset  = natural_done or time_limit_hit

        # done signal stored in buffer:
        # True  → model learns this is a genuine episode boundary
        # False → model treats next episode as continuation (correct for time limit)
        done_for_buffer = natural_done  # NOT time_limit_hit

        # Accumulate transition — obs from BEFORE the step
        seq["depth"].append(obs["depth"])
        seq["semantic"].append(obs["semantic"])
        seq["goal"].append(obs["goal"])
        seq["vector"].append(obs["vector"])
        seq["action"].append(action)
        seq["reward"].append(reward)
        seq["done"].append(done_for_buffer)

        # Flush complete sequence to disk
        if len(seq["action"]) == SEQ_LEN:
            save_sequence(seq, COLLECT_SAVE_DIR)
            seq_saved += 1
            seq = empty_seq()

        obs = next_obs

        if (step + 1) % COLLECT_LOG_EVERY == 0:
            print(f"  Step {step + 1:>6}/{COLLECT_TARGET_STEPS} | "
                  f"Ep: {episode_num} | "
                  f"Ep step: {episode_step}/{MAX_EPISODE_STEPS} | "
                  f"Seqs saved: {seq_saved} | "
                  f"Reward: {reward:.2f}")

        if should_reset:
            reason = "natural termination" if natural_done else "time limit"
            print(f"  Episode {episode_num} ended at step {step + 1} "
                  f"({reason}, ep_steps={episode_step}). Resetting...")
            obs, _ = env.reset()
            seq    = empty_seq()   # discard incomplete sequence at episode boundary
            episode_step = 0
            episode_num += 1

    print(f"\nCollection complete — {seq_saved} sequences saved to {COLLECT_SAVE_DIR}")
    print(f"  Total episodes: {episode_num}")
    env._cleanup()


if __name__ == "__main__":
    run_collection()