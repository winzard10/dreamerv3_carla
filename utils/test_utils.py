import numpy as np
import cv2
import torch
import carla

from params import *
from models.encoder import RGBEncoder
from models.rssm import RSSM
from models.actor_critic import Actor
from models.decoder import RGBDecoder

DETAILED_LOGGING = True

# =============================================================================
# Visualization
# =============================================================================

def show_reconstruction_windows(obs, deter, post_logits_flat, rssm, decoder) -> bool:
    """
    Renders GT vs reconstructed RGB side-by-side.
    Uses SOFT posterior probabilities for decoding visualization.
    Returns False if the user pressed 'q'.
    """
    with torch.no_grad():
        _, post_probs, _ = rssm.dist_from_logits_flat(post_logits_flat)
        stoch_soft_flat = post_probs.reshape(post_probs.shape[0], -1)

        recon_rgb, _, _ = decoder(deter, stoch_soft_flat)

    gt_rgb = obs["rgb"].astype(np.uint8)
    recon_rgb_np = (
        recon_rgb[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)

    rgb_vis = np.concatenate([gt_rgb, recon_rgb_np], axis=1)

    cv2.imshow("RGB (GT | Recon)", cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))

    return (cv2.waitKey(1) & 0xFF) != ord("q")


# =============================================================================
# Model loading
# =============================================================================

def build_models(model_dir: str = CKPT_DIR, model_name: str = TEST_MODEL):
    model_path = f"{model_dir}/{model_name}"
    print(f"Loading models from {model_path}...")
    Z_DIM = STOCH_CATEGORICALS * STOCH_CLASSES

    encoder = RGBEncoder(embed_dim=EMBED_DIM).to(DEVICE)
    rssm = RSSM(
        deter_dim=DETER_DIM, act_dim=2, embed_dim=EMBED_DIM, goal_dim=2,
        stoch_categoricals=STOCH_CATEGORICALS, stoch_classes=STOCH_CLASSES,
        unimix_ratio=0.01, kl_balance=0.8, free_nats=FREE_NATS,
    ).to(DEVICE)
    actor = Actor(
        deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2, action_dim=2,
        hidden_dim=512, min_std=0.1, init_std=1.0,
    ).to(DEVICE)
    decoder = RGBDecoder(
        deter_dim=DETER_DIM,
        stoch_dim=Z_DIM,
    ).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    rssm.load_state_dict(ckpt["rssm"])
    actor.load_state_dict(ckpt["actor"])
    decoder.load_state_dict(ckpt["decoder"])

    for m in [encoder, rssm, actor, decoder]:
        m.eval()

    return encoder, rssm, actor, decoder


# =============================================================================
# Observation preprocessing
# =============================================================================

def preprocess_obs(obs):
    rgb = (
        torch.as_tensor(obs["rgb"].copy())
        .permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    )  # [1, 3, H, W]

    vec = torch.as_tensor(
        obs.get("vector", np.zeros(3, dtype=np.float32)).copy()
    ).unsqueeze(0).float().to(DEVICE)

    goal = torch.as_tensor(
        obs.get("goal", np.zeros(2, dtype=np.float32)).copy()
    ).unsqueeze(0).float().to(DEVICE)

    return rgb, vec, goal


# =============================================================================
# Evaluation
# =============================================================================

def run_evaluation(env, encoder, rssm, actor, decoder,
                   num_episodes: int = TEST_NUM_EPISODES) -> dict:
    """
    Run `num_episodes` episodes and return aggregated metrics.
    """
    carla_map = env.world.get_map()
    spectator = env.world.get_spectator()

    results = {
        "speeds": [],
        "center_distances": [],
        "travel_distances": [],
        "rewards": [],
        "steer_means": [],
        "steer_stds": [],
        "throttle_means": [],
        "throttle_stds": [],
        "r_progress": [],
        "r_speed": [],
        "r_center": [],
        "r_heading": [],
        "r_driving": [],
        "r_ctrl_mag": [],
        "r_ctrl_rate": [],
        "r_collision": [],
        "r_offroad": [],
        "r_stall": [],
        "lane_invasions": [],
        "coll_per_meter": [],
    }
    keep_showing = SHOW_RECON

    for ep in range(num_episodes):
        obs, _ = env.reset()

        deter, stoch = rssm.initial(1, device=DEVICE)
        prev_action = torch.zeros(1, 2, device=DEVICE)

        ep_rewards, ep_speeds, ep_center_dist = [], [], []
        ep_steers, ep_throttles = [], []
        ep_r_progress = []
        ep_r_speed = []
        ep_r_center = []
        ep_r_heading = []
        ep_r_driving = []
        ep_r_ctrl_mag = []
        ep_r_ctrl_rate = []
        ep_r_collision = []
        ep_r_offroad = []
        ep_r_stall = []
        ep_lane_invasions = []
        ep_collision_count = 0

        total_dist = 0.0
        prev_loc = env.vehicle.get_location()
        done, step = False, 0

        while not done:
            with torch.no_grad():
                rgb, vec, goal = preprocess_obs(obs)
                embed = encoder(rgb, vec, goal)

                deter, stoch, post_logits_flat, _ = rssm.obs_step(
                    deter, stoch, prev_action, embed, goal
                )
                action, _, _, _ = actor(
                    deter, rssm.flatten_stoch(stoch), goal, sample=False
                )
                prev_action = action.detach()

                act_np = action.detach().cpu().numpy()[0]
                steer = float(act_np[0])
                throttle_cmd = float((act_np[1] + 1.0) / 2.0)

                ep_steers.append(steer)
                ep_throttles.append(throttle_cmd)

            if keep_showing and step % SHOW_EVERY_N_STEPS == 0:
                keep_showing = show_reconstruction_windows(
                    obs, deter, post_logits_flat, rssm, decoder
                )

            obs, reward, terminated, truncated, info = env.step(act_np)
            rc = info.get("reward_components", {})

            ep_collision_count += (1 if rc.get("r_collision", 0.0) < 0 else 0)

            ep_r_progress.append(float(rc.get("r_progress", 0.0)))
            ep_r_speed.append(float(rc.get("r_speed", 0.0)))
            ep_r_center.append(float(rc.get("r_center", 0.0)))
            ep_r_heading.append(float(rc.get("r_heading", 0.0)))
            ep_r_driving.append(float(rc.get("r_driving", 0.0)))
            ep_r_ctrl_mag.append(float(rc.get("r_ctrl_mag", 0.0)))
            ep_r_ctrl_rate.append(float(rc.get("r_ctrl_rate", 0.0)))
            ep_r_collision.append(float(rc.get("r_collision", 0.0)))
            ep_r_offroad.append(float(rc.get("r_offroad", 0.0)))
            ep_r_stall.append(float(rc.get("r_stall", 0.0)))
            ep_lane_invasions.append(info.get("lane_invasions", 0))

            done = terminated or truncated

            curr_loc = env.vehicle.get_location()
            total_dist += curr_loc.distance(prev_loc)
            prev_loc = curr_loc

            waypoint = carla_map.get_waypoint(curr_loc)
            ep_center_dist.append(curr_loc.distance(waypoint.transform.location))
            ep_speeds.append(float(obs["vector"][0]) * 10.0 * 3.6)
            ep_rewards.append(float(reward))

            if SHOW_SPECTATOR:
                v_t = env.vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    v_t.location - v_t.get_forward_vector() * 8 + carla.Location(z=4),
                    v_t.rotation,
                ))

            step += 1

        steer_mean = float(np.mean(ep_steers)) if ep_steers else 0.0
        steer_std = float(np.std(ep_steers)) if ep_steers else 0.0
        throttle_mean = float(np.mean(ep_throttles)) if ep_throttles else 0.0
        throttle_std = float(np.std(ep_throttles)) if ep_throttles else 0.0

        results["speeds"].append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        results["center_distances"].append(float(np.mean(ep_center_dist)) if ep_center_dist else 0.0)
        results["travel_distances"].append(float(total_dist))
        results["rewards"].append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)
        results["steer_means"].append(steer_mean)
        results["steer_stds"].append(steer_std)
        results["throttle_means"].append(throttle_mean)
        results["throttle_stds"].append(throttle_std)
        results["r_progress"].append(float(np.mean(ep_r_progress)) if ep_r_progress else 0.0)
        results["r_speed"].append(float(np.mean(ep_r_speed)) if ep_r_speed else 0.0)
        results["r_center"].append(float(np.mean(ep_r_center)) if ep_r_center else 0.0)
        results["r_heading"].append(float(np.mean(ep_r_heading)) if ep_r_heading else 0.0)
        results["r_driving"].append(float(np.mean(ep_r_driving)) if ep_r_driving else 0.0)
        results["r_ctrl_mag"].append(float(np.mean(ep_r_ctrl_mag)) if ep_r_ctrl_mag else 0.0)
        results["r_ctrl_rate"].append(float(np.mean(ep_r_ctrl_rate)) if ep_r_ctrl_rate else 0.0)
        results["r_collision"].append(float(np.mean(ep_r_collision)) if ep_r_collision else 0.0)
        results["r_offroad"].append(float(np.mean(ep_r_offroad)) if ep_r_offroad else 0.0)
        results["r_stall"].append(float(np.mean(ep_r_stall)) if ep_r_stall else 0.0)
        results["lane_invasions"].append(ep_lane_invasions[-1] if ep_lane_invasions else 0)

        coll_per_m = ep_collision_count / (total_dist + 1e-6)
        results["coll_per_meter"].append(coll_per_m)

        print(
            f"  Ep {ep + 1} | "
            f"Dist: {total_dist:.1f}m | "
            f"Avg Speed: {np.mean(ep_speeds):.1f} km/h | "
            f"Avg Reward: {np.mean(ep_rewards):.2f}"
            f"Total Reward: {np.sum(ep_rewards):.2f} | "
        )
        if DETAILED_LOGGING:
            print(
                f"     steer mean/std: {steer_mean:.3f} / {steer_std:.3f} | "
                f"throttle mean/std: {throttle_mean:.3f} / {throttle_std:.3f}"
            )
            print(f"     first 20 steer: {np.round(np.array(ep_steers[:20]), 3).tolist()}")
            print(f"     first 20 throttle: {np.round(np.array(ep_throttles[:20]), 3).tolist()}")
            print(
                f"     reward parts avg | "
                f"progress: {np.mean(ep_r_progress):.3f}, "
                f"speed: {np.mean(ep_r_speed):.3f}, "
                f"center: {np.mean(ep_r_center):.3f}, "
                f"heading: {np.mean(ep_r_heading):.3f}, "
                f"driving: {np.mean(ep_r_driving):.3f}"
            )
            print(
                f"     control avg     | "
                f"ctrl_mag: {np.mean(ep_r_ctrl_mag):.3f}, "
                f"ctrl_rate: {np.mean(ep_r_ctrl_rate):.3f}"
            )
            print(
                f"     hard penalties  | "
                f"collision: {np.mean(ep_r_collision):.3f}, "
                f"offroad: {np.mean(ep_r_offroad):.3f}, "
                f"stall: {np.mean(ep_r_stall):.3f}"
            )
    return results


def print_report(final_report: dict):
    print("\n" + "=" * 40)
    print("EVALUATION REPORT")
    print("=" * 40)

    # collect for global stats
    all_metrics = {
        "speeds": [],
        "center_distances": [],
        "travel_distances": [],
        "rewards": [],
        "lane_invasions": [],
        "coll_per_meter": [],
        "steer_means": [],
        "steer_stds": [],
        "throttle_means": [],
        "throttle_stds": [],
    }

    for town_name, metrics in final_report.items():
        print(f"[{town_name}]")

        def mean_std(x):
            return np.mean(x), np.std(x)

        sp_m, sp_s = mean_std(metrics["speeds"])
        cd_m, cd_s = mean_std(metrics["center_distances"])
        td_m, td_s = mean_std(metrics["travel_distances"])
        rw_m, rw_s = mean_std(metrics["rewards"])
        li_m, li_s = mean_std(metrics["lane_invasions"])
        cp_m, cp_s = mean_std(metrics["coll_per_meter"])

        st_m, st_s = mean_std(metrics["steer_means"])
        ststd_m, ststd_s = mean_std(metrics["steer_stds"])

        th_m, th_s = mean_std(metrics["throttle_means"])
        thstd_m, thstd_s = mean_std(metrics["throttle_stds"])

        print(f"  Avg Speed:           {sp_m:.2f} ± {sp_s:.2f} km/h")
        print(f"  Center Distance:     {cd_m:.2f} ± {cd_s:.2f} m")
        print(f"  Travel Distance:     {td_m:.2f} ± {td_s:.2f} m")
        print(f"  Episode Reward:      {rw_m:.2f} ± {rw_s:.2f}")
        print(f"  Lane Invasions:      {li_m:.2f} ± {li_s:.2f}")
        print(f"  Collisions / meter:  {cp_m:.3f} ± {cp_s:.3f}")

        print(f"  Steer mean:          {st_m:.3f} ± {st_s:.3f}")
        print(f"  Steer std:           {ststd_m:.3f} ± {ststd_s:.3f}")
        print(f"  Throttle mean:       {th_m:.3f} ± {th_s:.3f}")
        print(f"  Throttle std:        {thstd_m:.3f} ± {thstd_s:.3f}")

        print("-" * 40)

        # accumulate global
        for k in all_metrics:
            all_metrics[k].extend(metrics[k])

    # ===== GLOBAL STATS =====
    print("\n" + "=" * 40)
    print("OVERALL (ALL TOWNS)")
    print("=" * 40)

    def g(x):
        return np.mean(x), np.std(x)

    print(f"  Avg Speed:           {g(all_metrics['speeds'])[0]:.2f} ± {g(all_metrics['speeds'])[1]:.2f}")
    print(f"  Center Distance:     {g(all_metrics['center_distances'])[0]:.2f} ± {g(all_metrics['center_distances'])[1]:.2f}")
    print(f"  Travel Distance:     {g(all_metrics['travel_distances'])[0]:.2f} ± {g(all_metrics['travel_distances'])[1]:.2f}")
    print(f"  Episode Reward:      {g(all_metrics['rewards'])[0]:.2f} ± {g(all_metrics['rewards'])[1]:.2f}")
    print(f"  Lane Invasions:      {g(all_metrics['lane_invasions'])[0]:.2f} ± {g(all_metrics['lane_invasions'])[1]:.2f}")
    print(f"  Collisions / meter:  {g(all_metrics['coll_per_meter'])[0]:.3f} ± {g(all_metrics['coll_per_meter'])[1]:.3f}")

    print(f"  Steer mean:          {g(all_metrics['steer_means'])[0]:.3f} ± {g(all_metrics['steer_means'])[1]:.3f}")
    print(f"  Steer std:           {g(all_metrics['steer_stds'])[0]:.3f} ± {g(all_metrics['steer_stds'])[1]:.3f}")
    print(f"  Throttle mean:       {g(all_metrics['throttle_means'])[0]:.3f} ± {g(all_metrics['throttle_means'])[1]:.3f}")
    print(f"  Throttle std:        {g(all_metrics['throttle_stds'])[0]:.3f} ± {g(all_metrics['throttle_stds'])[1]:.3f}")