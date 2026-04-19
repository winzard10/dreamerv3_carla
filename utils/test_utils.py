# utils/test_utils.py
import numpy as np
import cv2
import torch
import carla

from params import (
    DEVICE, NUM_CLASSES, H, W,
    DETER_DIM, EMBED_DIM, STOCH_CATEGORICALS, STOCH_CLASSES, CKPT_PATH,
    SHOW_RECON, SHOW_SPECTATOR, SHOW_EVERY_N_STEPS, TEST_NUM_EPISODES,
)
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor
from models.decoder import MultiModalDecoder


# =============================================================================
# Visualization
# =============================================================================

def colorize_segmentation(seg_ids: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    rng    = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    seg_ids = np.clip(seg_ids, 0, num_classes - 1)
    return colors[seg_ids]


def show_reconstruction_windows(obs, deter, stoch, rssm, decoder) -> bool:
    """
    Renders GT vs reconstructed depth and semantic side-by-side.
    Returns False if the user pressed 'q' (signal to stop showing).
    """
    with torch.no_grad():
        stoch_flat = rssm.flatten_stoch(stoch)
        recon_depth, recon_segm_logits, _, _ = decoder(deter, stoch_flat)

    gt_depth = obs["depth"][:, :, 0].astype(np.uint8)
    gt_segm  = obs["semantic"][:, :, 0].astype(np.uint8)

    recon_depth_np = (recon_depth[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    recon_segm_ids = torch.argmax(recon_segm_logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    depth_vis    = np.concatenate([gt_depth, recon_depth_np], axis=1)
    segm_vis     = np.concatenate([colorize_segmentation(gt_segm), colorize_segmentation(recon_segm_ids)], axis=1)
    segm_vis_bgr = cv2.cvtColor(segm_vis, cv2.COLOR_RGB2BGR)

    cv2.imshow("Depth (GT | Recon)",    depth_vis)
    cv2.imshow("Semantic (GT | Recon)", segm_vis_bgr)

    return (cv2.waitKey(1) & 0xFF) != ord("q")


# =============================================================================
# Model loading
# =============================================================================

def build_models(model_path: str = CKPT_PATH):
    Z_DIM = STOCH_CATEGORICALS * STOCH_CLASSES

    encoder = MultiModalEncoder(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, sem_embed_dim=16).to(DEVICE)
    rssm    = RSSM(
        deter_dim=DETER_DIM, act_dim=2, embed_dim=EMBED_DIM, goal_dim=2,
        stoch_categoricals=STOCH_CATEGORICALS, stoch_classes=STOCH_CLASSES,
        unimix_ratio=0.01, kl_balance=0.8, free_nats=0.0,
    ).to(DEVICE)
    actor   = Actor(
        deter_dim=DETER_DIM, stoch_dim=Z_DIM, goal_dim=2, action_dim=2,
        hidden_dim=512, min_std=0.1, init_std=1.0,
    ).to(DEVICE)
    decoder = MultiModalDecoder(deter_dim=DETER_DIM, stoch_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)

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
    depth = (
        torch.as_tensor(obs["depth"].copy())
        .permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    )  # [1, 1, H, W]

    sem = (
        torch.as_tensor(obs["semantic"].copy())
        .permute(2, 0, 1).unsqueeze(0).long().to(DEVICE)
        .clamp(0, NUM_CLASSES - 1)
    )  # [1, 1, H, W]

    vec  = torch.as_tensor(obs.get("vector", np.zeros(3, dtype=np.float32)).copy()).unsqueeze(0).float().to(DEVICE)
    goal = torch.as_tensor(obs.get("goal",   np.zeros(2, dtype=np.float32)).copy()).unsqueeze(0).float().to(DEVICE)

    return depth, sem, vec, goal


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

    results = {"speeds": [], "center_distances": [], "travel_distances": [], "rewards": []}
    keep_showing = SHOW_RECON

    for ep in range(num_episodes):
        obs, _ = env.reset()

        deter, stoch = rssm.initial(1, device=DEVICE)
        prev_action  = torch.zeros(1, 2, device=DEVICE)

        ep_rewards, ep_speeds, ep_center_dist = [], [], []
        total_dist = 0.0
        prev_loc   = env.vehicle.get_location()
        done, step = False, 0

        while not done:
            with torch.no_grad():
                depth, sem, vec, goal = preprocess_obs(obs)
                embed = encoder(depth, sem, vec, goal)   # [1, EMBED_DIM]

                deter, stoch, _, _ = rssm.obs_step(deter, stoch, prev_action, embed, goal)
                action, _, _, _    = actor(deter, rssm.flatten_stoch(stoch), goal, sample=False)
                prev_action        = action

            if keep_showing and step % SHOW_EVERY_N_STEPS == 0:
                keep_showing = show_reconstruction_windows(obs, deter, stoch, rssm, decoder)

            obs, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy()[0])
            done = terminated or truncated

            curr_loc   = env.vehicle.get_location()
            total_dist += curr_loc.distance(prev_loc)
            prev_loc   = curr_loc

            waypoint = carla_map.get_waypoint(curr_loc)
            ep_center_dist.append(curr_loc.distance(waypoint.transform.location))
            ep_speeds.append(float(obs["vector"][0]))
            ep_rewards.append(float(reward))

            if SHOW_SPECTATOR:
                v_t = env.vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    v_t.location - v_t.get_forward_vector() * 8 + carla.Location(z=4),
                    v_t.rotation,
                ))

            step += 1

        results["speeds"].append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        results["center_distances"].append(float(np.mean(ep_center_dist)) if ep_center_dist else 0.0)
        results["travel_distances"].append(float(total_dist))
        results["rewards"].append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)

        print(
            f"  Ep {ep + 1} | "
            f"Dist: {total_dist:.1f}m | "
            f"Avg Speed: {np.mean(ep_speeds):.1f} km/h | "
            f"Avg Reward: {np.mean(ep_rewards):.2f}"
        )

    return results


def print_report(final_report: dict):
    print("\n" + "=" * 40)
    print("EVALUATION REPORT")
    print("=" * 40)
    for town_name, metrics in final_report.items():
        print(f"[{town_name}]")
        print(f"  Avg Speed:           {np.mean(metrics['speeds']):.2f} km/h")
        print(f"  Avg Center Distance: {np.mean(metrics['center_distances']):.2f} m")
        print(f"  Avg Travel Dist:     {np.mean(metrics['travel_distances']):.2f} m")
        print(f"  Avg Episode Reward:  {np.mean(metrics['rewards']):.2f}")
        print("-" * 20)