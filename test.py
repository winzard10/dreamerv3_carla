import torch
import carla
import numpy as np
import cv2

from env.carla_wrapper import CarlaEnv
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor
from models.decoder import MultiModalDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./checkpoints/dreamerv3/dreamerv3_latest.pth"
NUM_CLASSES = 28
TEST_TOWN = "Town10HD"   # e.g. "Town01", "Town02", "Town10HD"

# Match these to training
H, W = 128, 128
DETER_DIM = 512
EMBED_DIM = 1024
STOCH_CATEGORICALS = 32
STOCH_CLASSES = 32
Z_DIM = STOCH_CATEGORICALS * STOCH_CLASSES

# Visualization flags
SHOW_RECON = True
SHOW_SPECTATOR = True
SHOW_EVERY_N_STEPS = 3   # reduce GUI overhead


def colorize_segmentation(seg_ids: np.ndarray, num_classes: int = 28) -> np.ndarray:
    """
    Convert [H,W] semantic IDs into a color image [H,W,3].
    """
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)  # class 0 -> black

    seg_ids = np.clip(seg_ids, 0, num_classes - 1)
    return colors[seg_ids]


def build_models():
    encoder = MultiModalEncoder(
        latent_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        sem_embed_dim=16,
    ).to(DEVICE)

    rssm = RSSM(
        deter_dim=DETER_DIM,
        act_dim=2,
        embed_dim=EMBED_DIM,
        goal_dim=2,
        stoch_categoricals=STOCH_CATEGORICALS,
        stoch_classes=STOCH_CLASSES,
        unimix_ratio=0.01,
        kl_balance=0.8,
        free_nats=0.1,
    ).to(DEVICE)

    actor = Actor(
        deter_dim=DETER_DIM,
        stoch_dim=Z_DIM,
        goal_dim=2,
        action_dim=2,
        hidden_dim=512,
        min_std=0.1,
        init_std=1.0,
    ).to(DEVICE)

    decoder = MultiModalDecoder(
        deter_dim=DETER_DIM,
        stoch_dim=Z_DIM,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    encoder.load_state_dict(checkpoint["encoder"])
    rssm.load_state_dict(checkpoint["rssm"])
    actor.load_state_dict(checkpoint["actor"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    rssm.eval()
    actor.eval()
    decoder.eval()

    return encoder, rssm, actor, decoder


def show_reconstruction_windows(obs, deter, stoch, rssm, decoder):
    with torch.no_grad():
        stoch_flat = rssm.flatten_stoch(stoch)
        recon_depth, recon_segm_logits = decoder(deter, stoch_flat, out_hw=(H, W))

    # Ground truth
    gt_depth = obs["depth"][:, :, 0].astype(np.uint8)         # [H,W]
    gt_segm = obs["semantic"][:, :, 0].astype(np.uint8)       # [H,W]

    # Reconstruction
    recon_depth_np = (
        recon_depth[0, 0].detach().cpu().numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)

    recon_segm_ids = torch.argmax(recon_segm_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    # Depth side-by-side
    depth_vis = np.concatenate([gt_depth, recon_depth_np], axis=1)

    # Semantic side-by-side, colorized
    gt_segm_color = colorize_segmentation(gt_segm, NUM_CLASSES)
    recon_segm_color = colorize_segmentation(recon_segm_ids, NUM_CLASSES)
    segm_vis = np.concatenate([gt_segm_color, recon_segm_color], axis=1)
    segm_vis_bgr = cv2.cvtColor(segm_vis, cv2.COLOR_RGB2BGR)

    cv2.imshow("Depth (GT | Recon)", depth_vis)
    cv2.imshow("Semantic (GT | Recon)", segm_vis_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    return True


def run_evaluation(town_name: str, num_episodes: int = 5):
    print(f"\n>>> Starting Evaluation on {town_name} <<<")

    env = CarlaEnv(town=town_name)
    world = env.world
    carla_map = world.get_map()
    spectator = world.get_spectator()

    available_maps = env.client.get_available_maps()
    print("Available Maps:")
    for map_name in available_maps:
        print(f" - {map_name}")

    encoder, rssm, actor, decoder = build_models()

    town_results = {
        "speeds": [],
        "center_distances": [],
        "travel_distances": [],
        "rewards": [],
    }

    keep_showing = SHOW_RECON

    for ep in range(num_episodes):
        obs, _ = env.reset()

        deter, stoch = rssm.initial(1, device=DEVICE)
        prev_action = torch.zeros(1, 2, device=DEVICE)

        ep_rewards = []
        ep_speeds = []
        ep_center_dist = []
        total_dist = 0.0
        prev_loc = env.vehicle.get_location()

        done = False
        step = 0

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

            if keep_showing and step % SHOW_EVERY_N_STEPS == 0:
                keep_showing = show_reconstruction_windows(obs, deter, stoch, rssm, decoder)

            obs, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy()[0])
            done = terminated or truncated

            # Metrics
            ep_speeds.append(obs["vector"][0])

            curr_loc = env.vehicle.get_location()
            total_dist += curr_loc.distance(prev_loc)
            prev_loc = curr_loc

            waypoint = carla_map.get_waypoint(curr_loc)
            ep_center_dist.append(curr_loc.distance(waypoint.transform.location))

            ep_rewards.append(reward)

            if SHOW_SPECTATOR:
                v_trans = env.vehicle.get_transform()
                spectator.set_transform(
                    carla.Transform(
                        v_trans.location - v_trans.get_forward_vector() * 8 + carla.Location(z=4),
                        v_trans.rotation,
                    )
                )

            step += 1

        town_results["speeds"].append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        town_results["center_distances"].append(float(np.mean(ep_center_dist)) if ep_center_dist else 0.0)
        town_results["travel_distances"].append(float(total_dist))
        town_results["rewards"].append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)

        print(
            f"  Ep {ep + 1} | "
            f"Dist: {total_dist:.1f}m | "
            f"Avg Speed: {np.mean(ep_speeds):.1f}km/h | "
            f"Avg Reward: {np.mean(ep_rewards):.2f}"
        )

    cv2.destroyAllWindows()
    env._cleanup()

    return town_results


def test_all_towns(town):
    final_report = {}
    final_report[town] = run_evaluation(town)

    print("\n" + "=" * 40)
    print("FINAL MULTI-TOWN PERFORMANCE REPORT")
    print("=" * 40)

    for town_name, metrics in final_report.items():
        print(f"[{town_name}]")
        print(f"  Avg Speed:           {np.mean(metrics['speeds']):.2f} km/h")
        print(f"  Avg Center Distance: {np.mean(metrics['center_distances']):.2f} m")
        print(f"  Avg Travel Dist:     {np.mean(metrics['travel_distances']):.2f} m")
        print(f"  Avg Episode Reward:  {np.mean(metrics['rewards']):.2f}")
        print("-" * 20)


if __name__ == "__main__":
    test_all_towns(TEST_TOWN)