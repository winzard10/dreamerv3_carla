# test.py
import cv2

from params import TEST_TOWN, TEST_NUM_EPISODES
from utils.test_utils import build_models, run_evaluation, print_report
from env.carla_wrapper import CarlaEnv


# def colorize_segmentation(seg_ids: np.ndarray, num_classes: int = 28) -> np.ndarray:
#     """
#     Convert [H,W] semantic IDs into a color image [H,W,3].
#     """
#     rng = np.random.default_rng(0)
#     colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
#     colors[0] = np.array([0, 0, 0], dtype=np.uint8)  # class 0 -> black

#     seg_ids = np.clip(seg_ids, 0, num_classes - 1)
#     return colors[seg_ids]


# def build_models():
#     encoder = MultiModalEncoder(
#         embed_dim=EMBED_DIM,
#         num_classes=NUM_CLASSES,
#         sem_embed_dim=16,
#     ).to(DEVICE)

#     rssm = RSSM(
#         deter_dim=DETER_DIM,
#         act_dim=2,
#         embed_dim=EMBED_DIM,
#         goal_dim=2,
#         stoch_categoricals=STOCH_CATEGORICALS,
#         stoch_classes=STOCH_CLASSES,
#         unimix_ratio=0.01,
#         kl_balance=0.8,
#         free_nats=0.1,
#     ).to(DEVICE)

#     actor = Actor(
#         deter_dim=DETER_DIM,
#         stoch_dim=Z_DIM,
#         goal_dim=2,
#         action_dim=2,
#         hidden_dim=512,
#         min_std=0.1,
#         init_std=1.0,
#     ).to(DEVICE)

#     decoder = MultiModalDecoder(
#         deter_dim=DETER_DIM,
#         stoch_dim=Z_DIM,
#         num_classes=NUM_CLASSES,
#     ).to(DEVICE)

#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

#     encoder.load_state_dict(checkpoint["encoder"])
#     rssm.load_state_dict(checkpoint["rssm"])
#     actor.load_state_dict(checkpoint["actor"])
#     decoder.load_state_dict(checkpoint["decoder"])

#     encoder.eval()
#     rssm.eval()
#     actor.eval()
#     decoder.eval()

#     return encoder, rssm, actor, decoder


# def show_reconstruction_windows(obs, deter, stoch, rssm, decoder):
#     with torch.no_grad():
#         stoch_flat = rssm.flatten_stoch(stoch)
#         recon_depth, recon_segm_logits, _,_ = decoder(deter, stoch_flat, out_hw=(H, W))

#     # Ground truth
#     gt_depth = obs["depth"][:, :, 0].astype(np.uint8)         # [H,W]
#     gt_segm = obs["semantic"][:, :, 0].astype(np.uint8)       # [H,W]

#     # Reconstruction
#     recon_depth_np = (
#         recon_depth[0, 0].detach().cpu().numpy() * 255.0
#     ).clip(0, 255).astype(np.uint8)

#     recon_segm_ids = torch.argmax(recon_segm_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

#     # Depth side-by-side
#     depth_vis = np.concatenate([gt_depth, recon_depth_np], axis=1)

#     # Semantic side-by-side, colorized
#     gt_segm_color = colorize_segmentation(gt_segm, NUM_CLASSES)
#     recon_segm_color = colorize_segmentation(recon_segm_ids, NUM_CLASSES)
#     segm_vis = np.concatenate([gt_segm_color, recon_segm_color], axis=1)
#     segm_vis_bgr = cv2.cvtColor(segm_vis, cv2.COLOR_RGB2BGR)

#     cv2.imshow("Depth (GT | Recon)", depth_vis)
#     cv2.imshow("Semantic (GT | Recon)", segm_vis_bgr)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         return False
#     return True


# def run_evaluation(town_name: str, num_episodes: int = 5):
#     print(f"\n>>> Starting Evaluation on {town_name} <<<")

#     env = CarlaEnv(town=town_name)
#     world = env.world
#     carla_map = world.get_map()
#     spectator = world.get_spectator()

#     available_maps = env.client.get_available_maps()
#     print("Available Maps:")
#     for map_name in available_maps:
#         print(f" - {map_name}")

def main():
    print(f"\n>>> Starting Evaluation on {TEST_TOWN} <<<")

    env = CarlaEnv(town=TEST_TOWN)
    encoder, rssm, actor, decoder = build_models()

    try:
        results = run_evaluation(env, encoder, rssm, actor, decoder,
                                 num_episodes=TEST_NUM_EPISODES)
        print_report({TEST_TOWN: results})
    finally:
        cv2.destroyAllWindows()
        env._cleanup()


if __name__ == "__main__":
    main()