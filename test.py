import torch
import carla
import numpy as np
from env.carla_wrapper import CarlaEnv
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/dreamerv3/dreamerv3_ep899.pth"

NUM_CLASSES = 28

def test(num_episodes=5):
    env = CarlaEnv()
    world = env.world
    spectator = world.get_spectator()

    # Ensure sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # -------- Load models --------
    encoder = MultiModalEncoder(latent_dim=1024, num_classes=NUM_CLASSES).to(DEVICE)
    rssm = RSSM().to(DEVICE)
    actor = Actor(goal_dim=2).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder"])
    rssm.load_state_dict(checkpoint["rssm"])
    actor.load_state_dict(checkpoint["actor"])

    encoder.eval()
    rssm.eval()
    actor.eval()

    total_velocities = []

    for ep in range(num_episodes):
        obs, _ = env.reset()

        # Initialize RSSM state
        deter, stoch = rssm.initial(1, device=DEVICE)
        prev_action = torch.zeros(1, 2, device=DEVICE)

        ep_velocity = []
        done = False

        while not done:
            with torch.no_grad():
                # ---- Prepare inputs ----
                depth = (
                    torch.as_tensor(obs["depth"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(DEVICE)
                    / 255.0
                )

                # IMPORTANT: semantic is class ID — do NOT divide
                sem = (
                    torch.as_tensor(obs["semantic"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .long()
                    .to(DEVICE)
                    .clamp(0, NUM_CLASSES - 1)
                )

                vec = torch.as_tensor(obs["vector"]).unsqueeze(0).float().to(DEVICE)
                goal = torch.as_tensor(obs["goal"]).unsqueeze(0).float().to(DEVICE)

                # ---- Encode ----
                embed = encoder(depth, sem, vec, goal)

                # ---- RSSM update from real observation ----
                deter, stoch, _, _ = rssm.obs_step(
                    deter, stoch, prev_action, embed, goal
                )

                stoch_flat = rssm.flatten_stoch(stoch)

                # ---- Actor (deterministic for evaluation) ----
                action, _, _, _ = actor(deter, stoch_flat, goal, sample=False)

                prev_action = action

            # ---- Step environment ----
            act_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            # ---- Spectator chase cam ----
            v_trans = env.vehicle.get_transform()
            cam_pos = (
                v_trans.location
                - (v_trans.get_forward_vector() * 8.0)
                + carla.Location(z=4.0)
            )
            spectator.set_transform(carla.Transform(cam_pos, v_trans.rotation))

            ep_velocity.append(obs["vector"][0])

        avg_speed = np.mean(ep_velocity)
        total_velocities.append(avg_speed)
        print(f"Episode {ep+1} | Avg Speed: {avg_speed:.2f} km/h")

    print(f"\nFinal Average Velocity: {np.mean(total_velocities):.2f} km/h")


if __name__ == "__main__":
    test()