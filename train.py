import torch
import torch.optim as optim
from env.carla_wrapper import CarlaEnv
from models.encoder import MultiModalEncoder
from models.rssm import RSSM
from models.actor_critic import Actor, Critic
from models.decoder import MultiModalDecoder
from utils.buffer import SequenceBuffer

# 1. Hyperparameters & Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 50 
BATCH_SIZE = 16
LEARNING_RATE = 3e-4

def train():
    # 2. Initialize Environment and Buffer
    env = CarlaEnv()
    buffer = SequenceBuffer(capacity=10000, seq_len=SEQ_LEN, obs_shape=(160, 160, 1), action_dim=2, device=DEVICE)

    # 3. Initialize Models onto RTX 4070
    encoder = MultiModalEncoder().to(DEVICE)
    rssm = RSSM().to(DEVICE)
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    decoder = MultiModalDecoder().to(DEVICE)

    # Combined Optimizer for the World Model
    model_params = list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(model_params, lr=LEARNING_RATE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=8e-5)

    for episode in range(1000):
        obs = env.reset()
        state = rssm.get_initial_state(1, DEVICE)
        
        # --- PHASE 1: Real-World Collection ---
        for step in range(500):
            # Encode and get action
            with torch.no_grad():
                embed = encoder(
                    torch.as_tensor(obs['depth']).to(DEVICE).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
                    torch.as_tensor(obs['semantic']).to(DEVICE).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
                    torch.as_tensor(obs['vector']).to(DEVICE).unsqueeze(0)
                )
                _, state = rssm.observe(embed, torch.zeros(1, 2).to(DEVICE), state) # Simplified action input
                action = actor(state[0], state[1])
            
            next_obs, reward, done, _ = env.step(action.cpu().numpy()[0])
            buffer.add(obs['depth'], obs['semantic'], action.cpu().numpy()[0], reward, done)
            obs = next_obs
            if done: break

        # --- PHASE 2: World Model & Imagination Training ---
        if buffer.full or buffer.idx > BATCH_SIZE:
            # Sample sequences from buffer
            depths, sems, actions, rewards, terms = buffer.sample(BATCH_SIZE)

            # 1. Update World Model (Encoder + RSSM + Decoder)
            # This is where the model learns how physics and the road work
            optimizer.zero_grad()
            # ... (RSSM sequence processing and reconstruction loss calculation)
            # loss_total.backward()
            optimizer.step()

            # 2. Update Actor-Critic (Imagination Training)
            # The agent "dreams" inside the RSSM to optimize its policy
            actor_optimizer.zero_grad()
            # ... (Value estimation and policy gradient)
            # actor_loss.backward()
            actor_optimizer.step()

            print(f"Episode {episode} Training Step Complete.")

if __name__ == "__main__":
    train()