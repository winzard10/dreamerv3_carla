import torch
import torch.nn as nn

class RSSM(nn.Module):
    def __init__(self, hidden_dim=512, state_dim=32, act_dim=2, embed_dim=1024, goal_dim=2):
        super(RSSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        
        self.transition_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.representation_model = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.gru = nn.GRUCell(state_dim + act_dim, hidden_dim)
        
        self.reward_model = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_initial_state(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim).to(device)

    def predict_reward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.reward_model(state)

    def observe_sequence(self, embeds, actions, goals):
        batch_size, seq_len, _ = embeds.shape
        device = embeds.device
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        # We now collect separate lists for h and z
        post_h, post_z = [], []
        prior_h, prior_z = [], []
        
        for t in range(seq_len):
            # 1. Prior (Prediction)
            prior_z_t = self.transition_model(h)
            
            # 2. Posterior (Reality)
            post_z_t = self.representation_model(torch.cat([h, embeds[:, t], goals[:, t]], dim=-1))
            
            # Store the state (h, z) used at this timestep
            post_h.append(h); post_z.append(post_z_t)
            prior_h.append(h); prior_z.append(prior_z_t)
            
            # 3. Update memory for next step
            h = self.gru(torch.cat([post_z_t, actions[:, t]], dim=-1), h)
        
        # Return tuples: (h_seq, z_seq)
        return (torch.stack(post_h, dim=1), torch.stack(post_z, dim=1)), \
               (torch.stack(prior_h, dim=1), torch.stack(prior_z, dim=1))
    
    def imagine(self, start_h, start_z, goal, actor, horizon=15):
        h, z = start_h, start_z
        imag_h, imag_z = [], []
        
        for _ in range(horizon):
            action = actor(h, z, goal)
            # 1. Update memory using the previous state and action
            h = self.gru(torch.cat([z, action], dim=-1), h)
            # 2. Imagine the next stochastic state from that memory
            z = self.transition_model(h)
            imag_h.append(h); imag_z.append(z)
            
        return torch.stack(imag_h, dim=1), torch.stack(imag_z, dim=1)

    def kl_loss(self, posteriors, priors, free_nats=0.01, alpha=0.8):
        # 1. KL(Post || Prior) - forcing Memory to match Reality
        # Detach posteriors so we only move the Priors
        kl_prior = torch.pow(posteriors.detach() - priors, 2).mean()
        
        # 2. KL(Post || Prior) - keeping Reality organized
        # Detach priors so we only move the Posteriors
        kl_post = torch.pow(posteriors - priors.detach(), 2).mean()
        
        # 3. Balancing: Force memory to learn faster than perception is squashed
        kl = alpha * kl_prior + (1 - alpha) * kl_post
        
        # 4. Use a MUCH smaller free_nats to force visual attention
        return torch.max(kl, torch.tensor(free_nats).to(kl.device))