import torch
import torch.nn as nn

class RSSM(nn.Module):
    def __init__(self, latent_dim=1024, deter_dim=1024, stoch_dim=32, discrete_dim=32):
        super(RSSM, self).__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        
        # 1. GRU Cell: The deterministic "memory"
        # Takes (Action + Stochastic State) and updates the Deterministic State
        self.gru = nn.GRUCell(latent_dim + (stoch_dim * discrete_dim), deter_dim)
        
        # 2. Transition Model: Predicts the NEXT stochastic state (The Imagination)
        # Takes the current Deterministic State and predicts distribution for next step
        self.img_prior = nn.Sequential(
            nn.Linear(deter_dim, 512),
            nn.ReLU(),
            nn.Linear(512, stoch_dim * discrete_dim)
        )
        
        # 3. Representation Model: Corrects the state based on REAL observations
        # Takes (Deterministic State + Current Observation) and gives a better Stochastic State
        self.obs_posterior = nn.Sequential(
            nn.Linear(deter_dim + latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, stoch_dim * discrete_dim)
        )

    def get_initial_state(self, batch_size, device):
        # Initialize the memory (zeros)
        return (torch.zeros(batch_size, self.deter_dim).to(device),
                torch.zeros(batch_size, self.stoch_dim, self.discrete_dim).to(device))

    def observe(self, embed, action, state):
        """Used during REAL training with CARLA observations."""
        deter, stoch = state
        # Flat stoch for GRU input
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        
        # Update deterministic state (Memory)
        deter = self.gru(torch.cat([stoch_flat, action], dim=-1), deter)
        
        # Get posterior (The "Now I see it" state)
        logits = self.obs_posterior(torch.cat([deter, embed], dim=-1))
        stoch_post = self._sample_stochastic(logits)
        
        return deter, stoch_post

    def imagine(self, action, state):
        """Used during IMAGINED training (predicting the future)."""
        deter, stoch = state
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        
        deter = self.gru(torch.cat([stoch_flat, action], dim=-1), deter)
        
        # Get prior (The "I think this will happen" state)
        logits = self.img_prior(deter)
        stoch_prior = self._sample_stochastic(logits)
        
        return deter, stoch_prior

    def _sample_stochastic(self, logits):
        # DreamerV3 uses discrete (Categorical) stochastic states for stability
        logits = logits.view(-1, self.stoch_dim, self.discrete_dim)
        # Gumbel-Softmax trick for backpropagation through discrete sampling
        probs = torch.softmax(logits, dim=-1)
        return probs # In a full implementation, you'd sample or use Straight-Through gradients