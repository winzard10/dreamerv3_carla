import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, hidden_dim=512, state_dim=32, action_dim=2, goal_dim=2):
        super(Actor, self).__init__()
        # Input is combined RSSM state: h (hidden) + z (stochastic) + goal
        input_dim = hidden_dim + state_dim + goal_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Keeps steering/throttle between -1 and 1
        )

    def forward(self, h, z, goal):
        # Concatenate the deterministic and stochastic states
        x = torch.cat([h, z, goal], dim=-1)
        return self.layers(x)

class Critic(nn.Module):
    def __init__(self, hidden_dim=512, state_dim=32):
        super(Critic, self).__init__()
        input_dim = hidden_dim + state_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1) # Outputs a single 'Value' score
        )

    def forward(self, h, z):
        x = torch.cat([h, z], dim=-1)
        return self.layers(x)