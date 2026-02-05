import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, deter_dim=1024, stoch_dim=32, discrete_dim=32, action_dim=2):
        super(Actor, self).__init__()
        # Input is the combined state: Deterministic + Stochastic
        input_dim = deter_dim + (stoch_dim * discrete_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh() # Scales output to [-1.0, 1.0] for Steer and Throttle/Brake
        )

    def forward(self, deter, stoch):
        state = torch.cat([deter, stoch.reshape(stoch.shape[0], -1)], dim=-1)
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, deter_dim=1024, stoch_dim=32, discrete_dim=32):
        super(Critic, self).__init__()
        input_dim = deter_dim + (stoch_dim * discrete_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Outputs a single scalar value (expected return)
        )

    def forward(self, deter, stoch):
        state = torch.cat([deter, stoch.reshape(stoch.shape[0], -1)], dim=-1)
        return self.net(state)