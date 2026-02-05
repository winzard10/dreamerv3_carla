import torch
import torch.nn as nn

class MultiModalDecoder(nn.Module):
    def __init__(self, deter_dim=1024, stoch_dim=32, discrete_dim=32):
        super(MultiModalDecoder, self).__init__()
        input_dim = deter_dim + (stoch_dim * discrete_dim)
        
        self.fc = nn.Linear(input_dim, 256 * 8 * 8)
        
        # Mirror of the Encoder (Deconvolutional layers)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), # 18x18
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),  # 38x38
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),   # 79x79
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, output_padding=1), # 160x160
            nn.Sigmoid() # Normalizes output to [0, 1] for image reconstruction
        )

    def forward(self, deter, stoch):
        state = torch.cat([deter, stoch.reshape(stoch.shape[0], -1)], dim=-1)
        x = self.fc(state).view(-1, 256, 8, 8)
        return self.deconv(x)