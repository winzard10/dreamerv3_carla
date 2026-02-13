import torch
import torch.nn as nn

class MultiModalDecoder(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=32):
        super(MultiModalDecoder, self).__init__()
        # FIX 1: Match input size to RSSM (512 + 32 = 544)
        input_dim = deter_dim + stoch_dim
        
        self.fc = nn.Linear(input_dim, 256 * 8 * 8)
        
        self.deconv = nn.Sequential(
            # 8x8 -> 18x18
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), 
            nn.ReLU(),
            # 18x18 -> 38x38
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 38x38 -> 79x79
            # FIX 2: Added output_padding=1 here to match the Encoder's shape perfectly
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            # 79x79 -> 160x160
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2),
            nn.Sigmoid() 
        )

    def forward(self, deter, stoch):
        # FIX 3: Concatenate properly without reshaping (since we aren't using discrete vars)
        x = torch.cat([deter, stoch], dim=-1)
        x = self.fc(x).view(-1, 256, 8, 8)
        
        reconstruction = self.deconv(x)
        
        # FIX 4: Split the 2-channel output into Depth and Semantic
        # This allows: recon_depth, recon_sem = decoder(...)
        recon_depth = reconstruction[:, 0:1, :, :]
        recon_sem = reconstruction[:, 1:2, :, :]
        
        return recon_depth, recon_sem