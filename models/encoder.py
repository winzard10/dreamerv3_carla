import torch
import torch.nn as nn

class MultiModalEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(MultiModalEncoder, self).__init__()
        
        # 1. Vision Branch: Processes Depth + Semantic Segmentation
        # Input shape: (Batch, 2 Channels, 160, 160)
        # We stack Depth and Semantic along the channel dimension
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2), # Output: 79x79
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Output: 38x38
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # Output: 18x18
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), # Output: 8x8
            nn.ReLU(),
            nn.Flatten() # Output: 256 * 8 * 8 = 16384
        )
        
        # 2. Vector Branch: Processes Speed/Heading
        # Input shape: (Batch, 3)
        self.vector_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        
        # 3. Fusion Layer: Merges Vision and Vector data
        self.fusion = nn.Sequential(
            nn.Linear(16384 + 64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(self, depth, semantic, vector):
        # Stack images along the channel dimension
        # Depth: (B, 1, 160, 160), Semantic: (B, 1, 160, 160) -> (B, 2, 160, 160)
        x_vision = torch.cat([depth, semantic], dim=1)
        
        vision_features = self.cnn(x_vision)
        vector_features = self.vector_fc(vector)
        
        # Concatenate and map to latent dimension
        fused = torch.cat([vision_features, vector_features], dim=1)
        latent = self.fusion(fused)
        
        return latent