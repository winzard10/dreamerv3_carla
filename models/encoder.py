import torch
import torch.nn as nn

class MultiModalEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(MultiModalEncoder, self).__init__()
        
        # 1. Vision Branch: Depth + Semantic Segmentation
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Flatten() # Output: 16384
        )
        
        # 2. State Branch: Processes Speed/Heading (Size 3)
        self.vector_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )

        # 3. NEW: Navigation Intent Branch: Processes Local Waypoint Vector (dx, dy)
        self.goal_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # 4. Fusion Layer: Merges Perception + State + Intent
        # Total size = 16384 (Vision) + 64 (State) + 32 (Goal)
        self.fusion = nn.Sequential(
            nn.Linear(16384 + 64 + 32, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(self, depth, semantic, vector, goal):
        # Vision features
        x_vision = torch.cat([depth, semantic], dim=1)
        vision_features = self.cnn(x_vision)
        
        # State features (Speed/Heading)
        vector_features = self.vector_fc(vector)

        # Goal features (The local waypoint dx, dy)
        goal_features = self.goal_fc(goal)
        
        # Concatenate all three branches
        fused = torch.cat([vision_features, vector_features, goal_features], dim=1)
        latent = self.fusion(fused)
        
        return latent