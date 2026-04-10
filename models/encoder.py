import torch
import torch.nn as nn

class MultiModalEncoder(nn.Module):

    K:int = 4       # Conv kernel size (default: 4)
    d_final = 5     # final output size of encoder (default: 4)

    def __init__(self, latent_dim=1024, num_classes=28, sem_embed_dim=16):
        super().__init__()

        self.sem_embed = nn.Embedding(num_classes, sem_embed_dim)

        # CNN now gets: depth(1) + sem_emb(E) channels
        K = self.K
        D = self.d_final
        in_ch = 1 + sem_embed_dim

        self.cnn = nn.Sequential(
            # Initial Implementation
            nn.Conv2d(in_ch, 32, K, 2), nn.ReLU(),  # H=W=160 -> 80
            nn.Conv2d(32, 64, K, 2), nn.ReLU(),     # H=W=80 -> 40
            nn.Conv2d(64, 128, K, 2), nn.ReLU(),    # H=W=40 -> 20
            nn.Conv2d(128, 256, K, 2), nn.ReLU(),   # H=W=20 -> 10
            # nn.AdaptiveAvgPool2d((4, 4)),
            nn.AdaptiveMaxPool2d((D, D)),
            nn.Flatten(),  # 256 *D*D 

            # # Deeper Implementation
            # nn.Conv2d(in_ch, in_ch, K, 1), nn.ReLU(),
            # nn.Conv2d(in_ch, 32, K, 2), nn.ReLU(),  # H=W=160 -> 80
            # nn.Conv2d(32, 32, K, 1), nn.ReLU(),
            # nn.Conv2d(32, 64, K, 2), nn.ReLU(),     # H=W=80 -> 40
            # nn.Conv2d(64, 64, K, 1), nn.ReLU(),
            # nn.Conv2d(64, 128, K, 2), nn.ReLU(),    # H=W=40 -> 20
            # nn.Conv2d(128, 128, K, 1), nn.ReLU(),
            # nn.Conv2d(128, 256, K, 2), nn.ReLU(),   # H=W=20 -> 10
            # nn.AdaptiveMaxPool2d((D, D)),
            # nn.Flatten(),  # 256 *D*D 
        )

        self.vector_fc = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.goal_fc   = nn.Sequential(nn.Linear(2, 32), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.Linear(256*(D**2) + 64 + 32, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, depth, sem_ids, vector, goal):
        # depth:  [B,1,H,W] float in [0,1]
        # sem_ids:[B,1,H,W] long in [0..num_classes-1]

        sem_ids = sem_ids.squeeze(1).long()          # [B,H,W]
        sem_emb = self.sem_embed(sem_ids)            # [B,H,W,E]
        sem_emb = sem_emb.permute(0, 3, 1, 2)        # [B,E,H,W]

        x_vision = torch.cat([depth, sem_emb], dim=1)  # [B,1+E,H,W]
        vision_features = self.cnn(x_vision)

        vector_features = self.vector_fc(vector)
        goal_features   = self.goal_fc(goal)

        fused = torch.cat([vision_features, vector_features, goal_features], dim=1)
        return self.fusion(fused)