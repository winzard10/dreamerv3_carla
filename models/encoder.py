import torch
import torch.nn as nn

class MultiModalEncoder(nn.Module):
    def __init__(self, latent_dim=1024, num_classes=28, sem_embed_dim=16):
        super().__init__()

        self.sem_embed = nn.Embedding(num_classes, sem_embed_dim)

        # CNN now gets: depth(1) + sem_emb(E) channels
        in_ch = 1 + sem_embed_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),  # 256*4*4=4096
        )

        self.vector_fc = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.goal_fc   = nn.Sequential(nn.Linear(2, 32), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.Linear(4096 + 64 + 32, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, depth, sem_ids, vector, goal):
        # depth:  [B,1,H,W] float in [0,1]
        # sem_ids:[B,1,H,W] long in [0..num_classes-1]

        sem_ids = sem_ids.squeeze(1).long()                 # [B,H,W]
        sem_emb = self.sem_embed(sem_ids)            # [B,H,W,E]
        sem_emb = sem_emb.permute(0, 3, 1, 2)        # [B,E,H,W]

        x_vision = torch.cat([depth, sem_emb], dim=1)  # [B,1+E,H,W]
        vision_features = self.cnn(x_vision)

        vector_features = self.vector_fc(vector)
        goal_features   = self.goal_fc(goal)

        fused = torch.cat([vision_features, vector_features, goal_features], dim=1)
        return self.fusion(fused)