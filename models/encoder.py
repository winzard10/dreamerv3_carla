import torch
import torch.nn as nn

class MultiModalEncoder(nn.Module):
    d_final = 4

    def __init__(self, embed_dim=1024, num_classes=28, sem_embed_dim=16):
        super().__init__()

        D = self.d_final
        in_ch = 1 + sem_embed_dim

        self.sem_embed = nn.Embedding(num_classes, sem_embed_dim)

        self.cnn = nn.Sequential(
            # 128 x 128
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(),      # 128 -> 64

            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),         # 64 -> 32

            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),        # 32 -> 16

            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),       # 16 -> 8

            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1), nn.ReLU(),       # 8 -> 4

            nn.Flatten(),                                  # 256 * 4 * 4 = 4096
        )

        self.vector_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )

        self.goal_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 * (D ** 2) + 64 + 32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, depth, sem_ids, vector, goal):
        sem_ids = sem_ids.squeeze(1).long()          # [B,H,W]
        sem_emb = self.sem_embed(sem_ids)            # [B,H,W,E]
        sem_emb = sem_emb.permute(0, 3, 1, 2)        # [B,E,H,W]

        x_vision = torch.cat([depth, sem_emb], dim=1)
        vision_features = self.cnn(x_vision)

        vector_features = self.vector_fc(vector)
        goal_features = self.goal_fc(goal)

        fused = torch.cat([vision_features, vector_features, goal_features], dim=1)
        return self.fusion(fused)