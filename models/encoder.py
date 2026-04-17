import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiModalEncoder(nn.Module):
    def __init__(self, embed_dim=1024, num_classes=28, sem_embed_dim=16):
        super().__init__()


        self.sem_embed = nn.Embedding(num_classes, sem_embed_dim)

        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

        self.sem_stem = nn.Sequential(
            nn.Conv2d(sem_embed_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

        self.stage1 = ConvBlock(32, 32, stride=2)    # 128 -> 64
        self.stage2 = ConvBlock(32, 64, stride=2)    # 64 -> 32
        self.stage3 = ConvBlock(64, 128, stride=2)   # 32 -> 16
        self.stage4 = ConvBlock(128, 256, stride=2)  # 16 -> 8

        self.flatten = nn.Flatten()

        self.vector_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ELU(),
        )

        self.goal_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 * 8 * 8 + 64 + 32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )

    def forward(self, depth, sem_ids, vector, goal):
        if sem_ids.dim() == 4:
            sem_ids = sem_ids.squeeze(1)
        sem_ids = sem_ids.long()

        sem_emb = self.sem_embed(sem_ids)
        sem_emb = sem_emb.permute(0, 3, 1, 2)

        depth_feat = self.depth_stem(depth)
        sem_feat = self.sem_stem(sem_emb)

        x = torch.cat([depth_feat, sem_feat], dim=1)

        x64 = self.stage1(x)
        x32 = self.stage2(x64)
        x16 = self.stage3(x32)
        x8 = self.stage4(x16)

        vision_features = self.flatten(x8)
        vector_features = self.vector_fc(vector)
        goal_features = self.goal_fc(goal)

        fused = torch.cat([vision_features, vector_features, goal_features], dim=1)
        embed = self.fusion(fused)

        return embed