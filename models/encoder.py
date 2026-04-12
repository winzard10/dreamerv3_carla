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
    """
    Encodes:
      - depth: [B, 1, 128, 128]
      - sem_ids: [B, 1, 128, 128] or [B, 128, 128]
      - vector: [B, 3]
      - goal:   [B, 2]

    Returns:
      {
        "embed":   [B, embed_dim],
        "skip16":  [B, 128, 16, 16],
        "skip32":  [B, 64,  32, 32],
        "skip64":  [B, 32,  64, 64],
      }
    """

    def __init__(self, embed_dim=1024, num_classes=28, sem_embed_dim=16):
        super().__init__()

        self.sem_embed = nn.Embedding(num_classes, sem_embed_dim)

        # Separate modality stems
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

        # Fused visual backbone
        self.stage1 = ConvBlock(32, 32, stride=2)    # 128 -> 64
        self.stage2 = ConvBlock(32, 64, stride=2)    # 64 -> 32
        self.stage3 = ConvBlock(64, 128, stride=2)   # 32 -> 16
        self.stage4 = ConvBlock(128, 256, stride=2)  # 16 -> 8

        # Keep stage5 defined only if you want to inspect x4 later,
        # but it is NOT used for the final embedding in this test.
        self.stage5 = ConvBlock(256, 256, stride=2)  # 8 -> 4

        self.flatten = nn.Flatten()

        self.vector_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ELU(),
        )

        self.goal_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ELU(),
        )

        # IMPORTANT:
        # Use x8 (256, 8, 8) instead of x4 (256, 4, 4)
        self.fusion = nn.Sequential(
            nn.Linear(256 * 8 * 8 + 64 + 32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )

    def forward(self, depth, sem_ids, vector, goal):
        if sem_ids.dim() == 4:
            sem_ids = sem_ids.squeeze(1)
        sem_ids = sem_ids.long()

        sem_emb = self.sem_embed(sem_ids)       # [B,H,W,E]
        sem_emb = sem_emb.permute(0, 3, 1, 2)   # [B,E,H,W]

        depth_feat = self.depth_stem(depth)     # [B,16,128,128]
        sem_feat = self.sem_stem(sem_emb)       # [B,16,128,128]

        x = torch.cat([depth_feat, sem_feat], dim=1)  # [B,32,128,128]

        x64 = self.stage1(x)    # [B,32,64,64]
        x32 = self.stage2(x64)  # [B,64,32,32]
        x16 = self.stage3(x32)  # [B,128,16,16]
        x8  = self.stage4(x16)  # [B,256,8,8]

        # Optional: still compute x4 for debugging, but do not use it
        # x4 = self.stage5(x8)

        vision_features = self.flatten(x8)

        vector_features = self.vector_fc(vector)
        goal_features = self.goal_fc(goal)

        fused = torch.cat([vision_features, vector_features, goal_features], dim=1)
        embed = self.fusion(fused)

        return {
            "embed": embed,
            "skip16": x16,
            "skip32": x32,
            "skip64": x64,
        }