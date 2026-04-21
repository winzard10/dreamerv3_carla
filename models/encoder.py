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


class RGBEncoder(nn.Module):
    """
    Baseline encoder:
      rgb   : [B, 3, 128, 128]
      vector: [B, 3]
      goal  : [B, 2]

    Output:
      embed : [B, embed_dim]
    """
    def __init__(self, embed_dim=1024):
        super().__init__()

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
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

    def forward(self, rgb, vector, goal):
        rgb_feat = self.rgb_stem(rgb)

        x64 = self.stage1(rgb_feat)
        x32 = self.stage2(x64)
        x16 = self.stage3(x32)
        x8  = self.stage4(x16)

        vision_features = self.flatten(x8)
        vector_features = self.vector_fc(vector)
        goal_features   = self.goal_fc(goal)

        fused = torch.cat([vision_features, vector_features, goal_features], dim=1)
        embed = self.fusion(fused)

        return embed