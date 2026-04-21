import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRefine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = ConvRefine(in_ch, out_ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.refine(x)


class RGBDecoder(nn.Module):
    """
    Baseline decoder:
      deter: [B, deter_dim]
      stoch: [B, stoch_dim]

    Outputs:
      rgb        : [B, 3, 128, 128]
      goal_pred  : [B, 2]
      vector_pred: [B, 3]
    """
    def __init__(self, deter_dim=512, stoch_dim=1024):
        super().__init__()
        in_dim = deter_dim + stoch_dim

        self.fc = nn.Linear(in_dim, 128 * 16 * 16)

        self.init_refine = ConvRefine(128, 128)  # 16x16
        self.up32 = UpBlock(128, 96)             # 16 -> 32
        self.up64 = UpBlock(96, 64)              # 32 -> 64
        self.up128 = UpBlock(64, 48)             # 64 -> 128
        self.shared_refine = ConvRefine(48, 48)

        self.rgb_branch = nn.Sequential(
            ConvRefine(48, 32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

        self.goal_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2),
        )

        self.vector_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 3),
        )

    def forward(self, deter, stoch):
        flat = torch.cat([deter, stoch], dim=-1)
        bsz = flat.shape[0]

        x = self.fc(flat).view(bsz, 128, 16, 16)
        x = self.init_refine(x)
        x = self.up32(x)
        x = self.up64(x)
        x = self.up128(x)
        x = self.shared_refine(x)

        rgb = torch.sigmoid(self.rgb_branch(x))

        goal_pred   = self.goal_head(flat)
        vector_pred = self.vector_head(flat)

        return rgb, goal_pred, vector_pred