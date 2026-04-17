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


class MultiModalDecoder(nn.Module):

    """
    No-skip version:
      deter: [B, deter_dim]
      stoch: [B, stoch_dim]

    Outputs:
      depth: [B, 1, 128, 128]
      segm_logits: [B, num_classes, 128, 128]
    """

    def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
        super().__init__()
        in_dim = deter_dim + stoch_dim

        # Start from 16x16 instead of 8x8
        self.fc = nn.Linear(in_dim, 128 * 16 * 16)

        # Shared decoding trunk
        self.init_refine = ConvRefine(128, 128)  # 16x16
        self.up32 = UpBlock(128, 96)             # 16 -> 32
        self.up64 = UpBlock(96, 64)              # 32 -> 64
        self.up128 = UpBlock(64, 48)             # 64 -> 128
        self.shared_refine = ConvRefine(48, 48)

        # Depth branch
        self.depth_branch = nn.Sequential(
            ConvRefine(48, 32),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

        # Segmentation branch
        self.segm_branch = nn.Sequential(
            ConvRefine(48, 32),
            nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1),
        )
        
        # Low-dim reconstruction heads
        self.goal_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2),   # goal is [speed_target, angle] or whatever your 2D goal is
        )

        self.vector_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 3),   # velocity vector
        )

    def forward(self, deter, stoch):
        flat = torch.cat([deter, stoch], dim=-1)  # [B, deter_dim + stoch_dim] — save this
        bsz = flat.shape[0]

        # Conv path uses flat → spatial
        x = self.fc(flat).view(bsz, 128, 16, 16)
        x = self.init_refine(x)
        x = self.up32(x)
        x = self.up64(x)
        x = self.up128(x)
        x = self.shared_refine(x)

        depth       = torch.sigmoid(self.depth_branch(x))
        segm_logits = self.segm_branch(x)

        # Low-dim heads use the original flat latent — not the spatial x
        goal_pred   = self.goal_head(flat)    # ← flat, not x
        vector_pred = self.vector_head(flat)  # ← flat, not x

        return depth, segm_logits, goal_pred, vector_pred