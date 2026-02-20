import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalDecoder(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
        super().__init__()
        in_dim = deter_dim + stoch_dim

        self.fc = nn.Linear(in_dim, 256 * 8 * 8)

        # 8 -> 16 -> 32 -> 64 -> 128
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16
            nn.ELU(),
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),  # 32
            nn.ELU(),
            nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1),  # 64
            nn.ELU(),
            nn.ConvTranspose2d( 32,  32, kernel_size=4, stride=2, padding=1),  # 128
            nn.ELU(),
        )

        # Heads at 128x128
        self.depth_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.segm_head  = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, deter, stoch, out_hw=(160, 160)):
        x = torch.cat([deter, stoch], dim=-1)                # [B, D+Z]
        x = self.fc(x).view(-1, 256, 8, 8)                   # [B,256,8,8]
        feat = self.deconv(x)                                # [B,32,128,128]

        depth = self.depth_head(feat)                        # [B,1,128,128]
        segm_logits = self.segm_head(feat)                   # [B,C,128,128]

        # resize to 160x160
        depth = F.interpolate(depth, size=out_hw, mode="bilinear", align_corners=False)
        segm_logits = F.interpolate(segm_logits, size=out_hw, mode="nearest")

        # If your depth target is already /255 and in [0,1], keep sigmoid.
        depth = torch.sigmoid(depth)

        return depth, segm_logits
