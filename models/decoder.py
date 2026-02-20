import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalDecoder(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
        super().__init__()
        in_dim = deter_dim + stoch_dim

        self.fc = nn.Linear(in_dim, 256 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d( 32,  32, 4, 2, 1), nn.ELU(),
        )

        self.depth_head = nn.Conv2d(32, 1, 3, padding=1)  # logits -> sigmoid -> depth in [0,1]
        self.segm_head  = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, deter, stoch, out_hw=(160, 160)):
        B = deter.shape[0]
        x = torch.cat([deter, stoch], dim=-1)        # [B, D+Z]
        x = self.fc(x).view(B, 256, 8, 8)
        feat = self.deconv(x)                        # [B,32,128,128]

        depth_mean = self.depth_head(feat)           # [B,1,128,128] (raw mean)
        segm_logits = self.segm_head(feat)           # [B,C,128,128]

        depth_mean  = F.interpolate(depth_mean,  size=out_hw, mode="bilinear", align_corners=False)
        segm_logits = F.interpolate(segm_logits, size=out_hw, mode="nearest")
        
        depth = torch.sigmoid(depth_mean)
        return depth, segm_logits
