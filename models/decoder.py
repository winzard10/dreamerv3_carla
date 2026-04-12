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


class SkipFuse(nn.Module):
    def __init__(self, dec_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = ConvRefine(dec_ch + skip_ch, out_ch)

    def forward(self, dec_feat, skip_feat):
        x = torch.cat([dec_feat, skip_feat], dim=1)
        return self.fuse(x)


class MultiModalDecoder(nn.Module):
    """
    Inputs:
      deter: [B, deter_dim]
      stoch: [B, stoch_dim]
      skips: dict with keys skip16, skip32, skip64
    Outputs:
      depth: [B, 1, 128, 128]
      segm_logits: [B, num_classes, 128, 128]
    """

    def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
        super().__init__()
        in_dim = deter_dim + stoch_dim

        # Start from 16x16 instead of 8x8
        self.fc = nn.Linear(in_dim, 128 * 16 * 16)

        # Initial latent refinement at 16x16
        self.init_refine = ConvRefine(128, 128)

        # Fuse with encoder skips
        self.fuse16 = SkipFuse(dec_ch=128, skip_ch=128, out_ch=128)

        self.up32 = UpBlock(128, 96)                 # 16 -> 32
        self.fuse32 = SkipFuse(dec_ch=96, skip_ch=64, out_ch=96)

        self.up64 = UpBlock(96, 64)                  # 32 -> 64
        self.fuse64 = SkipFuse(dec_ch=64, skip_ch=32, out_ch=64)

        self.up128 = UpBlock(64, 48)                 # 64 -> 128
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

    def forward(self, deter, stoch, skips):
        x = torch.cat([deter, stoch], dim=-1)
        bsz = x.shape[0]

        x = self.fc(x).view(bsz, 128, 16, 16)
        x = self.init_refine(x)

        # Fuse 16x16 skip
        x = self.fuse16(x, skips["skip16"])

        # 16 -> 32, fuse skip
        x = self.up32(x)
        x = self.fuse32(x, skips["skip32"])

        # 32 -> 64, fuse skip
        x = self.up64(x)
        x = self.fuse64(x, skips["skip64"])

        # 64 -> 128
        x = self.up128(x)
        x = self.shared_refine(x)

        depth = torch.sigmoid(self.depth_branch(x))
        segm_logits = self.segm_branch(x)

        return depth, segm_logits

# import torch
# import torch.nn as nn

# class MultiModalDecoder(nn.Module):
#     def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
#         super().__init__()
#         in_dim = deter_dim + stoch_dim

#         self.fc = nn.Linear(in_dim, 256 * 8 * 8)

#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),  # 8 -> 16
#             nn.Conv2d(128, 128, 3, 1, 1), nn.ELU(),

#             nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ELU(),  # 16 -> 32
#             nn.Conv2d(64, 64, 3, 1, 1), nn.ELU(),

#             nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ELU(),  # 32 -> 64
#             nn.Conv2d(32, 32, 3, 1, 1), nn.ELU(),

#             nn.ConvTranspose2d( 32,  32, 4, 2, 1), nn.ELU(),  # 64 -> 128
#             nn.Conv2d(32, 32, 3, 1, 1), nn.ELU(),
#         )

#         self.depth_head = nn.Conv2d(32, 1, 3, padding=1)
#         self.segm_head  = nn.Conv2d(32, num_classes, 3, padding=1)

#     def forward(self, deter, stoch, out_hw=(128, 128)):
#         B = deter.shape[0]
#         x = torch.cat([deter, stoch], dim=-1)
#         x = self.fc(x).view(B, 256, 8, 8)
#         feat = self.deconv(x)                  # [B,32,128,128]

#         depth = torch.sigmoid(self.depth_head(feat))
#         segm_logits = self.segm_head(feat)
#         return depth, segm_logits