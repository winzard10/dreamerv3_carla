import torch
import torch.nn as nn

class MultiModalDecoder(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
        super().__init__()
        in_dim = deter_dim + stoch_dim

        self.fc = nn.Linear(in_dim, 256 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),  # 8 -> 16
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ELU(),  # 16 -> 32
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ELU(),  # 32 -> 64
            nn.ConvTranspose2d( 32,  32, 4, 2, 1), nn.ELU(),  # 64 -> 128
        )

        self.depth_head = nn.Conv2d(32, 1, 3, padding=1)
        self.segm_head  = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, deter, stoch, out_hw=(128, 128)):
        B = deter.shape[0]
        x = torch.cat([deter, stoch], dim=-1)
        x = self.fc(x).view(B, 256, 8, 8)
        feat = self.deconv(x)

        depth = torch.sigmoid(self.depth_head(feat))
        segm_logits = self.segm_head(feat)
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
#             nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ELU(),  # 16 -> 32
#             nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ELU(),  # 32 -> 64
#             nn.ConvTranspose2d( 32,  32, 4, 2, 1), nn.ELU(),  # 64 -> 128
#         )

#         # Slightly stronger depth branch
#         self.depth_branch = nn.Sequential(
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(32, 16, 3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(16, 1, 3, padding=1),
#         )

#         # Slightly stronger semantic branch
#         self.segm_branch = nn.Sequential(
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(32, num_classes, 3, padding=1),
#         )

#     def forward(self, deter, stoch, out_hw=(128, 128)):
#         B = deter.shape[0]
#         x = torch.cat([deter, stoch], dim=-1)
#         x = self.fc(x).view(B, 256, 8, 8)
#         feat = self.deconv(x)  # [B,32,128,128]

#         depth_logits = self.depth_branch(feat)
#         segm_logits = self.segm_branch(feat)

#         depth = torch.sigmoid(depth_logits)
#         return depth, segm_logits