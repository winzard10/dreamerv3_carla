import torch
import torch.nn as nn

class MultiModalDecoder(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
        super().__init__()
        in_dim = deter_dim + stoch_dim

        self.fc = nn.Linear(in_dim, 256 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),  # 8 -> 16
            nn.Conv2d(128, 128, 3, 1, 1), nn.ELU(),

            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ELU(),  # 16 -> 32
            nn.Conv2d(64, 64, 3, 1, 1), nn.ELU(),

            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ELU(),  # 32 -> 64
            nn.Conv2d(32, 32, 3, 1, 1), nn.ELU(),

            nn.ConvTranspose2d( 32,  32, 4, 2, 1), nn.ELU(),  # 64 -> 128
            nn.Conv2d(32, 32, 3, 1, 1), nn.ELU(),
        )
        self.depth_head = nn.Conv2d(32, 1, 3, padding=1)
        self.segm_head  = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, deter, stoch, out_hw=(128, 128)):
        B = deter.shape[0]
        x = torch.cat([deter, stoch], dim=-1)
        x = self.fc(x).view(B, 256, 8, 8)
        feat = self.deconv(x)                  # [B,32,128,128]

        depth = torch.sigmoid(self.depth_head(feat))
        segm_logits = self.segm_head(feat)
        return depth, segm_logits

# import torch
# import torch.nn as nn

# class MultiModalDecoder(nn.Module):
    
#     K:int = 4   # Conv kernel size (default: 4)
    
#     def __init__(self, deter_dim=512, stoch_dim=1024, num_classes=28):
#         super().__init__()
#         in_dim = deter_dim + stoch_dim

#         self.fc = nn.Linear(in_dim, 256 * 8 * 8)

#         K = self.K

#         # # Initial Implementation
#         # # Decoder truck
#         # self.deconv = nn.Sequential(
#         #     nn.ConvTranspose2d(256, 128, K, 2, 1), nn.ELU(),
#         #     nn.ConvTranspose2d(128,  64, K, 2, 1), nn.ELU(),
#         #     nn.ConvTranspose2d( 64,  32, K, 2, 1), nn.ELU(),
#         #     nn.ConvTranspose2d( 32,  32, K, 2, 1), nn.ELU(),
#         # )
#         # # Depth head
#         # self.depth_head = nn.Conv2d(32, 1, 3, padding=1)  # logits -> sigmoid -> depth in [0,1]
#         # # Semantic Segmentation head
#         # self.segm_head = nn.Conv2d(32, num_classes, 3, padding=1)

#         # Deeper Implementation
#         # Decoder Trunk
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, K, 2, 1), nn.ELU(),
#             nn.Conv2d(128, 128, K, 1), nn.ELU(),
#             nn.ConvTranspose2d(128,  64, K, 2, 1), nn.ELU(),
#             nn.Conv2d(64, 64, K, 1), nn.ELU(),
#             nn.ConvTranspose2d( 64,  32, K, 2, 1), nn.ELU(),
#             nn.Conv2d(32, 32, K, 1), nn.ELU(),
#             nn.ConvTranspose2d( 32,  17, K, 2, 1), nn.ELU(),
#             nn.Conv2d(17, 17, K, 1), nn.ELU(),
#         )
#         # Depth head
#         self.depth_head = nn.Conv2d(17, 1, 3, padding=1)  # logits -> sigmoid -> depth in [0,1]
#         # Semantic Segmentation head
#         self.segm_head = nn.Conv2d(17, num_classes, 3, padding=1)


#     def forward(self, deter, stoch, out_hw=(128, 128)):
#         B = deter.shape[0]
#         x = torch.cat([deter, stoch], dim=-1)
#         x = self.fc(x).view(B, 256, 8, 8)
#         feat = self.deconv(x)

#         depth = torch.sigmoid(self.depth_head(feat))
#         segm_logits = self.segm_head(feat)
#         return depth, segm_logits