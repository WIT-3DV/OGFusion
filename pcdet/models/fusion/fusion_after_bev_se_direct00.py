import torch
import torch.nn as nn
from torch.nn import functional as F


class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


# 类似BEVFusion simple 中的融合模块 Sequeeze and Excitation
class FusionAfterBEVSEDirect(nn.Module):
    def __init__(self, model_cfg, num_bev_features,
                 image_in_channels,
                 image_out_channels,
                 radar_in_channels,
                 radar_out_channels,
                 **kwargs):
        super().__init__()

        if isinstance(image_in_channels, list):
            image_in_channels = sum(image_in_channels)

        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features
        # self.img_conv = nn.Sequential(
        #     nn.Conv2d(image_in_channels, image_out_channels, [1, 1]),
        #     nn.BatchNorm2d(image_out_channels),
        #     nn.ReLU()
        # )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                      image_out_channels + radar_out_channels,
                      [3, 3], padding=1),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU()
        )
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')

    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"]  # [B, 128, 320, 320]
        radar_features = batch_dict['pillar_features_scattered']  # [B, 128, 160, 160]

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features, radar_features.shape[-2:],
                                           mode='bilinear')  # [B, 128, 160, 160]

        fuse_features = torch.concat([image_features, radar_features], dim=1)  # [B, 256, 160, 160]
        fuse_features = self.fuse_conv(fuse_features)  # [B, 256, 160, 160]

        fuse_features = self.se_block(fuse_features)  # [B, 256, 160, 160]

        batch_dict[self.feature_name] = fuse_features
        return batch_dict