import numpy as np
import torch
import torch.nn as nn


from .resnet_modules.custom_resnet import CustomResNet
from .resnet_modules.lss_fpn import FPN_LSS

class BEVResNet(nn.Module):
    def __init__(self, model_cfg, **kwargs) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = CustomResNet(numC_input=self.model_cfg.ResNet.numC_input,
                                     num_channels=self.model_cfg.ResNet.num_channels,
                                     backbone_output_ids=self.model_cfg.ResNet.backbone_output_ids)
        self.fpn = FPN_LSS(in_channels=self.model_cfg.FPN.in_channels,
                           out_channels=self.model_cfg.FPN.out_channels,
                           input_feature_index=self.model_cfg.FPN.input_feature_index)
        self.num_bev_features = self.model_cfg.FPN.out_channels


    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features'] # [B, 128, 320, 320]

        features = self.backbone(spatial_features) # [B, 128, 160, 160] [B, 512, 40, 40]
        spatial_features_2d = self.fpn(features) # [B, 128, 320, 320]

        data_dict['spatial_features_2d'] = spatial_features_2d
        return data_dict