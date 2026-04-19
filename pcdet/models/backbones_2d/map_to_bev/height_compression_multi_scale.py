import torch.nn as nn


class HeightCompressionMultiScale(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.in_channels = self.model_cfg.IN_CHANNELS
        self.out_channels = self.model_cfg.OUT_CHANNELS
        self.stage = len(self.in_channels)
        self.down_conv = nn.ModuleList()
        for in_num, out_num in zip(self.in_channels, self.out_channels):
            self.down_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_num, out_num, 1),
                    nn.BatchNorm2d(out_num),
                    nn.ReLU()
                )
            )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        spatial_features = []
        spatial_out = []
        spatial_features.append(batch_dict['multi_scale_3d_features']['x_conv3'].dense())
        spatial_features.append(batch_dict['multi_scale_3d_features']['x_conv4'].dense())
        spatial_features.append(batch_dict['encoded_spconv_tensor'].dense())
        for idx, feat in enumerate(spatial_features):
            N, C, D, H, W = feat.shape
            feat = feat.view(N, C * D, H, W)
            spatial_out.append(
                self.down_conv[idx](feat)
            )
        batch_dict['spatial_features'] = spatial_out
        return batch_dict
