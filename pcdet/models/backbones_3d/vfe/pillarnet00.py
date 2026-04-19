import spconv.utils
import torch
import torch.nn as nn
from spconv.pytorch import functional as fsp
from pcdet.models.backbones_3d.vfe.pillarnet_modules.dynamic_pillar_encoder import DynamicPillarFeatureNet
from pcdet.models.backbones_3d.vfe.pillarnet_modules.pcnres18 import SpMiddlePillarEncoder18
from pcdet.models.backbones_3d.vfe.pillarnet_modules.rpn import RPNV2
from pcdet.models.backbones_3d.vfe.pillarnet_modules.lss_fpn import FPN_LSS


class PillarNet(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.module_name_list = ['reader', 'backbone', 'neck']
        self.virtual = model_cfg.READER.get('USE_VIRTUAL_POINT', False)
        self.output_spatial_feature = self.model_cfg.get('OUTPUT_SPATIAL_FEATURES', False)
        for module_name in self.module_name_list:
            getattr(self, 'build_%s' % module_name)(
                model_cfg=model_cfg
            )

    def build_reader(self, model_cfg):
        self.reader = DynamicPillarFeatureNet(
            num_input_features=model_cfg.READER.NUM_INPUT_FEATURES,
            num_filters=model_cfg.READER.NUM_FILTERS,
            pillar_size=model_cfg.READER.PILLAR_SIZE,
            pc_range=model_cfg.READER.PC_RANGE,
            virtual=self.virtual,
            encoding_type=self.model_cfg.READER.get('ENCODING_TYPE', 'split'),
            dataset=self.model_cfg.READER.get('DATASET', 'vod')
        )

    def build_backbone(self, model_cfg):
        self.backbone = SpMiddlePillarEncoder18(
            in_planes=model_cfg.BACKBONE.IN_PLANES,
            ds_factor=model_cfg.BACKBONE.DS_FACTOR,
            out_indices=model_cfg.BACKBONE.get('OUT_INDICES', [1, 2, 3])
        )

    def build_neck(self, model_cfg):
        self.neck = FPN_LSS(
            in_channels=model_cfg.NECK.IN_CHANNELS,
            out_channels=model_cfg.NECK.OUT_CHANNELS,
            scale_factor=model_cfg.NECK.SCALE_FACTOR,
            extra_upsample=None,
        )

    def get_output_feature_dim(self):
        return 64

    def forward(self, batch_dict):
        batch_size = batch_dict['points'][:, 0].max().item()
        points = []
        for i in range(int(batch_size) + 1):
            mask = batch_dict['points'][:, 0] == i
            points.append(batch_dict['points'][mask][:, 1:])

        input_dict = dict(points=points)
        sp_tensor = self.reader(input_dict)
        features_backbone = self.backbone(sp_tensor)
        if self.output_spatial_feature:
            batch_dict['spatial_features_2d'] = self.neck(features_backbone)
        else:
            batch_dict['pillar_features_scattered'] = self.neck(features_backbone)  # [2, 128, 160, 160]
        return batch_dict