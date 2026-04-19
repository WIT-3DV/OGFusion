from collections import OrderedDict
from pathlib import Path
from torch import hub

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class DDNResNet50(nn.Module):

    def __init__(self, feat_extract_layer, depth, num_stages, out_indices, frozen_stages, norm_cfg, norm_eval, style,
                 dcn, stage_with_dcn,
                 in_channels, out_channels, start_level, add_extra_convs, num_outs, relu_before_extra_convs,
                  pretrained_path=None, aux_loss=None, use_lidar_depth=False,
                  use_pooling=False, use_depth=True, **kwargs):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.feat_extract_layer = feat_extract_layer
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        self.use_lidar_depth = use_lidar_depth
        self.use_pooling = use_pooling
        self.use_depth = use_depth

        self.freeze_backbone = kwargs.get('freeze_backbone', False)
        self.freeze_fpn = kwargs.get('freeze_fpn', False)
        self.pretrained_backbone = kwargs.get('pretrained_backbone', True)
        self.pretrained_fpn = kwargs.get('pretrained_fpn', False)

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # Model
        self.model = ResNet(
            depth=depth,
            num_stages=num_stages,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            style=style,
            dcn=dict(dcn),
            stage_with_dcn=stage_with_dcn,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
        )

        self.FPN = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=start_level,
            add_extra_convs=add_extra_convs,
            num_outs=num_outs,
            relu_before_extra_convs=relu_before_extra_convs
        )

        if self.pretrained_backbone or self.pretrained_fpn:
            assert self.pretrained_path is not None
            pretrained_dict = torch.load(self.pretrained_path)['state_dict']
            pretrained_dict_filtered_backbone = dict()
            pretrained_dict_filtered_neck = dict()

            for key in pretrained_dict.keys():
                if key.find('backbone') != -1:
                    pretrained_dict_filtered_backbone[key.replace('backbone.', '')] = pretrained_dict[key]
                if key.find('neck') != -1:
                    pretrained_dict_filtered_neck[key.replace('neck.', '')] = pretrained_dict[key]

            if self.pretrained_backbone:
                model_dict_backbone = self.model.state_dict()
                pretrained_dict_filtered_backbone = self.filter_state_dict(pretrained_dict_filtered_backbone, model_dict_backbone)
                model_dict_backbone.update(pretrained_dict_filtered_backbone)
                self.model.load_state_dict(model_dict_backbone)

            if self.pretrained_fpn:
                model_dict_neck = self.FPN.state_dict()
                pretrained_dict_filtered_neck = self.filter_state_dict(pretrained_dict_filtered_neck, model_dict_neck)
                model_dict_neck.update(pretrained_dict_filtered_neck)
                self.FPN.load_state_dict(model_dict_neck)

        if self.freeze_backbone:
            self.freeze_model_func(self.model)
        else:
            self.un_freeze_model_func(self.model)

        if self.freeze_fpn:
            self.freeze_model_func(self.FPN)
        else:
            self.un_freeze_model_func(self.FPN)

    def freeze_model_func(self, model):
        for _, para in model.named_parameters():
            para.requires_grad_(False)

    def un_freeze_model_func(self, model):
        for _, para in model.named_parameters():
            para.requires_grad_(True)

    def filter_state_dict(self, pretrained_state_dict, model_state_dict):
        filtered_pretrained_state_dict = dict()
        for key in pretrained_state_dict:
            if key in model_state_dict.keys():
                filtered_pretrained_state_dict[key] = pretrained_state_dict[key]
        return filtered_pretrained_state_dict
                

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        features = self.model(x) # [B, 256, 129, 484] [B, 512, 65, 242] [B, 1024, 33, 121]
        features_fpn = self.FPN(features) # [B, 256, 65, 242] [B, 256, 33, 121] [B, 256, 17, 61]

        if isinstance(self.feat_extract_layer, list):
            for index, feature_name in enumerate(self.feat_extract_layer): 
                result[feature_name] = features_fpn[index]
        else:
            raise NotImplementedError

        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            # Create a mask for padded pixels
            mask = (x == 0)

            # Match ResNet pretrained preprocessing
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            x[mask] = 0

        return x
