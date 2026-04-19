from .ddn_template import DDNTemplate

try:
    import torchvision
except:
    pass
from collections import OrderedDict
from pathlib import Path
from torch import hub
# from yoloxpafpn import YOLOXPAFPN
from pcdet.models.backbones_3d.vfe.image_vfe_modules.ffn.ddn.yoloxpafpn import YOLOXPAFPN
from mmdet.models import CSPDarknet

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')
    
class DDN_YOLOX(nn.Module):

    def __init__(self, backbone_name, num_classes,
                 deepen_factor, widen_factor, out_indices, use_depthwise, 
                 spp_kernal_sizes, norm_cfg, act_cfg,
                 in_channels, out_channels, num_csp_blocks, use_depthwise_PAN, 
                 upsample_cfg, norm_cfg_PAN, act_cfg_PAN,
                 pretrained_path=None, aux_loss=None, **kwargs):
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
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        # self.pretrained_path_deeplabv3 = pretrained_path_deeplabv3
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        self.freeze_backbone = kwargs.get('freeze_backbone', False)
        self.freeze_pan = kwargs.get('freeze_pan', False)
        self.pretrained_backbone = kwargs.get('pretrained_backbone', True)
        self.pretrained_pan = kwargs.get('pretrained_pan', False)

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        if self.pretrained_backbone or self.pretrained_pan:
            pretrained_dict = torch.load(self.pretrained_path)['state_dict']

            pretrained_dict_filtered_backbone = dict()
            pretrained_dict_filtered_neck = dict()

            for key in pretrained_dict.keys():
                if key.find('backbone') != -1:
                    pretrained_dict_filtered_backbone[key.replace('backbone.', '')] = pretrained_dict[key]
                if key.find('neck') != -1:
                    pretrained_dict_filtered_neck[key.replace('neck.', '')] = pretrained_dict[key]

        self.model = CSPDarknet(deepen_factor=deepen_factor, 
                                widen_factor=widen_factor,
                                out_indices=out_indices,
                                use_depthwise=use_depthwise,
                                spp_kernal_sizes=spp_kernal_sizes,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        if self.pretrained_backbone:
            model_dict_backbone = self.model.state_dict()
            model_dict_backbone.update(pretrained_dict_filtered_backbone)
            self.model.load_state_dict(model_dict_backbone)
        if self.freeze_backbone:
            self.freeze_model_func(self.model)

        if out_indices == [1, 2, 3]:
            feature_map_size=[[65, 242],
                             [129, 484]]
        elif out_indices == [2, 3, 4]:
            feature_map_size=[[33, 121],
                            [65, 242]]
        self.PAN = YOLOXPAFPN(in_channels=in_channels,
                              out_channels=out_channels,
                              num_csp_blocks=num_csp_blocks,
                              use_depthwise=use_depthwise_PAN,
                              upsample_cfg=upsample_cfg,
                              norm_cfg=norm_cfg_PAN,
                              act_cfg=act_cfg_PAN,
                              feature_map_size=feature_map_size)
        if self.pretrained_pan:
            model_dict_neck = self.PAN.state_dict()
            pretrained_dict_filtered_neck = self.filter_state_dict(pretrained_dict_filtered_neck, model_dict_neck)
            model_dict_neck.update(pretrained_dict_filtered_neck)
            self.PAN.load_state_dict(model_dict_neck)
        if self.freeze_pan:
            self.freeze_model_func(self.PAN)

        self.image_conv = \
            nn.Conv2d(out_channels, 256, kernel_size=1)
        self.depth_conv = \
            nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def freeze_model_func(self, model):
        for _, para in model.named_parameters():
            para.requires_grad_(False)

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
        x = self.preprocess(images) # [4,3,516,1936]

        # Extract features
        result = OrderedDict()
        features = self.model(x) # [[4,64,129,484],[4,128,65,242],[4,256,33,121]]
        features = self.PAN(features) # [4,128,129,484]

        result['image_view_features'] = features #TODO: 判断是否用pooling

        result['features'] = self.image_conv(features) # [1, 256,129,484]
        result["logits"] = self.depth_conv(features) # [1, 81,129,484]

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
