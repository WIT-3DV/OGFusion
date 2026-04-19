import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from . import ddn, ddn_loss
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class DepthFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor, use_lidar_depth, use_pooling=False, use_depth=True):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.use_lidar_depth = use_lidar_depth
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor
        self.use_pooling = use_pooling
        self.use_depth = use_depth
        self.feature_extract_layers = self.model_cfg.DDN.ARGS.get('feat_extract_layer', None)
        self.use_multi_scale_features = isinstance(self.feature_extract_layers, list)

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            use_lidar_depth=use_lidar_depth,
            use_pooling=use_pooling,
            use_depth=use_depth,
            **model_cfg.DDN.ARGS
        )
        if self.use_multi_scale_features:
            assert isinstance(model_cfg.CHANNEL_REDUCE['in_channels'], list)
            self.channel_reduce = nn.ModuleList()
            for in_channels in model_cfg.CHANNEL_REDUCE['in_channels']:
                self.channel_reduce.append(
                    BasicBlock2D(in_channels=in_channels,
                                 out_channels=model_cfg.CHANNEL_REDUCE['out_channels'],
                                 kernel_size=model_cfg.CHANNEL_REDUCE['kernel_size'],
                                 stride=model_cfg.CHANNEL_REDUCE['stride'],
                                 bias=model_cfg.CHANNEL_REDUCE['bias'])
                )
        else:
            self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
            disc_cfg=self.disc_cfg,
            downsample_factor=downsample_factor,
            **model_cfg.LOSS.ARGS
        )
        self.forward_ret_dict = {}
        self.use_foreground = self.model_cfg.get('FOREGROUND_MASK', False)

    def get_output_feature_dim(self):
        if self.use_multi_scale_features:
            return sum([x.out_channels for x in self.channel_reduce])
        else:
            return self.channel_reduce.out_channels
    
    def pseudocost_from_mono(self, monodepth): # monodepth[2,1,129,484]
        B, H, W = monodepth.shape
        monodepth = monodepth.reshape([B, 1, H, W])
        mode = self.disc_cfg["mode"]
        depth_max = self.disc_cfg["depth_max"]
        depth_min = self.disc_cfg["depth_min"]
        num_bins = self.disc_cfg["num_bins"]

        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((monodepth - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (monodepth - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + monodepth) - math.log(1 + depth_min)) / \
                (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError
        
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins
        indices = indices.type(torch.int64)

        pseudo_cost = monodepth.new_zeros([B, num_bins + 1, H, W]) # [2,81,129,484]
        ones = monodepth.new_ones([B, num_bins + 1, H, W])
        
        pseudo_cost.scatter_(dim = 1, index = indices, src = ones*10) # [2,81,129,484]
        
        return pseudo_cost

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"] # [2,3,516,1936]
        ddn_result = self.ddn(images)
        if self.use_multi_scale_features:
            image_features = []
            for feature_name in self.feature_extract_layers:
                image_features.append(ddn_result[feature_name])
        else:
            image_features = ddn_result["features"]  # [2,256,129,484]

        if self.use_lidar_depth:
            depth_logits = self.pseudocost_from_mono(batch_dict['depth_maps'])
        elif self.use_depth:
            depth_logits = ddn_result["logits"] # [2,81,129,484]

        # Channel reduce
        if self.channel_reduce is not None:
            if self.use_multi_scale_features:
                for i in range(len(self.feature_extract_layers)):
                    image_features[i] = self.channel_reduce[i](image_features[i])
            else:
                image_features = self.channel_reduce(image_features) # [2,80,129,484]

        if self.use_foreground:
            mask = batch_dict['foreground']
            target_shapes = [x.shape for x in image_features]
            for idx, target_shape in enumerate(target_shapes):
                image_features[idx] = image_features[idx] * F.interpolate(mask.unsqueeze(1), [*target_shape[-2:]])

        if self.use_pooling:
            batch_dict["features"] = image_features
            batch_dict["logits"] = ddn_result["logits"]
        elif self.use_depth:
            if self.use_multi_scale_features:
                frustum_features = []
                for feature in image_features:
                    if feature.shape[-2:] == depth_logits.shape[-2:]:
                        frustum_features.append(
                            self.create_frustum_features(image_features=feature,
                                                        depth_logits=depth_logits)
                        )
                    else:
                        frustum_features.append(
                            self.create_frustum_features(image_features=feature,
                                                        depth_logits=F.interpolate(depth_logits, feature.shape[-2:], mode='bilinear'))
                        )
                batch_dict["frustum_features"] = frustum_features
            else:
                # Create image feature plane-sweep volume
                frustum_features = self.create_frustum_features(image_features=image_features,
                                                                depth_logits=depth_logits) # [2,64,80,129,484]
                batch_dict["frustum_features"] = frustum_features
        else:
            batch_dict["features"] = image_features
            
        if self.training:
            if self.use_depth:
                self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
                self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
                self.forward_ret_dict["depth_logits"] = depth_logits
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict
