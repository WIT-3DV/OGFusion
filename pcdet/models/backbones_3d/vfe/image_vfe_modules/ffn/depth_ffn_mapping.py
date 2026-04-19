import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from . import ddn, ddn_loss
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D

# this file is discarded

def normalize(x: torch.Tensor, x_min, x_max) -> torch.Tensor:
    new_x = (x - x.min()) / (x.max() - x.min())
    new_x *= (x_max - x_min)
    new_x += x_min
    return new_x


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class DepthFFN_mapping(nn.Module):

    def __init__(self, model_cfg, downsample_factor, use_lidar_depth, use_pooling=False):
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

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            use_lidar_depth=use_lidar_depth,
            use_pooling=use_pooling,
            **model_cfg.DDN.ARGS
        )
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
            disc_cfg=self.disc_cfg,
            downsample_factor=downsample_factor,
            **model_cfg.LOSS.ARGS
        )
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
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
    
    def create_radar_mask(self, batch_dict):
        # get radar points
        points = batch_dict['points'][:, :5] # batch_num, xyz, rcs
        points_single_batch = points[points[:, 0] == 0, 1:4] # points in radar xyz
        rcs_single_batch = points[points[:, 0] == 0, 4] # points in radar rcs

        # convert to cam coor and project to image
        points_homo = torch.ones([points_single_batch.shape[0], 4]).to(points.device)
        points_homo[:, :3] = points_single_batch[...]
        inv_aug = torch.linalg.inv(batch_dict['lidar_aug_matrix'][0])
        points_homo = torch.matmul(inv_aug, points_homo.T).T
        
        radar2cam = batch_dict['trans_lidar_to_cam'][0] # since all the trans are the same
        cam2img = torch.eye(4).to(radar2cam.device)
        cam2img[:3, :] = batch_dict['trans_cam_to_img'][0]

        points_homo_cam = torch.matmul(radar2cam, points_homo.T).T # [N, 4]
        points_homo_img = torch.matmul(cam2img, points_homo_cam.T).T
        depth = torch.ones([points_homo_img.shape[0], 1]).to(points.device)
        depth[:, 0] = points_homo_img[:, 2]
        points_homo_img[:, :3] /= depth
        points_homo_img = torch.round(points_homo_img).int()

        # filter points outside image
        B, _, H, W = batch_dict["images"].shape
        valid = points_homo_img[:, 0] > 0
        valid &= points_homo_img[:, 0] < W
        valid &= points_homo_img[:, 1] > 0
        valid &= points_homo_img[:, 1] < H
        points_homo_img = points_homo_img[valid]
        depth = depth[valid]
        rcs_single_batch = rcs_single_batch[valid]

        rcs_single_batch = normalize(rcs_single_batch, 1, 5)
        depth = normalize(-depth, 1, 5)

        # create mask
        heatmap = torch.zeros([B, H, W]).to(batch_dict["images"].device)
        for index, (xy, rcs, d) in enumerate(zip(points_homo_img, rcs_single_batch, depth)):
            heatmap[0, ...] = draw_heatmap_gaussian(heatmap[0, ...], xy, int(rcs))
            heatmap[1, ...] = draw_heatmap_gaussian(heatmap[1, ...], xy, int(d))
            

        fig = plt.figure(figsize=(64, 16))
        fig.set_dpi(600)
        plt.clf()

        plt.subplot(411)
        plt.imshow(batch_dict["images"][0].permute([1, 2, 0]).cpu())

        plt.subplot(412)
        plt.scatter(points_homo_img[:, 0].cpu(), points_homo_img[:, 1].cpu(), c=-depth.cpu(), s=10)
        plt.imshow(batch_dict["images"][0].permute([1, 2, 0]).cpu())
        plt.axis('off')

        plt.subplot(413)
        plt.imshow(heatmap[0].cpu())

        plt.subplot(414)
        plt.imshow(heatmap[1].cpu())

        plt.savefig(batch_dict['frame_id'][0] + '.png', bbox_inches='tight', transparent=True)
        plt.close(fig)

        pass

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
        self.create_radar_mask(batch_dict)
        # Pixel-wise depth classification
        images = batch_dict["images"] # [2,3,516,1936]
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]  # [2,256,129,484]

        if self.use_lidar_depth:
            depth_logits = self.pseudocost_from_mono(batch_dict['depth_maps'])
        else:
            depth_logits = ddn_result["logits"] # [2,81,129,484]

        # Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features) # [2,80,129,484]

        if self.use_pooling:
            batch_dict["features"] = image_features
            batch_dict["logits"] = ddn_result["logits"]
        else:
        # Create image feature plane-sweep volume
            frustum_features = self.create_frustum_features(image_features=image_features,
                                                            depth_logits=depth_logits) # [2,64,80,129,484]
            batch_dict["frustum_features"] = frustum_features

        if self.training:
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
