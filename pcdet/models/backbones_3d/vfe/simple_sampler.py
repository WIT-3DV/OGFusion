import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

class SimpleSampler(nn.Module):
    def __init__(self, model_cfg, 
                 point_cloud_range, 
                 voxel_size,
                 use_virtual) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pcr = point_cloud_range # point cloud range
        self.voxel_size = torch.Tensor(voxel_size)
        self.use_virtual = use_virtual

        self.voxel_space_range = torch.Tensor([self.pcr[3] - self.pcr[0],
                                                self.pcr[4] - self.pcr[1],
                                                self.pcr[5] - self.pcr[2]]).reshape([1, 3])
        self.grid_size = torch.round(self.voxel_space_range / self.voxel_size).int().squeeze().numpy()
        self.lower_bound = torch.Tensor([self.pcr[0],
                                         self.pcr[1],
                                         self.pcr[2]]).reshape([1, 3])
        self.upper_bound = torch.Tensor([self.pcr[3],
                                         self.pcr[4],
                                         self.pcr[5]]).reshape([1, 3])

        self.sampler = partial(F.grid_sample, mode=self.model_cfg.get('MODE', 'BILINEAR'),
                                              padding_mode=self.model_cfg.get('PADDING_MODE', 'ZEROS'),
                                              align_corners=True)
        self.fuse_mode = self.model_cfg.get('FUSE_MODE', 'ADD')
        if self.fuse_mode == 'SE':
            self.fuse = SE_Block(self.get_output_feature_dim() * 2)
            self.fuse_squeeze = nn.Sequential(
                nn.Conv3d(self.get_output_feature_dim() * 2, self.get_output_feature_dim(), 1),
                nn.BatchNorm3d(self.get_output_feature_dim()),
                nn.ReLU()
            )

    def get_output_feature_dim(self):
        return self.model_cfg.OUTPUT_FEATURE_NUM
    
        
    def forward(self, batch_dict):
        # image voxel features  (B, C, Z, Y, X)  [B, 64, 31, 320, 320]
        image_voxel_features = batch_dict['voxel_features']
        B, C, Z, Y, X = image_voxel_features.shape
        assert C == self.get_output_feature_dim()
        cur_device = image_voxel_features.device

        # radar points [N, 8] [batch_index, xyz, rcs, vr, vr_com, time]
        points = batch_dict['points'].detach().clone()
        if self.use_virtual:
            valid = points[:, -2] == 0
            points = points[valid]

        radar_sampled_features = torch.zeros([B, 1, Z, Y, X], device=image_voxel_features.device)

        # move device of tensors
        self.lower_bound = self.lower_bound.to(cur_device)
        self.voxel_size = self.voxel_size.to(cur_device)
        self.voxel_space_range = self.voxel_space_range.to(cur_device)
        coor_bound_high = torch.tensor(image_voxel_features.shape[4:1:-1], device=points.device) - 1
        coor_bound_low = torch.zeros([3], device=points.device)

        for i in range(B):
            valid_mask = (points[:, 0] == i)
            pts_xyz = points[valid_mask, 1:4]

            center_loc = pts_xyz - self.lower_bound
            center_loc /= self.voxel_size
            center_loc = torch.round(center_loc).clamp(min=coor_bound_low, max=coor_bound_high).long()

            x, y, z = center_loc[:, 0], center_loc[:, 1], center_loc[:, 2]
            batch_idx = torch.ones_like(x) * i
            radar_sampled_features[batch_idx, 0, z, y, x] = 1

        # DEBUG
        # print('debugging!')
        # from matplotlib import pyplot as plt
        # pts_size = 1
        # plt.clf()
        # plt.subplots(figsize=(5, 10), dpi=500)
        # plt.axis('off')
        
        # ax = plt.subplot(2, 1, 1)
        # ax.set_xlim(0, 50)
        # ax.set_ylim(-25, 25)
        # plt.scatter(pts_xyz[:, 0].cpu(), pts_xyz[:, 1].cpu(), pts_size)

        # ax = plt.subplot(2, 1, 2)
        # plt.imshow(radar_sampled_features[0].sum(dim=0).sum(dim=0).detach().cpu().flip(dims=[0]))

        # plt.savefig('test.jpg')
        
        radar_sampled_features = torch.mul(radar_sampled_features, image_voxel_features)
        if self.fuse_mode == 'ADD':
            batch_dict["voxel_features"] = radar_sampled_features + image_voxel_features
        elif self.fuse_mode == 'SE':
            batch_dict["voxel_features"] = self.fuse_squeeze(self.fuse(torch.concat([radar_sampled_features, image_voxel_features], dim=1)))
        else:
            raise NotImplementedError

        return batch_dict