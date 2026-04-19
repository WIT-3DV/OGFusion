import torch
import torch.nn as nn

from .frustum_grid_generator import FrustumGridGenerator
from .sampler import Sampler


class FrustumToVoxel(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg, use_depth=True, **kwargs):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(grid_size=grid_size,
                                                   pc_range=pc_range,
                                                   disc_cfg=disc_cfg)
        self.sampler = Sampler(**model_cfg.SAMPLER)
        self.use_depth = use_depth

    def forward(self, batch_dict):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                frustum_features: (B, C, D, H_image, W_image), Image frustum features
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        # if self.use_depth:
        #     feature_shape = batch_dict["frustum_features"].shape[-2:]
        # else:
        #     feature_shape = batch_dict["features"].shape[-2:]
        # Generate sampling grid for frustum volume
        grid = self.grid_generator(lidar_to_cam=batch_dict["trans_lidar_to_cam"],
                                   cam_to_img=batch_dict["trans_cam_to_img"], # Lidar-X*Y*Z  [pixel-uv, depth] [W, H, D]
                                   image_shape=batch_dict["image_shape"])  # (B, X_D, Y_W, Z_H, 3) [2, 160, 160, 16, 3]

        # test code TODO: comment these
        # new_grid = 2 * torch.ones_like(grid) # [2, 320, 320, 31, 3]  D W H
        # new_grid[0, :, 160, 15, :] = torch.Tensor([0.25, 0.25, 0]).to(grid.device).repeat([320, 1])
        # new_grid[0, 160, :, 15, :] = torch.Tensor([0.5, 0.5, 0]).to(grid.device).repeat([320, 1])
        # new_grid[0, 160, 160, :, :] = torch.Tensor([0.75, 0.75, 0]).to(grid.device).repeat([31, 1])

        # new_grid[0, :, 150, 14, :] = torch.stack([torch.ones([320]) * 0.25,
        #                                           torch.ones([320]) * 0.25,
        #                                           torch.linspace(-1,1,320)], dim=1).to(grid.device)
        # B, C, W, H = batch_dict["features"].shape
        # new_features = self.sampler(input_features=batch_dict["features"].reshape([B, C, 1, W, H]), 
        #                             grid=new_grid)

        # test code

        if self.use_depth:
            # Sample frustum volume to generate voxel volume
            voxel_features = self.sampler(input_features=batch_dict["frustum_features"], # feature: D*H*W [2, 80, 80, 65, 242]
                                        grid=grid)  # (B, C, X, Y, Z) [2, 80, 160, 160, 16]   D W H
        else:
            # grid[..., 2] = 0
            B, C, W, H = batch_dict["features"].shape
            voxel_features = self.sampler(input_features=batch_dict["features"].reshape([B, C, 1, W, H]), 
                                        grid=grid)

        # (B, C, X, Y, Z) -> (B, C, Z, Y, X)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2) # [B, 64, 31, 320, 320]
        batch_dict["voxel_features"] = voxel_features
        return batch_dict
