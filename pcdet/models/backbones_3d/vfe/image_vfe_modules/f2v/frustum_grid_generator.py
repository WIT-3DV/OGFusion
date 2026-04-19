import torch
import torch.nn as nn

try:
    from kornia.utils.grid import create_meshgrid3d
    from kornia.geometry.linalg import transform_points
except Exception as e:
    # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
    # print('Warning: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
    pass

from pcdet.utils import transform_utils

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 3, 4, dtype=torch.float32, device=fx.device)
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    return K

class FrustumGridGenerator(nn.Module):

    def __init__(self, grid_size, pc_range, disc_cfg, bev_aug=False):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()
        try:
            import kornia
        except Exception as e:
            # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
            print('Error: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. '
                  'Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
            exit(-1)

        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size, dtype=self.dtype)
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg
        self.bev_aug = bev_aug

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        # Create voxel grid
        self.depth, self.width, self.height = self.grid_size.int()
        self.voxel_grid = create_meshgrid3d(depth=self.depth,
                                                         height=self.height,
                                                         width=self.width,
                                                         normalized_coordinates=False) # D*H*W [D W H]

        self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ # D*W*H [D W H]

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
        Returns:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)

        return unproject

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img, bda=None):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid: (B, X, Y, Z, 3), Voxel sampling grid
            grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
        Returns:
            frustum_grid: (B, X, Y, Z, 3), Frustum sampling grid
        """
        B = lidar_to_cam.shape[0]

        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)

        # Reshape to match dimensions
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)
        V_G = V_G.reshape([1, 4, 4]).repeat_interleave(repeats=B, dim=0)

        # Transform to camera frame
        lidar_grid = transform_points(trans_01=V_G.reshape(B, 1, 1, 4, 4), points_1=voxel_grid)
        if self.bev_aug:
            bda = torch.linalg.inv(bda)
            lidar_grid = transform_points(trans_01=bda.reshape(B, 1, 1, 4, 4), points_1=lidar_grid)
        camera_grid = transform_points(trans_01=C_V.reshape(B, 1, 1, 4, 4), points_1=lidar_grid)

        # Project to image
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = transform_utils.project_to_image(project=I_C, points=camera_grid)

        # Convert depths to depth bins
        image_depths = transform_utils.bin_depths(depth_map=image_depths, **self.disc_cfg)

        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return frustum_grid

    def forward(self, lidar_to_cam, cam_to_img, image_shape, bda=None):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
            image_shape: (B, 2), Image shape [H, W]
        Returns:
            frustum_grid (B, X, Y, Z, 3), Sampling grids for frustum features
        """
        # sx = feature_shape[0] / image_shape[0, 0]
        # sy = feature_shape[1] / image_shape[0, 1]
        # rescaled_intrinsics = scale_intrinsics(cam_to_img, sx, sy).detach()

        frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid.to(lidar_to_cam.device),
                                           grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
                                           lidar_to_cam=lidar_to_cam,# lidar-XYZ -> pixel-uv depth
                                           cam_to_img=cam_to_img,
                                           bda=bda) #        D*W*H -> D*W*H

        # Normalize grid
        image_shape, _ = torch.max(image_shape, dim=0)
        image_depth = torch.tensor([self.disc_cfg["num_bins"]],
                                   device=image_shape.device,
                                   dtype=image_shape.dtype)
        # frustum_shape = torch.cat((image_depth, torch.Tensor([feature_shape[0],feature_shape[1]]).to(image_depth.device)))
        frustum_shape = torch.cat((image_depth, image_shape))
        frustum_grid = transform_utils.normalize_coords(coords=frustum_grid, shape=frustum_shape)

        # Replace any NaNs or infinites with out of bounds
        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid
