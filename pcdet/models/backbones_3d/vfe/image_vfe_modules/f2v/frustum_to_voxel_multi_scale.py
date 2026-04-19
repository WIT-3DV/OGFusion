import torch
import torch.nn as nn

from .frustum_grid_generator import FrustumGridGenerator
from .sampler import Sampler


class FrustumToVoxelMultiScale(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg,
                 use_depth=True, feature_num=None, fuse_mode=None,
                 in_channels=None, out_channels=None, **kwargs):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        assert feature_num is not None
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.bev_augment = self.model_cfg.get('BEV_AUG', False)
        self.grid_generator = FrustumGridGenerator(grid_size=grid_size,
                                                   pc_range=pc_range,
                                                   disc_cfg=disc_cfg,
                                                   bev_aug=self.bev_augment)
        self.sampler = Sampler(**model_cfg.SAMPLER)
        self.use_depth = use_depth
        self.feature_num = feature_num
        self.fuse_mode = fuse_mode
        self.in_channels = in_channels * feature_num
        self.out_channels = out_channels
        if self.fuse_mode == 'CONCAT':
            assert self.in_channels is not None
            assert self.out_channels is not None
            self.channel_reduce = nn.Sequential(
                nn.Conv3d(self.in_channels, self.out_channels, (1, 1, 1)),
                nn.BatchNorm3d(num_features=self.out_channels),
                nn.ReLU()
            )



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
        # Generate sampling grid for frustum volume
        grid = self.grid_generator(lidar_to_cam=batch_dict["trans_lidar_to_cam"],
                                   cam_to_img=batch_dict["trans_cam_to_img"],
                                   # Lidar-X*Y*Z  [pixel-uv, depth] [W, H, D]
                                   image_shape=batch_dict["image_shape"],
                                   bda=batch_dict['lidar_aug_matrix'])  # (B, X_D, Y_W, Z_H, 3) [2, 160, 160, 16, 3]

        # # DEBUG code
        # print("debug code!")
        # batch_dict["features"][0][0, 0, :, :] = 0
        # batch_dict["features"][0][0, 0, 50:70, 220:260] = 1
        # batch_dict["features"][1][0, 0, :, :] = 0
        # batch_dict["features"][1][0, 0, 25:35, 110:130] = 1
        # batch_dict["features"][2][0, 0, :, :] = 0
        # batch_dict["features"][2][0, 0, 25:35, 110:130] = 1
        # batch_dict["features"][3][0, 0, :, :] = 0
        # batch_dict["features"][3][0, 0, 25:35, 110:130] = 1

        if self.use_depth:
            voxel_features = []
            for i in range(self.feature_num):
                voxel_features.append(
                    self.sampler(input_features=batch_dict["frustum_features"][i],
                                 grid=grid).permute(0, 1, 4, 3, 2)
                )
            batch_dict.pop('frustum_features')
        else:
            voxel_features = []
            for i in range(self.feature_num):
                B, C, W, H = batch_dict["features"][i].shape
                voxel_features.append(self.sampler(input_features=batch_dict["features"][i].reshape([B, C, 1, W, H]),
                                                   grid=grid).permute(0, 1, 4, 3, 2))  # [B, 64, 31, 320, 320]
        if self.fuse_mode == 'ADD':
            voxel_features = torch.stack(voxel_features)  # [4,1,64,31,320,320]
            voxel_features = torch.sum(voxel_features, dim=0)  # [1,64,31,320,320]
        elif self.fuse_mode == 'CONCAT':
            voxel_features = torch.concat(voxel_features, dim=1)  # [1,256,31,320,320]
            voxel_features = self.channel_reduce(voxel_features)  # [1,128,31,320,320]
        else:
            raise NotImplementedError

        # print("debug")
        # from matplotlib import pyplot as plt
        # for idx, feat in enumerate(voxel_features):
        #     heatmap = feat[0][0].sum(dim=0)
        #     plt.imshow(heatmap.cpu().detach().numpy())
        #     plt.savefig(f'{idx}.png')

        batch_dict["voxel_features"] = voxel_features
        #
        # def _p(name, x):
        #     if x is None:
        #         print(f"{name}: None")
        #         return
        #     if isinstance(x, (list, tuple)):
        #         print(f"{name}: list/tuple len={len(x)}")
        #         for i, t in enumerate(x[:6]):
        #             if hasattr(t, "shape"):
        #                 print(f"  [{i}] shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
        #             else:
        #                 print(f"  [{i}] type={type(t)}")
        #         return
        #     if hasattr(x, "shape"):
        #         print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        #         # 打印一个小片段避免刷屏
        #         try:
        #             print(f"  sample={x.flatten()[:6].detach().cpu().numpy()}")
        #         except Exception:
        #             pass
        #     else:
        #         print(f"{name}: type={type(x)}")
        #
        # print("=== batch_dict keys ===")
        # print(sorted(list(batch_dict.keys())))
        #
        # # 你当前明确有的
        # _p("trans_lidar_to_cam", batch_dict.get("trans_lidar_to_cam"))
        # _p("trans_cam_to_img", batch_dict.get("trans_cam_to_img"))
        # _p("image_shape", batch_dict.get("image_shape"))
        # _p("lidar_aug_matrix / bda", batch_dict.get("lidar_aug_matrix"))
        #
        # # 下面这些是 BEVDet/BEVPoolv2 常用的（有就打印，没有也没关系）
        # for k in [
        #     "cam2img", "cam_to_img", "cam2imgs", "intrinsics", "camera_intrinsics",
        #     "lidar2cam", "lidar_to_cam", "sensor2ego", "cam2ego", "extrinsics",
        #     "post_rots", "post_trans", "img_aug_matrix", "img_aug_rot", "img_aug_trans",
        #     "flip", "scale", "crop", "resize", "rot"
        # ]:
        #     _p(k, batch_dict.get(k))
        #
        # # feature / depth 相关：BEVPoolv2 需要 (depth, feat) 而不是 (C,D,H,W) 的 frustum feature 体
        # _p("frustum_features", batch_dict.get("frustum_features"))
        # _p("features", batch_dict.get("features"))
        # _p("depth", batch_dict.get("depth"))
        # _p("depth_probs", batch_dict.get("depth_probs"))
        # _p("depth_logits", batch_dict.get("depth_logits"))
        # print("=== print done ===")
        return batch_dict
