# import torch
# import torch.nn as nn
#
# class ForegroundSampler(nn.Module):
#     def __init__(self, model_cfg=None, point_cloud_range=None, voxel_size=None):
#         super().__init__()
#         self.threshold = model_cfg.get('threshold', 0.5)
#
#     def forward(self, batch_dict):
#         # 示例：从雷达或融合特征里采样 “前景” 点
#         # 假设 batch_dict['points'] 是 (B, N, D)
#         pts = batch_dict.get('points', None)
#         if pts is None:
#             return batch_dict
#
#         # 简单过滤逻辑示例：
#         mask = pts[:, :, 3] > self.threshold  # 假设第 4 维是强度
#         batch_dict['foreground_points'] = pts[mask]
#         return batch_dict

import torch
import torch.nn as nn

class ForegroundSampler(nn.Module):
    def __init__(self, model_cfg=None):
        super().__init__()
        # 阈值配置：可同时对强度和多普勒速度进行筛选
        self.power_threshold = model_cfg.get('power_threshold', 0.5)
        self.doppler_threshold = model_cfg.get('doppler_threshold', 0.1)
        # 是否保留batch维度输出
        self.keep_batch = model_cfg.get('keep_batch', True)

    def forward(self, batch_dict):
        """
        batch_dict['points']: (B, N, 8) 对应 [x, y, z, D, P, R, A, E]
        返回 batch_dict，增加 'foreground_points'
        """
        pts = batch_dict.get('points', None)
        if pts is None:
            return batch_dict

        if self.keep_batch:
            foreground_points = []
            for b in range(pts.shape[0]):
                mask = (pts[b, :, 4] > self.power_threshold) & \
                       (pts[b, :, 3].abs() > self.doppler_threshold)
                foreground_points.append(pts[b][mask])
            batch_dict['foreground_points'] = foreground_points
        else:
            # 打平所有batch
            mask = (pts[:, :, 4] > self.power_threshold) & \
                   (pts[:, :, 3].abs() > self.doppler_threshold)
            batch_dict['foreground_points'] = pts[mask]

        return batch_dict
