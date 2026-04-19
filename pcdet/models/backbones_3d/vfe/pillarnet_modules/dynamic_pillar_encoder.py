import torch
from torch import nn
# from ..registry import READERS
# from ..utils import build_norm_layer
from pcdet.ops.pillar_ops.pillar_modules import PillarMaxPooling



class DynamicPillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features=2,
        num_filters=(32,),
        pillar_size=0.1,
        virtual=False,
        pc_range=(0, -40, -3, 70.4, 40, 1),
        encoding_type='split',
        dataset='vod',
        **kwargs
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param pillar_size: (<float>: 3). Size of pillars.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.pc_range = pc_range
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_filters = [9 + num_input_features] + list(num_filters)
        self.pfn_layers = PillarMaxPooling(
            mlps=num_filters,
            pillar_size=pillar_size,
            point_cloud_range=pc_range
        )

        self.virtual = virtual
        self.encoding_type = encoding_type
        self.dataset = dataset
        print("pillar_encoder")
    @torch.no_grad()
    def absl_to_relative(self, absolute):
        relative = absolute.detach().clone()
        relative[..., 0] -= self.pc_range[0]
        relative[..., 1] -= self.pc_range[1]
        relative[..., 2] -= self.pc_range[2]

        return relative

    def forward(self, example, **kwargs):
        points_list = example.pop("points")
        device = points_list[0].device

        xyz = []
        xyz_batch_cnt = []
        points_padded_list = []

        for points in points_list:
            if self.virtual:
                if self.encoding_type == 'split':
                    # 1, 1: 真实点; 0, 0: gt框内的真实点; 0, 1: virtual 点
                    # 保留所有点云特征（除末尾2个标志位外），按 real/virtual 分块写入
                    feature_block = points.shape[1] - 3 - 2
                    if feature_block <= 0:
                        raise ValueError(
                            f"[PillarNet] invalid feature size: points.shape[1]={points.shape[1]} (need xyz + feats + 2 flags)"
                        )
                    expected_num_input = 3 + 2 * feature_block + 2
                    if self.num_input < expected_num_input:
                        raise ValueError(
                            f"[PillarNet] READER.NUM_INPUT_FEATURES too small: {self.num_input} < {expected_num_input}. "
                            "Please update config to keep all point features."
                        )

                    points_padded = torch.zeros([points.shape[0], self.num_input], device=points.device)
                    virtual_point_mask = points[:, -2] < 0.5
                    points_padded[:, :3] = points[:, :3]

                    real_range = slice(3, 3 + feature_block)
                    virtual_range = slice(3 + feature_block, 3 + 2 * feature_block)
                    points_padded[~virtual_point_mask, real_range] = points[~virtual_point_mask, 3:3 + feature_block]
                    points_padded[virtual_point_mask, virtual_range] = points[virtual_point_mask, 3:3 + feature_block]

                    points_padded[:, -2] = points[:, -2]
                    points_padded[:, -1] = points[:, -1]

                    real_points = self.absl_to_relative(points_padded) # [N, 12]
                    xyz_batch_cnt.append(len(real_points))
                    xyz.append(real_points[:, :3])
                    points_padded_list.append(points_padded)
                elif self.encoding_type == 'mixed':
                    points = self.absl_to_relative(points) # [N, 5]

                    xyz_batch_cnt.append(len(points))
                    xyz.append(points[:, :3])
                elif self.encoding_type == 'direct':
                    points = self.absl_to_relative(points[:, :-2]) # [N, 5]

                    xyz_batch_cnt.append(len(points))
                    xyz.append(points[:, :3])
                else:
                    raise NotImplementedError

            else:
                points = self.absl_to_relative(points) # [N, 5]

                xyz_batch_cnt.append(len(points))
                xyz.append(points[:, :3])

        xyz = torch.cat(xyz, dim=0).contiguous()  # 大于0的radar坐标
        if self.virtual:
            if self.encoding_type == 'split':
                pt_features = torch.cat(points_padded_list, dim=0).contiguous()
            elif self.encoding_type == 'mixed':
                pt_features = torch.cat(points_list, dim=0).contiguous() # 真实的radar坐标
            elif self.encoding_type == 'direct':
                pt_features = torch.cat(points_list, dim=0)[:, :-2].contiguous()
            else:
                raise NotImplementedError
        else:
            pt_features = torch.cat(points_list, dim=0).contiguous() # 真实的radar坐标
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)

        sp_tensor = self.pfn_layers(xyz, xyz_batch_cnt, pt_features) # [N, 3] [N, ] [N, 7]
        return sp_tensor
