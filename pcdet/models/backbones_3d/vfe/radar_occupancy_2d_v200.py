import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

class Focal_Loss(nn.Module):
    """
    二分类Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
	
    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels:标签
        """
        eps=1e-7
        loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
        loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
        loss=loss_0+loss_1
        return torch.mean(loss)

def gaussian_2d(shape, range, sigma=1):
    ratio = [x / (y + 1) for (x, y) in zip(range, shape)]
    m, n = [(ss - 1.) / 2. for ss in shape] # radius
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    y, x = y*ratio[1], x*ratio[2]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gauss_blur(image, kernel):
    return F.conv2d(image, kernel, padding=kernel.shape[0] // 2)

class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        voxel_features = batch_dict["voxel_features"] # [2, 64, 31, 320, 320]
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features"] = bev_features # [2, 64, 320, 320]
        return batch_dict

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def box2corner_3d(bbox):
    bottom_center = np.array([bbox[0], bbox[1], bbox[2] - bbox[5] / 2])
    cos, sin = np.cos(bbox[6]), np.sin(bbox[6])
    pc0 = np.array([bbox[0] + cos * bbox[3] / 2 + sin * bbox[4] / 2,
                    bbox[1] + sin * bbox[3] / 2 - cos * bbox[4] / 2,
                    bbox[2] - bbox[5] / 2])
    pc1 = np.array([bbox[0] + cos * bbox[3] / 2 - sin * bbox[4] / 2,
                    bbox[1] + sin * bbox[3] / 2 + cos * bbox[4] / 2,
                    bbox[2] - bbox[5] / 2])
    pc2 = 2 * bottom_center - pc0
    pc3 = 2 * bottom_center - pc1
    return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]

def plot_gt_bev(gt_boxes):
    for gt_box in gt_boxes:
        gt_box = gt_box[:7]
        box_corner = box2corner_3d(gt_box)
        corners = np.array(box_corner)[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=np.array([191, 4, 54]) / 256,  linestyle='solid')

class RadarOccupancy2DV2(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size, occupancy_init = 0.01, radar_backbone='pillarnet', use_mask=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pcr = point_cloud_range
        self.voxel_size = voxel_size
        self.voxel_size = torch.Tensor(voxel_size)

        self.voxel_space_range = torch.Tensor([self.pcr[3] - self.pcr[0],
                                                self.pcr[4] - self.pcr[1],
                                                self.pcr[5] - self.pcr[2]]).reshape([1, 3])
        self.grid_size = torch.round(self.voxel_space_range / self.voxel_size).int().squeeze().numpy()
        self.D = self.grid_size[2]

        self.image_input_channels = self.model_cfg.get('IMAGE_INPUT_CHANNELS', 128)
        self.radar_input_channels = self.model_cfg.RADAR_INPUT_CHANNELS
        self.replace_image_voxel_feats = self.model_cfg.get('REPLACE_IMAGE_VOEXL_FEATS', True)
        self.fuse_mode = self.model_cfg.get('FUSE_MODE', 'MUL')
        if self.fuse_mode == 'SE':
            self.fuse = SE_Block(self.image_input_channels * 2)
            self.fuse_squeeze = nn.Sequential(
                nn.Conv2d(self.image_input_channels + self.radar_input_channels, self.image_input_channels, 1),
                nn.BatchNorm2d(self.image_input_channels),
                nn.ReLU()
            )
        self.use_aspp = self.model_cfg.get('USE_ASPP', False)
        self.use_dcn = self.model_cfg.get('USE_DCN', None)
        self.pred_occupancy = nn.Sequential(
                nn.Conv2d(self.radar_input_channels,
                          self.radar_input_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='zeros'),
                nn.BatchNorm2d(self.radar_input_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.radar_input_channels//2,
                          1,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.Sigmoid()
            )
        if self.use_aspp:
            self.aspp_dial = self.model_cfg.get('ASPP_DIAL', None)
            self.aspp_shape = self.model_cfg.get('ASPP_SHAPE', 6)
            # 串行ASPP
            if self.aspp_dial is None:
                self.pred_occupancy = nn.Sequential(
                    nn.Conv2d(self.radar_input_channels,
                            self.radar_input_channels,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                            padding=0,
                            dilation=1),
                    nn.BatchNorm2d(self.radar_input_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.radar_input_channels,
                            self.radar_input_channels,
                            kernel_size=3,
                            stride=1,
                            bias=False,
                            padding=self.aspp_shape,
                            dilation=self.aspp_shape),
                    nn.BatchNorm2d(self.radar_input_channels),
                    nn.ReLU(inplace=True),
                    *self.pred_occupancy)
            # 并行ASPP
            else:
                self.aspp = nn.ModuleList()
                # 使用DCN
                if self.use_dcn is not None:
                    assert len(self.aspp_dial) == len(self.use_dcn)
                    for dial, dcn in zip(self.aspp_dial, self.use_dcn):
                        if dcn:
                            self.aspp.append(
                                nn.Sequential(
                                    DCN(self.radar_input_channels,
                                            self.radar_input_channels,
                                            kernel_size=1 if dial == 1 else 3,
                                            stride=1,
                                            bias=False,
                                            padding=0 if dial == 1 else dial,
                                            dilation=dial),
                                    nn.BatchNorm2d(self.radar_input_channels),
                                    nn.ReLU(inplace=True)
                                )
                            )
                        else:
                            self.aspp.append(
                                nn.Sequential(
                                    nn.Conv2d(self.radar_input_channels,
                                            self.radar_input_channels,
                                            kernel_size=1 if dial == 1 else 3,
                                            stride=1,
                                            bias=False,
                                            padding=0 if dial == 1 else dial,
                                            dilation=dial),
                                    nn.BatchNorm2d(self.radar_input_channels),
                                    nn.ReLU(inplace=True)
                                )
                            )
                # 不使用DCN
                else:
                    for dial in self.aspp_dial:
                        self.aspp.append(
                            nn.Sequential(
                                nn.Conv2d(self.radar_input_channels,
                                        self.radar_input_channels,
                                        kernel_size=1 if dial == 1 else 3,
                                        stride=1,
                                        bias=False,
                                        padding=0 if dial == 1 else dial,
                                        dilation=dial),
                                nn.BatchNorm2d(self.radar_input_channels),
                                nn.ReLU(inplace=True)
                            )
                        )
                self.fuse_conv = nn.Sequential(
                    nn.Conv2d(
                        self.radar_input_channels * len(self.aspp_dial),
                        self.radar_input_channels,
                        kernel_size=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(self.radar_input_channels),
                    nn.ReLU()
                )
        backbone_feature = {'pillarnet': 'pillar_features_scattered',
                            'second': 'spatial_features_2d'}
        self.radar_feature_name = backbone_feature[radar_backbone]
        self.use_mask = use_mask
        if self.use_mask:
            sigma = 240
            X, Y, Z = self.grid_size
            y, x = torch.arange(0, Y) - Y // 2, torch.arange(0, X)
            y_mesh, x_mesh = torch.meshgrid(y, x)
            self.radar_ratio = torch.exp(-(x_mesh * x_mesh + y_mesh * y_mesh) / (2 * sigma * sigma))
            self.pred_occupancy[-3].bias.data.fill_(bias_init_with_prob(occupancy_init))
        else:
            self.pred_occupancy[-2].bias.data.fill_(bias_init_with_prob(occupancy_init))
        self.forward_ret_dict = {}
        self.occ_loss = Focal_Loss()
        self.map_to_bev = Conv2DCollapse(model_cfg=self.model_cfg.MAP_TO_BEV,
                                        grid_size=self.grid_size)
        
        # self.kernel = gaussian_2d(5, 5)
    
    def forward(self, batch_dict):
        batch_dict = self.map_to_bev(batch_dict)
        image_bev_features = batch_dict["spatial_features"] # [B, 128, 320, 320]
        radar_features = batch_dict[self.radar_feature_name] # [B, 128, 320, 320]
        
        if image_bev_features.shape[-2:] != radar_features.shape[-2:]:
            image_bev_features = F.interpolate(image_bev_features, 
                                                 size=[*radar_features.shape[-2:]], 
                                                 mode='bilinear') # [B, 128, 320, 320]
        if self.use_aspp and self.aspp_dial is not None:
            feat_list = []
            for layer in self.aspp:
                feat_list.append(layer(radar_features))
            radar_features = self.fuse_conv(torch.concat(feat_list, dim=1))

        radar_occupancy = self.pred_occupancy(radar_features) # [B, 1, 320, 320]
        if self.use_mask:
            radar_ratio = self.radar_ratio.to(image_bev_features.device)
            B = radar_occupancy.shape[0]
            radar_ratio = radar_ratio.repeat([B, 1, 1, 1])
            batch_dict["spatial_features_occupy"] = image_bev_features * (radar_occupancy * radar_ratio + (1 - radar_ratio))
        else:
            if self.fuse_mode == 'MUL':
                feats = image_bev_features * radar_occupancy
            elif self.fuse_mode == 'SE':
                feats = self.fuse_squeeze(self.fuse(torch.concat([image_bev_features * radar_occupancy, image_bev_features], dim=1)))
            if self.replace_image_voxel_feats:
                batch_dict["spatial_features"] = feats
            else:
                batch_dict["spatial_features_occupy"] = feats


        if self.training:
            self.forward_ret_dict["radar_occ"] = radar_occupancy
            self.forward_ret_dict["gt_boxes"] = batch_dict["gt_boxes"]

        # plt.subplots(figsize=(5, 10), dpi=500)
        # plt.axis('off')
        # ax = plt.subplot(2, 1, 1)
        # img = radar_occupancy.squeeze()
        # plt.imshow(img.cpu())
        # ax = plt.subplot(2, 1, 2)
        # ax.set_xlim(0, 50)
        # ax.set_ylim(-25, 25)
        # plot_gt_bev(batch_dict["gt_boxes"][0].cpu())
        # plt.savefig('occ.jpg')
        
        return batch_dict
    
    def get_loss(self):
        gt_boxes = self.forward_ret_dict["gt_boxes"]
        radar_occ = self.forward_ret_dict["radar_occ"]

        B, N, _ = gt_boxes.shape

        dims = gt_boxes[..., 3:6] / 2
        center = gt_boxes[..., :3]

        pcr_low = torch.from_numpy(self.pcr[:3]).to(gt_boxes.device)
        voxel_size = self.voxel_size.to(gt_boxes.device)
        coor_bound_high = torch.from_numpy(self.grid_size).to(gt_boxes.device)[[2, 1, 0]] - 1
        coor_bound_low = torch.zeros([3], device=gt_boxes.device)

        center = ((center - pcr_low) / voxel_size).floor()
        dims = (dims / voxel_size).floor()

        target = torch.zeros([B, *self.grid_size], device=gt_boxes.device).permute([0, 3, 2, 1]) # b z y x

        for i in range(B):
            valid_gt_mask = gt_boxes[i, :, -1] > 0
            idxs = torch.where(valid_gt_mask)[0]
            points_list = []
            for idx in idxs:
                dx = torch.arange(-dims[i, idx, 0], dims[i, idx, 0])
                dy = torch.arange(-dims[i, idx, 1], dims[i, idx, 1])
                dz = torch.zeros([1])
                points_ori = torch.stack(torch.meshgrid(dx, dy, dz), dim=3).reshape([-1, 3]).to(gt_boxes.device)
                points_ori = rotate_points_along_z(points_ori.unsqueeze(0), gt_boxes[i, idx, -2].reshape([1])).floor()
                points = center[i, idx] + points_ori
                points_list.append(points.squeeze())
            all_points = torch.concat(points_list, dim=0)[..., [2, 1, 0]]
            all_points = all_points.clamp(min=coor_bound_low, max=coor_bound_high).long() # z y x

            batch_idx = torch.ones([all_points.shape[0]], device=all_points.device, dtype=torch.long) * i
            z = all_points[:, 0]
            y = all_points[:, 1]
            x = all_points[:, 2]

            target[(batch_idx, z, y, x)] = 1
        
        # target = target.permute([0, 1, 3, 2])
        target = target.sum(dim=1).clamp(max=1).unsqueeze(1)

        # blurred_target = gauss_blur(target, self.kernel.to(target.device))

        # plt.subplots(figsize=(5, 10), dpi=500)
        # plt.axis('off')
        # ax = plt.subplot(2, 1, 1)
        # img = target.sum(dim=1).squeeze()
        # plt.imshow(img.flip(dims=[0]).cpu())
        # ax = plt.subplot(2, 1, 2)
        # ax.set_xlim(0, 69.12)
        # ax.set_ylim(-39.68, 39.68)
        # plot_gt_bev(self.forward_ret_dict["gt_boxes"][0].cpu())
        # plt.savefig('occ.jpg')
        # print('debugging!')
        
        # radar_occ_binary_class = torch.concat([1 - radar_occ, radar_occ], dim=1)
        loss = self.occ_loss(labels=target, preds=radar_occ)
        return loss