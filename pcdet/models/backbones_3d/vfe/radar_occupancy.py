import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt

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

class RadarOccupancy(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size, occupancy_init = 0.01, radar_backbone='pillarnet', use_mask=False) -> None:
        super().__init__()
        print("RadarOccupancy")
        self.model_cfg = model_cfg
        self.pcr = point_cloud_range
        self.voxel_size = voxel_size
        self.voxel_size = torch.Tensor(voxel_size)

        self.voxel_space_range = torch.Tensor([self.pcr[3] - self.pcr[0],
                                                self.pcr[4] - self.pcr[1],
                                                self.pcr[5] - self.pcr[2]]).reshape([1, 3])
        self.grid_size = torch.round(self.voxel_space_range / self.voxel_size).int().squeeze().numpy()
        self.D = self.grid_size[2]
        self.replace_image_voxel_feats = self.model_cfg.get('REPLACE_IMAGE_VOEXL_FEATS', True)
        self.feature_add = self.model_cfg.get('FEATURE_ADD', False)
        self.use_aspp = self.model_cfg.get('USE_ASPP', False)

        self.radar_input_channels = self.model_cfg.RADAR_INPUT_CHANNELS
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
                          self.D,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.Sigmoid()
            )
        if self.use_aspp:
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
                          padding=6,
                          dilation=6),
                nn.BatchNorm2d(self.radar_input_channels),
                nn.ReLU(inplace=True),
                *self.pred_occupancy)
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
    
    def forward(self, batch_dict):
        image_voxel_features = batch_dict["voxel_features"] # [B, 128, 31, 320, 320]
        radar_features = batch_dict[self.radar_feature_name] # [B, 128, 320, 320]
        
        if image_voxel_features.shape[-2:] != radar_features.shape[-2:]:
            image_voxel_features = F.interpolate(image_voxel_features, 
                                                 size=[image_voxel_features.shape[-3], *radar_features.shape[-2:]], 
                                                 mode='trilinear') # [B, 128, 31, 160, 160]

        radar_occupancy = self.pred_occupancy(radar_features).unsqueeze(1) # [B, 1, 31, 160, 160]
        if self.use_mask:
            radar_ratio = self.radar_ratio.to(image_voxel_features.device)
            B = radar_occupancy.shape[0]
            radar_ratio = radar_ratio.repeat([B, 1, 1, 1, 1])
            batch_dict["voxel_features"] = image_voxel_features * (radar_occupancy * radar_ratio + (1 - radar_ratio))
        elif self.feature_add:
            batch_dict["spatial_features"] = image_voxel_features * radar_occupancy + image_voxel_features
        elif self.replace_image_voxel_feats:
            batch_dict["spatial_features"] = image_voxel_features * radar_occupancy
        else:
            batch_dict["spatial_features_occupy"] = image_voxel_features * radar_occupancy

        if self.training:
            self.forward_ret_dict["radar_occ"] = radar_occupancy
            self.forward_ret_dict["gt_boxes"] = batch_dict["gt_boxes"]

        # plt.subplots(figsize=(5, 10), dpi=500)
        # plt.axis('off')
        # ax = plt.subplot(2, 1, 1)
        # img = radar_occupancy.sum(dim=2).squeeze()
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

        target = torch.zeros([B, *self.grid_size], device=gt_boxes.device).permute([0, 3, 2, 1])

        for i in range(B):
            valid_gt_mask = gt_boxes[i, :, -1] > 0
            idxs = torch.where(valid_gt_mask)[0]
            points_list = []
            for idx in idxs:
                dx = torch.arange(-dims[i, idx, 0], dims[i, idx, 0])
                dy = torch.arange(-dims[i, idx, 1], dims[i, idx, 1])
                dz = torch.arange(0, dims[i, idx, 2]*2)
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
        
        # plt.subplots(figsize=(5, 10), dpi=500)
        # plt.axis('off')
        # ax = plt.subplot(2, 1, 1)
        # img = target.sum(dim=1).squeeze()
        # plt.imshow(img.flip(dims=[0]).cpu())
        # ax = plt.subplot(2, 1, 2)
        # ax.set_xlim(0, 50)
        # ax.set_ylim(-25, 25)
        # plot_gt_bev(self.forward_ret_dict["gt_boxes"][0].cpu())
        # plt.savefig('occ.jpg')
        # print('debugging!')
        
        # radar_occ_binary_class = torch.concat([1 - radar_occ, radar_occ], dim=1)
        loss = self.occ_loss(labels=target.unsqueeze(1), preds=radar_occ)
        return loss