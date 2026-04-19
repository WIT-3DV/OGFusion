
from skimage import io
import random
import torch
import math
import numpy as np
import matplotlib.patheffects as pe
from pathlib import Path
from matplotlib import pyplot as plt
from pcdet.datasets.kitti import kitti_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils


data_root = Path('./data/vod_radar_5frames/')
save_path = './visual_result/vod_baseline/'
infos = np.load(data_root / 'kitti_infos_val.pkl', allow_pickle=True)
det_result = np.load('./output/tools/cfgs/fusion_models/fusion_larger_bev_ori/default/eval/epoch_15/val/default/result.pkl', allow_pickle=True)
classes = ['Car', 'Pedestrian', 'Cyclist']

def get_image(idx):
    img_file = data_root / 'training/image_2' / ('%s.jpg' % idx)
    assert img_file.exists()
    image = io.imread(img_file)
    image = image.astype(np.float32)
    image /= 255.0
    return image

def get_lidar(idx):
    lidar_file = data_root / 'training/velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 7)
    return points

def get_calib(idx):
    calib_file = data_root / 'training/calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

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

def plot_gt_bev(gt_boxes, color=np.array([255, 0, 0]) / 256, facecolor=None):
    def convert_center_to_leftdown(x_cord, y_cord, angle, width, height):
        xp = x_cord-(math.sqrt(width**2+height**2)/2)*math.cos(math.atan2(height, width)+angle/180*math.pi)
        yp = y_cord-(math.sqrt(width**2+height**2)/2)*math.sin(math.atan2(height, width)+angle/180*math.pi)
        return xp, yp
    for gt_box in gt_boxes:
        gt_box = gt_box[:7]
        
        x, y = gt_box[:2]
        angle = gt_box[-1] / np.pi * 180
        w, h = gt_box[3:5]
        xp, yp = convert_center_to_leftdown(x, y, angle, w, h)
        left_bottom = [xp, yp]
        plt.gca().add_patch(
            plt.Rectangle(left_bottom, w,
                  h, fill=facecolor is not None, facecolor=facecolor, angle=angle,
                  edgecolor=color, linewidth=1))


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
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


def boxes_to_corners_3d(boxes3d):
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def lidar2img(points:np.ndarray, lidar2cam, cam2img):
    points = points.copy()
    points_lidar_homo = np.ones([points.shape[0], 4])
    points_lidar_homo[:, :3] = points[:, :3]
    points_cam_homo = np.matmul(lidar2cam, points_lidar_homo.T).T
    points_cam = points_cam_homo[:, :3]
    depth = points_cam[:, 2]
    points_img = np.matmul(cam2img[:3, :3], points_cam.T).T
    points_img = points_img / points_img[:, 2].reshape([points_img.shape[0], 1])

    points_img_depth = np.zeros_like(points)
    points_img_depth[:, :2] = points_img[:, :2]
    points_img_depth[:, 2] = depth
    return points_img_depth

def face(corners: np.ndarray, color: tuple, alpha: float = 0.3):
    xs = corners[:, 0]
    ys = corners[:, 1]
    plt.fill(xs, ys, color=color, alpha=alpha)

def plot_boxes(boxes: list, colors=None):
    for j in range(len(boxes)):
        corners_img = np.array(boxes[j])

        if colors is not None:
            color = colors[j]
        else:
            color = (1.0, 0.0, 0.0)

        if color == (1.0, 1.0, 1.0):
            alpha = 0.15
        else:
            alpha = 0.2

        # draw the 6 faces
        face(corners_img[:4], color, alpha)
        face(corners_img[4:], color, alpha)
        face(np.array([corners_img[0], corners_img[1], corners_img[5], corners_img[4]]), color, alpha)
        face(np.array([corners_img[1], corners_img[2], corners_img[6], corners_img[5]]), color, alpha)
        face(np.array([corners_img[2], corners_img[3], corners_img[7], corners_img[6]]), color, alpha)
        face(np.array([corners_img[0], corners_img[3], corners_img[7], corners_img[4]]), color, alpha)
    return

def plot_gt_3d(gt_boxes, lidar2cam, cam2img, colors):
    boxes = []
    for gt_box in gt_boxes:
        gt_box = gt_box[:7]
        box_corner = boxes_to_corners_3d(gt_box.reshape([1, 7]))
        box_corner = box_corner.reshape([8, 3])
        pts_img = lidar2img(box_corner, lidar2cam, cam2img)
        boxes.append(pts_img)
    plot_boxes(boxes, colors)

# idxes = list(range(len(infos)))
# random.shuffle(idxes)
idxes = list(range(0, 1296, 10))


for i, idx in enumerate(idxes):
    
    # if i > 100:
    #     break
    # print(i)
    # idx = 260
    # idx = 780
    idx = 890
    gt, dt = infos[idx], det_result[idx]
    frame_id = gt['image']['image_idx']
    img = get_image(frame_id)
    pts = get_lidar(frame_id)
    calib = get_calib(frame_id)

    # plot gt in image
    annos = gt['annos']
    annos = common_utils.drop_info_with_name(annos, name='DontCare')
    mask = []
    colors = []
    for name in annos['name']:
        mask.append(name in classes)
        if name in classes:
            colors.append([0.494, 0.184, 0.556])
        # if name == 'Car':
        #     colors.append([0.494, 0.184, 0.556])
        # elif name == 'Pedestrian':
        #     colors.append([82/255, 141/255, 232/255])
        # elif name == 'Cyclist':
        #     colors.append([173/255, 216/255, 230/255])

    loc, dims, rots = annos['location'][mask], annos['dimensions'][mask], annos['rotation_y'][mask]
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    gt_boxes = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
    trans_lidar_to_cam, trans_cam_to_img = kitti_utils.calib_to_matricies(calib)

    plt.clf()
    fig, ax = plt.subplots(dpi=500)
    plot_gt_3d(gt_boxes, trans_lidar_to_cam, trans_cam_to_img, colors) # TODO
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(save_path + (f"/{frame_id}_3d_gt.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot gt in bev
    plt.clf()
    fig, ax = plt.subplots(dpi=500)
    gca = plt.gca()
    gca.set_xlim(0, 50)
    gca.set_ylim(-25, 25)

    x, y = pts[:, 0], pts[:, 1]
    for i in range(len(x)):
        circle = plt.Circle([x[i], y[i]], .2, facecolor=[92/255,156/255,255/255])
        ax.add_artist(circle)
        stroke_effect = [pe.Stroke(linewidth=1, foreground='black'), pe.Normal()]
        circle.set_path_effects(stroke_effect)

    mask_bbox = gt['annos']['num_points_in_gt'][mask] > 0
    plot_gt_bev(gt_boxes[mask_bbox])

    plt.axis('off')
    plt.savefig(save_path + (f"/{frame_id}_bev_gt.png"))
    plt.close()

    # plot dt in image
    annos = dt
    if 'frame_id' in annos:
        annos.pop('frame_id')
    annos = common_utils.drop_info_with_name(annos, name='DontCare')
    mask = []
    colors = []
    threshold = 0.4
    for name, conf in zip(annos['name'], annos['score']):
        if conf > threshold:
            mask.append(name in classes)
            if name == 'Car':
                colors.append([0.494, 0.184, 0.556])
            elif name == 'Pedestrian':
                colors.append([82/255, 141/255, 232/255])
            elif name == 'Cyclist':
                colors.append([173/255, 216/255, 230/255])
        else:
            mask.append(False)

    loc, dims, rots = annos['location'][mask], annos['dimensions'][mask], annos['rotation_y'][mask]
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
    trans_lidar_to_cam, trans_cam_to_img = kitti_utils.calib_to_matricies(calib)

    plt.clf()
    fig, ax = plt.subplots(dpi=500)
    plot_gt_3d(gt_boxes_lidar, trans_lidar_to_cam, trans_cam_to_img, colors) # TODO
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(save_path + (f"/{frame_id}_3d_dt.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot dt and gt in bev
    plt.clf()
    fig, ax = plt.subplots(dpi=500)
    gca = plt.gca()
    gca.set_xlim(0, 50)
    gca.set_ylim(-25, 25)

    x, y = pts[:, 0], pts[:, 1]
    for i in range(len(x)):
        circle = plt.Circle([x[i], y[i]], .2, facecolor=[92/255,156/255,255/255])
        ax.add_artist(circle)
        stroke_effect = [pe.Stroke(linewidth=1, foreground='black'), pe.Normal()]
        circle.set_path_effects(stroke_effect)

    plot_gt_bev(gt_boxes[mask_bbox], color=[0.494, 0.184, 0.556],
                facecolor=np.array([0.494, 0.184, 0.556, 0.3]))
    plot_gt_bev(gt_boxes_lidar)

    plt.axis('off')
    plt.savefig(save_path + (f"/{frame_id}_bev_dt.png"))
    plt.close()

    pass