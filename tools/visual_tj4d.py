import pickle
from tqdm import tqdm
import numpy as np
from skimage import io
from pathlib import Path
from pcdet.datasets.kitti.tj4d_utils import plot_points_gt
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.kitti import kitti_utils

root_split_path = Path('./data/tj4d/training/')

def get_image(idx):
    """
    Loads image for a sample
    Args:
        idx: int, Sample index
    Returns:
        image: (H, W, 3), RGB Image
    """
    img_file = root_split_path / 'image_2' / ('%s.png' % idx)
    assert img_file.exists()
    image = io.imread(img_file)
    image = image.astype(np.float32)
    image /= 255.0
    return image

def get_lidar(idx):
    lidar_file = root_split_path / 'velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    number_of_channels = 8  # ['x', 'y', 'z', 'V_r', 'Range', 'Power', 'Alpha', 'Beta']
    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)
    return points

def get_calib(idx):
    calib_file = root_split_path / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

with open("./data/tj4d/kitti_infos_trainval.pkl", 'rb') as f:
    infos = pickle.load(f)
for info in tqdm(infos):
    frame_id = info['point_cloud']['lidar_idx']
    img = get_image(frame_id)
    pts = get_lidar(frame_id)

    calib = get_calib(frame_id)
    annos = info['annos']
    annos = common_utils.drop_info_with_name(annos, name='DontCare')
    loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
    gt_names = annos['name']
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
    trans_lidar_to_cam, trans_cam_to_img = kitti_utils.calib_to_matricies(calib)

    plot_points_gt(frame_id, img, pts, gt_boxes_lidar, trans_lidar_to_cam, trans_cam_to_img)
    pass

