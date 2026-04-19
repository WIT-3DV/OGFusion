# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import argparse
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from nusc_image_projection import to_batch_tensor, to_tensor, projectionV2, get_obj


# =========================
# Dataset image size (VoD)
# =========================
H = 1216
W = 1936


# -------------------------
# Detectron2 / Mask2Former
# -------------------------

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg



def get_parser():
    parser = argparse.ArgumentParser()

    # output
    parser.add_argument(
        "--pts-save-path",
        required=True,
        help="Directory to save generated virtual point dict npy files.",
    )

    # segmentation
    parser.add_argument(
        "--config-file",
        default="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "./ckpts/model_final_dfa862.pkl"],
        nargs=argparse.REMAINDER,
    )

    # dataset
    parser.add_argument(
        "--info-path",
        default="data/view_of_delft_PUBLIC/vod_radar_5frames/kitti_infos_trainval.pkl",
        help="Path to VoD info pkl.",
    )
    parser.add_argument(
        "--image-root",
        default="data/view_of_delft_PUBLIC/vod_radar_5frames/training/image_2",
        help="Path to VoD images.",
    )
    parser.add_argument(
        "--pts-root",
        default="data/view_of_delft_PUBLIC/vod_radar_5frames/training/velodyne",
        help="Path to VoD radar/point bins.",
    )
    parser.add_argument(
        "--num-camera",
        type=int,
        default=1,
        help="Number of cameras used (VoD script is tested for 1).",
    )

    # RAPID hyper-params (VoD-tuned defaults)
    parser.add_argument("--rapid-num-virtual", type=int, default=140, help="Virtual points per instance.")
    parser.add_argument("--rapid-cand-per-real", type=int, default=32, help="Candidates per real foreground point.")
    parser.add_argument(
        "--rapid-power-index",
        type=int,
        default=3,
        help="Index of echo-strength feature in raw point. For VoD Nx7 radar, 3=RCS.",
    )

    parser.add_argument("--rapid-range-ref", type=float, default=40.0, help="Reference range (meters) for sigma scaling.")

    parser.add_argument("--rapid-sigma-theta-min-deg", type=float, default=0.5, help="Min azimuth sigma in degrees.")
    parser.add_argument("--rapid-sigma-theta-range-deg", type=float, default=0.9, help="Azimuth sigma scale for range.")
    parser.add_argument("--rapid-sigma-theta-power-deg", type=float, default=1.2, help="Azimuth sigma scale for low power.")

    parser.add_argument("--rapid-sigma-phi-min-deg", type=float, default=0.3, help="Min elevation sigma in degrees.")
    parser.add_argument("--rapid-sigma-phi-range-deg", type=float, default=0.45, help="Elevation sigma scale for range.")
    parser.add_argument("--rapid-sigma-phi-power-deg", type=float, default=0.8, help="Elevation sigma scale for low power.")

    parser.add_argument("--rapid-sigma-max-deg", type=float, default=5.5, help="Clamp max sigma (deg).")
    parser.add_argument("--rapid-delta-clip", type=float, default=2.2, help="Clip delta to +- delta_clip * sigma.")

    parser.add_argument("--rapid-mask-dilate", type=int, default=5, help="Mask tolerance (pixels) for w_mask.")
    parser.add_argument("--rapid-boundary-weight", type=float, default=0.12, help="Weight if only inside dilated mask.")

    parser.add_argument("--rapid-lambda-occ", type=float, default=0.25, help="Occlusion decay lambda for w_occ.")
    parser.add_argument("--rapid-min-weight", type=float, default=1e-5, help="Min final weight to keep candidate.")

    # Gaussian image prior from the original VoD version
    parser.add_argument("--rapid-gauss-shape", type=int, default=51, help="Gaussian kernel spatial size.")
    parser.add_argument("--rapid-gauss-sigma", type=float, default=7.0, help="Gaussian kernel sigma.")
    parser.add_argument(
        "--rapid-gauss-alpha",
        type=float,
        default=0.55,
        help="Mixing strength of Gaussian image prior. 0 disables prior, 1 uses full normalized prior.",
    )

    return parser


class DatasetVoD(Dataset):
    def __init__(self, info_path: str, predictor, image_root: str):
        self.sweeps = get_obj(info_path)
        self.predictor = predictor
        self.image_root = image_root

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]

        img_name = info["image"]["image_idx"] + ".jpg"
        img_path = os.path.join(self.image_root, img_name)

        original_image = cv2.imread(img_path)
        if original_image is None:
            return []

        if self.predictor.input_format == "RGB":
            original_image = original_image[:, :, ::-1]

        height, width = original_image.shape[:2]
        image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        return [info, inputs]

    def __len__(self):
        return len(self.sweeps)


# -------------------------
# Utilities
# -------------------------

def read_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 7)



def postprocess(res):
    """Extract labels/scores/masks from Detectron2 Instance outputs.

    Output masks are reshaped to (N_inst, W*H) with (x,y) indexing order
    matching the original VoD code.
    """
    result = res["instances"]
    labels = result.pred_classes
    scores = result.scores
    masks = result.pred_masks.reshape(scores.shape[0], W * H)

    empty_mask = masks.sum(dim=1) == 0
    labels = labels[~empty_mask]
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]

    masks = masks.reshape(-1, H, W).permute(0, 2, 1).reshape(-1, W * H)
    return labels, scores, masks



def is_within_mask(points_xyc: torch.Tensor, masks_with_cam: torch.Tensor, H=H, W=W) -> torch.Tensor:
    """Check if each projected point falls inside each instance mask (camera-consistent).

    points_xyc: (N,3) int/long, columns [x(u), y(v), camera_id]
    masks_with_cam: (M, W*H+1) last column is camera_id

    returns: (N, M) bool
    """
    seg_mask = masks_with_cam[:, :-1].reshape(-1, W, H)
    camera_id = masks_with_cam[:, -1]
    pts = points_xyc.long()
    valid = seg_mask[:, pts[:, 0], pts[:, 1]] * (camera_id[:, None] == pts[:, -1][None])
    return valid.transpose(1, 0)



def gaussian_2d(shape, sigma=1.0):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h



def deg2rad(x: torch.Tensor) -> torch.Tensor:
    return x * np.pi / 180.0



def cartesian_to_spherical(xyz: torch.Tensor):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = torch.sqrt(x * x + y * y + z * z + 1e-12)
    theta = torch.atan2(y, x)
    xy = torch.sqrt(x * x + y * y + 1e-12)
    phi = torch.atan2(z, xy)
    return r, theta, phi



def spherical_to_cartesian(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    cos_phi = torch.cos(phi)
    x = r * cos_phi * torch.cos(theta)
    y = r * cos_phi * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)



def project_points_single_camera(
    xyz: torch.Tensor,
    Tr_velo_to_cam: torch.Tensor,
    K: torch.Tensor,
    H: int = H,
    W: int = W,
):
    """Project lidar/radar points to a single pinhole camera."""
    device = xyz.device
    dtype = xyz.dtype

    ones = torch.ones((xyz.shape[0], 1), device=device, dtype=dtype)
    xyz1 = torch.cat([xyz, ones], dim=1)

    cam = (Tr_velo_to_cam @ xyz1.t()).t()[:, :3]
    depth = cam[:, 2]

    uvw = (K @ cam.t()).t()
    u = uvw[:, 0] / (uvw[:, 2] + 1e-6)
    v = uvw[:, 1] / (uvw[:, 2] + 1e-6)

    valid = depth > 1e-5
    valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, depth, valid



def dilate_masks(masks_wh: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return masks_wh.bool()
    x = masks_wh.float().unsqueeze(1)
    x = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return x.squeeze(1) > 0.5



def build_instance_gaussian_prior(
    inst_mask_wh: torch.Tensor,
    inst_proj_points_xy: torch.Tensor,
    gauss_kernel: torch.Tensor,
) -> torch.Tensor:
    """Create masked Gaussian prior map from projected real points.

    Returns a normalized prior in [0,1] with shape (W,H).
    """
    prior = torch.zeros_like(inst_mask_wh, dtype=torch.float32)
    if inst_proj_points_xy.numel() == 0:
        return prior

    gW, gH = gauss_kernel.shape
    xr = gW // 2
    yr = gH // 2

    for point in inst_proj_points_xy:
        x = int(point[0].item())
        y = int(point[1].item())

        x1, x2 = x - xr, x + xr + 1
        y1, y2 = y - yr, y + yr + 1

        px1 = max(x1, 0)
        py1 = max(y1, 0)
        px2 = min(x2, W)
        py2 = min(y2, H)

        gx1 = px1 - x1
        gy1 = py1 - y1
        gx2 = gx1 + (px2 - px1)
        gy2 = gy1 + (py2 - py1)

        if px1 < px2 and py1 < py2:
            prior[px1:px2, py1:py2] += gauss_kernel[gx1:gx2, gy1:gy2]

    prior = prior * inst_mask_wh.float()
    maxv = prior.max()
    if maxv > 0:
        prior = prior / maxv
    return prior


# -------------------------
# RAPID virtual point generation
# -------------------------

@torch.no_grad()
def add_virtual_rapid(
    masks_with_cam: torch.Tensor,
    inst_labels_11: torch.Tensor,
    proj_points: torch.Tensor,
    raw_points: torch.Tensor,
    *,
    num_virtual: int,
    cand_per_real: int,
    power_index: int,
    intrinsics: torch.Tensor,
    transforms: torch.Tensor,
    mask_dilate: int,
    boundary_weight: float,
    lambda_occ: float,
    min_weight: float,
    range_ref: float,
    sigma_theta_min_deg: float,
    sigma_theta_range_deg: float,
    sigma_theta_power_deg: float,
    sigma_phi_min_deg: float,
    sigma_phi_range_deg: float,
    sigma_phi_power_deg: float,
    sigma_max_deg: float,
    delta_clip: float,
    gauss_shape: int,
    gauss_sigma: float,
    gauss_alpha: float,
    num_camera: int = 1,
):
    """RAPID with parent-depth occlusion + Gaussian image prior.

    Inputs:
      - masks_with_cam: (M, W*H+1)
      - inst_labels_11: (M, 11)  -> onehot(10) + seg_score(1)
      - proj_points:    (C, N, 5)-> [u, v, depth, valid, cam_id]
      - raw_points:     (N, 7)   -> [x,y,z, feat1..feat4]

    Returns:
      virtual_points: (Nv, 18) -> [x,y,z] + inst_labels_11 + raw_feat4
      real_points:    (Nr, 18) -> raw_points7 + inst_labels_11
      foreground_indices: (Nr,)
      prob_map_all: (W,H) float
    """
    if num_camera != 1:
        raise NotImplementedError("This implementation is currently intended for num_camera=1 (VoD single cam setup).")

    device = raw_points.device

    proj_flat = proj_points.reshape(-1, 5)
    assert proj_flat.shape[0] == raw_points.shape[0], (
        "For num_camera=1, proj_flat should align 1-1 with raw_points. "
        f"Got proj_flat={proj_flat.shape}, raw_points={raw_points.shape}."
    )

    points_xyc = proj_flat[:, [0, 1, 4]]
    valid = is_within_mask(points_xyc, masks_with_cam)
    valid = valid * proj_flat[:, 3:4]

    foreground_mask = valid.sum(dim=1) > 0
    if foreground_mask.sum() == 0:
        return None

    # assign foreground point to best mask using seg score as tie-breaker
    mask_scores = inst_labels_11[:, -1].clamp(min=0)
    weighted = valid.float() * mask_scores.unsqueeze(0)
    assignment = torch.argmax(weighted, dim=1)

    foreground_indices = foreground_mask.nonzero(as_tuple=False).reshape(-1)
    real_labels = inst_labels_11[assignment[foreground_indices]]
    real_points = torch.cat([raw_points[foreground_indices], real_labels], dim=1)

    cam_ids = masks_with_cam[:, -1].long()
    masks_flat = masks_with_cam[:, :-1].bool()
    masks_wh = masks_flat.reshape(-1, W, H)
    masks_dilated_wh = dilate_masks(masks_wh, mask_dilate)
    masks_dilated_flat = masks_dilated_wh.reshape(-1, W * H)

    p_all = raw_points[:, power_index]
    p_min, p_max = p_all.min(), p_all.max()
    p_denom = (p_max - p_min).clamp(min=1e-6)

    gauss_kernel = torch.from_numpy(
        gaussian_2d([gauss_shape, gauss_shape], sigma=gauss_sigma)
    ).to(device=device, dtype=torch.float32)

    virtual_list = []
    prob_map_all = torch.zeros((W, H), device=device, dtype=torch.float32)

    num_inst = masks_with_cam.shape[0]
    for inst_id in range(num_inst):
        inst_real_mask = foreground_mask & (assignment == inst_id)
        if inst_real_mask.sum() == 0:
            continue

        cam_id = int(cam_ids[inst_id].item())
        inst_real_idx = inst_real_mask.nonzero(as_tuple=False).reshape(-1)

        # parent real points
        real_xyz = raw_points[inst_real_idx, :3]
        r, theta, phi = cartesian_to_spherical(real_xyz)

        p = raw_points[inst_real_idx, power_index]
        p_norm = ((p - p_min) / p_denom).clamp(0.0, 1.0)

        sigma_theta = (
            deg2rad(torch.tensor(sigma_theta_min_deg, device=device))
            + deg2rad(torch.tensor(sigma_theta_range_deg, device=device)) * (r / range_ref)
            + deg2rad(torch.tensor(sigma_theta_power_deg, device=device)) * (1.0 - p_norm)
        )
        sigma_phi = (
            deg2rad(torch.tensor(sigma_phi_min_deg, device=device))
            + deg2rad(torch.tensor(sigma_phi_range_deg, device=device)) * (r / range_ref)
            + deg2rad(torch.tensor(sigma_phi_power_deg, device=device)) * (1.0 - p_norm)
        )
        sigma_max = deg2rad(torch.tensor(sigma_max_deg, device=device))
        sigma_theta = sigma_theta.clamp(min=1e-6, max=sigma_max)
        sigma_phi = sigma_phi.clamp(min=1e-6, max=sigma_max)

        # Gaussian image prior from projected real points that belong to this instance
        inst_mask_wh = masks_wh[inst_id]
        inst_proj_xy = proj_flat[inst_real_idx, :2].long()
        gauss_prior = build_instance_gaussian_prior(inst_mask_wh, inst_proj_xy, gauss_kernel)

        n_real = real_xyz.shape[0]
        dtheta = torch.randn((n_real, cand_per_real), device=device) * sigma_theta[:, None]
        dphi = torch.randn((n_real, cand_per_real), device=device) * sigma_phi[:, None]
        dtheta = dtheta.clamp(-delta_clip * sigma_theta[:, None], delta_clip * sigma_theta[:, None])
        dphi = dphi.clamp(-delta_clip * sigma_phi[:, None], delta_clip * sigma_phi[:, None])

        theta_p = theta[:, None] + dtheta
        phi_p = phi[:, None] + dphi
        r_p = r[:, None].expand_as(theta_p)
        cand_xyz = spherical_to_cartesian(r_p, theta_p, phi_p).reshape(-1, 3)

        # parent real index for each candidate
        parent_idx = inst_real_idx[:, None].expand(n_real, cand_per_real).reshape(-1)
        parent_depth = proj_flat[parent_idx, 2]

        # project candidates to image
        u, v, depth, proj_valid = project_points_single_camera(
            cand_xyz,
            transforms[cam_id],
            intrinsics[cam_id],
            H=H,
            W=W,
        )

        u_int = u.long().clamp(0, W - 1)
        v_int = v.long().clamp(0, H - 1)
        pix_flat_idx = u_int * H + v_int

        # mask consistency weight
        in_mask = masks_flat[inst_id, pix_flat_idx]
        in_mask_dil = masks_dilated_flat[inst_id, pix_flat_idx]
        w_mask = torch.where(
            in_mask,
            torch.ones_like(u),
            torch.where(in_mask_dil, torch.full_like(u, boundary_weight), torch.zeros_like(u)),
        )

        # parent-aware occlusion weight
        dd = (depth - parent_depth).clamp(min=0.0)
        w_occ = torch.exp(-lambda_occ * dd)

        # perturbation likelihood weight
        dtheta_f = dtheta.reshape(-1)
        dphi_f = dphi.reshape(-1)
        sigma_theta_f = sigma_theta[:, None].expand_as(dtheta).reshape(-1)
        sigma_phi_f = sigma_phi[:, None].expand_as(dphi).reshape(-1)
        w_pert = torch.exp(-0.5 * ((dtheta_f / sigma_theta_f) ** 2 + (dphi_f / sigma_phi_f) ** 2))

        # Gaussian prior from original VoD logic
        gauss_val = gauss_prior[u_int, v_int]
        w_gauss = (1.0 - gauss_alpha) + gauss_alpha * gauss_val

        w = w_mask * w_occ * w_pert * w_gauss * proj_valid.float()

        keep = w > min_weight
        if keep.sum() == 0:
            continue

        w_keep = w[keep]
        prob = w_keep + 1e-6
        sel_local = torch.multinomial(prob, num_virtual, replacement=True)

        keep_indices = keep.nonzero(as_tuple=False).reshape(-1)
        sel = keep_indices[sel_local]

        sel_xyz = cand_xyz[sel]
        sel_parent = parent_idx[sel]

        inst_label = inst_labels_11[inst_id].unsqueeze(0).repeat(num_virtual, 1)
        parent_feat4 = raw_points[sel_parent, 3:]
        virtual_points = torch.cat([sel_xyz, inst_label, parent_feat4], dim=1)
        virtual_list.append(virtual_points)

        # visualization/debug map: selection counts + support prior
        prob_map_all += gauss_prior
        prob_map_all.index_put_((u_int[sel], v_int[sel]), torch.ones((num_virtual,), device=device), accumulate=True)

    if len(virtual_list) == 0:
        return None

    virtual_all = torch.cat(virtual_list, dim=0)
    return virtual_all, real_points, foreground_indices, prob_map_all


@torch.no_grad()
def process_one_frame(info, predictor, data, args):
    all_cams_from_lidar = [info["calib"]["Tr_velo_to_cam"]]
    all_cams_intrinsic = [info["calib"]["P2"][:3, :3]]

    pts_name = info["point_cloud"]["lidar_idx"] + ".bin"
    pts_path = os.path.join(args.pts_root, pts_name)
    pts_np = read_file(pts_path)

    one_hot_labels = []
    for i in range(10):
        one_hot = torch.zeros(10, device="cuda:0", dtype=torch.float32)
        one_hot[i] = 1
        one_hot_labels.append(one_hot)
    one_hot_labels = torch.stack(one_hot_labels, dim=0)

    masks = []
    labels = []

    result = predictor.model(data[1:])

    for camera_id in range(args.num_camera):
        pred_label, score, pred_mask = postprocess(result[camera_id])

        cam_id_tensor = torch.tensor(camera_id, dtype=torch.float32, device="cuda:0").reshape(1, 1).repeat(pred_mask.shape[0], 1)
        pred_mask = torch.cat([pred_mask, cam_id_tensor], dim=1)

        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)

        masks.append(pred_mask)
        labels.append(transformed_labels)

    if len(masks) == 0:
        return None

    masks = torch.cat(masks, dim=0)
    labels = torch.cat(labels, dim=0)

    P = projectionV2(
        to_tensor(pts_np),
        to_batch_tensor(all_cams_from_lidar),
        to_batch_tensor(all_cams_intrinsic),
        H=H,
        W=W,
    )
    cam_ids = torch.arange(args.num_camera, dtype=torch.float32, device="cuda:0").reshape(args.num_camera, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, cam_ids], dim=-1)

    res = add_virtual_rapid(
        masks,
        labels,
        P,
        to_tensor(pts_np),
        num_virtual=args.rapid_num_virtual,
        cand_per_real=args.rapid_cand_per_real,
        power_index=args.rapid_power_index,
        intrinsics=to_batch_tensor(all_cams_intrinsic),
        transforms=to_batch_tensor(all_cams_from_lidar),
        mask_dilate=args.rapid_mask_dilate,
        boundary_weight=args.rapid_boundary_weight,
        lambda_occ=args.rapid_lambda_occ,
        min_weight=args.rapid_min_weight,
        range_ref=args.rapid_range_ref,
        sigma_theta_min_deg=args.rapid_sigma_theta_min_deg,
        sigma_theta_range_deg=args.rapid_sigma_theta_range_deg,
        sigma_theta_power_deg=args.rapid_sigma_theta_power_deg,
        sigma_phi_min_deg=args.rapid_sigma_phi_min_deg,
        sigma_phi_range_deg=args.rapid_sigma_phi_range_deg,
        sigma_phi_power_deg=args.rapid_sigma_phi_power_deg,
        sigma_max_deg=args.rapid_sigma_max_deg,
        delta_clip=args.rapid_delta_clip,
        gauss_shape=args.rapid_gauss_shape,
        gauss_sigma=args.rapid_gauss_sigma,
        gauss_alpha=args.rapid_gauss_alpha,
        num_camera=args.num_camera,
    )

    if res is None:
        return None

    virtual_points, real_points, foreground_indices, prob_map_all = res
    return (
        virtual_points.cpu().numpy(),
        real_points.cpu().numpy(),
        foreground_indices.cpu().numpy(),
        prob_map_all.T.cpu().numpy(),
    )



def simple_collate(batch_list):
    assert len(batch_list) == 1
    return batch_list[0]



def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    os.makedirs(args.pts_save_path, exist_ok=True)

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    predictor = demo.predictor

    data_loader = DataLoader(
        DatasetVoD(args.info_path, predictor, args.image_root),
        batch_size=1,
        num_workers=1,
        collate_fn=simple_collate,
        pin_memory=False,
        shuffle=False,
    )

    for _, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue

        info = data[0]
        output_name = info["image"]["image_idx"] + ".pkl.npy"
        output_path = os.path.join(args.pts_save_path, output_name)

        res = process_one_frame(info, predictor, data, args)

        if res is not None:
            virtual_points, real_points, _, _ = res
        else:
            virtual_points = np.zeros((0, 18), dtype=np.float32)
            real_points = np.zeros((0, 18), dtype=np.float32)

        # Keep SAME FINAL FORMAT as original VoD script
        # virtual_points: [xyz(3)] + [label(11)] + [feat4(4)] => 18
        # final save: xyz(3) + feat4(4) + label_first8(8) => 15
        virtual_points_result = np.concatenate(
            [virtual_points[:, :3], virtual_points[:, -4:], virtual_points[:, 3:11]], axis=1
        ).astype(np.float32)

        # real_points: raw7 + label11 => 18
        # final save: raw7 + label_first8 => 15
        real_points_result = real_points[:, :15].astype(np.float32)

        data_dict = {
            "virtual_points": virtual_points_result,
            "real_points": real_points_result,
        }
        np.save(output_path, data_dict)


if __name__ == "__main__":
    main()