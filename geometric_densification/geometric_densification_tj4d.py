# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import argparse
import multiprocessing as mp
import os
import sys
import tempfile

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

# Size of the dataset image
H = 810
W = 1280


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
        default="/home/ccc/xx/base/HGSFusion/Mask2Former/configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml",
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
        default="data/tj4d/kitti_infos_trainval.pkl",
        help="Path to TJ4D info pkl.",
    )
    parser.add_argument(
        "--image-root",
        default="./data/tj4d/training/image_2/",
        help="Path to images.",
    )
    parser.add_argument(
        "--pts-root",
        default="./data/tj4d/training/velodyne/",
        help="Path to point cloud bin.",
    )
    parser.add_argument(
        "--num-camera",
        type=int,
        default=1,
        help="Number of cameras used (this script is mainly tested with 1).",
    )

    # RAPID hyper-params
    parser.add_argument("--rapid-num-virtual", type=int, default=100, help="Virtual points per instance.")
    parser.add_argument("--rapid-cand-per-real", type=int, default=16, help="Candidates per real foreground point.")
    parser.add_argument("--rapid-power-index", type=int, default=3, help="Index of echo power in raw point (default: 3).")

    parser.add_argument("--rapid-range-ref", type=float, default=50.0, help="Reference range (meters) for sigma scaling.")

    parser.add_argument("--rapid-sigma-theta-min-deg", type=float, default=0.8, help="Min azimuth sigma in degrees.")
    parser.add_argument("--rapid-sigma-theta-range-deg", type=float, default=1.2, help="Azimuth sigma scale for range.")
    parser.add_argument("--rapid-sigma-theta-power-deg", type=float, default=2.0, help="Azimuth sigma scale for low power.")

    parser.add_argument("--rapid-sigma-phi-min-deg", type=float, default=0.5, help="Min elevation sigma in degrees.")
    parser.add_argument("--rapid-sigma-phi-range-deg", type=float, default=0.8, help="Elevation sigma scale for range.")
    parser.add_argument("--rapid-sigma-phi-power-deg", type=float, default=1.5, help="Elevation sigma scale for low power.")

    parser.add_argument("--rapid-sigma-max-deg", type=float, default=8.0, help="Clamp max sigma (deg).")
    parser.add_argument("--rapid-delta-clip", type=float, default=3.0, help="Clip delta to +- delta_clip * sigma.")

    parser.add_argument("--rapid-mask-dilate", type=int, default=3, help="Mask tolerance (pixels) for w_mask.")
    parser.add_argument("--rapid-boundary-weight", type=float, default=0.2, help="Weight if only inside dilated mask.")

    parser.add_argument("--rapid-lambda-occ", type=float, default=0.15, help="Occlusion decay lambda for w_occ.")
    parser.add_argument("--rapid-min-weight", type=float, default=1e-6, help="Min final weight to keep candidate.")

    return parser


class DatasetTJ4D(Dataset):
    def __init__(self, info_path: str, predictor, image_root: str):
        self.sweeps = get_obj(info_path)
        self.predictor = predictor
        self.image_root = image_root

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]

        img_name = info["image"]["image_idx"] + ".png"
        img_path = os.path.join(self.image_root, img_name)

        original_image = cv2.imread(img_path)
        if original_image is None:
            # skip broken sample
            return []

        original_image = original_image[:H]

        if self.predictor.input_format == "RGB":
            # model expects RGB
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
    return np.fromfile(path, dtype=np.float32).reshape(-1, 8)


def postprocess(res):
    """Extract labels/scores/masks from Detectron2 Instance outputs.

    Output masks are reshaped to (N_inst, W*H) with (x,y) indexing order
    matching the original code.
    """
    result = res["instances"]
    labels = result.pred_classes
    scores = result.scores
    masks = result.pred_masks.reshape(scores.shape[0], W * H)

    # remove empty masks
    empty_mask = masks.sum(dim=1) == 0
    labels = labels[~empty_mask]
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]

    # (N, H, W) -> (N, W, H) -> (N, W*H)
    masks = masks.reshape(-1, H, W).permute(0, 2, 1).reshape(-1, W * H)
    return labels, scores, masks


def is_within_mask(points_xyc: torch.Tensor, masks_with_cam: torch.Tensor, H=H, W=W) -> torch.Tensor:
    """Check if each projected point falls inside each instance mask (camera-consistent).

    points_xyc: (N,3) int/long, columns [x(u), y(v), camera_id]
    masks_with_cam: (M, W*H+1) last column is camera_id

    returns: (N, M) bool
    """
    seg_mask = masks_with_cam[:, :-1].reshape(-1, W, H)  # (M,W,H)
    camera_id = masks_with_cam[:, -1]
    pts = points_xyc.long()
    valid = seg_mask[:, pts[:, 0], pts[:, 1]] * (camera_id[:, None] == pts[:, -1][None])
    return valid.transpose(1, 0)


def deg2rad(x: torch.Tensor) -> torch.Tensor:
    return x * np.pi / 180.0


def cartesian_to_spherical(xyz: torch.Tensor):
    """xyz -> (r, theta, phi)

    theta: azimuth in x-y plane
    phi: elevation
    """
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
    """Project lidar points to a single pinhole camera.

    Returns:
        u, v: float pixel coords
        depth: cam-z depth
        valid: bool mask within image and depth>0
    """
    device = xyz.device
    dtype = xyz.dtype

    ones = torch.ones((xyz.shape[0], 1), device=device, dtype=dtype)
    xyz1 = torch.cat([xyz, ones], dim=1)  # (N,4)

    cam = (Tr_velo_to_cam @ xyz1.t()).t()[:, :3]  # (N,3)
    depth = cam[:, 2]

    # pinhole projection
    uvw = (K @ cam.t()).t()  # (N,3)
    u = uvw[:, 0] / (uvw[:, 2] + 1e-6)
    v = uvw[:, 1] / (uvw[:, 2] + 1e-6)

    valid = depth > 1e-5
    valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    return u, v, depth, valid


def dilate_masks(masks_wh: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation using max-pooling.

    masks_wh: (M, W, H) bool/float
    returns: (M, W, H) bool
    """
    if radius <= 0:
        return masks_wh.bool()
    x = masks_wh.float().unsqueeze(1)  # (M,1,W,H)
    x = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (x.squeeze(1) > 0.5)


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
    num_camera: int = 1,
):
    """RAPID: radar-angle perturbation + image-driven posterior filtering.

    Inputs keep the same conventions as the original script:
      - masks_with_cam: (M, W*H+1)
      - inst_labels_11: (M, 11)  -> onehot(10) + seg_score(1)
      - proj_points:    (C, N, 5)-> [u, v, depth, valid, cam_id]
      - raw_points:     (N, 8)   -> [x,y,z, feat1..feat5]

    Returns (same structure as the original add_virtual_mask):
      virtual_points: (Nv, 19) -> [x,y,z] + inst_labels_11 + raw_feat5
      real_points:    (Nr, 19) -> raw_points8 + inst_labels_11
      foreground_indices: (Nr,)
      prob_map_all: (W,H) float, visualization map (counts)
    """

    # This code path is primarily used with one camera.
    if num_camera != 1:
        raise NotImplementedError(
            "This RAPID implementation is currently intended for num_camera=1 (TJ4D single cam setup)."
        )

    device = raw_points.device

    # Flatten projected points
    proj_flat = proj_points.reshape(-1, 5)  # (N,5)
    assert proj_flat.shape[0] == raw_points.shape[0], (
        "For num_camera=1, proj_flat should align 1-1 with raw_points. "
        f"Got proj_flat={proj_flat.shape}, raw_points={raw_points.shape}."
    )

    # (u, v, cam_id)
    points_xyc = proj_flat[:, [0, 1, 4]]

    # point-in-mask matrix
    valid = is_within_mask(points_xyc, masks_with_cam)  # (N, M)
    valid = valid * proj_flat[:, 3:4]  # apply image-FOV valid flag

    foreground_mask = (valid.sum(dim=1) > 0)
    if foreground_mask.sum() == 0:
        return None

    # Assign each foreground point to the best mask.
    # Use segmentation score as a soft tie-breaker to reduce random overlap assignment.
    mask_scores = inst_labels_11[:, -1].clamp(min=0)  # (M,)
    weighted = valid.float() * mask_scores.unsqueeze(0)
    assignment = torch.argmax(weighted, dim=1)  # (N,)

    foreground_indices = foreground_mask.nonzero(as_tuple=False).reshape(-1)

    # Real points output (same as original: keep only points inside any instance mask)
    real_labels = inst_labels_11[assignment[foreground_indices]]
    real_points = torch.cat([raw_points[foreground_indices], real_labels], dim=1)  # (Nr, 19)

    # Prepare masks (remove camera id) and optional dilation for tolerance
    cam_ids = masks_with_cam[:, -1].long()  # (M,)
    masks_flat = masks_with_cam[:, :-1].bool()  # (M, W*H)
    masks_wh = masks_flat.reshape(-1, W, H)  # (M, W, H)

    masks_dilated_wh = dilate_masks(masks_wh, mask_dilate)
    masks_dilated_flat = masks_dilated_wh.reshape(-1, W * H)

    # Power normalization for adaptive sigma
    # (if power feature is constant, fall back to 0.5)
    p_all = raw_points[:, power_index]
    p_min, p_max = p_all.min(), p_all.max()
    p_denom = (p_max - p_min).clamp(min=1e-6)

    # Output containers
    virtual_list = []
    prob_map_all = torch.zeros((W, H), device=device, dtype=torch.float32)

    # For each instance, generate candidates from its real points
    num_inst = masks_with_cam.shape[0]
    for inst_id in range(num_inst):
        # indices of real points belonging to this instance
        inst_real_mask = foreground_mask & (assignment == inst_id)
        if inst_real_mask.sum() == 0:
            continue

        # Instance camera id
        cam_id = int(cam_ids[inst_id].item())
        if cam_id != 0:
            # TJ4D single camera: cam_id should be 0
            # Keep this branch for safety.
            pass

        inst_real_idx = inst_real_mask.nonzero(as_tuple=False).reshape(-1)

        # Surface depth approximation (minimum depth among real points)
        surface_depth = proj_flat[inst_real_idx, 2].min()  # scalar

        # Real xyz and power
        real_xyz = raw_points[inst_real_idx, :3]
        r, theta, phi = cartesian_to_spherical(real_xyz)

        p = raw_points[inst_real_idx, power_index]
        p_norm = (p - p_min) / p_denom
        p_norm = p_norm.clamp(0.0, 1.0)

        # Adaptive sigma: increases with range, increases when power decreases
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

        n_real = real_xyz.shape[0]
        # sample perturbations
        dtheta = torch.randn((n_real, cand_per_real), device=device) * sigma_theta[:, None]
        dphi = torch.randn((n_real, cand_per_real), device=device) * sigma_phi[:, None]

        # clip deltas for stability
        dtheta = dtheta.clamp(-delta_clip * sigma_theta[:, None], delta_clip * sigma_theta[:, None])
        dphi = dphi.clamp(-delta_clip * sigma_phi[:, None], delta_clip * sigma_phi[:, None])

        theta_p = theta[:, None] + dtheta
        phi_p = phi[:, None] + dphi
        r_p = r[:, None].expand_as(theta_p)

        cand_xyz = spherical_to_cartesian(r_p, theta_p, phi_p).reshape(-1, 3)

        # parent real index for each candidate
        parent_idx = inst_real_idx[:, None].expand(n_real, cand_per_real).reshape(-1)

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

        # w_mask: inside mask -> 1, inside dilated only -> boundary_weight, else -> 0
        in_mask = masks_flat[inst_id, pix_flat_idx]
        in_mask_dil = masks_dilated_flat[inst_id, pix_flat_idx]
        w_mask = torch.where(
            in_mask,
            torch.ones_like(u),
            torch.where(in_mask_dil, torch.full_like(u, boundary_weight), torch.zeros_like(u)),
        )

        # w_occ: penalize candidates behind surface depth
        dd = (depth - surface_depth).clamp(min=0.0)
        w_occ = torch.exp(-lambda_occ * dd)

        # w_pert: likelihood of perturbation (Gaussian)
        dtheta_f = dtheta.reshape(-1)
        dphi_f = dphi.reshape(-1)
        sigma_theta_f = sigma_theta[:, None].expand_as(dtheta).reshape(-1)
        sigma_phi_f = sigma_phi[:, None].expand_as(dphi).reshape(-1)
        w_pert = torch.exp(-0.5 * ((dtheta_f / sigma_theta_f) ** 2 + (dphi_f / sigma_phi_f) ** 2))

        # final weight
        w = w_mask * w_occ * w_pert * proj_valid.float()

        keep = w > min_weight
        if keep.sum() == 0:
            continue

        w_keep = w[keep]
        # Sampling distribution (avoid all-zero)
        prob = w_keep + 1e-6
        sel_local = torch.multinomial(prob, num_virtual, replacement=True)

        keep_indices = keep.nonzero(as_tuple=False).reshape(-1)
        sel = keep_indices[sel_local]

        sel_xyz = cand_xyz[sel]
        sel_parent = parent_idx[sel]

        # Inherit instance semantic label + parent radar features
        inst_label = inst_labels_11[inst_id].unsqueeze(0).repeat(num_virtual, 1)
        parent_feat5 = raw_points[sel_parent, 3:]

        virtual_points = torch.cat([sel_xyz, inst_label, parent_feat5], dim=1)  # (num_virtual, 19)
        virtual_list.append(virtual_points)

        # update prob map (for debugging/visualization)
        prob_map_all.index_put_((u_int[sel], v_int[sel]), torch.ones((num_virtual,), device=device), accumulate=True)

    if len(virtual_list) == 0:
        return None

    virtual_all = torch.cat(virtual_list, dim=0)
    return virtual_all, real_points, foreground_indices, prob_map_all


@torch.no_grad()
def process_one_frame(info, predictor, data, args):
    # camera calibration
    all_cams_from_lidar = [info["calib"]["Tr_velo_to_cam"]]  # 4x4
    all_cams_intrinsic = [info["calib"]["P2"][:3, :3]]  # 3x3

    # load point cloud
    pts_name = info["point_cloud"]["lidar_idx"] + ".bin"
    pts_path = os.path.join(args.pts_root, pts_name)
    pts_np = read_file(pts_path)

    # instance segmentation
    one_hot_labels = []
    for i in range(10):
        one_hot = torch.zeros(10, device="cuda:0", dtype=torch.float32)
        one_hot[i] = 1
        one_hot_labels.append(one_hot)
    one_hot_labels = torch.stack(one_hot_labels, dim=0)  # (10,10)

    masks = []
    labels = []

    result = predictor.model(data[1:])

    for camera_id in range(args.num_camera):
        pred_label, score, pred_mask = postprocess(result[camera_id])

        cam_id_tensor = torch.tensor(camera_id, dtype=torch.float32, device="cuda:0").reshape(1, 1).repeat(pred_mask.shape[0], 1)
        pred_mask = torch.cat([pred_mask, cam_id_tensor], dim=1)  # (M, W*H+1)

        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)  # (M,11)

        masks.append(pred_mask)
        labels.append(transformed_labels)

    if len(masks) == 0:
        return None

    masks = torch.cat(masks, dim=0)
    labels = torch.cat(labels, dim=0)

    # project real points to image
    P = projectionV2(
        to_tensor(pts_np),
        to_batch_tensor(all_cams_from_lidar),
        to_batch_tensor(all_cams_intrinsic),
        H=H,
        W=W,
    )
    cam_ids = torch.arange(args.num_camera, dtype=torch.float32, device="cuda:0").reshape(args.num_camera, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, cam_ids], dim=-1)  # (C,N,5)

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
        DatasetTJ4D(args.info_path, predictor, args.image_root),
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
            virtual_points, real_points, indices, _ = res
        else:
            # Keep shapes consistent with RAPID output: (N,19)
            virtual_points = np.zeros((0, 19), dtype=np.float32)
            real_points = np.zeros((0, 19), dtype=np.float32)
            indices = np.zeros((0,), dtype=np.int64)

        # ---- IMPORTANT: Keep the SAME FINAL FORMAT as the original script ----
        # virtual_points: [xyz(3)] + [label(11)] + [feat5(5)] => 19
        # final save: xyz(3) + feat5(5) + label_first8(8) => 16
        virtual_points_result = np.concatenate(
            [virtual_points[:, :3], virtual_points[:, -5:], virtual_points[:, 3:11]], axis=1
        ).astype(np.float32)

        # real_points: raw8 + label11 => 19
        # final save: raw8 + label_first8 => 16
        real_points_result = real_points[:, :16].astype(np.float32)

        data_dict = {
            "virtual_points": virtual_points_result,
            "real_points": real_points_result,
        }

        np.save(output_path, data_dict)


if __name__ == "__main__":
    main()
