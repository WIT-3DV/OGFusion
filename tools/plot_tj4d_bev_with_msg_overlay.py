#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_IDS = [
    "000017", "000069",
    "300002", "300012", "340089", "410010"
]


@dataclass
class Object3D:
    cls_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox2d: np.ndarray
    hwl: np.ndarray
    loc: np.ndarray
    ry: float
    score: float = 1.0


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def parse_label_file(label_path: str,
                     score_thr: float = 0.0,
                     allowed_classes: Optional[List[str]] = None) -> List[Object3D]:
    objs = []
    if not os.path.exists(label_path):
        return objs

    with open(label_path, "r") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 15:
            continue

        cls_type = parts[0]
        if allowed_classes is not None and cls_type not in allowed_classes:
            continue

        truncation = _safe_float(parts[1], 0.0)
        occlusion = int(_safe_float(parts[2], 0))
        alpha = normalize_angle(_safe_float(parts[3], 0.0))
        bbox = np.array([_safe_float(v) for v in parts[4:8]], dtype=np.float32)
        h, w, l = [_safe_float(v) for v in parts[8:11]]
        x, y, z = [_safe_float(v) for v in parts[11:14]]
        ry = normalize_angle(_safe_float(parts[14], 0.0))
        score = _safe_float(parts[15], 1.0) if len(parts) > 15 else 1.0

        if score < score_thr:
            continue

        objs.append(
            Object3D(
                cls_type=cls_type,
                truncation=truncation,
                occlusion=occlusion,
                alpha=alpha,
                bbox2d=bbox,
                hwl=np.array([h, w, l], dtype=np.float32),
                loc=np.array([x, y, z], dtype=np.float32),
                ry=ry,
                score=score,
            )
        )
    return objs


def read_calib_file(calib_path: str) -> Dict[str, np.ndarray]:
    data = {}
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = np.array([float(x) for x in value.strip().split()], dtype=np.float32)
            data[key] = vals

    def reshape(vals: np.ndarray, rows: int, cols: int) -> np.ndarray:
        return vals.reshape(rows, cols)

    calib = {}

    if "P2" in data:
        calib["P2"] = reshape(data["P2"], 3, 4)
    elif "P_rect_02" in data:
        calib["P2"] = reshape(data["P_rect_02"], 3, 4)
    else:
        raise KeyError(f"Cannot find P2/P_rect_02 in {calib_path}")

    if "R0_rect" in data:
        calib["R0_rect"] = reshape(data["R0_rect"], 3, 3)
    elif "R_rect" in data:
        calib["R0_rect"] = reshape(data["R_rect"], 3, 3)
    elif "R_rect_00" in data:
        calib["R0_rect"] = reshape(data["R_rect_00"], 3, 3)
    else:
        calib["R0_rect"] = np.eye(3, dtype=np.float32)

    if "Tr_velo_to_cam" in data:
        calib["Tr_velo_to_cam"] = reshape(data["Tr_velo_to_cam"], 3, 4)
    elif "Tr_velo_cam" in data:
        calib["Tr_velo_to_cam"] = reshape(data["Tr_velo_cam"], 3, 4)
    elif "Tr_velo2cam" in data:
        calib["Tr_velo_to_cam"] = reshape(data["Tr_velo2cam"], 3, 4)
    else:
        raise KeyError(f"Cannot find Tr_velo_to_cam-like key in {calib_path}")

    return calib


def cart_to_hom(pts: np.ndarray) -> np.ndarray:
    return np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)


def lidar_to_rect(pts_lidar_xyz: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    tr = calib["Tr_velo_to_cam"]
    r0 = calib["R0_rect"]
    pts_h = cart_to_hom(pts_lidar_xyz[:, :3])
    pts_cam = pts_h @ tr.T
    pts_rect = pts_cam @ r0.T
    return pts_rect


def boxes3d_camera_corners(obj: Object3D) -> np.ndarray:
    h, w, l = obj.hwl
    x, y, z = obj.loc
    ry = obj.ry

    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float32)
    y_corners = np.array([   0,    0,    0,    0,   -h,   -h,   -h,   -h], dtype=np.float32)
    z_corners = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float32)

    c = math.cos(ry)
    s = math.sin(ry)
    rot = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)

    corners = np.stack([x_corners, y_corners, z_corners], axis=0)
    corners = (rot @ corners).T
    corners += np.array([x, y, z], dtype=np.float32)[None, :]
    return corners


def box3d_bev_polygon(obj: Object3D) -> np.ndarray:
    corners3d = boxes3d_camera_corners(obj)
    return corners3d[[0, 1, 2, 3], :][:, [0, 2]]


def load_radar_bin(bin_path: str) -> np.ndarray:
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Point file not found: {bin_path}")

    pts = np.fromfile(bin_path, dtype=np.float32)

    if pts.size % 8 == 0:
        pts = pts.reshape(-1, 8)
    elif pts.size % 5 == 0:
        pts = pts.reshape(-1, 5)
    elif pts.size % 4 == 0:
        pts = pts.reshape(-1, 4)
    else:
        raise ValueError(f"Unexpected point format in {bin_path}, total floats={pts.size}")

    return pts


def resolve_paths(data_root: str, sample_id: str) -> Tuple[str, str]:
    calib_path = os.path.join(data_root, "calib", f"{sample_id}.txt")
    radar_path = os.path.join(data_root, "velodyne", f"{sample_id}.bin")
    return calib_path, radar_path


def load_msg_overlay(msg_map_path: Optional[str]) -> Optional[np.ndarray]:
    if msg_map_path is None or not os.path.exists(msg_map_path):
        return None

    img = cv2.imread(msg_map_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def transform_msg_img(msg_img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return msg_img
    elif mode == "transpose":
        return np.transpose(msg_img, (1, 0, 2))
    elif mode == "flipud":
        return np.flipud(msg_img)
    elif mode == "fliplr":
        return np.fliplr(msg_img)
    elif mode == "transpose_flipud":
        return np.flipud(np.transpose(msg_img, (1, 0, 2)))
    elif mode == "transpose_fliplr":
        return np.fliplr(np.transpose(msg_img, (1, 0, 2)))
    else:
        raise ValueError(f"Unsupported msg transform mode: {mode}")


def save_bev_overlay_view(
    calib_path: str,
    radar_path: str,
    gt_objs: List[Object3D],
    base_objs: List[Object3D],
    ours_objs: List[Object3D],
    save_path: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    point_size: float = 0.8,
    gt_line_width: float = 1.5,
    baseline_line_width: float = 1.5,
    ours_line_width: float = 1.5,
    fig_w: float = 8.0,
    fig_h: float = 8.0,
    dpi: int = 220,
    msg_map_path: Optional[str] = None,
    msg_alpha: float = 0.38,
    msg_transform: str = "transpose_flipud",
):
    calib = read_calib_file(calib_path)
    pts = load_radar_bin(radar_path)
    pts_rect = lidar_to_rect(pts[:, :3], calib)

    px = pts_rect[:, 0]
    py = pts_rect[:, 2]
    mask = (px >= x_range[0]) & (px <= x_range[1]) & (py >= y_range[0]) & (py <= y_range[1])
    pts_plot = pts_rect[mask]

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    msg_img = load_msg_overlay(msg_map_path)
    if msg_img is not None:
        msg_img = transform_msg_img(msg_img, msg_transform)
        ax.imshow(
            msg_img,
            origin='lower',
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            alpha=msg_alpha,
            aspect='auto'
        )

    if pts_plot.shape[0] > 0:
        ax.scatter(
            pts_plot[:, 0],
            pts_plot[:, 2],
            s=point_size,
            c="blue",
            alpha=0.75,
            linewidths=0
        )

    ego_w = 1.8
    ego_l = 4.2
    ego_rect = np.array([
        [-ego_w / 2, 0.0],
        [ ego_w / 2, 0.0],
        [ ego_w / 2, ego_l],
        [-ego_w / 2, ego_l],
        [-ego_w / 2, 0.0],
    ])
    ax.plot(ego_rect[:, 0], ego_rect[:, 1], color='black', linewidth=1.5)
    ax.arrow(0, ego_l, 0, 2.5, head_width=0.5, head_length=0.8,
             fc='black', ec='black', length_includes_head=True)

    def _plot_objs(objs: List[Object3D], color: str, lw: float, label: str):
        first = True
        for obj in objs:
            corners = box3d_bev_polygon(obj)
            poly = np.vstack([corners, corners[0]])
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=lw, label=label if first else None)
            center = corners.mean(axis=0)
            front = (corners[0] + corners[1]) / 2.0
            ax.plot([center[0], front[0]], [center[1], front[1]], color=color, linewidth=lw)
            first = False

    _plot_objs(gt_objs, 'lime', gt_line_width, 'GT')
    _plot_objs(base_objs, 'red', baseline_line_width, 'Baseline')
    _plot_objs(ours_objs, 'yellow', ours_line_width, 'Ours')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.35)

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor='lime', linewidth=gt_line_width, label='GT'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=baseline_line_width, label='Baseline'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor='yellow', linewidth=ours_line_width, label='Ours'),
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Overlay OG-WCA message maps on TJ4D BEV detection plots")

    parser.add_argument("--data-root", type=str,
                        default="/home/ccc/xx/base/HGSFusion/data/tj4d/training")
    parser.add_argument("--gt-dir", type=str,
                        default="/home/ccc/xx/base/HGSFusion/data/tj4d/training/label_2")
    parser.add_argument("--baseline-dir", type=str,
                        default="/home/ccc/xx/base/HGSFusion/output_tj4dtest/tools/cfgs/hgsfusion/hgsfusion_tj4d/default/eval/epoch_4/val/default/final_result/data")
    parser.add_argument("--ours-dir", type=str,
                        default="/home/ccc/xx/base/HGSFusion/output2.04___pillarnet+windows_38.95/tools/cfgs/hgsfusion/hgsfusion_tj4d/default/eval/eval_with_train/epoch_21/val/final_result/data")

    parser.add_argument("--msg-map-dir", type=str,
                        default="/home/ccc/xx/base/HGSFusion/output/tools/cfgs/hgsfusion/hgsfusion_tj4d/default/vis/bev_msg/test",
                        help="directory containing *_msg_map.png generated by vis_bev_attn.py")

    parser.add_argument("--save-dir", type=str,
                        default="/home/ccc/xx/base/HGSFusion/vis_overlay_msg_transform")

    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--classes", nargs="*", default=None)

    parser.add_argument("--bev-x-range", type=float, nargs=2, default=[-30.0, 30.0])
    parser.add_argument("--bev-y-range", type=float, nargs=2, default=[0.0, 80.0])
    parser.add_argument("--bev-point-size", type=float, default=0.8)
    parser.add_argument("--bev-gt-line-width", type=float, default=1.2)
    parser.add_argument("--bev-baseline-line-width", type=float, default=1.2)
    parser.add_argument("--bev-ours-line-width", type=float, default=1.2)
    parser.add_argument("--bev-fig-w", type=float, default=8.0)
    parser.add_argument("--bev-fig-h", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=220)

    parser.add_argument("--msg-alpha", type=float, default=0.38,
                        help="overlay alpha for message map")
    parser.add_argument("--msg-transform", type=str, default="transpose_flipud",
                        choices=[
                            "none",
                            "transpose",
                            "flipud",
                            "fliplr",
                            "transpose_flipud",
                            "transpose_fliplr"
                        ],
                        help="transform mode applied to msg_map before overlay")

    args = parser.parse_args()

    save_bev = os.path.join(args.save_dir, f"bev_overlay_{args.msg_transform}")
    os.makedirs(save_bev, exist_ok=True)

    print(f"[Info] msg transform mode: {args.msg_transform}")
    print(f"[Info] save dir: {save_bev}")

    for idx, sid in enumerate(args.ids):
        calib_path, radar_path = resolve_paths(args.data_root, sid)

        gt_path = os.path.join(args.gt_dir, f"{sid}.txt")
        baseline_path = os.path.join(args.baseline_dir, f"{sid}.txt")
        ours_path = os.path.join(args.ours_dir, f"{sid}.txt")
        msg_map_path = os.path.join(args.msg_map_dir, f"{sid}_msg_map.png")

        gt_objs = parse_label_file(gt_path, score_thr=args.score_thr, allowed_classes=args.classes)
        base_objs = parse_label_file(baseline_path, score_thr=args.score_thr, allowed_classes=args.classes)
        ours_objs = parse_label_file(ours_path, score_thr=args.score_thr, allowed_classes=args.classes)

        try:
            save_bev_overlay_view(
                calib_path=calib_path,
                radar_path=radar_path,
                gt_objs=gt_objs,
                base_objs=base_objs,
                ours_objs=ours_objs,
                save_path=os.path.join(save_bev, f"{sid}.png"),
                x_range=tuple(args.bev_x_range),
                y_range=tuple(args.bev_y_range),
                point_size=args.bev_point_size,
                gt_line_width=args.bev_gt_line_width,
                baseline_line_width=args.bev_baseline_line_width,
                ours_line_width=args.bev_ours_line_width,
                fig_w=args.bev_fig_w,
                fig_h=args.bev_fig_h,
                dpi=args.dpi,
                msg_map_path=msg_map_path if os.path.exists(msg_map_path) else None,
                msg_alpha=args.msg_alpha,
                msg_transform=args.msg_transform,
            )
            print(f"[{idx+1}/{len(args.ids)}] saved: {os.path.join(save_bev, f'{sid}.png')}")
        except Exception as e:
            print(f"[Warn] Failed on {sid}: {e}")

    print(f"[Done] Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()