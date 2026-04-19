#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import _init_path
import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


DEFAULT_IDS = ["000017", "000069", "300002", "300012", "340089", "410010"]


def parse_config():
    parser = argparse.ArgumentParser(description="Visualize global attention maps")

    parser.add_argument("--cfg_file", type=str,
                        default="./tools/cfgs/hgsfusion/hgsfusion_tj4d.yaml")
    parser.add_argument("--ckpt", type=str,
                        default="/home/ccc/xx/base/HGSFusion/output2.04___pillarnet+windows_38.95/tools/cfgs/hgsfusion/hgsfusion_tj4d/default/ckpt/checkpoint_epoch_21.pth")
    parser.add_argument("--pretrained_model", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra_tag", type=str, default="default")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18888)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--vis_tag", type=str, default="global_attn_maps")

    parser.add_argument("--target_ids", type=str, nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--max_vis_samples", type=int, default=20)

    parser.add_argument("--save_focus", action="store_true", default=True)
    parser.add_argument("--save_occ_pref", action="store_true", default=True)
    parser.add_argument("--save_attendedness", action="store_true", default=True)
    parser.add_argument("--save_occ", action="store_true", default=False)

    parser.add_argument("--occ_thr", type=float, default=0.3)
    parser.add_argument("--gaussian_sigma", type=float, default=1.2)
    parser.add_argument("--gaussian_kernel", type=int, default=9)

    parser.add_argument("--upsample", action="store_true", default=True)

    parser.add_argument("--x_min", type=float, default=-30.0)
    parser.add_argument("--x_max", type=float, default=30.0)
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--y_max", type=float, default=80.0)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def gaussian_kernel2d(kernel_size=9, sigma=1.2, device="cpu", dtype=torch.float32):
    assert kernel_size % 2 == 1
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_2d(x, kernel_size=9, sigma=1.2):
    if sigma <= 0:
        return x
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float()
    kernel = gaussian_kernel2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    x = x.unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, kernel, padding=kernel_size // 2)
    return x[0, 0]


def robust_normalize(x, low=5, high=99):
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    vmin = np.percentile(x, low)
    vmax = np.percentile(x, high)
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-6)
    return x


def maybe_upsample_map(x, target_hw):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if list(x.shape[-2:]) == list(target_hw):
        return x
    x = F.interpolate(
        x.unsqueeze(0).unsqueeze(0).float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False
    )[0, 0]
    return x


def save_bev_map(
    bev_map,
    save_path,
    x_range=(-30, 30),
    y_range=(0, 80),
    gaussian_sigma=1.2,
    gaussian_kernel=9,
    title=None,
):
    if torch.is_tensor(bev_map):
        bev_map = bev_map.detach().float()
        if gaussian_sigma > 0:
            bev_map = gaussian_blur_2d(bev_map, kernel_size=gaussian_kernel, sigma=gaussian_sigma)
        bev_map = bev_map.cpu().numpy()

    bev_map = robust_normalize(bev_map, low=5, high=99)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 8))
    plt.imshow(
        bev_map,
        cmap="Reds",
        origin="lower",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        aspect="auto",
        vmin=0,
        vmax=1
    )
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def build_attention_focus_map(att, h, w, eps=1e-6):
    """
    att: [HW, P]
    return: [h, w]
    越大表示注意力越集中
    """
    entropy = -(att * torch.log(att + eps)).sum(dim=-1)   # [HW]
    focus = -entropy
    return focus.view(h, w)


def build_occ_preference_map(att, occ, h, w, win, occ_thr=0.3):
    """
    att: [HW, P]
    occ: [h, w]
    return: [h, w]
    每个 query 对高 occupancy 单元的注意力总和
    """
    pad = win // 2
    occ_pad = F.pad(
        occ.unsqueeze(0).unsqueeze(0),
        (pad, pad, pad, pad),
        mode="constant",
        value=0
    )[0, 0]

    vals = []
    idx = 0
    for y in range(h):
        for x in range(w):
            patch = occ_pad[y:y+win, x:x+win].reshape(-1)  # [P]
            high_mask = (patch > occ_thr).float()
            high_att = (att[idx] * high_mask).sum()
            vals.append(high_att)
            idx += 1

    vals = torch.stack(vals, dim=0)
    return vals.view(h, w)


def build_global_attendedness_map(att, h, w, win):
    """
    att: [HW, P]
    return: [h, w]
    哪些位置整体更常被关注
    """
    pad = win // 2
    heat = torch.zeros((h, w), device=att.device, dtype=att.dtype)
    count = torch.zeros((h, w), device=att.device, dtype=att.dtype)

    idx = 0
    for cy in range(h):
        for cx in range(w):
            local_att = att[idx].view(win, win)
            idx += 1

            for dy in range(win):
                for dx in range(win):
                    yy = cy + dy - pad
                    xx = cx + dx - pad
                    if 0 <= yy < h and 0 <= xx < w:
                        heat[yy, xx] += local_att[dy, dx]
                        count[yy, xx] += 1.0

    heat = heat / count.clamp(min=1.0)
    return heat


def get_sample_name(batch_dict, batch_idx=0, fallback_idx=0):
    frame_id = batch_dict.get("frame_id", None)
    if frame_id is None:
        return f"sample_{fallback_idx:04d}"

    if isinstance(frame_id, (list, tuple)):
        return str(frame_id[batch_idx])

    if isinstance(frame_id, np.ndarray):
        return str(frame_id[batch_idx])

    return str(frame_id)


def should_keep_sample(sample_name, target_ids_set):
    if target_ids_set is None or len(target_ids_set) == 0:
        return True
    return sample_name in target_ids_set


def forward_model(model, batch_dict):
    load_data_to_gpu(batch_dict)
    _ = model(batch_dict)


def find_align_module(model):
    for name, module in model.named_modules():
        if hasattr(module, "last_att") and hasattr(module, "last_occ") and hasattr(module, "last_hw"):
            if hasattr(module, "win"):
                return name, module
    return None, None


def vis_one_batch(model, batch_dict, save_dir, logger, args, global_idx_start=0):
    forward_model(model, batch_dict)

    align_name, align_module = find_align_module(model)
    if align_module is None:
        raise RuntimeError("No align module with cached attention found in the model.")

    frame_id = batch_dict.get("frame_id", None)
    if isinstance(frame_id, (list, tuple, np.ndarray)):
        batch_size = len(frame_id)
    else:
        batch_size = 1

    target_ids_set = set(args.target_ids) if args.target_ids is not None else None
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    vis_count = 0
    h, w = align_module.last_hw
    input_hw = align_module.last_input_hw
    win = align_module.win

    for bidx in range(batch_size):
        sample_name = get_sample_name(batch_dict, batch_idx=bidx, fallback_idx=global_idx_start + vis_count)
        if not should_keep_sample(sample_name, target_ids_set):
            continue

        att = align_module.last_att[bidx]           # [HW, P]
        occ = align_module.last_occ[bidx, 0] if align_module.last_occ is not None else None  # [h, w]

        focus_map = None
        occ_pref_map = None
        attended_map = None

        if args.save_focus:
            focus_map = build_attention_focus_map(att, h, w)

        if args.save_occ_pref and occ is not None:
            occ_pref_map = build_occ_preference_map(att, occ, h, w, win, occ_thr=args.occ_thr)

        if args.save_attendedness:
            attended_map = build_global_attendedness_map(att, h, w, win)

        occ_map = occ

        if args.upsample and input_hw is not None:
            if focus_map is not None:
                focus_map = maybe_upsample_map(focus_map, input_hw)
            if occ_pref_map is not None:
                occ_pref_map = maybe_upsample_map(occ_pref_map, input_hw)
            if attended_map is not None:
                attended_map = maybe_upsample_map(attended_map, input_hw)
            if occ_map is not None:
                occ_map = maybe_upsample_map(occ_map, input_hw)

        sample_dir = os.path.join(save_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        if focus_map is not None:
            save_bev_map(
                focus_map,
                os.path.join(sample_dir, f"{sample_name}_focus_map.png"),
                x_range=x_range,
                y_range=y_range,
                gaussian_sigma=args.gaussian_sigma,
                gaussian_kernel=args.gaussian_kernel,
                title=None
            )
            logger.info(f"Saved focus map: {os.path.join(sample_dir, f'{sample_name}_focus_map.png')}")

        if occ_pref_map is not None:
            save_bev_map(
                occ_pref_map,
                os.path.join(sample_dir, f"{sample_name}_occ_pref_map.png"),
                x_range=x_range,
                y_range=y_range,
                gaussian_sigma=args.gaussian_sigma,
                gaussian_kernel=args.gaussian_kernel,
                title=None
            )
            logger.info(f"Saved occupancy preference map: {os.path.join(sample_dir, f'{sample_name}_occ_pref_map.png')}")

        if attended_map is not None:
            save_bev_map(
                attended_map,
                os.path.join(sample_dir, f"{sample_name}_attendedness_map.png"),
                x_range=x_range,
                y_range=y_range,
                gaussian_sigma=args.gaussian_sigma,
                gaussian_kernel=args.gaussian_kernel,
                title=None
            )
            logger.info(f"Saved attendedness map: {os.path.join(sample_dir, f'{sample_name}_attendedness_map.png')}")

        if args.save_occ and occ_map is not None:
            save_bev_map(
                occ_map,
                os.path.join(sample_dir, f"{sample_name}_occ_map.png"),
                x_range=x_range,
                y_range=y_range,
                gaussian_sigma=args.gaussian_sigma,
                gaussian_kernel=args.gaussian_kernel,
                title=None
            )
            logger.info(f"Saved occupancy map: {os.path.join(sample_dir, f'{sample_name}_occ_map.png')}")

        vis_count += 1

    return vis_count


def main():
    args, cfg = parse_config()

    if args.launcher == "none":
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, f"init_dist_{args.launcher}")(
            args.tcp_port, args.local_rank, backend="nccl"
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = 1
    else:
        assert args.batch_size % total_gpus == 0
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_dir is None:
        vis_output_dir = output_dir / "vis" / args.vis_tag / args.split
    else:
        vis_output_dir = Path(args.save_dir)

    vis_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = vis_output_dir / ("log_vis_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info("********************** Start global attention visualization **********************")
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "ALL"
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:20} {}".format(key, val))
    log_config_to_file(cfg, logger=logger)

    try:
        cfg.MODEL.EXPORT_VIS = True
    except Exception:
        pass

    dataset_cfg = cfg.DATA_CONFIG
    if hasattr(dataset_cfg, "DATA_SPLIT") and args.split in dataset_cfg.DATA_SPLIT:
        dataset_cfg.DATA_SPLIT["test"] = dataset_cfg.DATA_SPLIT[args.split]

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=dist_test,
        pre_trained_path=args.pretrained_model
    )
    model.cuda()
    model.eval()

    target_ids_set = set(args.target_ids) if args.target_ids is not None else None
    logger.info(f"Target IDs: {sorted(list(target_ids_set)) if target_ids_set is not None else 'ALL'}")

    saved_num = 0
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(test_loader):
            cur_saved = vis_one_batch(
                model=model,
                batch_dict=batch_dict,
                save_dir=str(vis_output_dir),
                logger=logger,
                args=args,
                global_idx_start=saved_num
            )
            saved_num += cur_saved

            if target_ids_set is not None and saved_num >= len(target_ids_set):
                break
            if saved_num >= args.max_vis_samples:
                break

    logger.info(f"Visualization finished. Saved {saved_num} sample(s) to: {vis_output_dir}")


if __name__ == "__main__":
    main()