import _init_path
import argparse
from pathlib import Path
import sys
from pathlib import Path

this_dir = Path(__file__).resolve().parent
root_dir = this_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='visualize attention by given sample ids')
    parser.add_argument('--cfg_file', type=str,
                        default='./tools/cfgs/hgsfusion/hgsfusion_vod.yaml',
                        help='config file')
    parser.add_argument('--ckpt', type=str,
                        default='./output/tools/cfgs/hgsfusion/hgsfusion_vod/default/ckpt/checkpoint_epoch_25.pth',
                        help='checkpoint path')
    parser.add_argument('--pretrained_model', type=str, default=None, help='optional pretrained model path')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=1, help='recommend 1 for visualization')
    parser.add_argument('--save_dir', type=str, default='./output/vis_attn_by_id', help='save directory')

    parser.add_argument('--target_ids', type=str, nargs='+', required=True,
                        help='sample ids to visualize, e.g. --target_ids 00001 00015 00023')
    parser.add_argument('--id_key', type=str, default='frame_id',
                        help='key in batch_dict for sample id, e.g. frame_id')
    parser.add_argument('--occ_key', type=str, default='radar_occ_prob',
                        help='occ key in batch_dict')

    parser.add_argument('--query_y', type=int, default=None, help='query y on attn map')
    parser.add_argument('--query_x', type=int, default=None, help='query x on attn map')
    parser.add_argument('--query_mode', type=str, default='center', choices=['center', 'max_occ'],
                        help='how to choose query point if query_y/query_x not given')

    # 平滑参数
    parser.add_argument('--smooth_sigma', type=float, default=1.2,
                        help='gaussian smoothing sigma for full bev heatmap')
    parser.add_argument('--smooth_kernel', type=int, default=9,
                        help='gaussian kernel size, should be odd')

    # 只是记录，当前版本优先用 gt_boxes 画框
    parser.add_argument('--label_dir', type=str,
                        default='/home/ccc/xx/base/HGSFusion/data/tj4d/training/label_2',
                        help='label directory (reserved). current code prefers batch_dict gt_boxes for BEV boxes')

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def find_module_by_type(root_module, target_type_name):
    for name, module in root_module.named_modules():
        if module.__class__.__name__ == target_type_name:
            return name, module
    return None, None


def to_numpy_id(x):
    if isinstance(x, bytes):
        return x.decode('utf-8')
    return str(x)


def get_batch_sample_ids(batch_dict, id_key):
    """
    从 batch_dict 中取出当前 batch 的样本 ID 列表
    """
    if id_key not in batch_dict:
        raise KeyError(
            f"Cannot find id_key='{id_key}' in batch_dict. "
            f"Available keys: {list(batch_dict.keys())}"
        )

    ids = batch_dict[id_key]

    if isinstance(ids, (list, tuple)):
        return [to_numpy_id(x) for x in ids]

    if isinstance(ids, np.ndarray):
        return [to_numpy_id(x) for x in ids.tolist()]

    if torch.is_tensor(ids):
        return [to_numpy_id(x) for x in ids.cpu().tolist()]

    return [to_numpy_id(ids)]


def choose_query_point(occ_map, h, w, query_mode='center', qy=None, qx=None):
    if qy is not None and qx is not None:
        qy = int(np.clip(qy, 0, h - 1))
        qx = int(np.clip(qx, 0, w - 1))
        return qy, qx

    if query_mode == 'center':
        return h // 2, w // 2

    if query_mode == 'max_occ' and occ_map is not None:
        idx = np.argmax(occ_map)
        qy, qx = np.unravel_index(idx, occ_map.shape)
        return int(qy), int(qx)

    return h // 2, w // 2


def project_local_attn_to_global(att_vec, h, w, win, qy, qx):
    local = att_vec.reshape(win, win)
    full = np.zeros((h, w), dtype=np.float32)
    pad = win // 2

    y0 = max(0, qy - pad)
    y1 = min(h, qy + pad + 1)
    x0 = max(0, qx - pad)
    x1 = min(w, qx + pad + 1)

    ly0 = y0 - (qy - pad)
    ly1 = ly0 + (y1 - y0)
    lx0 = x0 - (qx - pad)
    lx1 = lx0 + (x1 - x0)

    full[y0:y1, x0:x1] = local[ly0:ly1, lx0:lx1]
    return full


def get_occ_map(batch_dict, occ_key, b_idx, h, w):
    if occ_key not in batch_dict:
        return None

    occ_tensor = batch_dict[occ_key]
    if not torch.is_tensor(occ_tensor):
        return None

    occ_tensor = occ_tensor.detach().cpu()
    if occ_tensor.ndim != 4:
        return None

    occ_map = occ_tensor[b_idx, 0].numpy()
    if occ_map.shape != (h, w):
        occ_t = torch.from_numpy(occ_map)[None, None].float()
        occ_t = F.interpolate(occ_t, size=(h, w), mode='bilinear', align_corners=False)
        occ_map = occ_t[0, 0].numpy()
    return occ_map


def get_base_map(batch_dict, b_idx, h, w, occ_map=None):
    if occ_map is not None:
        return occ_map

    if 'pillar_features_scattered' in batch_dict:
        radar_feat = batch_dict['pillar_features_scattered']
        if torch.is_tensor(radar_feat):
            radar_feat = radar_feat.detach().cpu()
            base_map = radar_feat[b_idx].norm(dim=0).numpy()
            if base_map.shape != (h, w):
                bt = torch.from_numpy(base_map)[None, None].float()
                bt = F.interpolate(bt, size=(h, w), mode='bilinear', align_corners=False)
                base_map = bt[0, 0].numpy()
            return base_map

    return None
def save_attention_figures_for_one_sample(
    att_module,
    batch_dict,
    b_idx,
    sample_id,
    save_root,
    logger,
    point_cloud_range,
    smooth_sigma=1.2,
    smooth_kernel=9,
    occ_key='radar_occ_prob',
    user_query_y=None,
    user_query_x=None,
    query_mode='center'
):
    if getattr(att_module, 'last_att', None) is None:
        logger.warning(f'last_att is None, skip sample {sample_id}')
        return

    att = att_module.last_att
    h, w = att_module.last_hw
    win = att_module.win
    pad = win // 2

    if torch.is_tensor(att):
        att = att.cpu()

    B, HW, P = att.shape
    assert HW == h * w, f'HW mismatch: {HW} vs {h}*{w}'
    assert P == win * win, f'P mismatch: {P} vs {win}*{win}'
    assert b_idx < B, f'b_idx={b_idx} out of range, batch size={B}'

    logger.info(f'[sample_id={sample_id}] attention map size: h={h}, w={w}, win={win}, HW={HW}')

    occ_map = get_occ_map(batch_dict, occ_key, b_idx, h, w)
    qy, qx = choose_query_point(occ_map, h, w, query_mode, user_query_y, user_query_x)
    q_idx = qy * w + qx

    # 单个 query 的局部图
    att_vec = att[b_idx, q_idx].numpy()
    local_heat = att_vec.reshape(win, win)
    global_heat_single = project_local_attn_to_global(att_vec, h, w, win, qy, qx)

    # 整张 BEV 的累计注意力
    full_bev_attn_sum = np.zeros((h, w), dtype=np.float32)
    contrib_count = np.zeros((h, w), dtype=np.float32)

    att_sample = att[b_idx].numpy()  # [HW, P]

    for yy in range(h):
        for xx in range(w):
            cur_idx = yy * w + xx
            cur_local = att_sample[cur_idx].reshape(win, win)

            y0 = max(0, yy - pad)
            y1 = min(h, yy + pad + 1)
            x0 = max(0, xx - pad)
            x1 = min(w, xx + pad + 1)

            ly0 = y0 - (yy - pad)
            ly1 = ly0 + (y1 - y0)
            lx0 = x0 - (xx - pad)
            lx1 = lx0 + (x1 - x0)

            patch = cur_local[ly0:ly1, lx0:lx1]
            full_bev_attn_sum[y0:y1, x0:x1] += patch
            contrib_count[y0:y1, x0:x1] += 1.0

    full_bev_attn_avg = full_bev_attn_sum / np.clip(contrib_count, a_min=1e-6, a_max=None)

    # 平滑
    full_bev_attn_sum_smooth = smooth_heatmap_np(full_bev_attn_sum, sigma=smooth_sigma, kernel_size=smooth_kernel)
    full_bev_attn_avg_smooth = smooth_heatmap_np(full_bev_attn_avg, sigma=smooth_sigma, kernel_size=smooth_kernel)

    # 归一化显示
    full_bev_attn_sum_vis = normalize_to_01(full_bev_attn_sum_smooth)
    full_bev_attn_avg_vis = normalize_to_01(full_bev_attn_avg_smooth)

    base_map = get_base_map(batch_dict, b_idx, h, w, occ_map)

    # 优先从 batch_dict 里拿 gt_boxes
    gt_boxes = get_gt_boxes_from_batch(batch_dict, b_idx)

    sample_dir = save_root / str(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)

    with open(sample_dir / 'attn_hw.txt', 'w') as f:
        f.write(f'h={h}\n')
        f.write(f'w={w}\n')
        f.write(f'win={win}\n')
        f.write(f'HW={HW}\n')
        f.write(f'query_y={qy}\n')
        f.write(f'query_x={qx}\n')

    # 局部图（保留）
    plt.figure(figsize=(5, 4))
    plt.imshow(local_heat, cmap='hot', interpolation='bicubic')
    plt.colorbar()
    plt.title(f'Local Attention @ q=({qy}, {qx}), win={win}')
    plt.xlabel('local x')
    plt.ylabel('local y')
    plt.tight_layout()
    plt.savefig(sample_dir / 'local_attention.png', dpi=200)
    plt.close()

    # 单个 query 投影
    plt.figure(figsize=(6, 6))
    plt.imshow(global_heat_single, cmap='jet', interpolation='bicubic')
    plt.scatter([qx], [qy], c='white', s=35)
    plt.colorbar()
    plt.title(f'Single Query Attention on BEV (h={h}, w={w})')
    plt.tight_layout()
    plt.savefig(sample_dir / 'global_attention_single_query.png', dpi=200)
    plt.close()

    # 全图 sum
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(full_bev_attn_sum_vis, cmap='jet', interpolation='bicubic')
    draw_gt_boxes_on_ax(ax, gt_boxes, point_cloud_range, h, w, color='cyan', linewidth=1.5)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Full BEV Attention Sum (h={h}, w={w})')
    plt.tight_layout()
    plt.savefig(sample_dir / 'full_bev_attention_sum.png', dpi=220)
    plt.close()

    # 全图 avg（推荐看这个）
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(full_bev_attn_avg_vis, cmap='jet', interpolation='bicubic')
    draw_gt_boxes_on_ax(ax, gt_boxes, point_cloud_range, h, w, color='cyan', linewidth=1.5)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Full BEV Attention Avg (h={h}, w={w})')
    plt.tight_layout()
    plt.savefig(sample_dir / 'full_bev_attention_avg.png', dpi=220)
    plt.close()

    # overlay 到底图
    if base_map is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(base_map, cmap='gray', interpolation='bicubic')
        ax.imshow(full_bev_attn_avg_vis, cmap='jet', alpha=0.45, interpolation='bicubic')
        draw_gt_boxes_on_ax(ax, gt_boxes, point_cloud_range, h, w, color='cyan', linewidth=1.5)
        ax.set_title(f'Full BEV Attention Overlay (h={h}, w={w})')
        plt.tight_layout()
        plt.savefig(sample_dir / 'full_bev_attention_overlay.png', dpi=220)
        plt.close()

    if occ_map is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(occ_map, cmap='viridis', interpolation='bicubic')
        draw_gt_boxes_on_ax(ax, gt_boxes, point_cloud_range, h, w, color='red', linewidth=1.2)
        ax.scatter([qx], [qy], c='white', s=25)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Occ map (h={h}, w={w})')
        plt.tight_layout()
        plt.savefig(sample_dir / 'occ_map.png', dpi=220)
        plt.close()

    np.save(sample_dir / 'local_attention.npy', local_heat)
    np.save(sample_dir / 'global_attention_single_query.npy', global_heat_single)
    np.save(sample_dir / 'full_bev_attention_sum.npy', full_bev_attn_sum)
    np.save(sample_dir / 'full_bev_attention_avg.npy', full_bev_attn_avg)
    np.save(sample_dir / 'full_bev_attention_sum_smooth.npy', full_bev_attn_sum_smooth)
    np.save(sample_dir / 'full_bev_attention_avg_smooth.npy', full_bev_attn_avg_smooth)

    if base_map is not None:
        np.save(sample_dir / 'base_map.npy', base_map)
    if occ_map is not None:
        np.save(sample_dir / 'occ_map.npy', occ_map)

    logger.info(f'saved attention figures for sample_id={sample_id} -> {sample_dir}')
def smooth_heatmap_np(arr, sigma=1.2, kernel_size=9):
    """
    用 torch 在 CPU 上做一个高斯平滑，输入输出都是 numpy。
    arr: [H, W]
    """
    if sigma <= 0:
        return arr

    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()

    x = torch.from_numpy(arr).float()[None, None]   # [1,1,H,W]
    kernel = kernel[None, None]                     # [1,1,K,K]

    out = F.conv2d(x, kernel, padding=radius)
    return out[0, 0].numpy()


def normalize_to_01(arr):
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def get_gt_boxes_from_batch(batch_dict, b_idx):
    """
    优先从 batch_dict 里读取 gt_boxes。
    期望格式通常是 [B, M, 7(+...)]，其中前 7 维是：
    x, y, z, dx, dy, dz, heading
    """
    if 'gt_boxes' not in batch_dict:
        return None

    gt_boxes = batch_dict['gt_boxes']
    if not torch.is_tensor(gt_boxes):
        return None

    gt_boxes = gt_boxes.detach().cpu().numpy()

    if gt_boxes.ndim != 3:
        return None

    boxes = gt_boxes[b_idx]   # [M, 7+]
    valid_boxes = []
    for box in boxes:
        if np.allclose(box[:7], 0):
            continue
        valid_boxes.append(box[:7])

    if len(valid_boxes) == 0:
        return None

    return np.array(valid_boxes, dtype=np.float32)


def lidar_xy_to_bev_pixel(x, y, point_cloud_range, h, w):
    """
    LiDAR 平面坐标 -> BEV 像素坐标
    point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    px = (x - x_min) / max(x_max - x_min, 1e-6) * (w - 1)
    py = (y - y_min) / max(y_max - y_min, 1e-6) * (h - 1)

    # 图像坐标 y 轴向下，所以翻一下
    py = (h - 1) - py
    return px, py


def box_to_bev_corners(box):
    """
    box: [x, y, z, dx, dy, dz, heading]
    返回 BEV 平面的 4 个角点（LiDAR xy 平面）
    """
    x, y, z, dx, dy, dz, heading = box[:7]

    # 以盒子中心为原点的四个角
    corners = np.array([
        [ dx / 2,  dy / 2],
        [ dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2,  dy / 2],
    ], dtype=np.float32)

    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)

    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def draw_gt_boxes_on_ax(ax, gt_boxes, point_cloud_range, h, w, color='cyan', linewidth=1.5):
    """
    把 LiDAR gt_boxes 画到当前 ax 上
    """
    if gt_boxes is None or len(gt_boxes) == 0:
        return

    for box in gt_boxes:
        corners = box_to_bev_corners(box)  # [4,2] in lidar xy
        pix = [lidar_xy_to_bev_pixel(cx, cy, point_cloud_range, h, w) for cx, cy in corners]
        pix = np.array(pix, dtype=np.float32)

        # 闭合多边形
        pix = np.vstack([pix, pix[0:1]])
        ax.plot(pix[:, 0], pix[:, 1], color=color, linewidth=linewidth)


def move_batch_to_cuda(batch_dict):
    """
    把 batch_dict 里的数值型数据搬到 GPU。
    对 images 做特殊处理：
    - 如果是数值型 numpy.ndarray，先转 tensor
    - 如果是字符串/对象类型 numpy.ndarray，跳过
    - 如果 images 是 NHWC，就转成 NCHW
    """
    for key, val in batch_dict.items():
        # numpy 数组
        if isinstance(val, np.ndarray):
            # 字符串、unicode、object 之类不能转 torch，直接跳过
            if val.dtype.kind in ['U', 'S', 'O']:
                continue

            val = torch.from_numpy(val)

        if torch.is_tensor(val):
            # images: [B, H, W, C] -> [B, C, H, W]
            if key == 'images' and val.ndim == 4 and val.shape[-1] in [1, 3]:
                val = val.permute(0, 3, 1, 2).contiguous()

            batch_dict[key] = val.cuda(non_blocking=True)

def main():
    args, cfg = parse_config()

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    logger = common_utils.create_logger(save_root / 'vis_attn_by_id_log.txt')
    logger.info('================ visualize attention by ids ================')
    for key, val in vars(args).items():
        logger.info(f'{key:16} {val}')

    target_ids = set([str(x) for x in args.target_ids])
    logger.info(f'target_ids: {sorted(list(target_ids))}')

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=False,
        pre_trained_path=args.pretrained_model
    )
    model.cuda()
    model.eval()

    module_name, att_module = find_module_by_type(model, 'LocalWinCrossAttnOcc')
    if att_module is None:
        raise RuntimeError('Cannot find LocalWinCrossAttnOcc in model. Check module name/path.')
    logger.info(f'found attention module: {module_name}')

    found_ids = set()

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(test_loader):
            batch_ids = get_batch_sample_ids(batch_dict, args.id_key)

            matched_local_indices = []
            matched_sample_ids = []
            for i, sid in enumerate(batch_ids):
                if sid in target_ids and sid not in found_ids:
                    matched_local_indices.append(i)
                    matched_sample_ids.append(sid)

            if len(matched_local_indices) == 0:
                continue

            logger.info(f'batch_idx={batch_idx}, matched_ids={matched_sample_ids}')

            move_batch_to_cuda(batch_dict)

            if 'images' in batch_dict and torch.is_tensor(batch_dict['images']):
                logger.info(f"images shape before model: {tuple(batch_dict['images'].shape)}")

            try:
                _ = model(batch_dict)
            except TypeError:
                _ = model.forward(batch_dict)

            for local_b_idx, sample_id in zip(matched_local_indices, matched_sample_ids):
                save_attention_figures_for_one_sample(
                    att_module=att_module,
                    batch_dict=batch_dict,
                    b_idx=local_b_idx,
                    sample_id=sample_id,
                    save_root=save_root,
                    logger=logger,
                    point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                    smooth_sigma=args.smooth_sigma,
                    smooth_kernel=args.smooth_kernel,
                    occ_key=args.occ_key,
                    user_query_y=args.query_y,
                    user_query_x=args.query_x,
                    query_mode=args.query_mode
                )
                found_ids.add(sample_id)

            if found_ids == target_ids:
                logger.info('all target ids have been processed')
                break

    missing_ids = target_ids - found_ids
    if len(missing_ids) > 0:
        logger.warning(f'some target ids were not found: {sorted(list(missing_ids))}')
    else:
        logger.info('all requested ids were successfully visualized')

if __name__ == '__main__':
    main()