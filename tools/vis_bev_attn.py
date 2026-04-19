import _init_path
import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Visualize BEV message/attention maps')
    parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/hgsfusion/hgsfusion_vod.yaml',
                        help='specify the config for visualization')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str,
                        default='./output/tools/cfgs/hgsfusion/hgsfusion_vod/default/ckpt/checkpoint_epoch_25.pth',
                        help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed testing')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed testing')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='dataset split used for visualization')
    parser.add_argument('--max_vis_samples', type=int, default=20,
                        help='maximum number of samples to visualize')
    parser.add_argument('--vis_tag', type=str, default='bev_msg',
                        help='sub-folder name for visualization results')

    # 输出控制
    parser.add_argument('--save_occ', action='store_true', default=False,
                        help='also save occupancy map')
    parser.add_argument('--save_attn', action='store_true', default=False,
                        help='also save attention map (not recommended as main figure)')
    parser.add_argument('--upsample', action='store_true', default=True,
                        help='upsample map to input BEV size before saving')

    # message map 设置
    parser.add_argument('--msg_reduce', type=str, default='mean_abs',
                        choices=['mean_abs', 'sum_abs', 'l2'],
                        help='how to reduce message over channels')
    parser.add_argument('--use_pre_gate_msg', action='store_true', default=False,
                        help='use message before occ gate')

    # 指定 frame id
    parser.add_argument('--target_ids', type=str, nargs='*', default=[
        "000017", "000069", "300002", "300012", "340089", "410010"
    ], help='only visualize these frame ids')

    # 高斯平滑
    parser.add_argument('--gaussian_sigma', type=float, default=1.2,
                        help='gaussian blur sigma, 0 means disabled')
    parser.add_argument('--gaussian_kernel', type=int, default=9,
                        help='gaussian kernel size, should be odd')

    # BEV 显示范围
    parser.add_argument('--x_min', type=float, default=-30.0)
    parser.add_argument('--x_max', type=float, default=30.0)
    parser.add_argument('--y_min', type=float, default=0.0)
    parser.add_argument('--y_max', type=float, default=80.0)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def gaussian_kernel2d(kernel_size=9, sigma=1.2, device='cpu', dtype=torch.float32):
    assert kernel_size % 2 == 1, 'gaussian kernel size should be odd'
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_2d(x, kernel_size=9, sigma=1.2):
    """
    x: [H, W]
    """
    if sigma <= 0:
        return x

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    x = x.float()
    device = x.device
    kernel = gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma, device=device, dtype=x.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

    x = x.unsqueeze(0).unsqueeze(0)            # [1,1,H,W]
    pad = kernel_size // 2
    x = F.conv2d(x, kernel, padding=pad)
    x = x[0, 0]
    return x


def robust_normalize(x, low=5, high=99):
    """
    分位数归一化，避免极少数异常点把整图压扁
    """
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()

    vmin = np.percentile(x, low)
    vmax = np.percentile(x, high)
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-6)
    return x


def maybe_upsample_map(x, target_hw):
    """
    x: [H, W]
    target_hw: (H_target, W_target)
    """
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if list(x.shape[-2:]) == list(target_hw):
        return x
    x = F.interpolate(
        x.unsqueeze(0).unsqueeze(0).float(),
        size=target_hw,
        mode='bilinear',
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
    title=None
):
    import matplotlib.pyplot as plt

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
        cmap='Reds',
        origin='lower',
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        aspect='auto',
        vmin=0,
        vmax=1
    )
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_sample_name(batch_dict, batch_idx=0, fallback_idx=0):
    frame_id = batch_dict.get('frame_id', None)
    if frame_id is None:
        return f'sample_{fallback_idx:04d}'

    if isinstance(frame_id, (list, tuple)):
        return str(frame_id[batch_idx])

    if isinstance(frame_id, np.ndarray):
        return str(frame_id[batch_idx])

    return str(frame_id)


def forward_model(model, batch_dict):
    load_data_to_gpu(batch_dict)
    output = model(batch_dict)
    return output


def should_keep_sample(sample_name, target_ids_set):
    if target_ids_set is None or len(target_ids_set) == 0:
        return True
    return sample_name in target_ids_set


def find_align_module(model):
    for name, module in model.named_modules():
        if hasattr(module, 'get_last_msg_map') and hasattr(module, 'last_msg_post_gate'):
            return name, module
    return None, None


def vis_one_batch(model, batch_dict, save_dir, logger, args, global_idx_start=0):
    _ = forward_model(model, batch_dict)

    align_name, align_module = find_align_module(model)
    if align_module is None:
        raise RuntimeError('No align module with cached message found in the model.')

    frame_id = batch_dict.get('frame_id', None)
    if isinstance(frame_id, (list, tuple, np.ndarray)):
        batch_size = len(frame_id)
    else:
        batch_size = 1

    vis_count = 0
    target_ids_set = set(args.target_ids) if args.target_ids is not None else None

    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    for bidx in range(batch_size):
        sample_name = get_sample_name(batch_dict, batch_idx=bidx, fallback_idx=global_idx_start + vis_count)

        if not should_keep_sample(sample_name, target_ids_set):
            continue

        use_post_gate = not args.use_pre_gate_msg
        msg_map = align_module.get_last_msg_map(
            batch_idx=bidx,
            upsample=False,
            use_post_gate=use_post_gate,
            reduce=args.msg_reduce
        )

        input_hw = align_module.last_input_hw
        occ = align_module.last_occ[bidx, 0] if align_module.last_occ is not None else None

        if args.save_attn:
            attn_map = align_module.get_last_global_attention_map(
                batch_idx=bidx,
                upsample=False,
                normalize=True
            )
        else:
            attn_map = None

        if args.upsample and input_hw is not None:
            msg_map = maybe_upsample_map(msg_map, input_hw)
            if occ is not None:
                occ = maybe_upsample_map(occ, input_hw)
            if attn_map is not None:
                attn_map = maybe_upsample_map(attn_map, input_hw)

        # 主输出：message map
        msg_path = os.path.join(save_dir, f'{sample_name}_msg_map.png')
        save_bev_map(
            msg_map,
            msg_path,
            x_range=x_range,
            y_range=y_range,
            gaussian_sigma=args.gaussian_sigma,
            gaussian_kernel=args.gaussian_kernel,
            title=None
        )
        logger.info(f'Saved message map: {msg_path}')

        # 可选输出：occ
        if args.save_occ and occ is not None:
            occ_path = os.path.join(save_dir, f'{sample_name}_occ.png')
            save_bev_map(
                occ,
                occ_path,
                x_range=x_range,
                y_range=y_range,
                gaussian_sigma=args.gaussian_sigma,
                gaussian_kernel=args.gaussian_kernel,
                title=None
            )
            logger.info(f'Saved occupancy map: {occ_path}')

        # 可选输出：attention
        if args.save_attn and attn_map is not None:
            attn_path = os.path.join(save_dir, f'{sample_name}_bev_attn.png')
            save_bev_map(
                attn_map,
                attn_path,
                x_range=x_range,
                y_range=y_range,
                gaussian_sigma=args.gaussian_sigma,
                gaussian_kernel=args.gaussian_kernel,
                title=None
            )
            logger.info(f'Saved attention map: {attn_path}')

        vis_count += 1

    return vis_count


def main():
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, f'init_dist_{args.launcher}')(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = 1
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_output_dir = output_dir / 'vis' / args.vis_tag / args.split
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = vis_output_dir / ('log_vis_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start visualization**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # 强制开启可视化导出
    try:
        cfg.MODEL.EXPORT_VIS = True
    except Exception:
        pass

    dataset_cfg = cfg.DATA_CONFIG
    if hasattr(dataset_cfg, 'DATA_SPLIT') and args.split in dataset_cfg.DATA_SPLIT:
        dataset_cfg.DATA_SPLIT['test'] = dataset_cfg.DATA_SPLIT[args.split]

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
    logger.info(f'Target IDs: {sorted(list(target_ids_set)) if target_ids_set is not None else "ALL"}')

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

    logger.info(f'Visualization finished. Saved {saved_num} sample(s) to: {vis_output_dir}')


if __name__ == '__main__':
    main()

"""
python tools/vis_bev_attn.py \
    --cfg_file ./tools/cfgs/hgsfusion/hgsfusion_tj4d.yaml \
    --ckpt /home/ccc/xx/base/HGSFusion/output2.04___pillarnet+windows_38.95/tools/cfgs/hgsfusion/hgsfusion_tj4d/default/ckpt/checkpoint_epoch_21.pth \
    --batch_size 1 \
    --max_vis_samples 10
"""