import _init_path
import argparse
import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import tqdm

from pcdet.config import cfg, cfg_from_yaml_file, cfg_from_list, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Diagnose evaluator vs model')
    parser.add_argument('--cfg_file', type=str, required=True, help='cfg yaml')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)

    parser.add_argument(
        '--mode',
        type=str,
        default='fake_gt',
        choices=['fake_gt', 'shuffle_gt', 'shift_half_right', 'real_model', 'all'],
        help='diagnose mode'
    )
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all samples')
    parser.add_argument('--max_batches', type=int, default=0, help='for real_model only, 0 means full loader')
    parser.add_argument('--save_result_pkl', action='store_true', default=False)
    parser.add_argument('--out_dir', type=str, default='./output/diagnose_eval_vs_model')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)
    return args, cfg


def make_logger(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'diagnose_log.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('********************** Diagnose Start **********************')
    log_config_to_file(cfg, logger=logger)
    return logger


def build_test_loader(args, logger, dist_test=False):
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False
    )
    return test_set, test_loader, sampler


def save_pkl(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def summarize_det_annos(det_annos, logger, prefix='det'):
    total_objs = 0
    class_counter = {}
    for anno in det_annos:
        names = anno.get('name', [])
        total_objs += len(names)
        for n in names:
            class_counter[str(n)] = class_counter.get(str(n), 0) + 1

    logger.info('[%s] num_samples = %d', prefix, len(det_annos))
    logger.info('[%s] avg objects/sample = %.4f', prefix, total_objs / max(1, len(det_annos)))
    logger.info('[%s] class histogram = %s', prefix, class_counter)


def filter_eval_classes_and_add_score(anno, allowed_classes=('Car', 'Pedestrian', 'Cyclist')):
    anno = copy.deepcopy(anno)
    names = np.array(anno['name'])
    valid_mask = np.array([str(n) in allowed_classes for n in names], dtype=bool)

    det = {}
    keys_to_keep = [
        'name', 'truncated', 'occluded', 'alpha', 'bbox',
        'dimensions', 'location', 'rotation_y'
    ]
    for k in keys_to_keep:
        v = anno[k]
        if isinstance(v, np.ndarray):
            det[k] = v[valid_mask].copy()
        else:
            det[k] = np.array(v)[valid_mask].copy()

    num = len(det['name'])
    if num > 0:
        det['score'] = np.linspace(1.0, 0.001, num, dtype=np.float32)
        det['score'] = det['score'] - np.arange(num, dtype=np.float32) * 1e-6
    else:
        det['score'] = np.zeros((0,), dtype=np.float32)
    return det


def get_subset_kitti_infos(dataset, max_samples=0):
    if max_samples is None or max_samples <= 0:
        return dataset.kitti_infos
    return dataset.kitti_infos[:max_samples]


def build_fake_det_annos_from_gt(dataset, max_samples=0):
    infos = get_subset_kitti_infos(dataset, max_samples=max_samples)
    gt_annos = [copy.deepcopy(info['annos']) for info in infos]
    fake_det_annos = [
        filter_eval_classes_and_add_score(x, allowed_classes=tuple(dataset.class_names))
        for x in gt_annos
    ]
    return gt_annos, fake_det_annos


def build_shuffled_det_annos_from_gt(dataset, class_names, max_samples=0):
    _, fake_det_annos = build_fake_det_annos_from_gt(dataset, max_samples=max_samples)
    class_names = list(class_names)

    shuffled = []
    for anno in fake_det_annos:
        cur = copy.deepcopy(anno)
        if len(cur['name']) > 0:
            new_names = []
            for n in cur['name']:
                idx = class_names.index(str(n))
                new_names.append(class_names[(idx + 1) % len(class_names)])
            cur['name'] = np.array(new_names)
        shuffled.append(cur)
    return shuffled


def build_shift_half_right_det_annos_from_gt(dataset, max_samples=0):
    """
    所有框向右平移“自身宽度的一半”
    当前 VOD annos['dimensions'] 约定为 [l, h, w]
    所以向相机坐标系 x 正方向平移 w / 2
    """
    _, fake_det_annos = build_fake_det_annos_from_gt(dataset, max_samples=max_samples)

    shifted = []
    for anno in fake_det_annos:
        cur = copy.deepcopy(anno)
        if len(cur['name']) > 0:
            # dimensions: [l, h, w]
            widths = cur['dimensions'][:, 2]
            cur['location'][:, 0] += widths / 10.0
        shifted.append(cur)
    return shifted


def run_dataset_evaluation(dataset, det_annos, class_names, logger, tag='eval'):
    summarize_det_annos(det_annos, logger, prefix=tag)

    orig_infos = dataset.kitti_infos
    dataset.kitti_infos = dataset.kitti_infos[:len(det_annos)]
    try:
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )
    finally:
        dataset.kitti_infos = orig_infos

    logger.info('\n========== %s RESULT_STR ==========', tag)
    logger.info('\n%s', result_str)
    logger.info('========== %s RESULT_DICT ==========', tag)
    for k, v in result_dict.items():
        logger.info('%s: %s', k, v)
    return result_str, result_dict


def run_real_model_eval(model, dataloader, dataset, class_names, logger, max_batches=0, max_samples=0):
    det_annos = []
    label_hist = {}
    score_list = []
    seen_samples = 0

    model.eval()
    pbar = tqdm.tqdm(total=len(dataloader), desc='real_model_eval', dynamic_ncols=True)
    with torch.no_grad():
        for i, batch_dict in enumerate(dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            if max_samples > 0 and seen_samples >= max_samples:
                break

            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names, output_path=None
            )
            det_annos.extend(annos)
            seen_samples += len(annos)

            for pred in pred_dicts:
                labels = pred['pred_labels'].detach().cpu().numpy()
                scores = pred['pred_scores'].detach().cpu().numpy()
                for lb in labels:
                    label_hist[int(lb)] = label_hist.get(int(lb), 0) + 1
                if len(scores) > 0:
                    score_list.extend(scores.tolist())

            pbar.update()

    pbar.close()

    logger.info('[real_model] raw pred label histogram = %s', label_hist)
    if len(score_list) > 0:
        score_np = np.array(score_list)
        logger.info('[real_model] score stats: min=%.6f max=%.6f mean=%.6f median=%.6f',
                    float(score_np.min()), float(score_np.max()),
                    float(score_np.mean()), float(np.median(score_np)))
    else:
        logger.info('[real_model] no scores found')

    if max_samples > 0:
        det_annos = det_annos[:max_samples]

    return det_annos


def main():
    args, _ = parse_config()

    if args.launcher == 'none':
        dist_test = False
    else:
        _, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    out_dir = Path(args.out_dir)
    logger = make_logger(out_dir)

    logger.info('cfg_file = %s', args.cfg_file)
    logger.info('ckpt = %s', args.ckpt)
    logger.info('mode = %s', args.mode)
    logger.info('max_samples = %s', args.max_samples)
    logger.info('max_batches = %s', args.max_batches)

    dataset, dataloader, _ = build_test_loader(args, logger, dist_test=dist_test)
    class_names = dataset.class_names
    logger.info('dataset class = %s', dataset.__class__.__name__)
    logger.info('class_names = %s', class_names)
    logger.info('dataset size = %d', len(dataset))

    if args.mode in ['fake_gt', 'all']:
        logger.info('\n\n#################### A. PERFECT GT AS DET ####################')
        _, fake_det_annos = build_fake_det_annos_from_gt(dataset, max_samples=args.max_samples)
        if args.save_result_pkl:
            save_pkl(fake_det_annos, out_dir / 'fake_gt_result.pkl')
        run_dataset_evaluation(dataset, fake_det_annos, class_names, logger, tag='fake_gt')

    if args.mode in ['shuffle_gt', 'all']:
        logger.info('\n\n#################### B. SHUFFLED GT AS DET ####################')
        shuffled_det_annos = build_shuffled_det_annos_from_gt(
            dataset, class_names, max_samples=args.max_samples
        )
        if args.save_result_pkl:
            save_pkl(shuffled_det_annos, out_dir / 'shuffle_gt_result.pkl')
        run_dataset_evaluation(dataset, shuffled_det_annos, class_names, logger, tag='shuffle_gt')

    if args.mode in ['shift_half_right', 'all']:
        logger.info('\n\n#################### C. SHIFT HALF WIDTH TO RIGHT ####################')
        shifted_det_annos = build_shift_half_right_det_annos_from_gt(
            dataset, max_samples=args.max_samples
        )
        if args.save_result_pkl:
            save_pkl(shifted_det_annos, out_dir / 'shift_half_right_result.pkl')
        run_dataset_evaluation(dataset, shifted_det_annos, class_names, logger, tag='shift_half_right')

    if args.mode in ['real_model', 'all']:
        logger.info('\n\n#################### D. REAL MODEL EVAL ####################')
        if args.ckpt is None:
            raise ValueError('--ckpt is required for real_model/all mode')

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        logger.info('loading checkpoint from %s', args.ckpt)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        real_det_annos = run_real_model_eval(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            class_names=class_names,
            logger=logger,
            max_batches=args.max_batches,
            max_samples=args.max_samples
        )
        if args.save_result_pkl:
            save_pkl(real_det_annos, out_dir / 'real_model_result.pkl')
        run_dataset_evaluation(dataset, real_det_annos, class_names, logger, tag='real_model')

    logger.info('\n\n==================== HOW TO READ RESULTS ====================')
    logger.info('1) fake_gt 很高 => evaluator 基本没坏')
    logger.info('2) shuffle_gt 很低 => evaluator 正常利用类别')
    logger.info('3) shift_half_right 很低 => evaluator 正常利用空间位置 / IoU')
    logger.info('4) fake_gt 很高但 real_model 很低 => 模型输出 / 权重 / 配置 / 数据不匹配')
    logger.info('============================================================')


if __name__ == '__main__':
    main()