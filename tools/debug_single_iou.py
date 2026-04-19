#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import _init_path
import argparse
import pickle
from pathlib import Path

import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.utils import vod_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Debug single-box IoU for VOD evaluator')
    parser.add_argument('--cfg_file', type=str, required=True, help='config yaml')
    parser.add_argument('--sample_idx', type=int, default=0, help='which sample in kitti_infos to inspect')
    parser.add_argument('--class_name', type=str, default='Car', choices=['Car', 'Pedestrian', 'Cyclist'],
                        help='target class to inspect')
    return parser.parse_args()


def build_dataset(logger):
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False
    )
    return test_set


def find_first_object_of_class(annos, class_name):
    names = annos['name']
    for i, n in enumerate(names):
        if str(n) == class_name:
            return i
    return None


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    logger = common_utils.create_logger()
    dataset = build_dataset(logger)

    logger.info('dataset class: %s', dataset.__class__.__name__)
    logger.info('num samples: %d', len(dataset.kitti_infos))
    logger.info('target class: %s', args.class_name)

    info = dataset.kitti_infos[args.sample_idx]
    annos = info['annos']

    obj_idx = find_first_object_of_class(annos, args.class_name)
    if obj_idx is None:
        raise ValueError(f'No object of class {args.class_name} found in sample {args.sample_idx}')

    print('\n========== RAW GT ANNO ==========')
    print('sample_idx:', args.sample_idx)
    print('frame_id:', info['point_cloud']['lidar_idx'])
    print('obj_idx:', obj_idx)
    print('name:', annos['name'][obj_idx])
    print('bbox:', annos['bbox'][obj_idx])
    print('alpha:', annos['alpha'][obj_idx])
    print('dimensions:', annos['dimensions'][obj_idx], '  # expected current format in annos')
    print('location:', annos['location'][obj_idx])
    print('rotation_y:', annos['rotation_y'][obj_idx])
    print('================================\n')

    # 1) 2D IoU: bbox with itself
    bbox = annos['bbox'][obj_idx:obj_idx + 1].astype(np.float64)
    iou_2d = vod_utils.image_box_overlap(bbox, bbox)
    print('2D IoU self-self =')
    print(iou_2d)

    # 2) BEV IoU: build one box in evaluator BEV format [x, z, dx, dz, rot]
    loc = annos['location'][obj_idx:obj_idx + 1]
    dims = annos['dimensions'][obj_idx:obj_idx + 1]
    rots = annos['rotation_y'][obj_idx:obj_idx + 1]

    bev_box = np.concatenate([
        loc[:, [0, 2]],
        dims[:, [0, 2]],
        rots[..., np.newaxis]
    ], axis=1).astype(np.float64)

    iou_bev = vod_utils.bev_box_overlap(bev_box, bev_box)
    print('\nBEV IoU self-self =')
    print(iou_bev)
    print('BEV box used =', bev_box)

    # 3) 3D IoU: evaluator format [x, y, z, dim0, dim1, dim2, rot]
    box3d = np.concatenate([
        annos['location'][obj_idx:obj_idx + 1],
        annos['dimensions'][obj_idx:obj_idx + 1],
        annos['rotation_y'][obj_idx:obj_idx + 1][..., np.newaxis]
    ], axis=1).astype(np.float64)

    iou_3d = vod_utils.d3_box_overlap(box3d, box3d)
    print('\n3D IoU self-self =')
    print(iou_3d)
    print('3D box used =', box3d)

    # 4) Swap dimension order tests
    # current annos stores dimensions as [l, h, w] according to your dataset code
    lhw = annos['dimensions'][obj_idx:obj_idx + 1].astype(np.float64)
    hwl = lhw[:, [1, 2, 0]]
    wlh = lhw[:, [2, 0, 1]]

    box3d_lhw = np.concatenate([
        annos['location'][obj_idx:obj_idx + 1],
        lhw,
        annos['rotation_y'][obj_idx:obj_idx + 1][..., np.newaxis]
    ], axis=1)

    box3d_hwl = np.concatenate([
        annos['location'][obj_idx:obj_idx + 1],
        hwl,
        annos['rotation_y'][obj_idx:obj_idx + 1][..., np.newaxis]
    ], axis=1)

    box3d_wlh = np.concatenate([
        annos['location'][obj_idx:obj_idx + 1],
        wlh,
        annos['rotation_y'][obj_idx:obj_idx + 1][..., np.newaxis]
    ], axis=1)


    from itertools import permutations

    print('\\n========== DIM/Y CONVENTION BRUTE FORCE ==========')

    orig_loc = annos['location'][obj_idx:obj_idx + 1].astype(np.float64)  # [x, y, z]
    orig_dims = annos['dimensions'][obj_idx:obj_idx + 1].astype(np.float64)  # current stored order
    orig_rot = annos['rotation_y'][obj_idx:obj_idx + 1].astype(np.float64)

    dim_names = ['d0', 'd1', 'd2']
    perm_list = list(permutations([0, 1, 2]))

    best_rows = []

    for perm in perm_list:
        dims_perm = orig_dims[:, list(perm)]  # reorder dims

        # 假设 perm 后的 dims_perm[:,1] 是“高度 h”
        h_assumed = dims_perm[:, 1:2]

        y_modes = {
            'y_raw': orig_loc[:, 1:2],
            'y_minus_h2': orig_loc[:, 1:2] - h_assumed / 2.0,
            'y_plus_h2': orig_loc[:, 1:2] + h_assumed / 2.0,
        }

        # BEV 里尝试三种取轴方式（从 perm 后的三个维度里挑两个）
        bev_axis_choices = [
            (0, 2),
            (0, 1),
            (1, 2),
        ]

        for y_mode_name, y_used in y_modes.items():
            # 3D box = [x, y, z, d0, d1, d2, rot]
            box3d = np.concatenate([
                orig_loc[:, 0:1],
                y_used,
                orig_loc[:, 2:3],
                dims_perm,
                orig_rot[..., np.newaxis]
            ], axis=1)

            iou_3d = float(vod_utils.d3_box_overlap(box3d, box3d)[0, 0])

            for bev_axes in bev_axis_choices:
                # BEV box = [x, z, dx, dz, rot]
                bev_box = np.concatenate([
                    orig_loc[:, [0, 2]],
                    dims_perm[:, [bev_axes[0], bev_axes[1]]],
                    orig_rot[..., np.newaxis]
                ], axis=1)

                iou_bev = float(vod_utils.bev_box_overlap(bev_box, bev_box)[0, 0])

                row = {
                    'perm': perm,
                    'perm_name': f'[{dim_names[perm[0]]},{dim_names[perm[1]]},{dim_names[perm[2]]}]',
                    'dims_used': dims_perm.copy(),
                    'y_mode': y_mode_name,
                    'bev_axes': bev_axes,
                    'bev_axes_name': f'({bev_axes[0]},{bev_axes[1]})',
                    'iou_bev': iou_bev,
                    'iou_3d': iou_3d,
                    'score': abs(iou_bev - 1.0) + abs(iou_3d - 1.0),
                    'box3d': box3d.copy(),
                    'bev_box': bev_box.copy(),
                }
                best_rows.append(row)

    # 按离 (1,1) 最近排序
    best_rows = sorted(best_rows, key=lambda x: x['score'])

    print('original dims stored in annos =', orig_dims)
    print('Top 15 closest combinations to (BEV=1, 3D=1):\\n')

    for i, row in enumerate(best_rows[:15]):
        print(f'[{i}] perm={row["perm_name"]}, y_mode={row["y_mode"]}, bev_axes={row["bev_axes_name"]}, '
              f'BEV={row["iou_bev"]:.6f}, 3D={row["iou_3d"]:.6f}')
        print('    box3d =', row['box3d'])
        print('    bev_box =', row['bev_box'])

    print('\\nBest candidate summary:')
    best = best_rows[0]
    print('perm =', best['perm_name'])
    print('y_mode =', best['y_mode'])
    print('bev_axes =', best['bev_axes_name'])
    print('best BEV IoU =', best['iou_bev'])
    print('best 3D IoU =', best['iou_3d'])
    print('best box3d =', best['box3d'])
    print('best bev_box =', best['bev_box'])
    print('===========================================================\\n')
    print('============================================\n')

    print('\\n========== ROTATE_IOU MINIMAL SANITY ==========')

    # 一个最简单的轴对齐矩形，不旋转
    simple_box = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    print('simple_box =', simple_box)
    print('rotate_iou_eval(simple, simple) =')
    print(vod_utils.rotate_iou_eval(simple_box, simple_box, -1))

    # 一个相同尺寸但旋转后的框，自己和自己算
    simple_rot = np.array([[0.0, 0.0, 2.0, 1.0, 0.3]], dtype=np.float32)
    print('simple_rot =', simple_rot)
    print('rotate_iou_eval(simple_rot, simple_rot) =')
    print(vod_utils.rotate_iou_eval(simple_rot, simple_rot, -1))

    # 再试试你刚才 best candidate 的 bev_box
    bev_best = best['bev_box'].astype(np.float32)
    print('best bev box =', bev_best)
    print('rotate_iou_eval(best_bev, best_bev) =')
    print(vod_utils.rotate_iou_eval(bev_best, bev_best, -1))

    print('===============================================\\n')

    print('Done.')


if __name__ == '__main__':
    main()