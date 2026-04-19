import pickle
import time

import numpy as np
import torch
import tqdm

from copy import deepcopy
from functools import partial
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor, augmentor_utils
from pcdet.utils.calibration_kitti import Calibration

class TTA():
    def __init__(self):
        self.tta_list = [
            self.random_world_flip(config=dict(ALONG_AXIS_LIST=['x'])),
            self.random_image_flip_fusion(config=dict(ALONG_AXIS_LIST=['horizontal'])),
            self.random_world_rotation(config=dict(WORLD_ROT_ANGLE=[-0.3926, 0.3926])),
            self.random_world_scaling(config=dict(WORLD_SCALE_RANGE=[0.95, 1.05]))
        ]
        self.tta_num = 4

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        gt_boxes_classes = gt_boxes[:, -1]
        gt_boxes = gt_boxes[:, :7]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_flip=True
            )
            data_dict['flip_%s'%cur_axis] = enable
            if 'roi_boxes' in data_dict.keys():
                num_frame, num_rois,dim = data_dict['roi_boxes'].shape
                roi_boxes, _, _ = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                data_dict['roi_boxes'].reshape(-1,dim), np.zeros([1,3]), return_flip=True, enable=enable
                )
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = np.concatenate([gt_boxes, gt_boxes_classes.reshape([-1, 1])], axis=1)
        data_dict['points'] = points
        return data_dict
    
    def random_image_flip_fusion(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip_fusion, config=config)
        assert data_dict['flip_x'] is not None
        if data_dict['flip_x']:
            return self.random_image_flip(data_dict, config, prob=1)
        else:
            return data_dict
        
    def random_image_flip(self, data_dict=None, config=None, prob=0.5):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        # depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, gt_boxes, gt_boxes2d, enable = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, gt_boxes, calib, prob, gt_boxes2d
            )
            if enable and 'foreground' in data_dict:
                data_dict['foreground'] = np.fliplr(data_dict['foreground'])
            if enable and 'depth_maps' in data_dict:
                data_dict['depth_maps'] = np.fliplr(data_dict['depth_maps'])

        data_dict['images'] = images
        # data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes2d'] = gt_boxes2d
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        def global_rotation(gt_boxes, points, rot_range, return_rot=False, noise_rotation=None):
            if noise_rotation is None: 
                noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
            points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
            gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
            gt_boxes[:, 6] += noise_rotation

            if return_rot:
                return gt_boxes, points, noise_rotation
            return gt_boxes, points

        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois,dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_rotation(
            data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]), rot_range=rot_range, return_rot=True, noise_rotation=noise_rot)
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict
    
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        if 'roi_boxes' in data_dict.keys():
            gt_boxes, roi_boxes, points, noise_scale = augmentor_utils.global_scaling_with_roi_boxes(
                data_dict['gt_boxes'], data_dict['roi_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict

    def copy(self, data):
        keys = list(data.keys())
        result = dict()
        for key in keys:
            if key == 'images':
                result[key] = np.ascontiguousarray(data[key].copy())
            elif isinstance(data[key], (np.ndarray, str, np.bool_, float)):
                result[key] = data[key].copy()
            elif isinstance(data[key], (Calibration, int)):
                result[key] = data[key]
            else:
                raise NotImplementedError
        return result

    def __call__(self, batch_dict):
        B = batch_dict['batch_size']
        assert B == 1
        keys = list(batch_dict.keys())
        data_list = [dict()]
        # unpack
        for i in range(B):
            for key in keys:
                if key == 'points':
                    valid_mask = batch_dict[key][:, 0] == i
                    data_list[i][key] = batch_dict[key][valid_mask] # [:, 1:]
                elif key == 'batch_size':
                    data_list[i][key] = batch_dict[key]
                else:
                    data_list[i][key] = batch_dict[key][i]
        # augment
        tta_data = []
        for i in range(self.tta_num):
            data = self.copy(data_list[0])
            for module in self.tta_list:
                data = module(data)
            for key in ['images']:
                data[key] = np.ascontiguousarray(np.expand_dims(data[key], axis=0))
            tta_data.append(data)
        # pack
        # result_dict = dict()
        # for key in keys:
        #     if key in ['trans_cam_to_img', 'trans_lidar_to_cam', 'frame_id', 'calib']:
        #         result_dict[key] = [x[key] for x in data_list]
        #     elif key == 'points':
        #         pts = []
        #         for i in range(B):
        #             batch_idx = np.ones([data_list[i][key].shape[0], 1]) * i
        #             pts.append(np.concatenate([batch_idx, data_list[i][key]], axis=1))
        #         pts = np.concatenate(pts)
        #     elif key == 'batch_size':
        #         result_dict[key] = B
        #     else:
        #         result_dict[key] = np.stack([x[key] for x in data_list])
        return tta_data

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch_tta(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    tta = TTA()
    for i, all_batch_dict in enumerate(dataloader):
        batch_dict_list = tta(all_batch_dict)
        for idx, batch_dict in enumerate(batch_dict_list):
            load_data_to_gpu(batch_dict)

            if getattr(args, 'infer_time', False):
                start_time = time.time()

            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)

            disp_dict = {}

            if getattr(args, 'infer_time', False):
                inference_time = time.time() - start_time
                infer_time_meter.update(inference_time * 1000)
                # use ms to measure inference time
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if args.save_to_file else None
            )
            det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
