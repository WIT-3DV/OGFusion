from .detector3d_template import Detector3DTemplate


class CaDDN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.fusion = self.model_cfg.get('FusionVFE', False)
        self.use_centerhead = self.model_cfg.DENSE_HEAD.NAME == 'CenterHead'
        if self.fusion:
            self.use_lidar_depth = self.model_cfg['FusionVFE']['ImageVFE']['FFN'].get('USE_LIDAR_DEPTH', False)
            self.use_depth = self.model_cfg['FusionVFE']['ImageVFE'].get('USE_DEPTH', True)
            self.use_occupancy = self.model_cfg.FusionVFE.get('RadarOccupancy', False)
            if self.use_occupancy:
                self.use_occupancy_loss = self.model_cfg.FusionVFE.RadarOccupancy.get('USE_OCC_LOSS', False)
            else:
                self.use_occupancy_loss = False
        else:
            self.use_lidar_depth = self.model_cfg['VFE']['FFN'].get('USE_LIDAR_DEPTH', False)
            self.use_depth = self.model_cfg['VFE'].get('USE_DEPTH', True)
            self.use_occupancy = False
            self.use_occupancy_loss = False

    def post_processing_centerhead(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.use_centerhead:
                pred_dicts, recall_dicts = self.post_processing_centerhead(batch_dict)
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):


        disp_dict = {}

        # 预测loss
        loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
        
        # 深度loss
        if (not self.use_lidar_depth) and self.use_depth:
            if self.vfe != None:
                loss_depth, tb_dict_depth = self.vfe.get_loss()
            else:
                loss_depth, tb_dict_depth = self.fusion_vfe.get_loss()

        # 占用网络loss
        if self.use_occupancy_loss:
            loss_occ = self.fusion_vfe.get_occ_loss()
        
        # 总loss
        tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict_rpn
            }
        loss = loss_rpn

        if (not self.use_lidar_depth) and self.use_depth:
            tb_dict.update({
                'loss_depth': loss_depth.item(),
                **tb_dict_depth
            })

            loss += loss_depth

        if self.use_occupancy_loss:
            tb_dict.update({
                'loss_occ': loss_occ.item()
            })

            loss = loss + loss_occ

        return loss, tb_dict, disp_dict
