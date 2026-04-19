import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion_MAP_TO_BEV(nn.Module):
    def __init__(self, model_cfg, Image_MAP_TO_BEV, Radar_MAP_TO_BEV, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        if Image_MAP_TO_BEV is None and Radar_MAP_TO_BEV is None:
            self.did_nothing = True
        else:
            self.map_module_list = [Image_MAP_TO_BEV, Radar_MAP_TO_BEV]
            self.add_module('Image_MAP_TO_BEV', Image_MAP_TO_BEV)
            self.add_module('Radar_MAP_TO_BEV', Radar_MAP_TO_BEV)
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
    
    def forward(self, batch_dict):
        batch_dict['image_features'] = self.map_module_list[0](batch_dict)['spatial_features'] # [2,80,160,160]

        if batch_dict.get('pillar_features_scattered', None) is None: # PointPillar 需要maptobev
            batch_dict['radar_features'] = self.map_module_list[1](batch_dict)['spatial_features'] # [2,64,320,320]
        else: # pillarnet 已经生成了bev feature
            batch_dict['radar_features'] = batch_dict['pillar_features_scattered'] # [2,128,160,160]

        if batch_dict['radar_features'].shape[-2:] != batch_dict['image_features'].shape[-2:]:
            batch_dict['radar_features'] = F.interpolate(batch_dict['radar_features'], size=batch_dict['image_features'].shape[-2:])

        batch_dict['spatial_features'] = torch.concat([batch_dict['image_features'], batch_dict['radar_features']], dim=1)
        # batch_dict['spatial_features'] = torch.concat([Image_spatial_features, Radar_spatial_features], dim=1) # [2,208,160,160]
        return batch_dict
