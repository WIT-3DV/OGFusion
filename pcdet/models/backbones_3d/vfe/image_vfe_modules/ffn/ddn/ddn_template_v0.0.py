from collections import OrderedDict
from pathlib import Path
from torch import hub

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class DDNTemplate(nn.Module):

    def __init__(self, constructor,  num_classes, feat_extract_layer='', pretrained_path=None, aux_loss=None, use_lidar_depth=False,
                 freeze_backbone=False, use_pooling=False, use_depth=True):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        self.use_lidar_depth = use_lidar_depth
        self.freeze_backbone = freeze_backbone
        self.use_pooling = use_pooling
        self.use_depth = use_depth

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # Model
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        
        if self.freeze_backbone:
            self.freeze_backbone_func()
            
        if isinstance(feat_extract_layer, list):
            self.model.backbone.return_layers.update({
                layer_name: f'features_{index}' for index, layer_name in enumerate(feat_extract_layer)
            })
        else:
            if self.use_lidar_depth or (not self.use_depth):
                self.model_simple = torch.nn.Sequential(
                    self.model.backbone.conv1,
                    self.model.backbone.bn1,
                    self.model.backbone.relu,
                    self.model.backbone.maxpool,
                    self.model.backbone.layer1
                )
                del self.model.backbone
                del self.model.classifier
            else:
                self.model.backbone.return_layers = {
                    feat_extract_layer: 'features',
                    **self.model.backbone.return_layers
                }

    def freeze_backbone_func(self):
        for name, para in self.model.named_parameters():
            if 'backbone' in name:
                para.requires_grad_(False)
                
    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        # Get model
        model = constructor(pretrained=False,
                            pretrained_backbone=False,
                            num_classes=self.num_classes,
                            aux_loss=self.aux_loss)

        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            
            # Download pretrained model if not available yet
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict,
                                                          pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if "aux_classifier" not in key}

        # Removes final conv layer from weights if number of classes are different
        model_num_classes = model_dict["classifier.4.weight"].shape[0]
        pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop("classifier.4.weight")
            pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        x = self.preprocess(images)

        # Extract features
        result = OrderedDict()

        if isinstance(self.feat_extract_layer, list):
            features = self.model.backbone(x)
            for index, feature_name in enumerate(self.feat_extract_layer): # [B, 256, 129, 484] [B, 512, 65, 242] [B, 1024, 65, 242] [B, 2048, 65, 242]
                result[feature_name] = features[f'features_{index}']
            if self.use_lidar_depth or (not self.use_depth):
                pass
            else:
                x = features[f'features_{len(self.feat_extract_layer) - 1}']
                feat_shape = features['features_0'].shape[-2:]
                x = self.model.classifier(x) # [2, 81, 65, 242]
                result["logits_small"] = x
                x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
                result["logits"] = x
        else:
            if self.use_lidar_depth or (not self.use_depth):
                features = self.model_simple(x)
                result['features'] = features # [1, 256, 129, 484]
            else:
                features = self.model.backbone(x)
                result['features'] = features['features'] # [1, 256, 129, 484]
                feat_shape = features['features'].shape[-2:]
                # Prediction classification logits
                x = features["out"] # [1, 2048, 65, 242]
                x = self.model.classifier(x) # [2, 81, 65, 242]
                x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
                result["logits"] = x

        # Prediction auxillary classification logits
        if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            # Create a mask for padded pixels
            mask = (x == 0)

            # Match ResNet pretrained preprocessing
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            x[mask] = 0

        return x
