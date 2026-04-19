import torch
from torch import nn
from enum import Enum
from typing import List, Optional, Tuple, Union
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SubMConv3d, SparseConv2d, SparseMaxPool2d, SparseInverseConv2d
except:
    import spconv
    from spconv import ops
    from spconv import SubMConv3d, SparseConv2d, SparseMaxPool2d, SparseInverseConv2d

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}

def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_cfg=None):
    m = spconv.SparseSequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
    

def conv2D3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding to keep the same input and output"""
    assert stride >= 1
    padding = dilation
    if stride == 1:
        return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
    else:
        return spconv.SparseConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )


class Sparse2DBasicBlockV(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlockV, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv0 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = replace_feature(x, self.relu(x.features))
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlock, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class ConvAlgo(Enum):
    Native = 0
    MaskImplicitGemm = 1
    MaskSplitImplicitGemm = 2

class SpMiddlePillarEncoder18(nn.Module):
    def __init__(self,
                 in_planes=32, name="SpMiddlePillarEncoder18",
                 out_indices=[1, 2, 3], **kwargs):
        super(SpMiddlePillarEncoder18, self).__init__()
        self.name = name
        self.out_indices = out_indices

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                64, 128, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                128, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        # norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
        #     build_norm_layer(norm_cfg, 256)[1],
        #     nn.ReLU(),
        #     dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
        #     dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
        # )

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            # 'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            # 'x_conv5': 16,
        }

    def forward(self, sp_tensor):# [320, 320]
        x_conv1 = self.conv1(sp_tensor) # [B, 32, 320, 320]
        x_conv2 = self.conv2(x_conv1) # [B, 64, 160, 160]
        x_conv3 = self.conv3(x_conv2) # [B, 128, 80, 80]
        x_conv4 = self.conv4(x_conv3) # [B, 256, 40, 40]
        # x_conv4 = x_conv4.dense()
        # x_conv5 = self.conv5(x_conv4) # [B, 256, 20, 20]
        out_list = []
        for ind in self.out_indices:
            out_list.append(eval(f'x_conv{ind + 1}'))
        return out_list
