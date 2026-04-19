# ------------------------------------------------------------------#
# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
# HS-FPN
# ├── HFP (High Frequency Perception Module)
# │   ├── DctSpatialInteraction (Spatial Path of HFP)
# │   └── DctChannelInteraction (Channel Path of HFP)
# └── SDP&SDP_Large (Spatial Dependency Perception Module)
# ------------------------------------------------------------------#

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import torch_dct as DCT
except ImportError:
    DCT = None

# ----------------------------- ConvModule ----------------------------- #
# 优先使用 mmcv 的 ConvModule；若没有则用简易版替代
try:
    from mmcv.cnn import ConvModule as MMCVConvModule
except ImportError:
    MMCVConvModule = None


class ConvModule(nn.Module):
    """简化版 ConvModule，用于 mmcv 不可用时的 fallback."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=True):
        super().__init__()

        # 如果 mmcv.ConvModule 可用，并且我们想完全对齐，可以直接用它
        if MMCVConvModule is not None:
            self.mod = MMCVConvModule(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=inplace
            )
            self.use_mmcv = True
        else:
            self.use_mmcv = False
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            )

            # norm
            self.bn = None
            if norm_cfg is not None:
                norm_type = norm_cfg.get('type', 'BN')
                if norm_type == 'BN':
                    self.bn = nn.BatchNorm2d(out_channels)
                # 其他 norm 类型需要的话可以自行扩展

            # act
            self.act = None
            if act_cfg is not None:
                act_type = act_cfg.get('type', 'ReLU')
                if act_type == 'ReLU':
                    self.act = nn.ReLU(inplace=inplace)
                elif act_type == 'LeakyReLU':
                    self.act = nn.LeakyReLU(inplace=inplace)
                elif act_type == 'ReLU6':
                    self.act = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        if self.use_mmcv:
            return self.mod(x)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


# ----------------------------- BaseModule & auto_fp16 ----------------------------- #
# mmcv.runner 不一定存在，做一个轻量替代
try:
    from mmcv.runner import BaseModule, auto_fp16 as mmcv_auto_fp16
except ImportError:
    mmcv_auto_fp16 = None

    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()
            self.init_cfg = init_cfg

    def auto_fp16():
        """简单版 auto_fp16 装饰器：当 self.fp16_enabled 为 True 时，把 float32 输入 cast 成 float16."""
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if getattr(self, 'fp16_enabled', False):
                    def _cast(t):
                        if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
                            return t.half()
                        return t

                    args = tuple(_cast(a) for a in args)
                    kwargs = {k: _cast(v) for k, v in kwargs.items()}
                return func(self, *args, **kwargs)
            return wrapper
        return decorator
else:
    # 如果 mmcv 存在，就直接用原来的 auto_fp16
    def auto_fp16():
        return mmcv_auto_fp16()


__all__ = ['HS_FPN']

# ------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctSpatialInteraction(BaseModule):
    def __init__(self,
                 in_channels,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DctSpatialInteraction, self).__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct  # true when in p1&p2, false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
            )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))

        if DCT is None:
            raise RuntimeError("torch_dct is required for DctSpatialInteraction, "
                               "but not found. Please install torch_dct or add a fallback.")

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')  # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctChannelInteraction(BaseModule):
    def __init__(self,
                 in_channels,
                 patch,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(DctChannelInteraction, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.channel2x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:  # true when in p3&p4
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        if DCT is None:
            raise RuntimeError("torch_dct is required for DctChannelInteraction, "
                               "but not found. Please install torch_dct or add a fallback.")

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        # TODO: The values of aavgp and amaxp appear to be on different scales.
        # Add is a better choice instead of concat.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------#
# High Frequency Perception Module HFP
# ------------------------------------------------------------------#
class HFP(BaseModule):
    def __init__(self,
                 in_channels,
                 ratio,
                 patch=(8, 8),
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HFP, self).__init__(init_cfg)
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
              nn.GroupNorm(32, in_channels)]
        )

    def forward(self, x):
        spatial = self.spatial(x)   # output of spatial path
        channel = self.channel(x)   # output of channel path
        return self.out(spatial + channel)


# ------------------------------------------------------------------#
# Spatial Dependency Perception Module SDP
# ------------------------------------------------------------------#
class SDP(BaseModule):
    def __init__(self,
                 dim=256,
                 inter_dim=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SDP, self).__init__(init_cfg)
        self.inter_dim = inter_dim
        if self.inter_dim is None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(
            *[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False),
              nn.GroupNorm(32, self.inter_dim)]
        )
        self.conv_k = nn.Sequential(
            *[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False),
              nn.GroupNorm(32, self.inter_dim)]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_low, x_high, patch_size):
        """
        x_low: 低层特征 (N, C, H_l, W_l)
        x_high: 高层特征 (N, C, H_h, W_h)，可能与 x_low 尺寸不同
        patch_size: [p1, p2]，期望的 patch 尺寸（只是一个建议值）
        """
        # 1) 先把高层特征 resize 到和低层一致的空间尺寸，避免 B*h*w 不一致
        b_, _, h_, w_ = x_low.size()
        if x_high.shape[-2:] != (h_, w_):
            x_high = F.interpolate(
                x_high, size=(h_, w_), mode='bilinear', align_corners=False
            )

        # 2) 选择实际使用的 patch 大小，使得 H, W 能被 patch_size 整除，整除不了就退回 1
        p1, p2 = patch_size
        # 防止给的 patch 比特征还大
        p1 = max(1, min(p1, h_))
        p2 = max(1, min(p2, w_))

        if h_ % p1 != 0:
            p1 = 1
        if w_ % p2 != 0:
            p2 = 1

        # 3) 计算 q, k, 注意力
        q = rearrange(
            self.conv_q(x_low),
            'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
            p1=p1, p2=p2
        )
        q = q.transpose(1, 2)  # (B*H*W, patch_area, C)

        k = rearrange(
            self.conv_k(x_high),
            'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
            p1=p1, p2=p2
        )

        attn = torch.matmul(q, k)  # (B*H*W, patch_area, patch_area)
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)

        v = k.transpose(1, 2)  # (B*H*W, patch_area, C)
        output = torch.matmul(attn, v)  # (B*H*W, patch_area, C)

        output = rearrange(
            output.transpose(1, 2).contiguous(),
            '(b h w) c (p1 p2) -> b c (h p1) (w p2)',
            p1=p1, p2=p2,
            h=h_ // p1, w=w_ // p2
        )
        return output + x_low


# ------------------------------------------------------------------#
# Improved Version of SDP (2025-03-15)
# ------------------------------------------------------------------#
class SDP_Improved(BaseModule):
    def __init__(self,
                 dim=256,
                 inter_dim=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SDP_Improved, self).__init__(init_cfg)
        self.inter_dim = inter_dim
        if self.inter_dim is None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(
            *[ConvModule(dim, self.inter_dim, 3, padding=1, bias=False),
              nn.GroupNorm(32, self.inter_dim)]
        )
        self.conv_k = nn.Sequential(
            *[ConvModule(dim, self.inter_dim, 3, padding=1, bias=False),
              nn.GroupNorm(32, self.inter_dim)]
        )
        self.conv = nn.Sequential(
            *[ConvModule(self.inter_dim, dim, 3, padding=1, bias=False),
              nn.GroupNorm(32, dim)]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low),
                      'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
                      p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1, 2)  # (B*H*W, patch_area, C)

        k = rearrange(self.conv_k(x_high),
                      'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
                      p1=patch_size[0], p2=patch_size[1])

        attn = torch.matmul(q, k)  # (B*H*W, patch_area, patch_area)
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)

        v = k.transpose(1, 2)  # (B*H*W, patch_area, C)
        output = torch.matmul(attn, v)  # (B*H*W, patch_area, C)

        output = rearrange(
            output.transpose(1, 2).contiguous(),
            '(b h w) c (p1 p2) -> b c (h p1) (w p2)',
            p1=patch_size[0], p2=patch_size[1],
            h=h_ // patch_size[0], w=w_ // patch_size[1]
        )
        output = self.conv(output + x_low)
        return output


# ------------------------------------------------------------------#
# HS_FPN
# ------------------------------------------------------------------#
class HS_FPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 ratio=(0.25, 0.25),
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HS_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        def interpolate(input):
            up_mode = 'nearest'
            return F.interpolate(
                input,
                scale_factor=2,
                mode='nearest',
                align_corners=False if up_mode == 'bilinear' else None
            )

        self.fpn_upsample = interpolate

        # HFP (SelfAttn)
        self.SelfAttn_p4 = HFP(out_channels, ratio=None, isdct=False)
        self.SelfAttn_p3 = HFP(out_channels, ratio=None, isdct=False)
        self.SelfAttn_p2 = HFP(out_channels, ratio=ratio, patch=(8, 8), isdct=True)
        self.SelfAttn_p1 = HFP(out_channels, ratio=ratio, patch=(16, 16), isdct=True)

        # SDP (CrossAttn)
        self.CrossAtten_p4_p3 = SDP(dim=out_channels)
        self.CrossAtten_p3_p2 = SDP(dim=out_channels)
        self.CrossAtten_p2_p1 = SDP(dim=out_channels)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_ch = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_ch = out_channels
                extra_fpn_conv = ConvModule(
                    in_ch,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor]): feature maps from backbone,
                each is a 4D-tensor (N, C_i, H_i, W_i).
                一般情况下是 4 层: [C2, C3, C4, C5]
        Returns:
            tuple[Tensor]: FPN 输出的多尺度特征图
        """
        assert len(inputs) == len(self.in_channels)

        # ---------------- 1. build laterals ----------------
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # laterals: list of [P2, P3, P4, P5] (尚未融合，通道已变成 out_channels)

        # ---------------- 2. HFP + SDP 融合 ----------------
        # 原版代码使用 p4 的 h,w 构造 patch_size，但在很多分辨率下不整除会导致 q/k 维度不匹配。
        # 这里用最稳定的 patch_size = [1, 1]，等价于不分块，直接对每个像素做 cross-attn。
        patch_size = [1, 1]

        # laterals[3]: 最顶层 P5
        laterals[3] = self.SelfAttn_p4(laterals[3])
        # P4 <- P5
        laterals[2] = self.CrossAtten_p4_p3(
            self.SelfAttn_p3(laterals[2]),
            self.fpn_upsample(laterals[3]),
            patch_size
        )
        # P3 <- P4
        laterals[1] = self.CrossAtten_p3_p2(
            self.SelfAttn_p2(laterals[1]),
            self.fpn_upsample(laterals[2]),
            patch_size
        )
        # P2 <- P3
        laterals[0] = self.CrossAtten_p2_p1(
            self.SelfAttn_p1(laterals[0]),
            self.fpn_upsample(laterals[1]),
            patch_size
        )

        used_backbone_levels = len(laterals)

        # ---------------- 3. 标准 FPN 自顶向下加和 ----------------
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # ---------------- 4. build outputs (原始层) ----------------
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # ---------------- 5. extra levels (如果 num_outs > backbone 层数) ----------------
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs (e.g., Faster/Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)

