# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

# kept for compatibility with older forks that import it
from torch.utils.checkpoint import checkpoint  # noqa: F401


# -----------------------------
# PillarDAN-style additions
# -----------------------------
class CoordEmbedding2D(nn.Module):
    """Add 2D coordinate embedding to a BEV feature map.

    Creates normalized (x, y) grids in [-1, 1], projects to C channels with 1x1 conv,
    and adds to input feature map (position awareness).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(2, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device

        xs = torch.linspace(-1.0, 1.0, w, device=device)
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)  # [B,2,H,W]

        return x + self.proj(grid)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """Channel + spatial attention (PFA-like)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class GlobalContextBlock(nn.Module):
    """Lightweight global interaction (GPA-like).

    Computes an attention-pooled global context vector over (H,W),
    transforms it, then adds back to x. Cheap alternative to full Transformer.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mask = self.conv_mask(x).view(b, 1, -1)              # [B,1,HW]
        attn = torch.softmax(mask, dim=-1)                   # [B,1,HW]
        x_flat = x.view(b, c, -1)                            # [B,C,HW]
        ctx = torch.bmm(x_flat, attn.transpose(1, 2))        # [B,C,1]
        ctx = ctx.view(b, c, 1, 1)
        return x + self.transform(ctx)


class FPN_LSS(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False,
                 # ---- New (optional) knobs for method-2 (PillarDAN-like) ----
                 use_coord_emb=True,
                 use_gpa=True,
                 use_pfa=True,
                 attn_reduction=16):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1

        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor

        self.conv34 = nn.Sequential(
            nn.Conv2d(
                in_channels[1],
                out_channels[1] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[1] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels[1] * channels_factor,
                out_channels[1] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[1] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )

        self.conv23 = nn.Sequential(
            nn.Conv2d(
                in_channels[0],
                out_channels[0] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[0] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels[0] * channels_factor,
                out_channels[0] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[0] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

        # -----------------------------
        # Method-2: apply on dense BEV x23
        # -----------------------------
        out_c = out_channels[0] * channels_factor
        self.use_coord_emb = bool(use_coord_emb)
        self.use_gpa = bool(use_gpa)
        self.use_pfa = bool(use_pfa)

        self.coord_emb = CoordEmbedding2D(out_c) if self.use_coord_emb else None
        self.gpa = GlobalContextBlock(out_c, reduction=attn_reduction) if self.use_gpa else None
        self.pfa = CBAM(out_c, reduction=attn_reduction) if self.use_pfa else None

    def forward(self, feats):
        # feats are sparse tensors from spconv backbone
        x2, x3, x4 = feats  # [B, 64, 160, 160] [B, 128, 80, 80] [B, 256, 40, 40]
        x2 = x2.dense()
        x3 = x3.dense()
        x4 = x4.dense()

        x4 = self.up(x4)  # [B, 256, 80, 80]

        x34 = torch.cat([x3, x4], dim=1)  # [B, 384, 80, 80]
        x34 = self.conv34(x34)            # [B, 128, 80, 80]
        x34 = self.up(x34)                # [B, 128, 160, 160]

        if x34.shape[-2:] != x2.shape[-2:]:
            x34 = nn.functional.interpolate(x34, size=x2.shape[-2:], mode='bilinear', align_corners=True)

        x23 = torch.cat([x2, x34], dim=1)  # [B, 192, 160, 160]
        x23 = self.conv23(x23)             # [B, 128, 160, 160]

        # ---- Method-2 additions (PillarDAN-like) ----
        if self.coord_emb is not None:
            x23 = self.coord_emb(x23)
        if self.gpa is not None:
            x23 = self.gpa(x23)
        if self.pfa is not None:
            x23 = self.pfa(x23)

        return x23
