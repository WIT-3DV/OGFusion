import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalWinCrossAttnOcc(nn.Module):
    """
    img(Query) <- radar(Key/Value), 每个 query 只看 k×k 局部窗口
    occ 用作：
      1) attention bias（更偏向 occ 高的位置）
      2) 输出 gate（occ 小就少注入）
    为了省显存：在 downsample 后做注意力，再 upsample 回去
    """
    def __init__(self,
                 c_img: int,
                 c_rad: int,
                 d_model: int = 64,
                 win: int = 7,
                 down_ratio: int = 4,
                 occ_bias_scale: float = 2.0,
                 init_alpha: float = 0.1):
        super().__init__()
        assert win % 2 == 1, "win must be odd"
        assert d_model > 0

        self.d_model = d_model
        self.win = win
        self.pad = win // 2
        self.down_ratio = down_ratio
        self.occ_bias_scale = occ_bias_scale

        self.q_proj = nn.Conv2d(c_img, d_model, 1, bias=False)
        self.k_proj = nn.Conv2d(c_rad, d_model, 1, bias=False)
        self.v_proj = nn.Conv2d(c_rad, d_model, 1, bias=False)
        self.out_proj = nn.Conv2d(d_model, c_img, 1, bias=False)

        # 2D 特征更适合用 GroupNorm（比 LayerNorm 省事）
        g = 8
        if d_model % g != 0:
            g = 1
        self.norm_q = nn.GroupNorm(g, d_model)
        self.norm_k = nn.GroupNorm(g, d_model)

        # 注入强度（小起步更稳）
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        self.last_att = None
        self.last_logits = None
        self.last_occ_nb = None
        self.last_hw = None

    def forward(self, img_bev, rad_bev, occ_prob=None):
        """
        img_bev: [B, C_img, H, W]
        rad_bev: [B, C_rad, H, W]
        occ_prob: [B, 1, H, W] (0~1) or None
        """
        print(">>> LocalWinCrossAttnOcc forward called")

        B, _, H, W = img_bev.shape
        eps = 1e-6

        # ---- downsample ----
        if self.down_ratio > 1:
            img = F.avg_pool2d(img_bev, self.down_ratio)
            rad = F.max_pool2d(rad_bev, self.down_ratio)
            occ = F.max_pool2d(occ_prob, self.down_ratio) if occ_prob is not None else None
        else:
            img, rad, occ = img_bev, rad_bev, occ_prob

        q = self.norm_q(self.q_proj(img))  # [B,d,h,w]
        k = self.norm_k(self.k_proj(rad))  # [B,d,h,w]
        v = self.v_proj(rad)  # [B,d,h,w]

        B, d, h, w = q.shape
        HW = h * w
        P = self.win * self.win

        k_nb = F.unfold(k, kernel_size=self.win, padding=self.pad)
        v_nb = F.unfold(v, kernel_size=self.win, padding=self.pad)

        k_nb = k_nb.view(B, d, P, HW).permute(0, 3, 2, 1).contiguous()
        v_nb = v_nb.view(B, d, P, HW).permute(0, 3, 2, 1).contiguous()

        q_flat = q.view(B, d, HW).permute(0, 2, 1).contiguous()

        logits = (q_flat.unsqueeze(2) * k_nb).sum(-1) / math.sqrt(d)

        occ_nb = None
        if occ is not None:
            occ_nb = F.unfold(occ, kernel_size=self.win, padding=self.pad)
            occ_nb = occ_nb.permute(0, 2, 1).contiguous()
            logits = logits + self.occ_bias_scale * torch.log(occ_nb.clamp(min=eps))

        att = torch.softmax(logits, dim=-1)
        print(">>> att shape =", att.shape)

        # 缓存给可视化脚本
        self.last_att = att.detach().cpu()
        self.last_logits = logits.detach().cpu()
        self.last_hw = (h, w)
        self.last_occ_nb = occ_nb.detach().cpu() if occ_nb is not None else None

        msg = (att.unsqueeze(-1) * v_nb).sum(2)

        msg = msg.permute(0, 2, 1).contiguous().view(B, d, h, w)
        msg = self.out_proj(msg)

        if occ is not None:
            gate = occ.clamp(0, 1)
            msg = msg * gate

        if self.down_ratio > 1:
            msg = F.interpolate(msg, size=(H, W), mode="bilinear", align_corners=False)

        return img_bev + self.alpha * msg

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)


class FusionAfterBEVSEDirect(nn.Module):
    """
    你的原版融合 + 方案A：局部窗口 cross-attn + occ
    """
    def __init__(self, model_cfg, num_bev_features,
                 image_in_channels,
                 image_out_channels,
                 radar_in_channels,
                 radar_out_channels,
                 **kwargs):
        super().__init__()

        if isinstance(image_in_channels, list):
            image_in_channels = sum(image_in_channels)

        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features

        # 如果 image_in != image_out，就先 1x1 适配
        if image_in_channels != image_out_channels:
            self.img_conv = nn.Sequential(
                nn.Conv2d(image_in_channels, image_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(image_out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.img_conv = nn.Identity()

        # ===== 方案A：局部窗口 cross-attn =====
        self.use_cross_attn = True
        self.occ_key = model_cfg.get("OCC_KEY", "radar_occ_prob")  # 你自己统一一下 key
        self.align = LocalWinCrossAttnOcc(
            c_img=image_out_channels,
            c_rad=radar_in_channels,
            d_model=model_cfg.get("XATTN_D_MODEL", 64),
            win=model_cfg.get("XATTN_WIN", 7),
            down_ratio=model_cfg.get("XATTN_DOWN", 4),
            occ_bias_scale=model_cfg.get("XATTN_OCC_BIAS", 2.0),
            init_alpha=model_cfg.get("XATTN_ALPHA_INIT", 0.1),
        )
        # ====================================

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                      image_out_channels + radar_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU(inplace=True)
        )
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')

    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"]              # [B, C_img_in, H, W]
        radar_features = batch_dict["pillar_features_scattered"]     # [B, C_rad_in, h, w]

        occ = batch_dict.get(self.occ_key, None)                     # [B,1,h,w] or None

        # 先做通道适配
        image_features = self.img_conv(image_features)

        # 尺寸对齐到 radar
        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features,
                                           size=radar_features.shape[-2:],
                                           mode='bilinear',
                                           align_corners=False)

        # occ 尺寸也对齐（若存在）
        if occ is not None and occ.shape[-2:] != radar_features.shape[-2:]:
            occ = F.interpolate(occ,
                                size=radar_features.shape[-2:],
                                mode='bilinear',
                                align_corners=False)

        # 方案A：img <- radar 的局部窗口 cross-attn（带 occ bias + gate）
        if self.use_cross_attn:
            image_features = self.align(image_features, radar_features, occ_prob=occ)

        # 你原来的 concat + conv + SE
        fuse_features = torch.cat([image_features, radar_features], dim=1)
        fuse_features = self.fuse_conv(fuse_features)
        fuse_features = self.se_block(fuse_features)

        batch_dict[self.feature_name] = fuse_features
        return batch_dict
