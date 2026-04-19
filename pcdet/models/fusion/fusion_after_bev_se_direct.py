# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class LocalWinCrossAttnOcc(nn.Module):
#     """
#     img(Query) <- radar(Key/Value), 每个 query 只看 k×k 局部窗口
#     occ 用作：
#       1) attention bias（更偏向 occ 高的位置）
#       2) 输出 gate（occ 小就少注入）
#     为了省显存：在 downsample 后做注意力，再 upsample 回去
#     """
#     def __init__(self,
#                  c_img: int,
#                  c_rad: int,
#                  d_model: int = 64,
#                  win: int = 11,
#                  down_ratio: int = 4,
#                  occ_bias_scale: float = 2.0,
#                  init_alpha: float = 0.1):
#         super().__init__()
#         assert win % 2 == 1, "win must be odd"
#         assert d_model > 0
#
#         self.d_model = d_model
#         self.win = win
#         self.pad = win // 2
#         self.down_ratio = down_ratio
#         self.occ_bias_scale = occ_bias_scale
#
#         self.q_proj = nn.Conv2d(c_img, d_model, 1, bias=False)
#         self.k_proj = nn.Conv2d(c_rad, d_model, 1, bias=False)
#         self.v_proj = nn.Conv2d(c_rad, d_model, 1, bias=False)
#         self.out_proj = nn.Conv2d(d_model, c_img, 1, bias=False)
#
#         # 2D 特征更适合用 GroupNorm（比 LayerNorm 省事）
#         g = 8
#         if d_model % g != 0:
#             g = 1
#         self.norm_q = nn.GroupNorm(g, d_model)
#         self.norm_k = nn.GroupNorm(g, d_model)
#
#         # 注入强度（小起步更稳）
#         self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
#
#     def forward(self, img_bev, rad_bev, occ_prob=None):
#         """
#         img_bev: [B, C_img, H, W]
#         rad_bev: [B, C_rad, H, W]
#         occ_prob: [B, 1, H, W] (0~1) or None
#         """
#         B, _, H, W = img_bev.shape
#         eps = 1e-6
#
#         # ---- downsample ----
#         if self.down_ratio > 1:
#             img = F.avg_pool2d(img_bev, self.down_ratio)
#             # radar 稀疏时 max_pool 往往更靠谱（保峰值）
#             rad = F.max_pool2d(rad_bev, self.down_ratio)
#             occ = F.max_pool2d(occ_prob, self.down_ratio) if occ_prob is not None else None
#         else:
#             img, rad, occ = img_bev, rad_bev, occ_prob
#
#         q = self.norm_q(self.q_proj(img))   # [B,d,h,w]
#         k = self.norm_k(self.k_proj(rad))   # [B,d,h,w]
#         v = self.v_proj(rad)                # [B,d,h,w]
#
#         B, d, h, w = q.shape
#         HW = h * w
#         P = self.win * self.win
#
#         # ---- unfold: 每个位置取局部窗口 ----
#         # K/V: [B, d*P, HW]
#         k_nb = F.unfold(k, kernel_size=self.win, padding=self.pad)
#         v_nb = F.unfold(v, kernel_size=self.win, padding=self.pad)
#
#         # reshape -> [B, HW, P, d]
#         k_nb = k_nb.view(B, d, P, HW).permute(0, 3, 2, 1).contiguous()
#         v_nb = v_nb.view(B, d, P, HW).permute(0, 3, 2, 1).contiguous()
#
#         # Q: [B, HW, d]
#         q_flat = q.view(B, d, HW).permute(0, 2, 1).contiguous()
#
#         # ---- attention logits: [B, HW, P] ----
#         logits = (q_flat.unsqueeze(2) * k_nb).sum(-1) / math.sqrt(d)
#
#         # ---- occ bias（窗口内 occ 越大越容易被关注）----
#         if occ is not None:
#             occ_nb = F.unfold(occ, kernel_size=self.win, padding=self.pad)  # [B, P, HW]
#             occ_nb = occ_nb.permute(0, 2, 1).contiguous()                   # [B, HW, P]
#             logits = logits + self.occ_bias_scale * torch.log(occ_nb.clamp(min=eps))
#
#         att = torch.softmax(logits, dim=-1)  # [B, HW, P]
#
#         # ---- message: [B, HW, d] ----
#         msg = (att.unsqueeze(-1) * v_nb).sum(2)
#
#         # ---- back to map: [B, d, h, w] -> [B, C_img, h, w] ----
#         msg = msg.permute(0, 2, 1).contiguous().view(B, d, h, w)
#         msg = self.out_proj(msg)
#
#         # ---- occ gate：occ 小就少注入（防噪声）----
#         if occ is not None:
#             gate = occ.clamp(0, 1)          # [B,1,h,w]
#             msg = msg * gate
#
#         # ---- upsample back ----
#         if self.down_ratio > 1:
#             msg = F.interpolate(msg, size=(H, W), mode="bilinear", align_corners=False)
#
#         return img_bev + self.alpha * msg
#
#
# class SE_Block(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, c, kernel_size=1, stride=1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         return x * self.att(x)
#
#
# class FusionAfterBEVSEDirect(nn.Module):
#     """
#     你的原版融合 + 方案A：局部窗口 cross-attn + occ
#     """
#     def __init__(self, model_cfg, num_bev_features,
#                  image_in_channels,
#                  image_out_channels,
#                  radar_in_channels,
#                  radar_out_channels,
#                  **kwargs):
#         super().__init__()
#
#         if isinstance(image_in_channels, list):
#             image_in_channels = sum(image_in_channels)
#
#         self.model_cfg = model_cfg
#         self.num_bev_features = num_bev_features
#
#         # 如果 image_in != image_out，就先 1x1 适配
#         if image_in_channels != image_out_channels:
#             self.img_conv = nn.Sequential(
#                 nn.Conv2d(image_in_channels, image_out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(image_out_channels),
#                 nn.ReLU(inplace=True)
#             )
#         else:
#             self.img_conv = nn.Identity()
#
#         # ===== 方案A：局部窗口 cross-attn =====
#         self.use_cross_attn = True
#         self.occ_key = model_cfg.get("OCC_KEY", "radar_occ_prob")  # 你自己统一一下 key
#         self.align = LocalWinCrossAttnOcc(
#             c_img=image_out_channels,
#             c_rad=radar_in_channels,
#             d_model=model_cfg.get("XATTN_D_MODEL", 64),
#             win=model_cfg.get("XATTN_WIN", 7),
#             down_ratio=model_cfg.get("XATTN_DOWN", 4),
#             occ_bias_scale=model_cfg.get("XATTN_OCC_BIAS", 2.0),
#             init_alpha=model_cfg.get("XATTN_ALPHA_INIT", 0.1),
#         )
#         # ====================================
#
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(image_out_channels + radar_in_channels,
#                       image_out_channels + radar_out_channels,
#                       kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(image_out_channels + radar_out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.se_block = SE_Block(image_out_channels + radar_out_channels)
#         self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
#
#     def forward(self, batch_dict):
#         image_features = batch_dict["spatial_features"]              # [B, C_img_in, H, W]
#         radar_features = batch_dict["pillar_features_scattered"]     # [B, C_rad_in, h, w]
#
#         occ = batch_dict.get(self.occ_key, None)                     # [B,1,h,w] or None
#
#         # 先做通道适配
#         image_features = self.img_conv(image_features)
#
#         # 尺寸对齐到 radar
#         if image_features.shape[-2:] != radar_features.shape[-2:]:
#             image_features = F.interpolate(image_features,
#                                            size=radar_features.shape[-2:],
#                                            mode='bilinear',
#                                            align_corners=False)
#
#         # occ 尺寸也对齐（若存在）
#         if occ is not None and occ.shape[-2:] != radar_features.shape[-2:]:
#             occ = F.interpolate(occ,
#                                 size=radar_features.shape[-2:],
#                                 mode='bilinear',
#                                 align_corners=False)
#
#         # 方案A：img <- radar 的局部窗口 cross-attn（带 occ bias + gate）
#         if self.use_cross_attn:
#             image_features = self.align(image_features, radar_features, occ_prob=occ)
#
#         # 你原来的 concat + conv + SE
#         fuse_features = torch.cat([image_features, radar_features], dim=1)
#         fuse_features = self.fuse_conv(fuse_features)
#         fuse_features = self.se_block(fuse_features)
#
#         batch_dict[self.feature_name] = fuse_features
#         return batch_dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_global_bev_attention_map(att, h, w, win, normalize=True):
    """
    将局部窗口注意力 [HW, P] 聚合为整张 BEV 热力图 [h, w]
    注意：这个函数保留，方便你以后还想看 attention。
    但当前更推荐看 message map。

    Args:
        att: [HW, P]
        h, w: spatial size
        win: window size
        normalize: whether normalize to [0,1]

    Returns:
        heat: [h, w]
    """
    assert att.dim() == 2, f"att should be [HW, P], got {att.shape}"
    HW, P = att.shape
    assert HW == h * w, f"HW mismatch: {HW} vs {h}*{w}"
    assert P == win * win, f"P mismatch: {P} vs {win}*{win}"

    device = att.device
    pad = win // 2

    heat = torch.zeros((h, w), device=device, dtype=att.dtype)
    count = torch.zeros((h, w), device=device, dtype=att.dtype)

    idx = 0
    for cy in range(h):
        for cx in range(w):
            local_att = att[idx].view(win, win)
            idx += 1

            for dy in range(win):
                for dx in range(win):
                    yy = cy + dy - pad
                    xx = cx + dx - pad
                    if 0 <= yy < h and 0 <= xx < w:
                        heat[yy, xx] += local_att[dy, dx]
                        count[yy, xx] += 1.0

    heat = heat / count.clamp(min=1.0)

    if normalize:
        heat = heat - heat.min()
        max_val = heat.max()
        if max_val > 0:
            heat = heat / max_val

    return heat


class LocalWinCrossAttnOcc(nn.Module):
    """
    img(Query) <- radar(Key/Value), 每个 query 只看 k×k 局部窗口
    occ 用作：
      1) attention bias（更偏向 occ 高的位置）
      2) 输出 gate（occ 小就少注入）

    为了省显存：在 downsample 后做注意力，再 upsample 回去

    新增：
      - 自动缓存注意力和 message，便于推理阶段可视化
    """
    def __init__(self,
                 c_img: int,
                 c_rad: int,
                 d_model: int = 64,
                 win: int = 11,
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

        g = 8
        if d_model % g != 0:
            g = 1
        self.norm_q = nn.GroupNorm(g, d_model)
        self.norm_k = nn.GroupNorm(g, d_model)

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        # ===== 可视化缓存 =====
        self.last_att = None              # [B, HW, P]
        self.last_occ = None              # [B, 1, h, w] or None
        self.last_msg_pre_gate = None     # [B, C_img, h, w]
        self.last_msg_post_gate = None    # [B, C_img, h, w]
        self.last_msg_up = None           # [B, C_img, H, W]
        self.last_hw = None               # (h, w)
        self.last_input_hw = None         # (H, W)
        # =====================

    def forward(self, img_bev, rad_bev, occ_prob=None):
        """
        img_bev: [B, C_img, H, W]
        rad_bev: [B, C_rad, H, W]
        occ_prob: [B, 1, H, W] or None
        """
        B, _, H, W = img_bev.shape
        eps = 1e-6

        # ---- downsample ----
        if self.down_ratio > 1:
            img = F.avg_pool2d(img_bev, self.down_ratio)
            rad = F.max_pool2d(rad_bev, self.down_ratio)
            occ = F.max_pool2d(occ_prob, self.down_ratio) if occ_prob is not None else None
        else:
            img, rad, occ = img_bev, rad_bev, occ_prob

        q = self.norm_q(self.q_proj(img))   # [B,d,h,w]
        k = self.norm_k(self.k_proj(rad))   # [B,d,h,w]
        v = self.v_proj(rad)                # [B,d,h,w]

        B, d, h, w = q.shape
        HW = h * w
        P = self.win * self.win

        # ---- unfold ----
        k_nb = F.unfold(k, kernel_size=self.win, padding=self.pad)   # [B, d*P, HW]
        v_nb = F.unfold(v, kernel_size=self.win, padding=self.pad)   # [B, d*P, HW]

        k_nb = k_nb.view(B, d, P, HW).permute(0, 3, 2, 1).contiguous()  # [B, HW, P, d]
        v_nb = v_nb.view(B, d, P, HW).permute(0, 3, 2, 1).contiguous()  # [B, HW, P, d]

        q_flat = q.view(B, d, HW).permute(0, 2, 1).contiguous()          # [B, HW, d]

        # ---- attention logits ----
        logits = (q_flat.unsqueeze(2) * k_nb).sum(-1) / math.sqrt(d)     # [B, HW, P]

        if occ is not None:
            occ_nb = F.unfold(occ, kernel_size=self.win, padding=self.pad)   # [B, P, HW]
            occ_nb = occ_nb.permute(0, 2, 1).contiguous()                    # [B, HW, P]
            logits = logits + self.occ_bias_scale * torch.log(occ_nb.clamp(min=eps))

        att = torch.softmax(logits, dim=-1)  # [B, HW, P]

        # ---- message ----
        msg = (att.unsqueeze(-1) * v_nb).sum(2)                             # [B, HW, d]
        msg = msg.permute(0, 2, 1).contiguous().view(B, d, h, w)            # [B, d, h, w]
        msg = self.out_proj(msg)                                             # [B, C_img, h, w]

        msg_pre_gate = msg

        if occ is not None:
            gate = occ.clamp(0, 1)
            msg = msg * gate

        msg_post_gate = msg

        if self.down_ratio > 1:
            msg = F.interpolate(msg, size=(H, W), mode="bilinear", align_corners=False)

        # ===== 缓存 =====
        self.last_att = att.detach()
        self.last_occ = occ.detach() if occ is not None else None
        self.last_msg_pre_gate = msg_pre_gate.detach()
        self.last_msg_post_gate = msg_post_gate.detach()
        self.last_msg_up = msg.detach()
        self.last_hw = (h, w)
        self.last_input_hw = (H, W)
        # ================

        return img_bev + self.alpha * msg

    @torch.no_grad()
    def get_last_global_attention_map(self, batch_idx=0, upsample=False, normalize=True):
        """
        从缓存中生成某个样本的 attention heatmap
        """
        assert self.last_att is not None, "No cached attention found. Please run forward first."
        assert self.last_hw is not None, "No cached spatial size found."

        h, w = self.last_hw
        att = self.last_att[batch_idx]
        heat = build_global_bev_attention_map(att, h, w, self.win, normalize=normalize)

        if upsample:
            assert self.last_input_hw is not None, "No cached input size found."
            H, W = self.last_input_hw
            heat = F.interpolate(
                heat.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )[0, 0]

            if normalize:
                heat = heat - heat.min()
                max_val = heat.max()
                if max_val > 0:
                    heat = heat / max_val

        return heat

    @torch.no_grad()
    def get_last_msg_map(self, batch_idx=0, upsample=False, use_post_gate=True, reduce="mean_abs"):
        """
        获取 message response map，当前最推荐可视化这个

        Args:
            batch_idx: batch index
            upsample: whether upsample to input size
            use_post_gate: True -> 用 gate 后的 msg；False -> 用 gate 前的 msg
            reduce:
                - "mean_abs": abs 后按通道平均
                - "sum_abs": abs 后按通道求和
                - "l2": 通道 L2 范数

        Returns:
            msg_map: [h, w] or [H, W]
        """
        feat = self.last_msg_post_gate if use_post_gate else self.last_msg_pre_gate
        assert feat is not None, "No cached message found. Please run forward first."

        x = feat[batch_idx]   # [C, h, w]

        if reduce == "mean_abs":
            msg_map = x.abs().mean(dim=0)
        elif reduce == "sum_abs":
            msg_map = x.abs().sum(dim=0)
        elif reduce == "l2":
            msg_map = torch.sqrt((x ** 2).sum(dim=0) + 1e-12)
        else:
            raise ValueError(f"Unsupported reduce type: {reduce}")

        if upsample:
            assert self.last_input_hw is not None, "No cached input size found."
            H, W = self.last_input_hw
            msg_map = F.interpolate(
                msg_map.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )[0, 0]

        return msg_map


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
    原版融合 + 局部窗口 cross-attn + occ

    新增：
      - 将可视化结果写入 batch_dict，便于 vis 脚本直接读取
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

        if image_in_channels != image_out_channels:
            self.img_conv = nn.Sequential(
                nn.Conv2d(image_in_channels, image_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(image_out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.img_conv = nn.Identity()

        self.use_cross_attn = True
        self.occ_key = model_cfg.get("OCC_KEY", "radar_occ_prob")

        self.align = LocalWinCrossAttnOcc(
            c_img=image_out_channels,
            c_rad=radar_in_channels,
            d_model=model_cfg.get("XATTN_D_MODEL", 64),
            win=model_cfg.get("XATTN_WIN", 7),
            down_ratio=model_cfg.get("XATTN_DOWN", 4),
            occ_bias_scale=model_cfg.get("XATTN_OCC_BIAS", 2.0),
            init_alpha=model_cfg.get("XATTN_ALPHA_INIT", 0.1),
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                      image_out_channels + radar_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU(inplace=True)
        )
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')

        self.export_vis = self.model_cfg.get("EXPORT_VIS", True)

    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"]              # [B, C_img_in, H, W]
        radar_features = batch_dict["pillar_features_scattered"]     # [B, C_rad_in, h, w]
        occ = batch_dict.get(self.occ_key, None)                     # [B,1,h,w] or None

        image_features = self.img_conv(image_features)

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(
                image_features,
                size=radar_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        if occ is not None and occ.shape[-2:] != radar_features.shape[-2:]:
            occ = F.interpolate(
                occ,
                size=radar_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        if self.use_cross_attn:
            image_features = self.align(image_features, radar_features, occ_prob=occ)

            if self.export_vis:
                batch_dict["vis_att"] = self.align.last_att
                batch_dict["vis_occ"] = self.align.last_occ
                batch_dict["vis_msg_pre_gate"] = self.align.last_msg_pre_gate
                batch_dict["vis_msg_post_gate"] = self.align.last_msg_post_gate
                batch_dict["vis_msg_up"] = self.align.last_msg_up
                batch_dict["vis_hw"] = self.align.last_hw
                batch_dict["vis_input_hw"] = self.align.last_input_hw

        fuse_features = torch.cat([image_features, radar_features], dim=1)
        fuse_features = self.fuse_conv(fuse_features)
        fuse_features = self.se_block(fuse_features)

        batch_dict[self.feature_name] = fuse_features
        return batch_dict