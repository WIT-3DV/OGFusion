import torch
import torch.nn as nn
import torch.nn.functional as F

from .hmb_lite import HybridMambaBlockLite


class FusionAfterBEVSEDirect(nn.Module):
    """
    Radar BEV + Image BEV 融合（单帧）：
    - Radar BEV 做 Query
    - Image BEV 做 Key / Value
    - 沿着 X 方向（一行一行）做 cross-attention

    方案：Cross-Attn + HMB（两路残差）
      - delta_attn: 由 cross-attn 后的 radar 表示直接映射得到（baseline 主路）
      - delta_hmb : 由 HMB 输出映射得到（增益分支）
      - radar_enhanced = radar + gamma * gate * delta_attn + gamma_hmb * gate * delta_hmb

    其它：
      - Progressive Focus: 每行 top-K image token
      - Gate: 由 [radar, image] 控制 correction 强度
      - range mask: 远处衰减（可选）
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

        # ===== 1) 注意力内部维度 =====
        attn_channels = min(image_out_channels, 128)
        self.attn_channels = attn_channels

        # ===== 2) 自动选择 num_heads =====
        num_heads = 1
        for h in [8, 4, 2, 1]:
            if attn_channels % h == 0:
                num_heads = h
                break
        self.num_heads = num_heads

        # ===== 3) 投影到注意力空间 =====
        self.img_proj = nn.Conv2d(image_out_channels, attn_channels, kernel_size=1, bias=False)
        self.rad_proj = nn.Conv2d(radar_in_channels, attn_channels, kernel_size=1, bias=False)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1_q = nn.LayerNorm(attn_channels)
        self.norm1_kv = nn.LayerNorm(attn_channels)

        self.ffn = nn.Sequential(
            nn.Linear(attn_channels, attn_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(attn_channels * 4, attn_channels),
        )
        self.norm2 = nn.LayerNorm(attn_channels)

        # ===== 4) Progressive Focus =====
        self.focus_ratio = 0.5
        self.min_focus_tokens = 16

        hidden = max(attn_channels // 2, 1)
        self.score_mlp = nn.Sequential(
            nn.Linear(attn_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

        # ===== 5) HMB（增益分支）=====
        # 你要更轻量：bidir=False, global_stride=4, global_dim=32
        # 你要更强：bidir=True, global_stride=2, global_dim=64
        self.hmb = HybridMambaBlockLite(
            d_model=self.attn_channels,
            d_state=16,
            d_conv=4,
            expand=2,
            bidir=True,
            use_local=True,
            local_dirs="x",
            use_global=True,
            global_stride=2,
            global_dim=min(64, self.attn_channels),
            global_order="snake",
        )

        # ===== 6) 两路 delta 映射回 radar_out_channels =====
        # delta_attn: cross-attn 后的 radar 表示 -> radar_out_channels
        # delta_hmb : HMB 输出 -> radar_out_channels
        self.rad_delta_proj = nn.Conv2d(attn_channels, radar_out_channels, kernel_size=1, bias=False)

        # 两个缩放因子：主路 gamma（attn），增益 gamma_hmb（hmb）
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_hmb = nn.Parameter(torch.zeros(1))

        # ===== 7) gate =====
        self.gate_conv = nn.Sequential(
            nn.Conv2d(radar_out_channels + image_out_channels, radar_out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # ===== 8) range mask =====
        self.use_range_mask = True
        self.range_decay = 0.7

        # ===== 9) 最终融合卷积 =====
        out_channels = radar_out_channels + image_out_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if hasattr(self.model_cfg, "get"):
            self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
        else:
            self.feature_name = 'spatial_features_2d'

    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"]              # [B, C_img, H, W] (可能需插值)
        radar_features = batch_dict['pillar_features_scattered']     # [B, C_r,   H, W]

        # 对齐分辨率
        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(
                image_features,
                radar_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        B, C_r, H, W = radar_features.shape

        # ====== 1) 投影到 attn 空间 ======
        img_attn = self.img_proj(image_features)    # [B, C_attn, H, W]
        rad_attn = self.rad_proj(radar_features)    # [B, C_attn, H, W]

        # ====== 2) 行级序列化 ======
        img_seq = img_attn.permute(0, 2, 3, 1).reshape(B * H, W, self.attn_channels)  # [B*H, W, C]
        rad_seq = rad_attn.permute(0, 2, 3, 1).reshape(B * H, W, self.attn_channels)  # [B*H, W, C]

        # ====== 3) Progressive Focus top-K ======
        if self.focus_ratio < 1.0:
            K = int(W * self.focus_ratio)
            K = max(self.min_focus_tokens, K)
            K = min(K, W)

            if K < W:
                score_input = torch.cat([rad_seq.detach(), img_seq], dim=-1)          # [B*H, W, 2C]
                score = self.score_mlp(score_input).squeeze(-1)                       # [B*H, W]
                _, topk_idx = torch.topk(score, k=K, dim=-1)                          # [B*H, K]
                idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, self.attn_channels)
                img_kv = torch.gather(img_seq, dim=1, index=idx_expand)               # [B*H, K, C]
            else:
                img_kv = img_seq
        else:
            img_kv = img_seq

        # ====== 4) Cross-Attention + FFN ======
        q = self.norm1_q(rad_seq)
        kv = self.norm1_kv(img_kv)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)                      # [B*H, W, C]

        rad_seq = rad_seq + attn_out
        rad_seq2 = self.norm2(rad_seq)
        rad_seq = rad_seq + self.ffn(rad_seq2)

        # ====== 5) reshape 回 BEV（cross-attn 后 radar 表示）=====
        rad_attn_bev = rad_seq.reshape(B, H, W, self.attn_channels).permute(0, 3, 1, 2).contiguous()
        img_attn_bev = img_attn

        # ====== 6) 两路 delta ======
        # baseline 主路：delta_attn
        delta_attn = self.rad_delta_proj(rad_attn_bev)                               # [B, C_r, H, W]

        # HMB 增益：delta_hmb
        hmb_out = self.hmb(rad_attn_bev, img_attn_bev)                               # [B, C_attn, H, W]
        delta_hmb = self.rad_delta_proj(hmb_out)                                     # [B, C_r, H, W]

        # ====== 7) gate + range mask ======
        gate_input = torch.cat([radar_features, image_features], dim=1)              # [B, C_r + C_i, H, W]
        gate = self.gate_conv(gate_input)                                            # [B, C_r, H, W]

        if self.use_range_mask:
            y = torch.arange(H, device=radar_features.device, dtype=radar_features.dtype)
            y_norm = y / (H - 1) if H > 1 else torch.zeros_like(y)
            w = (1.0 - self.range_decay * y_norm).clamp(min=0.3)
            range_mask = w.view(1, 1, H, 1).expand(B, 1, H, W)
            gate = gate * range_mask

        # ====== 8) 两路残差增强 ======
        radar_enhanced = radar_features \
            + self.gamma * gate * delta_attn \
            + self.gamma_hmb * gate * delta_hmb

        # ====== 9) 最终融合 ======
        fuse_input = torch.cat([radar_enhanced, image_features], dim=1)
        fuse_bev = self.fuse_conv(fuse_input)

        batch_dict[self.feature_name] = fuse_bev
        return batch_dict
