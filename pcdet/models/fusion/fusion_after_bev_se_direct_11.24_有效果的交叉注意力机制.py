import torch
import torch.nn as nn
from torch.nn import functional as F

class FusionAfterBEVSEDirect(nn.Module):
    """
    Radar BEV + Image BEV 融合（单帧）：
    - Radar BEV 做 Query
    - Image BEV 做 Key / Value
    - 沿着 X 方向（一行一行）做 cross-attention

    改进点：
      1）Cross-Attn 只作为 Radar 残差 correction（带可学习标量 gamma）
      2）增加跨模态 gate：由 [radar, image] 决定在哪些位置/通道更信任 Cross-Attn
      3）Progressive Focus：每行只对得分最高的 K 个 image token 做注意力，减少噪声和计算
      4）距离衰减 range mask：越远的 BEV 位置 Cross-Attn 影响越小，保护远处雷达几何
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

        # ===== 1) 注意力内部维度（直接在代码里控制） =====
        # 上限 128，太大显存扛不住；太小表达力差
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

        # ===== 4) Progressive Focus：先筛掉不重要的 image token 再做 Attention =====
        # 直接在代码里固定比例：每行保留 50% token，至少 16 个
        self.focus_ratio = 0.5
        self.min_focus_tokens = 16

        # 用一个轻量 MLP 做 per-token 打分：输入 concat(rad, img)
        hidden = max(attn_channels // 2, 1)
        self.score_mlp = nn.Sequential(
            nn.Linear(attn_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

        # ===== 5) 把 attn 输出映射回雷达通道，做残差 correction =====
        self.rad_delta_proj = nn.Conv2d(attn_channels, radar_out_channels, kernel_size=1, bias=False)
        # 学习一个缩放因子 gamma，初始化为 0，保证一开始近似 baseline
        self.gamma = nn.Parameter(torch.zeros(1))

        # 跨模态 gate：由 [radar, image] 决定 correction 的权重
        self.gate_conv = nn.Sequential(
            nn.Conv2d(radar_out_channels + image_out_channels,
                      radar_out_channels,
                      kernel_size=1,
                      bias=True),
            nn.Sigmoid()
        )

        # ===== 6) 距离衰减 range mask：越远的位置 Cross-Attn 权重越低 =====
        # 这里直接在代码里开关 + 衰减系数
        self.use_range_mask = True
        self.range_decay = 0.7  # 0~1，越大衰减越快

        # ===== 7) 最终融合卷积 =====
        out_channels = radar_out_channels + image_out_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 输出特征名字还是允许从 cfg 里读，如果没有就用默认
        if hasattr(self.model_cfg, "get"):
            self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
        else:
            self.feature_name = 'spatial_features_2d'

    def forward(self, batch_dict):
        # Image BEV: [B, C_img, H_img, W_img]
        image_features = batch_dict["spatial_features"]
        # Radar BEV: [B, C_rad, H_bev, W_bev]
        radar_features = batch_dict['pillar_features_scattered']

        # 对齐分辨率
        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(
                image_features,
                radar_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )  # [B, C_img, H, W]

        B, C_r, H, W = radar_features.shape
        _, C_i, _, _ = image_features.shape

        # ====== 1) Cross-Attention 只用来给 radar 做残差补偿 ======
        img_attn = self.img_proj(image_features)  # [B, C_attn, H, W]
        rad_attn = self.rad_proj(radar_features)  # [B, C_attn, H, W]

        # 行级序列化: [B, C, H, W] -> [B*H, W, C]
        img_seq = img_attn.permute(0, 2, 3, 1).reshape(B * H, W, self.attn_channels)
        rad_seq = rad_attn.permute(0, 2, 3, 1).reshape(B * H, W, self.attn_channels)

        # ====== 2) Progressive Focus: 每行只保留 top-K image token 做 K/V ======
        if self.focus_ratio < 1.0:
            K = int(W * self.focus_ratio)
            K = max(self.min_focus_tokens, K)
            K = min(K, W)

            if K < W:
                # score_input: [B*H, W, 2*C]
                # detach rad_seq 避免 score_mlp 把太多梯度反向到 Q
                score_input = torch.cat([rad_seq.detach(), img_seq], dim=-1)
                score = self.score_mlp(score_input).squeeze(-1)  # [B*H, W]

                # 每一行 top-K
                topk_vals, topk_idx = torch.topk(score, k=K, dim=-1)  # [B*H, K]
                idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, self.attn_channels)
                img_kv = torch.gather(img_seq, dim=1, index=idx_expand)  # [B*H, K, C]
            else:
                img_kv = img_seq
        else:
            # 不做 token 筛选，退化为 full attention
            img_kv = img_seq

        # ====== 3) Cross-Attention: Radar <- Image ======
        q = self.norm1_q(rad_seq)
        kv = self.norm1_kv(img_kv)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)  # [B*H, W, C_attn]

        rad_seq = rad_seq + attn_out  # 残差
        rad_seq2 = self.norm2(rad_seq)
        rad_ffn = self.ffn(rad_seq2)
        rad_seq = rad_seq + rad_ffn  # 再次残差

        # reshape 回 BEV
        rad_bev_delta = rad_seq.reshape(B, H, W, self.attn_channels).permute(0, 3, 1, 2)  # [B, C_attn, H, W]

        # 映射回雷达通道，作为 correction
        rad_bev_delta = self.rad_delta_proj(rad_bev_delta)  # [B, C_r, H, W]

        # ====== 4) Gate + 可选 range mask 控制 correction 强度 ======
        # 由当前 radar + image 共同预测 gate（逐通道、逐位置）
        gate_input = torch.cat([radar_features, image_features], dim=1)  # [B, C_r + C_i, H, W]
        gate = self.gate_conv(gate_input)                               # [B, C_r, H, W]

        # 根据距离做衰减 mask（简单用 H 维近似前后方向）
        if self.use_range_mask:
            # 假设 H 维是前后方向：0 行最近，H-1 行最远
            y = torch.arange(H, device=radar_features.device, dtype=radar_features.dtype)
            if H > 1:
                y_norm = y / (H - 1)
            else:
                y_norm = torch.zeros_like(y)
            # 距离越远权重越小，例如 w = 1 - decay * y_norm
            w = 1.0 - self.range_decay * y_norm
            w = w.clamp(min=0.3)  # 最远至少保留 0.3，避免完全关闭
            range_mask = w.view(1, 1, H, 1).expand(B, 1, H, W)  # [B,1,H,W]
            gate = gate * range_mask

        # 使用 gamma * gate 控制 correction 的强度
        radar_enhanced = radar_features + self.gamma * gate * rad_bev_delta

        # ====== 5) 最终融合：radar_enhanced + 原始 image_features ======
        fuse_input = torch.cat([radar_enhanced, image_features], dim=1)  # [B, C_r + C_i, H, W]
        fuse_bev = self.fuse_conv(fuse_input)  # [B, C_out, H, W]

        batch_dict[self.feature_name] = fuse_bev
        return batch_dict
