import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意：不要 from mamba_ssm import Mamba（会触发 LM/transformers 依赖）
from mamba_ssm.modules.mamba_simple import Mamba


def _snake_order_index(h: int, w: int, device):
    """
    通用的 2D->1D 顺序：偶数行 L->R，奇数行 R->L
    任何 H,W 都能用，比纯 row-major 更“连续”。
    """
    idx = torch.arange(h * w, device=device).view(h, w)
    if h > 1:
        idx[1::2] = torch.flip(idx[1::2], dims=[1])
    return idx.reshape(-1)


def _invert_perm(idx: torch.Tensor):
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=idx.device, dtype=idx.dtype)
    return inv


def _bidir_mamba(m: Mamba, x: torch.Tensor):
    """
    x: [B, L, C]
    双向：forward + reverse 平均（抵消单向 scan 的偏置）
    """
    y_fwd = m(x)
    y_bwd = torch.flip(m(torch.flip(x, dims=[1])), dims=[1])
    return 0.5 * (y_fwd + y_bwd)


class HybridMambaBlockLite(nn.Module):
    """
    Lite 版 HMB：Local + Global
    输入：
      - rad_bev: [B, C, H, W]  (建议是你 cross-attn 后的 radar 表示)
      - img_bev: [B, C, H, W]  (建议是你 img_proj 后的 image 表示)
    输出：
      - delta:  [B, C, H, W]  （给你外面 rad_delta_proj 用）

    轻量化设计：
      - Global 分支默认 stride 下采样 + 通道降维
      - Global 顺序默认 snake（可选 hilbert_idx）
      - Local 默认只扫 x（按行），可选 xy 双向更强但更慢
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidir: bool = True,

        # Local 分支
        use_local: bool = True,
        local_dirs: str = "x",   # "x" 或 "xy"
        share_local: bool = True,

        # Global 分支
        use_global: bool = True,
        global_stride: int = 2,          # 1=不下采样（最重），2/4 更轻
        global_dim: int = None,          # None -> min(64, d_model)
        global_order: str = "snake",     # "snake" or "row" or "hilbert"
        hilbert_idx: torch.Tensor = None # 传入 1D permutation（长度=H'*W'），可选
    ):
        super().__init__()
        assert local_dirs in ("x", "xy")
        assert global_order in ("snake", "row", "hilbert")

        self.d_model = d_model
        self.use_local = use_local
        self.local_dirs = local_dirs
        self.share_local = share_local

        self.use_global = use_global
        self.global_stride = int(global_stride)
        self.global_order = global_order

        # 跨模态融合：concat -> 1x1 压回 C
        self.fuse = nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False)

        # Local：在 full C 上做（更保点）
        if use_local:
            self.local_mamba_x = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            if local_dirs == "xy":
                self.local_mamba_y = self.local_mamba_x if share_local else \
                    Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # Global：先降维到 global_dim 再跑（更轻量）
        if use_global:
            gd = min(64, d_model) if global_dim is None else int(global_dim)
            self.global_dim = gd
            self.g_in  = nn.Conv2d(d_model, gd, kernel_size=1, bias=False)
            self.g_out = nn.Conv2d(gd, d_model, kernel_size=1, bias=False)
            self.global_mamba = Mamba(d_model=gd, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.global_dim = None
            self.g_in = None
            self.g_out = None
            self.global_mamba = None

        # 归一化（pre-norm 风格）
        self.norm_local  = nn.LayerNorm(d_model)
        self.norm_global = nn.LayerNorm(self.global_dim if self.global_dim is not None else d_model)

        self.bidir = bidir

        # 输出再做一次 1x1（让 delta 更“像修正量”）
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)

        # 保存可选 hilbert index（注意：必须与下采样后的 H',W' 对齐）
        self.register_buffer("hilbert_idx_buf", hilbert_idx if hilbert_idx is not None else None, persistent=False)

    def _run_mamba(self, m: Mamba, x: torch.Tensor):
        return _bidir_mamba(m, x) if self.bidir else m(x)

    def _local_x(self, x: torch.Tensor):
        # x: [B,C,H,W] -> [B*H, W, C]
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        y = self._run_mamba(self.local_mamba_x, self.norm_local(seq))
        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return y

    def _local_y(self, x: torch.Tensor):
        # x: [B,C,H,W] -> [B*W, H, C]
        B, C, H, W = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)  # (B,W,H,C)->(B*W,H,C)
        y = self._run_mamba(self.local_mamba_y, self.norm_local(seq))
        y = y.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()      # back to [B,C,H,W]
        return y

    def _global_order_index(self, h: int, w: int, device):
        L = h * w
        if self.global_order == "row":
            idx = torch.arange(L, device=device)
            inv = None
            return idx, inv

        if self.global_order == "hilbert" and self.hilbert_idx_buf is not None:
            idx = self.hilbert_idx_buf.to(device)
            if idx.numel() == L:
                inv = _invert_perm(idx)
                return idx, inv
            # 尺寸对不上时自动 fallback
        # default: snake
        idx = _snake_order_index(h, w, device)
        inv = _invert_perm(idx)
        return idx, inv

    def _global(self, x: torch.Tensor):
        """
        x: [B,C,H,W] -> global delta in [B,C,H,W]
        """
        B, C, H, W = x.shape
        stride = max(1, self.global_stride)

        # 1) 下采样（大幅减 token）
        if stride > 1:
            x_s = F.avg_pool2d(x, kernel_size=stride, stride=stride)
        else:
            x_s = x
        _, _, Hs, Ws = x_s.shape

        # 2) 降维跑 mamba
        x_g = self.g_in(x_s)                          # [B, gd, Hs, Ws]
        gd = x_g.shape[1]
        seq = x_g.permute(0, 2, 3, 1).contiguous().view(B, Hs * Ws, gd)  # [B,L,gd]

        # 3) 重排顺序（snake/hilbert/row）
        idx, inv = self._global_order_index(Hs, Ws, device=seq.device)
        if idx is not None:
            seq = seq[:, idx, :]

        # 4) Mamba（可双向）
        y = self._run_mamba(self.global_mamba, self.norm_global(seq))

        # 5) 还原顺序
        if inv is not None:
            y = y[:, inv, :]

        # 6) reshape 回 2D + 升维 + 上采样回原尺寸
        y = y.view(B, Hs, Ws, gd).permute(0, 3, 1, 2).contiguous()    # [B,gd,Hs,Ws]
        y = self.g_out(y)                                             # [B,C,Hs,Ws]
        if stride > 1:
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y

    def forward(self, rad_bev: torch.Tensor, img_bev: torch.Tensor):
        """
        返回 delta: [B,C,H,W]
        """
        x = self.fuse(torch.cat([rad_bev, img_bev], dim=1))  # [B,C,H,W]

        delta = 0.0
        if self.use_local:
            lx = self._local_x(x)
            if self.local_dirs == "xy":
                ly = self._local_y(x)
                delta = delta + 0.5 * (lx + ly)
            else:
                delta = delta + lx

        if self.use_global:
            delta = delta + self._global(x)

        delta = self.out_proj(delta)
        return delta
