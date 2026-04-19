import math
import spconv.utils
import torch
import torch.nn as nn
from spconv.pytorch import functional as fsp  # noqa: F401

from pcdet.models.backbones_3d.vfe.pillarnet_modules.dynamic_pillar_encoder import DynamicPillarFeatureNet
from pcdet.models.backbones_3d.vfe.pillarnet_modules.pcnres18 import SpMiddlePillarEncoder18
from pcdet.models.backbones_3d.vfe.pillarnet_modules.rpn import RPNV2  # noqa: F401
from pcdet.models.backbones_3d.vfe.pillarnet_modules.lss_fpn import FPN_LSS


class SparsePillarSelfAttention(nn.Module):
    """
    Always-on sparse pillar self-attention between reader and backbone.

    - Treat all non-empty pillars as tokens: sp_tensor.features is [N, C]
    - Group tokens by batch id (sp_tensor.indices[:, 0])
    - Self-attention inside each batch
    - Write updated features back; indices/spatial_shape unchanged

    Safety:
    - If tokens per batch > max_tokens, only top-K (by feature norm) participate in attention.
      The rest remain unchanged.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0, max_tokens: int = 4096):
        super().__init__()
        self.channels = int(channels)
        self.max_tokens = int(max_tokens)

        # Ensure heads divides channels; otherwise fallback to 1 head.
        nh = int(num_heads)
        if self.channels % nh != 0:
            nh = 1
        self.num_heads = nh

        self.mha = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.ln = nn.LayerNorm(self.channels)

    @staticmethod
    def _get_batch_idx(indices: torch.Tensor) -> torch.Tensor:
        # indices: [N, 4] (b,z,y,x) usually, batch at col0
        return indices[:, 0].to(torch.long)

    @staticmethod
    def _replace_feature(sp_tensor, new_feat: torch.Tensor):
        # spconv 2.x: replace_feature exists
        if hasattr(sp_tensor, "replace_feature"):
            return sp_tensor.replace_feature(new_feat)

        # fallback: reconstruct SparseConvTensor if possible; else assign
        try:
            import spconv.pytorch as spconv
            return spconv.SparseConvTensor(
                features=new_feat,
                indices=sp_tensor.indices,
                spatial_shape=sp_tensor.spatial_shape,
                batch_size=sp_tensor.batch_size
            )
        except Exception:
            sp_tensor.features = new_feat
            return sp_tensor

    def forward(self, sp_tensor):
        feats = sp_tensor.features  # [N, C]
        if feats is None or feats.numel() == 0:
            return sp_tensor

        idx = sp_tensor.indices
        if idx is None or idx.numel() == 0:
            return sp_tensor

        batch_idx = self._get_batch_idx(idx)
        # batch size preference: sp_tensor.batch_size, else infer
        if hasattr(sp_tensor, "batch_size") and sp_tensor.batch_size is not None:
            B = int(sp_tensor.batch_size)
        else:
            B = int(batch_idx.max().item()) + 1

        new_feats = feats.clone()

        # process each batch independently (variable token counts)
        for b in range(B):
            sel = (batch_idx == b).nonzero(as_tuple=False).squeeze(-1)
            n = int(sel.numel())
            if n <= 1:
                continue

            x = feats[sel]  # [n, C]

            # token cap: top-K by L2 norm (keep the strongest responses)
            if n > self.max_tokens:
                norms = torch.norm(x, p=2, dim=1)  # [n]
                topk = torch.topk(norms, k=self.max_tokens, largest=True).indices
                sel_attn = sel[topk]
                x_attn = feats[sel_attn]
            else:
                sel_attn = sel
                x_attn = x

            # MHA is safer in fp32; cast back after
            orig_dtype = x_attn.dtype
            x_in = x_attn.float().unsqueeze(0)  # [1, T, C]

            attn_out, _ = self.mha(x_in, x_in, x_in, need_weights=False)
            y = self.ln(x_in + self.dropout(attn_out)).squeeze(0)  # [T, C]

            y = y.to(dtype=orig_dtype)
            new_feats[sel_attn] = y

        return self._replace_feature(sp_tensor, new_feats)


class PillarNet(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.module_name_list = ['reader', 'backbone', 'neck']
        self.virtual = model_cfg.READER.get('USE_VIRTUAL_POINT', False)
        self.output_spatial_feature = self.model_cfg.get('OUTPUT_SPATIAL_FEATURES', False)

        for module_name in self.module_name_list:
            getattr(self, 'build_%s' % module_name)(model_cfg=model_cfg)

        # ===== Always enable sparse pillar attention (no cfg switch) =====
        # Use backbone IN_PLANES as token channel dim (reader output should match this).
        c = int(model_cfg.BACKBONE.IN_PLANES)
        # simple, stable defaults
        self.pillar_attn = SparsePillarSelfAttention(
            channels=c,
            num_heads=4,       # will fall back to 1 if not divisible
            dropout=0.0,
            max_tokens=4096,
        )
        # ===============================================================

    def build_reader(self, model_cfg):
        self.reader = DynamicPillarFeatureNet(
            num_input_features=model_cfg.READER.NUM_INPUT_FEATURES,
            num_filters=model_cfg.READER.NUM_FILTERS,
            pillar_size=model_cfg.READER.PILLAR_SIZE,
            pc_range=model_cfg.READER.PC_RANGE,
            virtual=self.virtual,
            encoding_type=self.model_cfg.READER.get('ENCODING_TYPE', 'split'),
            dataset=self.model_cfg.READER.get('DATASET', 'vod')
        )

    def build_backbone(self, model_cfg):
        self.backbone = SpMiddlePillarEncoder18(
            in_planes=model_cfg.BACKBONE.IN_PLANES,
            ds_factor=model_cfg.BACKBONE.DS_FACTOR,
            out_indices=model_cfg.BACKBONE.get('OUT_INDICES', [1, 2, 3])
        )

    def build_neck(self, model_cfg):
        self.neck = FPN_LSS(
            in_channels=model_cfg.NECK.IN_CHANNELS,
            out_channels=model_cfg.NECK.OUT_CHANNELS,
            scale_factor=model_cfg.NECK.SCALE_FACTOR,
            extra_upsample=None,
        )

    def get_output_feature_dim(self):
        return 64

    def forward(self, batch_dict):
        batch_size = batch_dict['points'][:, 0].max().item()
        points = []
        for i in range(int(batch_size) + 1):
            mask = batch_dict['points'][:, 0] == i
            points.append(batch_dict['points'][mask][:, 1:])

        input_dict = dict(points=points)

        # reader -> sparse pillar attention -> backbone -> neck
        sp_tensor = self.reader(input_dict)
        sp_tensor = self.pillar_attn(sp_tensor)     # <<< always-on global interaction
        features_backbone = self.backbone(sp_tensor)

        if self.output_spatial_feature:
            batch_dict['spatial_features_2d'] = self.neck(features_backbone)
        else:
            batch_dict['pillar_features_scattered'] = self.neck(features_backbone)  # [B, 128, 160, 160]
        return batch_dict
