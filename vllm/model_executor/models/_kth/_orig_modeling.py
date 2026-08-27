# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import torch.nn.functional as F
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


import os as _gate_os


def _force_gate_override():
    """FORCE_GATE_INFER env var → scalar override for inference-time gate sweep."""
    v = _gate_os.environ.get("FORCE_GATE_INFER")
    return None if v is None or v == "" else float(v)


class ScalarGate(nn.Module):
    """Learnable scalar gate (zero-init, tanh 적용, float32 유지)."""
    def __init__(self, init_value=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x, base_tokens=None):
        ov = _force_gate_override()
        if ov is not None:
            return torch.tensor(ov, dtype=x.dtype, device=x.device) * x
        gate = torch.tanh(self.weight.float())
        return gate.to(x.dtype) * x


class PerTokenGate(nn.Module):
    """Per-spatial-token learnable gate (N scalars, tanh, float32).
    각 spatial token마다 독립 gate. 어떤 토큰이 유용한지 학습."""
    def __init__(self, num_tokens=12, init_value=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_tokens,), init_value, dtype=torch.float32))

    def forward(self, x, base_tokens=None):
        # x: (B, num_tokens, D)
        ov = _force_gate_override()
        if ov is not None:
            return torch.tensor(ov, dtype=x.dtype, device=x.device) * x
        gate = torch.tanh(self.weight.float()).view(1, -1, 1)  # (1, num_tokens, 1)
        return gate.to(x.dtype) * x


class ContentAwareGate(nn.Module):
    """[v10] Bounded content-aware gate:  g = sigmoid( τ·cos(w, LN(pool)) + b )

    ── 왜 바꿨나 (v9 실측, 2026-07-14) ─────────────────────────────────────────
    v9 게이트 sigmoid(W·mean_pool + b) 는 학습된 모델 3개 전부에서 완전 포화했다:
        모델        bias    ‖W‖      로짓     실측 gate
        KTH_720k    5.000    1.81    11.6      1.0000
        v2_c        5.000   14.73    67.2      1.0000
        v2_b        2.009    1.54    10.0      0.9999
      로짓 최솟값 8.1 — 어떤 이미지에서도 gate 가 0.9997 아래로 내려간 적이 없다.
      즉 content-aware 라우팅은 한 번도 일어나지 않았고 KTH 는 항상 100% 주입됐다.

    원인:
      (1) pool 노름이 커서 W·pool 이 +6~+62 까지 뜬다 (LN 부재)
      (2) init_bias=5 → σ(5)=0.993. bias 만으로 이미 포화이고, 실측상 bias 는
          학습으로 거의 안 움직인다 (5.000/5.000/2.009 = 초기값 그대로).
      포화 구간 gradient ∝ g(1-g) ≈ 0.0066 → 게이트가 학습 신호를 못 받고,
      W 는 표류하다 저수준 통계(타일 수/zoom, r=-0.735)에 정렬됐다.

    ── v10: 포화를 '감시'하지 말고 '불가능'하게 ───────────────────────────────
      z = τ·cos(w, LN(pool)) + b       cos ∈ [-1,1] → z ∈ [b-τ, b+τ] (구조적 보장)
      τ=2, b=1.5 (둘 다 고정) → z ∈ [-0.5, 3.5], gate ∈ [0.38, 0.97]
        · 초기 gate 0.82 : KTH 주입 유지(학습 효과 보존)
        · 최소 gradient 0.03 : v9(0.0066)의 5배 → 게이트가 실제로 학습된다
        · ‖w‖ 가 얼마든 무관(방향만 사용) → wd 튜닝 불필요, 재포화 불가능
      loss 에 usage 항 없음: 학습분포에서 KTH ablate 시 CE 변화 +0.0005 라
      CE 가 하한을 못 지킨다 → cap 을 걸면 게이트가 0 으로 붕괴한다.

    판정: std(gate) > 0.05 → content-aware 최초 성립.
          std(gate) < 0.02 가 200 step 지속 → 게이트 사망, 학습 중단.
    env: GATE_TAU(2.0) GATE_BIAS_FIXED(1.5) GATE_LEGACY=1(v9 동작 폴백)
    """
    def __init__(self, hidden_dim, init_bias=-3.0):
        super().__init__()
        self.legacy = _gate_os.environ.get("GATE_LEGACY") == "1"
        self.tau = float(_gate_os.environ.get("GATE_TAU", "2.0"))
        self.bias_fixed = float(_gate_os.environ.get("GATE_BIAS_FIXED", "1.5"))

        self.proj = nn.Linear(hidden_dim, 1, bias=True)
        if self.legacy:
            nn.init.zeros_(self.proj.weight)
            nn.init.constant_(self.proj.bias, init_bias)
        else:
            # 방향만 쓰이므로 랜덤 초기화. bias 는 미사용(고정 스칼라 b 로 대체)
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.proj.bias)
            self.proj.bias.requires_grad_(False)
            self.gate_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj.weight.data = self.proj.weight.data.float()
        self.proj.bias.data = self.proj.bias.data.float()

    def compute_gate(self, base_tokens):
        # base_tokens: (B=타일, N_base=256, D) → (B, 1, 1)
        pooled = base_tokens.float().mean(dim=1)             # (B, D)
        if self.legacy:
            return torch.sigmoid(self.proj(pooled)).unsqueeze(-1)

        h = self.gate_ln(pooled)                             # 크기 정규화
        w = self.proj.weight.float().squeeze(0)              # (D,)
        cos = F.cosine_similarity(h, w.unsqueeze(0), dim=-1)  # (B,) ∈ [-1,1]
        logit = self.tau * cos + self.bias_fixed             # ∈ [b-τ, b+τ]
        return torch.sigmoid(logit).view(-1, 1, 1)           # (B, 1, 1)

    def forward(self, x, base_tokens=None):
        # x: (B, num_spatial_tokens, D)
        ov = _force_gate_override()
        if ov is not None:
            return torch.tensor(ov, dtype=x.dtype, device=x.device) * x
        if base_tokens is None:
            raise RuntimeError("ContentAwareGate requires base_tokens")
        gate = self.compute_gate(base_tokens)            # (B, 1, 1)
        return gate.to(x.dtype) * x


class LearnableResidualGate(nn.Module):
    """Learnable zero-init residual gate (sigmoid, 학습 가능).
    시작 시 sigmoid(init_value) ≈ 0 → 점진적으로 학습.
    bf16 환경에서도 float32로 계산."""
    def __init__(self, init_value=-5.0):
        super().__init__()
        # init=-5.0 → sigmoid≈0.0067 (거의 0에서 시작)
        self.weight = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        gate = torch.sigmoid(self.weight.float())
        return gate.to(x.dtype) * x


def _make_spatial_gate(num_tokens, hidden_dim, default_type="scalar"):
    """GATE_TYPE env var (scalar | per_token | content_aware) → 적절한 gate 생성.
    env var 없으면 default_type (config의 spatial_gate_type) 사용.
    GATE_INIT_BIAS env var로 ContentAwareGate 초기 bias 조정 (default -3.0).
    100% open으로 시작하려면 GATE_INIT_BIAS=5 정도 (sigmoid(5)≈0.993)."""
    gate_type = _gate_os.environ.get("GATE_TYPE", default_type).lower()
    init_bias = float(_gate_os.environ.get("GATE_INIT_BIAS", "-3.0"))
    if gate_type == "per_token":
        return PerTokenGate(num_tokens=num_tokens, init_value=0.0)
    if gate_type == "content_aware":
        return ContentAwareGate(hidden_dim=hidden_dim, init_bias=init_bias)
    return ScalarGate(init_value=0.0)


class RegisterTokens(nn.Module):
    """Pure register tokens with configurable init/freeze modes.

    Init modes (REGISTER_INIT env or config.register_init_mode):
      - "small_random" (default): std=0.001 normal, learnable
      - "zero": exactly 0, learnable (default of variant_A's gated-zero pattern)
      - "fixed_zero": exactly 0, FROZEN (no learning) — register effect ablation baseline
      - "fixed_random": std=0.5 random, FROZEN — "random ≠ learned" comparison
    """
    def __init__(self, num_tokens=12, hidden_dim=4096, init_std=0.001, init_mode="small_random"):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(num_tokens, hidden_dim, dtype=torch.float32))

        mode = (_gate_os.environ.get("REGISTER_INIT", init_mode)).lower()
        if mode == "zero" or mode == "fixed_zero":
            nn.init.zeros_(self.tokens)
        elif mode == "fixed_random":
            nn.init.normal_(self.tokens, mean=0.0, std=0.5)
        else:  # small_random (default)
            nn.init.normal_(self.tokens, mean=0.0, std=init_std)

        # Frozen modes: param 자체는 learnable로 두고 sft_train.py에서 requires_grad=False
        self.init_mode = mode
        self.num_tokens = num_tokens

    def forward(self, base_tokens):
        B = base_tokens.shape[0]
        regs = self.tokens.unsqueeze(0).expand(B, -1, -1).to(base_tokens.dtype)
        return torch.cat([regs, base_tokens], dim=1)


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        _raw_sfm = getattr(config, "spatial_feature_mode", "none")
        # 구버전 체크포인트 호환: 옛 llava_sp_* 이름 → koni_* 로 정규화 (KTH_720k 등 기존 config 로드용)
        _SFM_ALIAS = {"llava_sp_both": "koni_token_hier", "llava_sp_pooling": "koni_pool", "llava_sp_cropping": "koni_crop"}
        self.spatial_feature_mode = _SFM_ALIAS.get(_raw_sfm, _raw_sfm)
        self.spatial_pool_size = self._normalize_hw(
            getattr(config, "spatial_pool_size", None), "spatial_pool_size"
        )
        self.spatial_crop_size = self._normalize_hw(
            getattr(config, "spatial_crop_size", None), "spatial_crop_size"
        )
        self.spatial_crop_stride = self._normalize_hw(
            getattr(config, "spatial_crop_stride", None), "spatial_crop_stride"
        )
        # LLaVA-SP parameters
        self.sfe_intermediate_size = getattr(config, "sfe_intermediate_size", 512)
        self.sfe_scales = getattr(config, "sfe_scales", None)
        self.dfi_detail_kernel = getattr(config, "dfi_detail_kernel", 16)
        self.dfi_detail_stride = getattr(config, "dfi_detail_stride", 4)
        # v5 attention config (defaults False for backward compat with v4)
        self._use_spatial_attention_cfg = getattr(config, "use_spatial_attention", False)
        self._use_channel_attention_cfg = getattr(config, "use_channel_attention", False)
        self._use_token_self_attention_cfg = getattr(config, "use_token_self_attention", False)
        self._channel_attention_reduction = getattr(config, "channel_attention_reduction", 16)
        self._token_self_attention_heads = getattr(config, "token_self_attention_heads", 8)
        # Cross-attention fusion config
        self._use_cross_attention_fusion_cfg = getattr(config, "use_cross_attention_fusion", False)
        self._cross_attention_fusion_heads = getattr(config, "cross_attention_fusion_heads", 8)
        self._cross_attention_fusion_concat_kth = getattr(config, "cross_attention_fusion_concat_kth", True)
        # Actual flags set in LLaVA-SP block below (default off)
        self._use_spatial_attention = False
        self._use_channel_attention = False
        self._use_token_self_attention = False
        self._use_cross_attention_fusion = False

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        base_grid_size = int((image_size // patch_size) * self.downsample_ratio)
        base_num_image_token = base_grid_size * base_grid_size
        extra_tokens_per_image = self._infer_extra_tokens(base_grid_size)
        self.num_image_token = base_num_image_token + extra_tokens_per_image

        logger.info(f'num_image_token: {self.num_image_token}')
        if extra_tokens_per_image > 0:
            logger.info(
                f'spatial_feature_mode: {self.spatial_feature_mode}, '
                f'extra_tokens_per_image: {extra_tokens_per_image}'
            )
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if getattr(config, "llm_use_pretrained", False):
                if not config.llm_model_name_or_path:
                    raise ValueError("llm_use_pretrained=True requires llm_model_name_or_path")
                try:
                    from transformers import modeling_rope_utils
                    if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
                        modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = (
                            modeling_rope_utils._compute_default_rope_parameters
                        )
                except Exception:
                    # Best-effort fallback; if the mapping is missing in older versions,
                    # this prevents a KeyError for rope_type="default".
                    pass
                llm_config = AutoConfig.from_pretrained(
                    config.llm_model_name_or_path,
                    trust_remote_code=getattr(config, "llm_trust_remote_code", False),
                )
                # HyperCLOVA models may store rope_scaling/rope_type as None or "default".
                # Force linear scaling with factor=1.0 to avoid ROPE_INIT_FUNCTIONS["default"] issues.
                rope_scaling = getattr(llm_config, "rope_scaling", None)
                rope_scaling_type = None
                if isinstance(rope_scaling, dict):
                    rope_scaling_type = rope_scaling.get("rope_type")
                elif rope_scaling is not None:
                    rope_scaling_type = getattr(rope_scaling, "rope_type", None)
                if rope_scaling is None or rope_scaling_type == "default":
                    llm_config.rope_scaling = {"rope_type": "linear", "factor": 1.0}
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    config.llm_model_name_or_path,
                    trust_remote_code=getattr(config, "llm_trust_remote_code", False),
                    config=llm_config,
                    torch_dtype=getattr(llm_config, "torch_dtype", None),
                    low_cpu_mem_usage=True,
                )
                config.llm_config = llm_config
            else:
                architecture: str = config.llm_config.architectures[0]
                if architecture == 'LlamaForCausalLM':
                    self.language_model = LlamaForCausalLM(config.llm_config)
                elif architecture == 'Qwen2ForCausalLM':
                    self.language_model = Qwen2ForCausalLM(config.llm_config)
                elif architecture == 'Qwen3MoeForCausalLM':
                    self.language_model = Qwen3MoeForCausalLM(config.llm_config)
                elif architecture == 'Qwen3ForCausalLM':
                    self.language_model = Qwen3ForCausalLM(config.llm_config)
                else:
                    raise NotImplementedError(f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # ── v8 ablation: register_only mode ──
        if self.spatial_feature_mode == "register_only":
            n_reg = getattr(config, "register_num_tokens", 12)
            init_mode = getattr(config, "register_init_mode", "small_random")
            self.register_tokens = RegisterTokens(
                num_tokens=n_reg, hidden_dim=llm_hidden_size, init_mode=init_mode
            )
            logger.info(
                f'register_only mode: n={n_reg} init={init_mode} '
                f'(params={n_reg * llm_hidden_size:,})'
            )

        # ── LLaVA-SP: SFE (Spatial Feature Extractor) + optional DFI ──
        _KONI_MODES = ("koni_pool", "koni_crop", "koni_token_hier")
        if self.spatial_feature_mode in _KONI_MODES:
            pre_ps_grid = image_size // patch_size  # 32 for 448/14
            sfe_scales = self.sfe_scales or [4, 8, 14, 20, 26, pre_ps_grid]
            sfe_dim = self.sfe_intermediate_size  # 512

            need_pooling = self.spatial_feature_mode in ("koni_pool", "koni_token_hier")
            need_cropping = self.spatial_feature_mode in ("koni_crop", "koni_token_hier")

            # ── [v10 local-branch fix] center-crop(동심원, 중앙편향) → dense windows ──
            #   진단(2026-07-16): 기존 local은 항상 정중앙 k×k crop만 → 주변부 안 봄
            #   (peripheryFrac 0.299), 보완장치 DFI는 inert. LOCAL_BRANCH env로 분기.
            import os as _lb_os
            self._local_mode = _lb_os.environ.get("LOCAL_BRANCH", "center_crop")
            self._pre_ps_grid = pre_ps_grid

            def _build_sfe_convs():
                convs = nn.ModuleList()
                for k in sfe_scales:
                    if k < pre_ps_grid:
                        convs.append(nn.Sequential(
                            nn.AdaptiveAvgPool2d(k),
                            nn.Conv2d(vit_hidden_size, sfe_dim, k, bias=False),
                        ))
                    else:
                        convs.append(
                            nn.Conv2d(vit_hidden_size, sfe_dim, k, bias=False)
                        )
                return convs

            # Pooling SFE
            if need_pooling:
                self.sfe_pool_convs = _build_sfe_convs()
                self.sfe_pool_proj = nn.Sequential(
                    nn.GELU(), nn.Linear(sfe_dim, llm_hidden_size, bias=False)
                )

            # Cropping SFE + DFI  (Control=center_crop) 또는 [v10] dense local windows
            if need_cropping and self._local_mode == "center_crop":
                self.sfe_crop_convs = _build_sfe_convs()
                dk = self.dfi_detail_kernel
                ds = self.dfi_detail_stride
                self.dfi_conv_cm = nn.Conv2d(
                    vit_hidden_size, sfe_dim, kernel_size=dk, stride=ds, bias=False
                )
                self.dfi_query_proj = nn.Sequential(
                    nn.LayerNorm(sfe_dim), nn.Linear(sfe_dim, sfe_dim)
                )
                self.dfi_key_proj = nn.Sequential(
                    nn.LayerNorm(sfe_dim), nn.Linear(sfe_dim, sfe_dim)
                )
                self.dfi_value_proj = nn.Sequential(
                    nn.LayerNorm(sfe_dim), nn.Linear(sfe_dim, sfe_dim)
                )
                # SFE(512) concat DFI(512) → 1024
                self.sfe_crop_proj = nn.Sequential(
                    nn.GELU(), nn.Linear(sfe_dim * 2, llm_hidden_size, bias=False)
                )
            elif need_cropping and self._local_mode == "global_ca":
                # [v10 E6] local도 전역 멀티스케일 pooling(pool과 동일 기계, 별도 가중치) + channel attn.
                #   spatial attn은 안 씀 → 두 브랜치가 attention 종류(spatial vs channel)로만 구분.
                #   "로컬 detail이 정말 필요한가 vs 전역 두 관점으로 충분한가" 검증. 6토큰.
                self.local_pool_convs = _build_sfe_convs()
                self.local_dense_proj = nn.Sequential(
                    nn.GELU(), nn.Linear(sfe_dim, llm_hidden_size, bias=False)
                )
                logger.info(f'[v10] E6 global_ca local: 전역 pooling + channel attn, '
                            f'local_tokens={self._local_token_count()}')
            elif need_cropping and self._local_mode == "none":
                # [v10 E7] pool-only: local branch 완전 제거(global 6토큰만). 아무 local 모듈도 안 지음.
                #   "local이 애초에 도움되나"의 바닥선. tsa는 pool 6토큰 위에서만 돎.
                logger.info('[v10] E7 pool-only: local branch 없음 (global 6토큰만)')
            elif need_cropping and self._local_mode == "hybrid":
                # [v11 H1] hybrid: center-crop(foveal, DFI 없음) 6토큰 + dense_4x4 전역 16토큰 = 22.
                #   pair_match(dense) + identification/remedy(center-crop) 둘 다 노림.
                self.sfe_crop_convs = _build_sfe_convs()           # 중앙 다중스케일 crop 6
                self.dense_convs = nn.ModuleList([                 # 전역 4×4 윈도우 16
                    nn.Conv2d(vit_hidden_size, sfe_dim,
                              kernel_size=pre_ps_grid // 4, stride=pre_ps_grid // 4, bias=False)
                ])
                self.local_dense_proj = nn.Sequential(
                    nn.GELU(), nn.Linear(sfe_dim, llm_hidden_size, bias=False)
                )
                logger.info(f'[v11] hybrid local: center-crop 6 + dense_4x4 16 = '
                            f'{self._local_token_count()}토큰')
            elif need_cropping:
                # [v10 dense local windows] 32×32를 겹치지 않는(또는 겹치는) 윈도우 → 각 1토큰.
                #   dense_4x4=16, dense_2x2=4, dense_8x8=64, dense_ms=16+4=20,
                #   dense_4x4_satt=16(+로컬 spatial attn), dense_4x4_overlap=16(겹침)
                if self._local_mode == "dense_4x4_overlap":
                    # [E9] 겹치는 4×4: kernel 12 · stride 8 · pad 2 → 4×4=16, 윈도우가 겹침
                    self.dense_convs = nn.ModuleList([
                        nn.Conv2d(vit_hidden_size, sfe_dim, kernel_size=12, stride=8, padding=2, bias=False)
                    ])
                    _dbg = "overlap(k12s8p2)"
                else:
                    _Ps = {"dense_4x4": [4], "dense_4x4_satt": [4], "dense_2x2": [2],
                           "dense_ms": [4, 2], "dense_8x8": [8], "dense_4x4_pos": [4]}[self._local_mode]
                    self.dense_convs = nn.ModuleList([
                        nn.Conv2d(vit_hidden_size, sfe_dim,
                                  kernel_size=pre_ps_grid // P, stride=pre_ps_grid // P, bias=False)
                        for P in _Ps
                    ])
                    _dbg = f"Ps={_Ps}"
                self.local_dense_proj = nn.Sequential(
                    nn.GELU(), nn.Linear(sfe_dim, llm_hidden_size, bias=False)
                )
                if self._local_mode == "dense_4x4_satt":
                    self.local_satt = nn.Sequential(
                        nn.Conv2d(vit_hidden_size, 1, kernel_size=1, bias=True), nn.Sigmoid()
                    )
                if self._local_mode == "dense_4x4_pos":
                    # [H5] 16개 윈도우(4×4)에 대한 학습된 2D 위치 임베딩
                    self.local_pos_emb = nn.Parameter(torch.zeros(1, 16, sfe_dim))
                    nn.init.normal_(self.local_pos_emb, mean=0.0, std=0.02)
                logger.info(f'[v10] dense local windows: mode={self._local_mode} '
                            f'{_dbg} local_tokens={self._local_token_count()}')

            # ── v5: Attention modules ──
            self._use_spatial_attention = (self._use_spatial_attention_cfg and need_pooling)
            # [E10] USE_CHANNEL_ATTN=0 → 로컬 branch channel attention 제거 ablation
            self._use_channel_attention = (self._use_channel_attention_cfg and need_cropping
                                           and _lb_os.environ.get("USE_CHANNEL_ATTN", "1") != "0")
            self._use_token_self_attention = (
                self._use_token_self_attention_cfg
                and self.spatial_feature_mode == "koni_token_hier"
            )

            if self._use_spatial_attention:
                self.sfe_spatial_attn = nn.Sequential(
                    nn.Conv2d(vit_hidden_size, 1, kernel_size=1, bias=True),
                    nn.Sigmoid()
                )
                logger.info('v5: Spatial Attention enabled for pooling branch')

            if self._use_channel_attention:
                r = self._channel_attention_reduction
                self.sfe_channel_pool = nn.AdaptiveAvgPool2d(1)
                self.sfe_channel_fc = nn.Sequential(
                    nn.Linear(vit_hidden_size, vit_hidden_size // r),
                    nn.ReLU(inplace=True),
                    nn.Linear(vit_hidden_size // r, vit_hidden_size),
                    nn.Sigmoid()
                )
                logger.info(f'v5: Channel Attention enabled for cropping branch (reduction={r})')

            if self._use_token_self_attention:
                if self._local_mode == "center_crop":
                    self.sfe_crop_to_sfe = nn.Linear(sfe_dim * 2, sfe_dim, bias=False)
                self.sfe_token_ln = nn.LayerNorm(sfe_dim)
                self.sfe_token_self_attn = nn.MultiheadAttention(
                    embed_dim=sfe_dim, num_heads=self._token_self_attention_heads,
                    batch_first=True
                )
                self.sfe_token_ffn = nn.Sequential(
                    nn.LayerNorm(sfe_dim),
                    nn.Linear(sfe_dim, sfe_dim * 2),
                    nn.GELU(),
                    nn.Linear(sfe_dim * 2, sfe_dim)
                )
                self.sfe_token_proj = nn.Sequential(
                    nn.GELU(), nn.Linear(sfe_dim, llm_hidden_size, bias=False)
                )
                logger.info(
                    f'v5: Token Self-Attention enabled '
                    f'(dim={sfe_dim}, heads={self._token_self_attention_heads})'
                )

            # Spatial token output normalization + learnable gate
            self.sfe_output_ln = nn.LayerNorm(llm_hidden_size)
            # [실험 b] KTH_ZERO_INIT_OUTPUT=1 → KTH 출력을 0에서 시작(ReZero식).
            #   초기 노이즈 제거 → 낮은 bias여도 게이트 안 죽고 KTH가 유용한 방향으로만 성장.
            #   (fresh init에서만 적용; stage2 resume 시 학습된 값이 load로 덮어씀)
            import os as _kth_os
            if _kth_os.environ.get("KTH_ZERO_INIT_OUTPUT") == "1":
                nn.init.zeros_(self.sfe_output_ln.weight)
                nn.init.zeros_(self.sfe_output_ln.bias)
                print("  [KTH_ZERO_INIT_OUTPUT] sfe_output_ln zero-init → KTH 출력 0에서 시작")
            # GATE_TYPE env var (scalar/per_token/content_aware)로 dispatch
            _num_spatial_tokens = (
                len(sfe_scales) + self._local_token_count()
                if self.spatial_feature_mode == "koni_token_hier"
                else len(sfe_scales))
            self.sfe_spatial_gate = _make_spatial_gate(
                num_tokens=_num_spatial_tokens, hidden_dim=llm_hidden_size,
                default_type=getattr(config, "spatial_gate_type", "scalar"),
            )
            logger.info(
                f'spatial_gate type: {type(self.sfe_spatial_gate).__name__} '
                f'(GATE_TYPE={_gate_os.environ.get("GATE_TYPE", "scalar")})'
            )

            # ── Cross-attention fusion: base tokens attend to KTH tokens ──
            self._use_cross_attention_fusion = (
                self._use_cross_attention_fusion_cfg
                and self.spatial_feature_mode in ("koni_pool", "koni_crop", "koni_token_hier")
            )
            if self._use_cross_attention_fusion:
                self.sfe_xattn_ln_base = nn.LayerNorm(llm_hidden_size)
                self.sfe_xattn_ln_kth = nn.LayerNorm(llm_hidden_size)
                self.sfe_xattn = nn.MultiheadAttention(
                    embed_dim=llm_hidden_size,
                    num_heads=self._cross_attention_fusion_heads,
                    batch_first=True
                )
                self.sfe_xattn_ffn = nn.Sequential(
                    nn.LayerNorm(llm_hidden_size),
                    nn.Linear(llm_hidden_size, llm_hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(llm_hidden_size * 2, llm_hidden_size)
                )
                self.sfe_xattn_scale = 0.05  # fixed residual scale (gate 대신)
                logger.info(
                    f'Cross-attention fusion enabled '
                    f'(heads={self._cross_attention_fusion_heads}, '
                    f'concat_kth={self._cross_attention_fusion_concat_kth})'
                )

            # Backward compat aliases for single-mode access
            if self.spatial_feature_mode == "koni_pool":
                self.sfe_convs = self.sfe_pool_convs
                self.sfe_proj = self.sfe_pool_proj
            elif self.spatial_feature_mode == "koni_crop":
                self.sfe_convs = self.sfe_crop_convs
                self.sfe_proj = self.sfe_crop_proj

            self._sfe_scales = sfe_scales
            n = len(sfe_scales)
            self._sfe_num_tokens = (n + self._local_token_count()
                                    if self.spatial_feature_mode == "koni_token_hier" else n)

            logger.info(
                f'LLaVA-SP mode: {self.spatial_feature_mode}, '
                f'scales: {sfe_scales}, sfe_dim: {sfe_dim}, '
                f'spatial_tokens: {self._sfe_num_tokens}'
            )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Backward compat: nn.Parameter → ScalarGate 키 변환
        old_key = prefix + 'sfe_spatial_gate'
        new_key = prefix + 'sfe_spatial_gate.weight'
        gate_module = getattr(self, 'sfe_spatial_gate', None)
        if old_key in state_dict and new_key not in state_dict:
            val = state_dict.pop(old_key)
            if isinstance(gate_module, ScalarGate):
                state_dict[new_key] = val
            # 다른 gate type이면 무시 (fresh init)
        # 새 gate type 전환 시 shape 불일치 키 제거 (fresh init 유도)
        if isinstance(gate_module, PerTokenGate) and new_key in state_dict:
            if state_dict[new_key].shape != gate_module.weight.shape:
                state_dict.pop(new_key)
        if isinstance(gate_module, ContentAwareGate) and new_key in state_dict:
            # ContentAware는 .proj.weight / .proj.bias 사용 → 구 .weight 무시
            state_dict.pop(new_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def tie_weights(self):
        """accelerate device_map='auto' 후 meta device에 남은 파라미터를 GPU로 이동."""
        super().tie_weights()
        target = None
        for p in self.parameters():
            if p.device.type not in ('meta', 'cpu'):
                target = p.device
                break
        if target is None:
            return
        for name, param in list(self.named_parameters()):
            if param.device.type == 'meta':
                parts = name.split('.')
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                # learnable 모듈 (sfe_spatial_gate, register_tokens)는 requires_grad=True 유지
                if 'sfe_spatial_gate' in name:
                    # ContentAwareGate의 proj.bias는 -3.0으로 초기화 (sigmoid≈0.047)
                    if 'sfe_spatial_gate.proj.bias' in name:
                        init_val = -3.0
                    else:
                        init_val = getattr(module, '_init_value', 0.0) if hasattr(module, '_init_value') else 0.0
                    new_data = torch.full(param.shape, init_val, dtype=param.dtype, device=target)
                    setattr(module, parts[-1], nn.Parameter(new_data, requires_grad=True))
                elif 'register_tokens' in name:
                    # v8: register_tokens는 std=0.001 small init, requires_grad=True 유지
                    new_data = torch.empty(param.shape, dtype=param.dtype, device=target)
                    nn.init.normal_(new_data, mean=0.0, std=0.001)
                    setattr(module, parts[-1], nn.Parameter(new_data, requires_grad=True))
                else:
                    new_data = torch.zeros(param.shape, dtype=param.dtype, device=target)
                    setattr(module, parts[-1], nn.Parameter(new_data, requires_grad=False))

    def _normalize_hw(self, value, name: str):
        if value is None:
            return None
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(f"{name} must be an int or a tuple/list of length 2.")

    def _local_token_count(self) -> int:
        """[v10] local(cropping) branch 토큰 수. env LOCAL_BRANCH로 분기.
        forward 출력 개수와 반드시 일치해야 num_image_token(=256+extra)이 맞음.
        center_crop=기존(len(sfe_scales)), dense_4x4=16, dense_2x2=4, dense_ms=20."""
        import os as _lb_os
        mode = _lb_os.environ.get("LOCAL_BRANCH", "center_crop")
        # none=E7(local 없음), center_crop=동심원, global_ca=E6(전역 pooling)
        if mode == "none":
            return 0
        if mode in ("center_crop", "global_ca"):
            return len(self.sfe_scales or [4, 8, 14, 20, 26, 32])
        if mode == "hybrid":
            # [H1] center-crop(foveal) + dense_4x4(전역) concat
            return len(self.sfe_scales or [4, 8, 14, 20, 26, 32]) + 16
        counts = {"dense_4x4": 16, "dense_4x4_satt": 16, "dense_4x4_overlap": 16,
                  "dense_2x2": 4, "dense_ms": 20, "dense_8x8": 64, "dense_4x4_pos": 16}
        if mode not in counts:
            raise ValueError(f"Unknown LOCAL_BRANCH={mode!r}")
        return counts[mode]

    def _infer_extra_tokens(self, grid_size: int) -> int:
        if self.spatial_feature_mode == "none":
            return 0
        if self.spatial_feature_mode == "register_only":
            return getattr(self.config, "register_num_tokens", 12)
        if self.spatial_feature_mode in ("koni_pool", "koni_crop", "koni_token_hier"):
            sfe_scales = self.sfe_scales or [4, 8, 14, 20, 26, grid_size * 2]
            n = len(sfe_scales)
            if self.spatial_feature_mode == "koni_token_hier":
                return n + self._local_token_count()
            if self.spatial_feature_mode == "koni_crop":
                return self._local_token_count()
            return n
        if self.spatial_feature_mode == "pool":
            pool_size = self.spatial_pool_size or (2, 2)
            return int(pool_size[0] * pool_size[1])
        if self.spatial_feature_mode == "crop":
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), grid_size)
            crop_w = min(int(crop_size[1]), grid_size)
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            num_h = 1 + (grid_size - crop_h) // stride_h
            num_w = 1 + (grid_size - crop_w) // stride_w
            return int(num_h * num_w)
        if self.spatial_feature_mode == "pool_crop":
            pool_size = self.spatial_pool_size or (2, 2)
            pool_tokens = int(pool_size[0] * pool_size[1])
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), grid_size)
            crop_w = min(int(crop_size[1]), grid_size)
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            num_h = 1 + (grid_size - crop_h) // stride_h
            num_w = 1 + (grid_size - crop_w) // stride_w
            crop_tokens = int(num_h * num_w)
            return pool_tokens + crop_tokens
        raise ValueError(f"Unsupported spatial_feature_mode: {self.spatial_feature_mode}")

    def _extract_spatial_tokens(self, grid: torch.Tensor) -> Optional[torch.Tensor]:
        if self.spatial_feature_mode == "none":
            return None

        features = grid.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        if self.spatial_feature_mode == "pool":
            pool_size = self.spatial_pool_size or (2, 2)
            pooled = F.adaptive_avg_pool2d(features, pool_size)
            return pooled.flatten(2).transpose(1, 2)
        if self.spatial_feature_mode == "crop":
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), features.size(2))
            crop_w = min(int(crop_size[1]), features.size(3))
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            windows = features.unfold(2, crop_h, stride_h).unfold(3, crop_w, stride_w)
            pooled = windows.mean(dim=(-1, -2))
            tokens = pooled.permute(0, 2, 3, 1).contiguous()
            return tokens.view(tokens.size(0), -1, tokens.size(-1))
        if self.spatial_feature_mode == "pool_crop":
            pool_size = self.spatial_pool_size or (2, 2)
            pooled = F.adaptive_avg_pool2d(features, pool_size)
            pool_tokens = pooled.flatten(2).transpose(1, 2)
            crop_size = self.spatial_crop_size or (2, 2)
            crop_stride = self.spatial_crop_stride or crop_size
            crop_h = min(int(crop_size[0]), features.size(2))
            crop_w = min(int(crop_size[1]), features.size(3))
            stride_h = max(1, min(int(crop_stride[0]), crop_h))
            stride_w = max(1, min(int(crop_stride[1]), crop_w))
            windows = features.unfold(2, crop_h, stride_h).unfold(3, crop_w, stride_w)
            pooled_windows = windows.mean(dim=(-1, -2))
            crop_tokens = pooled_windows.permute(0, 2, 3, 1).contiguous()
            crop_tokens = crop_tokens.view(crop_tokens.size(0), -1, crop_tokens.size(-1))
            return torch.cat([pool_tokens, crop_tokens], dim=1)

        raise ValueError(f"Unsupported spatial_feature_mode: {self.spatial_feature_mode}")

    def _sfe_pooling_forward(self, features: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        """SFE-Pooling: (optional spatial attn) + AdaptiveAvgPool2d(K) + Conv2d → project."""
        if self._use_spatial_attention:
            attn_map = self.sfe_spatial_attn(features)  # (B, 1, H, W)
            features = features * attn_map
        tokens = []
        for conv in self.sfe_pool_convs:
            t = conv(features).squeeze(-1).squeeze(-1)  # (B, sfe_dim)
            tokens.append(t)
        spatial = torch.stack(tokens, dim=1).nan_to_num()  # (B, N, sfe_dim)
        if return_raw:
            return spatial
        return self.sfe_pool_proj(spatial)  # (B, N, llm_hidden)

    def _sfe_cropping_dfi_forward(self, features: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        """SFE-Cropping + DFI: (optional channel attn) + center crops → Conv2d → cross-attention → project."""
        if self._use_channel_attention:
            gap = self.sfe_channel_pool(features).flatten(1)  # (B, C)
            w = self.sfe_channel_fc(gap)  # (B, C)
            features = features * w.unsqueeze(-1).unsqueeze(-1)
        B, C, H, W = features.shape
        center_h, center_w = H // 2, W // 2
        tokens = []
        for conv, k in zip(self.sfe_crop_convs, self._sfe_scales):
            if k >= H:
                crop = features
            else:
                half = k // 2
                h_start = min(max(0, center_h - half), H - k)
                w_start = min(max(0, center_w - half), W - k)
                crop = features[:, :, h_start:h_start+k, w_start:w_start+k]
            t = conv(crop).squeeze(-1).squeeze(-1)  # (B, sfe_dim)
            tokens.append(t)
        sfe_tokens = torch.stack(tokens, dim=1).nan_to_num()  # (B, N, sfe_dim)

        # DFI: detail tokens from full grid
        detail = self.dfi_conv_cm(features)  # (B, sfe_dim, dH, dW)
        detail = detail.flatten(2).transpose(1, 2).nan_to_num()  # (B, D, sfe_dim)

        # Cross-attention: Q=sfe, K/V=detail
        Q = self.dfi_query_proj(sfe_tokens)
        K = self.dfi_key_proj(detail)
        V = self.dfi_value_proj(detail)
        scale = K.shape[-1] ** 0.5
        attn = (Q @ K.transpose(-1, -2) / scale).nan_to_num().softmax(dim=-1)
        dfi_tokens = attn @ V  # (B, N, sfe_dim)

        fused = torch.cat([sfe_tokens, dfi_tokens], dim=-1)  # (B, N, sfe_dim*2)
        if return_raw:
            return fused
        return self.sfe_crop_proj(fused)  # (B, N, llm_hidden)

    def _local_forward(self, features: torch.Tensor) -> torch.Tensor:
        """[v10] dense local windows → (B, M, sfe_dim). center-crop 대체(중앙편향 제거).
        32×32 그리드를 P×P 겹치지 않는 윈도우로 나눠 각 윈도우를 1토큰(strided conv).
        channel attention은 center_crop과 동일하게 유지. dense_4x4_satt는 로컬 spatial attn 가중."""
        if self._use_channel_attention:
            gap = self.sfe_channel_pool(features).flatten(1)      # (B, C)
            wv = self.sfe_channel_fc(gap)                         # (B, C)
            features = features * wv.unsqueeze(-1).unsqueeze(-1)
        if self._local_mode == "global_ca":
            # [E6] 전역 멀티스케일 pooling(AdaptiveAvgPool(k)+Conv) → 6토큰. spatial attn 없음.
            toks = [conv(features).squeeze(-1).squeeze(-1) for conv in self.local_pool_convs]
            return torch.stack(toks, dim=1).nan_to_num()          # (B, 6, sfe_dim)
        if self._local_mode == "hybrid":
            # [H1] center-crop 6(foveal, 중앙 다중스케일) + dense_4x4 16(전역) concat = 22
            B, C, H, W = features.shape
            ch, cw = H // 2, W // 2
            crop_toks = []
            for conv, k in zip(self.sfe_crop_convs, self._sfe_scales):
                if k >= H:
                    crop = features
                else:
                    half = k // 2
                    hs = min(max(0, ch - half), H - k); ws = min(max(0, cw - half), W - k)
                    crop = features[:, :, hs:hs + k, ws:ws + k]
                crop_toks.append(conv(crop).squeeze(-1).squeeze(-1))
            crop_tokens = torch.stack(crop_toks, dim=1)           # (B, 6, sfe_dim)
            dense_toks = [c(features).flatten(2).transpose(1, 2) for c in self.dense_convs]
            dense_tokens = torch.cat(dense_toks, dim=1)           # (B, 16, sfe_dim)
            return torch.cat([crop_tokens, dense_tokens], dim=1).nan_to_num()  # (B, 22, sfe_dim)
        if self._local_mode == "dense_4x4_satt":
            features = features * self.local_satt(features)       # (B,1,H,W) saliency 가중
        toks = []
        for conv in self.dense_convs:
            t = conv(features)                                    # (B, sfe_dim, P, P)
            t = t.flatten(2).transpose(1, 2)                      # (B, P*P, sfe_dim)
            toks.append(t)
        out = torch.cat(toks, dim=1)                              # (B, M, sfe_dim)
        if self._local_mode == "dense_4x4_pos":
            # [H5] 각 윈도우 토큰에 학습된 2D 위치 임베딩 더함 → LLM이 "어느 구역" 토큰인지 grounding
            out = out + self.local_pos_emb.to(out.dtype)          # (1, 16, sfe_dim) broadcast
        return out.nan_to_num()                                   # (B, M, sfe_dim)

    def _extract_koni_tokens(self, vit_grid: torch.Tensor) -> torch.Tensor:
        """LLaVA-SP spatial token extraction on pre-pixel-shuffle ViT grid.

        Args:
            vit_grid: (B, H, W, C) — raw ViT patch grid (e.g. 32x32, dim=1024)

        Returns:
            Spatial tokens projected to LLM dim: (B, num_tokens, llm_hidden)
        """
        features = vit_grid.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        if self.spatial_feature_mode == "koni_pool":
            return self._sfe_pooling_forward(features)

        elif self.spatial_feature_mode == "koni_crop":
            return self._sfe_cropping_dfi_forward(features)

        elif self.spatial_feature_mode == "koni_token_hier":
            if self._use_token_self_attention:
                # Get raw tokens at sfe_dim level for interaction
                pool_raw = self._sfe_pooling_forward(features, return_raw=True)     # (B, 6, sfe_dim)
                if self._local_mode == "none":
                    combined = pool_raw                              # [E7] pool-only, local 없음
                elif self._local_mode == "center_crop":
                    crop_raw = self._sfe_cropping_dfi_forward(features, return_raw=True)  # (B, 6, sfe_dim*2)
                    local_reduced = self.sfe_crop_to_sfe(crop_raw)  # (B, 6, sfe_dim)
                    combined = torch.cat([pool_raw, local_reduced], dim=1)
                else:
                    local_reduced = self._local_forward(features)   # (B, M, sfe_dim)
                    combined = torch.cat([pool_raw, local_reduced], dim=1)  # (B, 6+M, sfe_dim)
                # Self-attention with residual
                residual = combined
                combined = self.sfe_token_ln(combined)
                combined, _ = self.sfe_token_self_attn(combined, combined, combined)
                combined = residual + combined
                # FFN with residual
                combined = combined + self.sfe_token_ffn(combined)
                result = self.sfe_token_proj(combined)  # (B, 6+M, llm_hidden)
                return result
            else:
                pool_tokens = self._sfe_pooling_forward(features)      # (B, 6, llm_hidden)
                if self._local_mode == "none":
                    return pool_tokens                                 # [E7] pool-only
                elif self._local_mode == "center_crop":
                    crop_tokens = self._sfe_cropping_dfi_forward(features)  # (B, 6, llm_hidden)
                else:
                    crop_tokens = self.local_dense_proj(self._local_forward(features))  # (B, M, llm_hidden)
                return torch.cat([pool_tokens, crop_tokens], dim=1)     # (B, 6+M, llm_hidden)

        raise ValueError(f"Unsupported LLaVA-SP mode: {self.spatial_feature_mode}")

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _fix_meta_params(self, target_device):
        """accelerate device_map='auto' 후 meta에 남은 파라미터/버퍼를 GPU로 이동."""
        for name, param in list(self.named_parameters()):
            if param.device.type == 'meta':
                parts = name.split('.')
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                # learnable 모듈 보존
                if 'sfe_spatial_gate' in name or 'register_tokens' in name:
                    new_data = torch.empty(param.shape, dtype=param.dtype, device=target_device)
                    if 'register_tokens' in name:
                        nn.init.normal_(new_data, mean=0.0, std=0.001)
                    else:
                        new_data.zero_()
                    setattr(module, parts[-1], nn.Parameter(new_data, requires_grad=True))
                else:
                    new_data = torch.zeros(param.shape, dtype=param.dtype, device=target_device)
                    setattr(module, parts[-1], nn.Parameter(new_data, requires_grad=False))
        for name, buf in list(self.named_buffers()):
            if buf.device.type == 'meta':
                parts = name.split('.')
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                new_data = torch.zeros(buf.shape, dtype=buf.dtype, device=target_device)
                module.register_buffer(parts[-1], new_data)

    def extract_feature(self, pixel_values):
        # 첫 호출 시 meta device 파라미터 수정 (accelerate 호환)
        if not getattr(self, '_meta_fixed', False):
            self._fix_meta_params(pixel_values.device)
            self._meta_fixed = True

        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        # ── v8 ablation: register_only mode (12 learnable empty tokens, no spatial info) ──
        if self.spatial_feature_mode == "register_only":
            ps_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            ps_embeds = ps_embeds.reshape(ps_embeds.shape[0], -1, ps_embeds.shape[-1])
            base_tokens = self.mlp1(ps_embeds)  # (B, 256, llm_hidden)
            # register_tokens module이 12 learnable empty tokens prepend
            return self.register_tokens(base_tokens)  # (B, 268, llm_hidden)

        # ── LLaVA-SP: SFE on pre-pixel-shuffle grid, separate projection ──
        if self.spatial_feature_mode in ("koni_pool", "koni_crop", "koni_token_hier"):
            # Branch 1: SFE spatial tokens (on 32x32 raw ViT grid)
            spatial_tokens = self._extract_koni_tokens(vit_embeds)  # (B, 12, llm_hidden)

            # Normalize spatial tokens + learnable gate (scale mismatch 방지)
            spatial_tokens = self.sfe_output_ln(spatial_tokens)

            # Branch 2: Base tokens (existing pixel_shuffle + MLP1 pipeline)
            ps_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            ps_embeds = ps_embeds.reshape(ps_embeds.shape[0], -1, ps_embeds.shape[-1])
            base_tokens = self.mlp1(ps_embeds)  # (B, 256, llm_hidden)

            # ─── [ABLATION] Gate/Base mixing strategy ─────────────────────
            # ABLATION_VARIANT 환경변수로 3가지 변형 switching
            #   A (default): baseline → spatial = tanh(gate) × x, base 그대로
            #   B (base_scaling): α × base (curriculum, α 0.5→1.0), spatial = tanh(gate) × x
            #   C (convex): spatial = sigmoid(gate) × x, base = (1-sigmoid(gate)) × x
            import os as _ab_os
            _ab_variant = _ab_os.environ.get("ABLATION_VARIANT", "A").upper()

            if _ab_variant == "C":
                # Convex combination: (1-g)×base + g×kth (g = sigmoid)
                # 주의: ScalarGate에서만 동작. 새 gate 타입은 Variant A 강제.
                if not isinstance(self.sfe_spatial_gate, ScalarGate):
                    raise ValueError(
                        "Variant C requires GATE_TYPE=scalar; "
                        f"got {type(self.sfe_spatial_gate).__name__}"
                    )
                _g = torch.sigmoid(self.sfe_spatial_gate.weight.float())
                _g = _g.to(spatial_tokens.dtype)
                spatial_tokens = _g * spatial_tokens
                base_tokens = (1.0 - _g).to(base_tokens.dtype) * base_tokens
            elif _ab_variant == "B":
                # Base scaling curriculum + 기존 tanh gate
                spatial_tokens = self.sfe_spatial_gate(spatial_tokens, base_tokens=base_tokens)
                _alpha_min = float(_ab_os.environ.get("BASE_SCALE_MIN", "0.5"))
                _alpha_warmup = int(_ab_os.environ.get("BASE_SCALE_WARMUP", "700"))
                _cur_step = int(getattr(self, "_current_train_step", 0))
                if _cur_step < _alpha_warmup:
                    _alpha = _alpha_min + (1.0 - _alpha_min) * (_cur_step / max(_alpha_warmup, 1))
                else:
                    _alpha = 1.0
                base_tokens = _alpha * base_tokens
            else:
                # A (baseline): gate 적용 (scalar / per_token / content_aware 모두 지원)
                spatial_tokens = self.sfe_spatial_gate(spatial_tokens, base_tokens=base_tokens)

            # dtype 통일 (새로 초기화된 LN/gate가 fp32일 수 있음)
            spatial_tokens = spatial_tokens.to(base_tokens.dtype)

            # ── Cross-attention fusion: base tokens selectively attend to KTH ──
            if self._use_cross_attention_fusion:
                q = self.sfe_xattn_ln_base(base_tokens)
                kv = self.sfe_xattn_ln_kth(spatial_tokens)
                attn_out, _ = self.sfe_xattn(q, kv, kv)
                attn_out = attn_out + self.sfe_xattn_ffn(attn_out)
                base_tokens = base_tokens + self.sfe_xattn_scale * attn_out

                if self._cross_attention_fusion_concat_kth:
                    # concat 모드: base(256) + KTH(12) = 268 토큰
                    return torch.cat([spatial_tokens, base_tokens], dim=1)
                else:
                    # fusion only 모드: base(256)만 반환 (KTH 정보는 attention으로 흡수됨)
                    return base_tokens

            # Prepend spatial tokens (LLaVA-SP convention)
            return torch.cat([spatial_tokens, base_tokens], dim=1)  # (B, 268, llm_hidden)

        # ── Legacy modes: pool / crop / pool_crop ──
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        extra_tokens = self._extract_spatial_tokens(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        if extra_tokens is not None:
            vit_embeds = torch.cat([vit_embeds, extra_tokens.to(vit_embeds.dtype)], dim=1)
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
