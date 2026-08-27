"""Regression: KTH extra-token count must come from the vendored modeling code.

The bug this pins: ``_kth_extra_tokens`` used to hardcode the count from the
``llava_sp_*`` mode names. Checkpoints later renamed those modes to ``koni_*`` and
made the cropping-branch count depend on the ``LOCAL_BRANCH`` env var. The hardcoded
version then returned 0 for every real checkpoint — the model loaded, ran, and
answered with the entire spatial-feature contribution missing and no error raised.

Any future edit that reimplements the count instead of delegating will fail here.
"""
import os

import pytest

from vllm.model_executor.models.kth_internvl import _kth_extra_tokens
from vllm.model_executor.models._kth import _orig_modeling as _orig


class _Cfg:
    """Just the fields the token-count path reads."""

    class _Vision:
        image_size = 448
        patch_size = 14

    def __init__(self, mode, scales=(4, 8, 14, 20, 26, 32)):
        self.spatial_feature_mode = mode
        self.sfe_scales = list(scales)
        self.vision_config = self._Vision()
        self.downsample_ratio = 0.5


# grid = (448 // 14) * 0.5 = 16 -> 256 base tokens per tile
@pytest.mark.parametrize(
    "local_branch,expected",
    [
        ("center_crop", 12),   # 6 scales + 6 concentric crops
        ("global_ca", 12),     # 6 scales + 6 (global pooling + channel attn)
        ("none", 6),           # scales only, no local branch
        ("hybrid", 28),        # 6 + (6 + 16)
        ("dense_4x4", 22),     # 6 + 16
        ("dense_8x8", 70),     # 6 + 64
    ],
)
def test_koni_token_hier_follows_local_branch(monkeypatch, local_branch, expected):
    monkeypatch.setenv("LOCAL_BRANCH", local_branch)
    assert _kth_extra_tokens(_Cfg("koni_token_hier")) == expected


def test_legacy_llava_sp_names_are_aliased(monkeypatch):
    """Pre-rename checkpoints must resolve to the same count as their koni_* name."""
    monkeypatch.setenv("LOCAL_BRANCH", "center_crop")
    assert _kth_extra_tokens(_Cfg("llava_sp_both")) == _kth_extra_tokens(
        _Cfg("koni_token_hier"))


def test_plain_internvl_gets_no_extra_tokens():
    assert _kth_extra_tokens(_Cfg("none")) == 0


def test_delegates_to_vendored_modeling(monkeypatch):
    """The count must equal what the checkpoint's own module would compute."""
    monkeypatch.setenv("LOCAL_BRANCH", "global_ca")
    cfg = _Cfg("koni_token_hier")

    class _Shim:
        config = cfg
        spatial_feature_mode = "koni_token_hier"
        sfe_scales = cfg.sfe_scales
        spatial_pool_size = spatial_crop_size = spatial_crop_stride = None
        _local_token_count = _orig.InternVLChatModel._local_token_count
        _infer_extra_tokens = _orig.InternVLChatModel._infer_extra_tokens

    assert _kth_extra_tokens(cfg) == _Shim()._infer_extra_tokens(16)


def test_unknown_local_branch_is_rejected(monkeypatch):
    """Silently guessing a token count would desync the image placeholders."""
    monkeypatch.setenv("LOCAL_BRANCH", "not_a_real_branch")
    with pytest.raises(ValueError, match="LOCAL_BRANCH"):
        _kth_extra_tokens(_Cfg("koni_token_hier"))
