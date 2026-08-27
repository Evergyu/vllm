"""Regression: KTH vision code must come from the checkpoint, never a vendored copy.

The bug this pins. The extra-token count was hardcoded from the ``llava_sp_*`` mode
names and the modeling code was vendored under ``_kth/``. Checkpoints later renamed
those modes to ``koni_*``, replaced the content-aware gate, and made the cropping
branch selectable through the ``LOCAL_BRANCH`` env var. The vendored copy went stale,
so on a current checkpoint the count came out 0, the spatial modules were never built,
and their weights were dropped — while the token bookkeeping stayed self-consistent.
The model loaded, ran, and answered with the whole KTH contribution missing, and
nothing raised.

The fix is structural: load the class out of the checkpoint at runtime. These tests
fail if anyone reintroduces a reimplementation or a vendored import.
"""
import os

import pytest

from vllm.model_executor.models._kth import kth_vision
from vllm.model_executor.models.kth_internvl import _kth_extra_tokens


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


def test_unknown_local_branch_is_rejected(monkeypatch):
    """Silently guessing a token count would desync the image placeholders."""
    monkeypatch.setenv("LOCAL_BRANCH", "not_a_real_branch")
    with pytest.raises(ValueError, match="LOCAL_BRANCH"):
        _kth_extra_tokens(_Cfg("koni_token_hier"))


def test_class_is_loaded_from_the_checkpoint(tmp_path, monkeypatch):
    """A checkpoint that ships modeling code must win over the vendored fallback."""
    shipped = tmp_path / "modeling_internvl_chat.py"
    shipped.write_text(
        "class InternVLChatModel:\n"
        "    _loaded_from_checkpoint = True\n"
        "    def tie_weights(self):\n"
        "        pass\n"
    )
    seen = {}

    def _fake(qualname, path, **kwargs):
        seen["path"] = path
        ns: dict = {}
        exec(shipped.read_text(), ns)
        return ns["InternVLChatModel"]

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module", _fake)
    cls = kth_vision.load_kth_class(str(tmp_path))
    assert getattr(cls, "_loaded_from_checkpoint", False)
    assert seen["path"] == str(tmp_path)


def test_falls_back_only_when_checkpoint_ships_no_code(tmp_path):
    """No remote code in the checkpoint -> vendored copy, and it must still work."""
    cls = kth_vision.load_kth_class(str(tmp_path))
    assert cls.__name__ == "InternVLChatModel"
    assert not getattr(cls, "_loaded_from_checkpoint", False)


def test_internvl_arch_routes_to_kth():
    """KTH checkpoints declare architectures=["InternVLChatModel"].

    Serving them must not require editing config.json, so that name has to resolve
    to the KTH class (which falls back to stock behaviour for plain InternVL).
    """
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
    assert _MULTIMODAL_MODELS["InternVLChatModel"] == (
        "kth_internvl", "KTHInternvlChatModel")


# ---------------------------------------------------------------- tiling parity
def _harness_ratios(min_num: int, max_num: int):
    """How the reference harness builds candidates: a set, iterated as-is."""
    return {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }


def test_tiling_matches_the_reference_harness():
    """Tie-breaks must resolve the way every published KONI-V number was measured.

    ``find_closest_aspect_ratio`` picks the last tied candidate that passes the area
    test, so the iteration order decides the tile count. Upstream sorts by block
    count; the harness passes the set. They disagree on ~10% of real benchmark
    images (e.g. a 1000x1000 image: 1 tile vs 9).
    """
    from PIL import Image

    from vllm.model_executor.models._kth.koni_tiling import koni_target_ratios
    from vllm.transformers_utils.processors.internvl import (
        dynamic_preprocess_internvl,
        get_internvl_target_ratios,
    )

    sizes = [(800, 600), (1920, 1080), (448, 448), (1000, 1000), (1500, 1500),
             (2000, 400), (640, 1400), (1024, 768), (1280, 720), (300, 900)]
    upstream_differs = 0
    for w, h in sizes:
        img = Image.new("RGB", (w, h))
        koni = dynamic_preprocess_internvl(
            img, target_ratios=koni_target_ratios(1, 12),
            image_size=448, use_thumbnail=True)
        harness = dynamic_preprocess_internvl(
            img, target_ratios=_harness_ratios(1, 12),
            image_size=448, use_thumbnail=True)
        upstream = dynamic_preprocess_internvl(
            img, target_ratios=get_internvl_target_ratios(1, 12),
            image_size=448, use_thumbnail=True)

        assert len(koni) == len(harness), (w, h, len(koni), len(harness))
        if len(upstream) != len(harness):
            upstream_differs += 1

    # If this ever hits 0, upstream converged on the harness order and this whole
    # module can go away. Until then it must stay.
    assert upstream_differs > 0


def test_square_image_tiles_like_the_harness():
    """The clearest single case: a large square image is one tile, not nine."""
    from PIL import Image

    from vllm.model_executor.models._kth.koni_tiling import koni_target_ratios
    from vllm.transformers_utils.processors.internvl import dynamic_preprocess_internvl

    tiles = dynamic_preprocess_internvl(
        Image.new("RGB", (1000, 1000)), target_ratios=koni_target_ratios(1, 12),
        image_size=448, use_thumbnail=True)
    assert len(tiles) == 1


def test_chat_template_injects_the_koni_system_prompt():
    """The checkpoint's own template omits it, so serving would drop it silently."""
    from pathlib import Path

    from jinja2 import Template

    path = Path(__file__).parents[4] / "examples/template/koni_v.jinja"
    tpl = Template(path.read_text(encoding="utf-8"))

    out = tpl.render(messages=[{"role": "user", "content": "hi"}],
                     add_generation_prompt=True)
    assert out.startswith("<|im_start|>system\n")
    assert "KONI-V" in out

    # An explicit system message from the caller must win.
    out2 = tpl.render(
        messages=[{"role": "system", "content": "You are X."},
                  {"role": "user", "content": "hi"}],
        add_generation_prompt=True)
    assert out2.startswith("<|im_start|>system\nYou are X.")
    assert "KONI-V" not in out2
