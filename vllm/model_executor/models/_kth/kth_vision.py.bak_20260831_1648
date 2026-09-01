"""Build the encoder-modified InternVL vision tower for native vLLM use.

**The modeling code is loaded from the checkpoint itself, never vendored.**

These checkpoints ship their own ``modeling_internvl_chat.py`` (HF remote code) and
it keeps changing: the spatial-feature modes were renamed ``llava_sp_*`` -> ``koni_*``,
the content-aware gate was replaced, and the cropping branch became selectable via the
``LOCAL_BRANCH`` env var. A vendored copy silently goes stale — and because the token
bookkeeping stays self-consistent, the model then loads, runs, and answers with the
whole spatial contribution missing and no error anywhere.

So we do what the reference harness does: pull the class out of the checkpoint at
runtime with ``get_class_from_dynamic_module``. The vision math is then byte-identical
to the HuggingFace reference by construction, not by keeping a copy in sync.

The class is instantiated with a stub LLM so only the vision tower (ViT + mlp1 + the
extra spatial modules) is built; vLLM runs the decoder on its own engine.
"""
import os

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)



class _StubLLM(nn.Module):
    """Placeholder LM so the original ``__init__`` skips building the real decoder."""

    def __init__(self, llm_config):
        super().__init__()
        self.config = llm_config
        self._emb = nn.Embedding(1, 1)

    def get_input_embeddings(self):
        return self._emb


def has_kth(config) -> bool:
    """True if this is an encoder-modified (spatial-feature) InternVL checkpoint."""
    mode = getattr(config, "spatial_feature_mode", "none")
    return mode not in (None, "none", "")


def _apply_tf_compat(cls) -> None:
    """Let an older custom ``InternVLChatModel`` construct under transformers 5.x.

    ``tie_weights`` is called with ``missing_keys=...`` in 5.x but the legacy class
    overrides it with no ``**kwargs``. Swallow the extra arguments.
    """
    if getattr(cls, "_kth_tie_patched", False):
        return
    _orig_tie = cls.tie_weights
    cls.tie_weights = lambda self, *a, **kw: _orig_tie(self)
    cls._kth_tie_patched = True


def load_kth_class(model_path: str | None, config=None):
    """Return the checkpoint's own ``InternVLChatModel`` class.

    Works for a local directory and for a Hugging Face Hub repo id alike: the
    ``auto_map`` entry in ``config.json`` is the signal, and
    ``get_class_from_dynamic_module`` resolves either form (downloading the module
    from the Hub when needed).

    Raises if the checkpoint ships no remote code: there is deliberately no vendored
    fallback, because a stale copy loses the spatial modules without any error.
    """
    ref = None
    if config is not None:
        auto_map = getattr(config, "auto_map", None) or {}
        ref = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if ref is None and model_path and os.path.isfile(
            os.path.join(model_path, "modeling_internvl_chat.py")):
        ref = "modeling_internvl_chat.InternVLChatModel"

    if ref and model_path:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module(ref, model_path)
        _apply_tf_compat(cls)
        return cls

    raise ValueError(
        f"No remote modeling code found for {model_path!r}. This model class runs the "
        "checkpoint's own vision tower, so the checkpoint must ship "
        "modeling_*.py and declare it in config.json's auto_map. There is no vendored "
        "fallback on purpose: a copy goes stale, and when it does the spatial modules "
        "are silently not built and their weights are dropped without an error.")


def build_kth_vision(config, model_path: str | None = None) -> nn.Module:
    """Build the modified vision tower on the meta device (weights filled by vLLM).

    The returned module owns ``vision_model`` / ``mlp1`` / the spatial modules; its
    ``extract_feature(pixel_values)`` yields ``(num_patches, num_image_token, hidden)``
    including the extra spatial tokens.
    """
    cls = load_kth_class(model_path, config)
    stub = _StubLLM(config.llm_config)
    with torch.device("meta"):
        m = cls(config, language_model=stub, use_flash_attn=False)
    m.language_model = None  # drop the stub so it is excluded from weight loading
    return m


def infer_extra_tokens(config, model_path: str | None = None) -> int:
    """Extra spatial tokens per tile, computed by the checkpoint's own code.

    Do NOT reimplement this. The count depends on ``spatial_feature_mode``, on
    ``sfe_scales`` and — for the cropping branch — on the ``LOCAL_BRANCH`` env var
    (center_crop/global_ca = len(scales), dense_4x4 = 16, dense_8x8 = 64,
    hybrid = len(scales)+16, none = 0).
    """
    cls = load_kth_class(model_path, config)
    raw = getattr(config, "spatial_feature_mode", "none")
    alias = {"llava_sp_both": "koni_token_hier",
             "llava_sp_pooling": "koni_pool",
             "llava_sp_cropping": "koni_crop"}

    class _Shim:
        pass

    shim = _Shim()
    shim.config = config
    shim.spatial_feature_mode = alias.get(raw, raw)
    shim.sfe_scales = getattr(config, "sfe_scales", None)
    shim.spatial_pool_size = getattr(config, "spatial_pool_size", None)
    shim.spatial_crop_size = getattr(config, "spatial_crop_size", None)
    shim.spatial_crop_stride = getattr(config, "spatial_crop_stride", None)
    shim._local_token_count = cls._local_token_count.__get__(shim)
    vc = config.vision_config
    grid = int((vc.image_size // vc.patch_size) * float(getattr(config, "downsample_ratio", 0.5)))
    return int(cls._infer_extra_tokens(shim, grid))
