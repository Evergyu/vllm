"""KTHInternvlChatModel: native vLLM support for encoder-modified InternVL.

These checkpoints are InternVL with extra spatial vision modules (LLaVA-SP style)
added to the encoder, then SFT'd. This class subclasses the vLLM InternVL model and,
when the config declares a spatial-feature mode:

- swaps the vision tower for the checkpoint's original (eager, numerically-exact)
  module so ``extract_feature`` returns the extra spatial tokens per tile,
- runs the Qwen3 decoder on vLLM's KV-cache / paged-attention engine,
- adjusts the processor's ``image_seq_length`` so image placeholders match,
- routes ``language_model.*`` weights to the vLLM LLM and the vision/spatial params
  to the original module.

For plain InternVL configs the same class falls back to stock vLLM behavior.
"""
import os
from collections.abc import Iterable

import torch

from .internvl import (
    InternVLChatModel as _VLLMInternVL,
    InternVLProcessingInfo,
    InternVLProcessor,
    InternVLDummyInputsBuilder,
    InternVLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ._kth.kth_vision import has_kth, build_kth_vision
from ._kth import _orig_modeling as _orig

# old checkpoints (KTH_720k, FullFT_*) name the modes llava_sp_*; v10 renamed them koni_*
_SFM_ALIAS = {
    "llava_sp_both": "koni_token_hier",
    "llava_sp_pooling": "koni_pool",
    "llava_sp_cropping": "koni_crop",
}


def _kth_extra_tokens(config) -> int:
    """Extra spatial tokens added on top of the base image tokens per tile.

    Do NOT reimplement this. The count depends on ``spatial_feature_mode``, on
    ``sfe_scales`` and — for the cropping branch — on the ``LOCAL_BRANCH`` env var
    (center_crop/global_ca = len(scales), dense_4x4 = 16, dense_8x8 = 64, hybrid =
    len(scales)+16, none = 0). Delegate to the checkpoint's own ``_infer_extra_tokens``
    so this can never drift from the vendored modeling code.
    """
    return _infer_extra_tokens_via_model(config)


class _ExtraTokenShim:
    """Minimal stand-in exposing exactly what ``_infer_extra_tokens`` reads."""

    def __init__(self, config):
        self.config = config
        raw = getattr(config, "spatial_feature_mode", "none")
        # old checkpoints use the llava_sp_* names; the model normalizes them to koni_*
        self.spatial_feature_mode = _SFM_ALIAS.get(raw, raw)
        self.sfe_scales = getattr(config, "sfe_scales", None)
        self.spatial_pool_size = getattr(config, "spatial_pool_size", None)
        self.spatial_crop_size = getattr(config, "spatial_crop_size", None)
        self.spatial_crop_stride = getattr(config, "spatial_crop_stride", None)

    _local_token_count = _orig.InternVLChatModel._local_token_count
    _infer_extra_tokens = _orig.InternVLChatModel._infer_extra_tokens


def _infer_extra_tokens_via_model(config) -> int:
    vc = config.vision_config
    downsample_ratio = float(getattr(config, "downsample_ratio", 0.5))
    grid = int((vc.image_size // vc.patch_size) * downsample_ratio)
    return int(_ExtraTokenShim(config)._infer_extra_tokens(grid))


class KTHInternvlProcessingInfo(InternVLProcessingInfo):
    def get_hf_processor(self, **kwargs) -> InternVLProcessor:
        proc = super().get_hf_processor(**kwargs)
        config = self.get_hf_config()
        if has_kth(config):
            proc.image_seq_length = int(proc.image_seq_length) + _kth_extra_tokens(config)
        return proc


@MULTIMODAL_REGISTRY.register_processor(
    InternVLMultiModalProcessor,
    info=KTHInternvlProcessingInfo,
    dummy_inputs=InternVLDummyInputsBuilder,
)
class KTHInternvlChatModel(_VLLMInternVL):
    """Encoder-modified InternVL: original vision tower + vLLM Qwen3 decode."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        self._is_kth = has_kth(config)
        if self._is_kth:
            # original vision tower on meta device; weights filled in load_weights
            self._kth = build_kth_vision(config)
            self.num_image_token = int(self._kth.num_image_token)
            # stock vision/mlp1 unused (weights go to _kth) -> avoid conflicts
            self.vision_model = None
            self.mlp1 = None

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_is_kth", False):
            return self._kth.extract_feature(pixel_values)
        return super().extract_feature(pixel_values)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not getattr(self, "_is_kth", False):
            return super().load_weights(weights)

        # split language_model.* (vLLM LLM) from vision/spatial params (original module)
        llm_weights = []
        kth_state = {}
        for name, w in weights:
            if name.startswith("language_model."):
                llm_weights.append((name[len("language_model."):], w))
            else:
                kth_state[name] = w

        loaded = set()
        llm_loaded = self.language_model.load_weights(llm_weights)
        loaded |= {f"language_model.{n}" for n in (llm_loaded or [])}

        loaded |= self._load_kth_state(kth_state)
        return loaded

    def _load_kth_state(self, kth_state: dict) -> set[str]:
        """Materialize the meta params/buffers of the vision module from checkpoint.

        This must be strict. The vision tower here is the checkpoint's *own* module,
        so a name that does not match means the vendored modeling code has drifted
        from the checkpoint — and silently zero-filling it would produce a model that
        loads without error and answers with the spatial features missing.
        """
        dev = next(self.language_model.parameters()).device
        own = dict(self._kth.named_parameters())
        own_buf = dict(self._kth.named_buffers())
        loaded = set()
        unexpected = []
        for name, w in kth_state.items():
            tgt = own.get(name)
            is_buf = False
            if tgt is None:
                tgt = own_buf.get(name)
                is_buf = True
            if tgt is None:
                unexpected.append(name)
                continue
            if tuple(tgt.shape) != tuple(w.shape):
                raise ValueError(
                    f"KTH vision weight shape mismatch for {name!r}: "
                    f"module expects {tuple(tgt.shape)}, checkpoint has {tuple(w.shape)}. "
                    "The vendored modeling code does not match this checkpoint."
                )
            data = w.to(device=dev, dtype=tgt.dtype if tgt.dtype.is_floating_point else w.dtype)
            _set_submodule_tensor(self._kth, name, data, is_buffer=is_buf)
            loaded.add(name)

        # anything still on meta got no checkpoint weight -> that is a load failure,
        # not something to zero-fill. Buffers may legitimately be recomputed.
        missing = [n for n, p in self._kth.named_parameters() if p.device.type == "meta"]
        if missing or unexpected:
            raise ValueError(
                "KTH vision weight loading is incomplete — refusing to run with an "
                "unverified vision tower.\n"
                f"  spatial_feature_mode = {getattr(self.config, 'spatial_feature_mode', None)!r}\n"
                f"  LOCAL_BRANCH         = {os.environ.get('LOCAL_BRANCH', 'center_crop')!r}\n"
                f"  {len(missing)} param(s) with no checkpoint weight: {missing[:8]}\n"
                f"  {len(unexpected)} checkpoint key(s) with no module param: {unexpected[:8]}\n"
                "Re-vendor vllm/model_executor/models/_kth/ from this checkpoint's "
                "remote code (modeling_internvl_chat.py -> _orig_modeling.py)."
            )
        _materialize_remaining_meta(self._kth, dev)  # buffers only, by now
        return {f"_kth.{n}" for n in loaded}


def _set_submodule_tensor(root, dotted: str, data: torch.Tensor, is_buffer=False):
    *path, leaf = dotted.split(".")
    mod = root
    for p in path:
        mod = getattr(mod, p)
    if is_buffer:
        mod._buffers[leaf] = data
    else:
        import torch.nn as nn
        mod._parameters[leaf] = nn.Parameter(data, requires_grad=False)


def _materialize_remaining_meta(root, dev):
    import torch.nn as nn
    for name, p in list(root.named_parameters()):
        if p.device.type == "meta":
            *path, leaf = name.split(".")
            mod = root
            for x in path:
                mod = getattr(mod, x)
            mod._parameters[leaf] = nn.Parameter(
                torch.zeros(p.shape, dtype=p.dtype if p.dtype.is_floating_point else torch.bfloat16, device=dev),
                requires_grad=False)
    for name, b in list(root.named_buffers()):
        if b is not None and b.device.type == "meta":
            *path, leaf = name.split(".")
            mod = root
            for x in path:
                mod = getattr(mod, x)
            mod._buffers[leaf] = torch.zeros(b.shape, dtype=b.dtype, device=dev)
