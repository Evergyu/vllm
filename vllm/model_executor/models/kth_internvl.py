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


def _kth_extra_tokens(config) -> int:
    """Extra spatial tokens added on top of the base image tokens per tile."""
    mode = getattr(config, "spatial_feature_mode", "none")
    scales = getattr(config, "sfe_scales", None) or [4, 8, 14, 20, 26, 32]
    n = len(scales)
    if mode == "llava_sp_both":
        return n * 2
    if mode in ("llava_sp_pooling", "llava_sp_cropping"):
        return n
    if mode == "register_only":
        return 12
    return 0


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
        """Materialize the meta params/buffers of the vision module from checkpoint."""
        dev = next(self.language_model.parameters()).device
        own = dict(self._kth.named_parameters())
        own_buf = dict(self._kth.named_buffers())
        loaded = set()
        for name, w in kth_state.items():
            tgt = own.get(name)
            is_buf = False
            if tgt is None:
                tgt = own_buf.get(name)
                is_buf = True
            if tgt is None:
                continue  # unknown key (e.g. stub leftover) -> skip
            data = w.to(device=dev, dtype=tgt.dtype if tgt.dtype.is_floating_point else w.dtype)
            _set_submodule_tensor(self._kth, name, data, is_buffer=is_buf)
            loaded.add(name)
        # remaining meta params (e.g. learned register inits without weights) -> zero init
        _materialize_remaining_meta(self._kth, dev)
        # report every vision-module param/buffer under the model path (_kth.*) as loaded
        full = {f"_kth.{n}" for n, _ in self._kth.named_parameters()}
        full |= {f"_kth.{n}" for n, _ in self._kth.named_buffers()}
        return full


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
