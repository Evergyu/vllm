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
from dataclasses import replace

from vllm.platforms import current_platform

import torch

from .internvl import (
    InternVLChatModel as _VLLMInternVL,
    InternVLProcessingInfo,
    InternVLProcessor,
    InternVLDummyInputsBuilder,
    InternVLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ._kth.kth_vision import has_kth, build_kth_vision, infer_extra_tokens
from ._kth.kth_tiling import KTHInternVLImageProcessor, kth_target_ratios

# old checkpoints (KTH_720k, FullFT_*) name the modes llava_sp_*; v10 renamed them koni_*
_SFM_ALIAS = {
    "llava_sp_both": "koni_token_hier",
    "llava_sp_pooling": "koni_pool",
    "llava_sp_cropping": "koni_crop",
}


def _kth_extra_tokens(config, model_path: str | None = None) -> int:
    """Extra spatial tokens per tile — computed by the checkpoint's own code.

    Never reimplement this. See ``_kth/kth_vision.infer_extra_tokens``.
    """
    return infer_extra_tokens(config, model_path)


class KTHInternvlProcessingInfo(InternVLProcessingInfo):
    def get_image_processor(self, **kwargs):
        """Use the reference harness's tie-break order when tiling.

        See ``_kth/kth_tiling`` — upstream picks a different tied ratio, which
        changes the tile count on ~10% of real benchmark images. Serving has to
        reproduce the published numbers.
        """
        proc = super().get_image_processor(**kwargs)
        if not has_kth(self.get_hf_config()):
            return proc
        return KTHInternVLImageProcessor(
            image_size=proc.image_size,
            min_dynamic_patch=proc.min_dynamic_patch,
            max_dynamic_patch=proc.max_dynamic_patch,
            dynamic_image_size=proc.dynamic_image_size,
            use_thumbnail=proc.use_thumbnail,
        )

    def get_video_processor(self, **kwargs):
        """Drop image-only kwargs before they reach the video processor.

        ``KTHInternVLMultiModalProcessor`` injects ``max_dynamic_patch`` into
        ``hf_processor_mm_kwargs`` to share one tile budget across a request's images.
        ``InternVLProcessingInfo.get_hf_processor`` forwards *all* of those kwargs to
        ``get_video_processor``, and ``InternVLVideoProcessor.__init__`` does not take
        ``max_dynamic_patch`` — so the whole request raised

            TypeError: InternVLVideoProcessor.__init__() got an unexpected keyword
                       argument 'max_dynamic_patch'

        The injection only fires when ``dyn < max_dynamic_patch``, i.e. from **6 images
        up** (64 // 6 = 10 < 12). So single-image and 4~5 image requests never hit it,
        but multi-page documents (mmlongbench_doc, MMLB_MAX_PAGES=8) did — and
        lmms-eval swallows the exception and exits 0, so it looked like a bad score
        rather than a crash. (2026-08-31)
        """
        return super().get_video_processor(
            **{k: v for k, v in kwargs.items() if k != "max_dynamic_patch"})

    def get_hf_processor(self, **kwargs) -> InternVLProcessor:
        proc = super().get_hf_processor(**kwargs)
        config = self.get_hf_config()
        if not has_kth(config):
            return proc

        proc.image_seq_length = (int(proc.image_seq_length)
                                 + _kth_extra_tokens(config, self.model_id))

        # The placeholder count must be derived from the same candidate order the
        # pixels are tiled with, or the image tokens desync from the tiles.
        _orig_resolve = proc.resolve_target_ratios

        def _resolve(*args, **kw):
            ratios = _orig_resolve(*args, **kw)
            mn = min(a * b for a, b in ratios)
            mx = max(a * b for a, b in ratios)
            return kth_target_ratios(mn, mx)

        proc.resolve_target_ratios = _resolve
        return proc


# Tile budget shared across the images of one request. The reference harness
# (`lmms_eval/models/simple/internvl3.py`, `max_num=12`, `total_max_num=64`) applies it
# per request in the KTH wrapper:
#
#     dyn = max(1, min(max_num, total_max_num // n_images))
#
# Upstream vLLM gives *every* image the full `max_dynamic_patch`, so on an 8-page
# document the reference tiles each page into at most 8 tiles while vLLM would use 12.
# The published numbers were measured with the reference budget; serving has to
# reproduce it. For 1-5 images the two agree (64 // 5 = 12), so single-image
# benchmarks are untouched.
_KTH_TOTAL_MAX_NUM = int(os.environ.get("KTH_TOTAL_MAX_NUM", "64"))


class KTHInternVLMultiModalProcessor(InternVLMultiModalProcessor):
    """Share one tile budget across the images of a request, as the reference does.

    The cap is injected into ``hf_processor_mm_kwargs`` **once**, at the entry point,
    so the pixel tiling and the placeholder count are both derived from it. Setting it
    in only one of the two paths desyncs image tokens from tiles.
    """

    def apply(self, inputs, *args, **kwargs):
        items = inputs.mm_data_items.get("image")
        n = len(items) if items is not None else 0
        # An explicit per-request `max_dynamic_patch` from the caller wins.
        if n > 1 and "max_dynamic_patch" not in inputs.hf_processor_mm_kwargs:
            config = self.info.get_hf_config()
            if has_kth(config):
                mx = int(getattr(config, "max_dynamic_patch", 12) or 12)
                dyn = max(1, min(mx, _KTH_TOTAL_MAX_NUM // n))
                if dyn < mx:
                    inputs = replace(
                        inputs,
                        hf_processor_mm_kwargs={
                            **inputs.hf_processor_mm_kwargs,
                            "max_dynamic_patch": dyn,
                        },
                    )
        return super().apply(inputs, *args, **kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    KTHInternVLMultiModalProcessor,
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
            # Build inside _mark_tower_model so the tower is skipped when the server
            # runs text-only (--limit-mm-per-prompt image=0), same as the stock
            # vision_model it replaces. Marking it separately would leave the stale
            # marking from super().__init__(), which named vision_model/mlp1 — the
            # modules we are about to drop.
            model_path = getattr(vllm_config.model_config, "model", None)
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                # the checkpoint's own vision tower, on meta; weights arrive in
                # load_weights
                self._kth = build_kth_vision(config, model_path)
            self.num_image_token = int(self._kth.num_image_token)
            # stock vision/mlp1 unused (their weights go to _kth) -> avoid conflicts
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
        # Not necessarily the language model's device: under pipeline parallelism a
        # stage can hold no language-model parameters at all. Fall back to the
        # device the runner put us on.
        dev = next((p.device for p in self.language_model.parameters()), None)
        if dev is None:
            dev = next((p.device for p in self.parameters()
                        if p.device.type != "meta"), None) or current_platform.device_type
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
