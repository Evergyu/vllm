"""Build the encoder-modified InternVL vision tower for native vLLM use.

**The modeling code is loaded from the checkpoint itself, never vendored.**

These checkpoints ship their own ``modeling_internvl_chat.py`` (HF remote code) and
it keeps changing: the spatial-feature modes were renamed from the ``llava_sp_*``
spelling to the current one (see ``SPATIAL_MODE_ALIASES`` below), the content-aware
gate was replaced, and the cropping branch became selectable via the
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

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for the checkpoint-side spatial-mode vocabulary.
#
# These are NOT vLLM's names. ``spatial_feature_mode`` is written into the
# checkpoint's ``config.json`` by the training stack, and the checkpoint's own
# remote code dispatches on it. The strings below must therefore match that data
# byte-for-byte — renaming them here without re-exporting the checkpoints makes
# every affected checkpoint fail to load with "Unsupported spatial_feature_mode".
#
# The v10 training run renamed the modes; checkpoints produced before it still
# carry the older ``llava_sp_*`` spelling, so we map old -> current.
#
# TO RE-BRAND: change the three values on the right-hand side here and nothing
# else. This dict is the only place they appear in the tree.
# ---------------------------------------------------------------------------
SPATIAL_MODE_ALIASES = {
    "llava_sp_both": "koni_token_hier",
    "llava_sp_pooling": "koni_pool",
    "llava_sp_cropping": "koni_crop",
}



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


def _make_batch_invariant_safe(m) -> int:
    """Route the vision attention through SDPA so ``VLLM_BATCH_INVARIANT`` can be used.

    ``enable_batch_invariant_mode()`` overrides ``aten::softmax``/``aten::_softmax`` on
    CUDA process-wide, and that implementation hits an illegal memory access on this
    vision tower's attention shapes::

        attn = attn.softmax(dim=-1)                     # modeling_intern_vit._naive_attn
        input_max = torch.amax(input, dim=dim, ...)     # softmax_batch_invariant
        torch.AcceleratorError: CUDA error: an illegal memory access was encountered

    The reference pipeline never hits it: it runs the vision tower with HF in the
    *main* process, so only the decoder (in EngineCore) sees the patch. Serving runs
    vision inside EngineCore, so it does.

    That matters for parity, not just for the crash — the reference harness turns the
    flag on whenever it batches (``run_eval.sh``: GEN_BATCH>1 -> VLLM_BATCH_INVARIANT=1),
    and its deterministic decode kernels are worth **+4.17 points on PRISMM pair_match**
    (2026-08-31 실측: 래퍼 78.65 with, 74.48 without).

    ``F.scaled_dot_product_attention`` is the same function without the explicit
    ``aten::softmax`` call, so the patch cannot reach it. Applied only when the flag is
    on — default runs are untouched. Returns how many modules were patched.
    """
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("1", "true", "True"):
        return 0
    import torch.nn.functional as _F

    def _sdpa_attn(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        if getattr(self, "qk_normalization", False):
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        o = _F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
        o = o.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(o))

    seen, n = set(), 0
    for mod in m.modules():
        cls = type(mod)
        if cls in seen or not hasattr(cls, "_naive_attn"):
            continue
        seen.add(cls)
        if getattr(cls, "_kth_bi_safe", False):
            continue
        cls._naive_attn = _sdpa_attn
        cls._kth_bi_safe = True
        n += 1

    # ``mean_batch_invariant`` returns float32 unconditionally
    # (``result = input.to(torch.float32)``), so the spatial module's pooled tensor
    # arrives at a bf16 ``Linear`` and torch raises
    #     RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
    # (``sfe_channel_fc(gap)`` in the global_ca channel attention).
    # Coerce Linear inputs to the weight dtype; a no-op when they already match.
    import torch.nn as _nn
    k = 0
    for mod in m.modules():
        if not isinstance(mod, _nn.Linear) or getattr(mod, "_kth_bi_cast", False):
            continue
        _orig = mod.forward

        def _fwd(x, _o=_orig, _m=mod):
            if x.dtype != _m.weight.dtype:
                x = x.to(_m.weight.dtype)
            return _o(x)

        mod.forward = _fwd
        mod._kth_bi_cast = True
        k += 1
    if n or k:
        logger.info("KTH: VLLM_BATCH_INVARIANT compat — attention->SDPA (%d class), "
                    "Linear dtype-coerce (%d module)", n, k)
    return n


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
    _make_batch_invariant_safe(m)
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
    alias = SPATIAL_MODE_ALIASES

    class _Shim:
        pass

    shim = _Shim()
    shim.config = config
    # --- 2026-08-31: 구세대 KTH 체크포인트 호환 (수정 2곳 중 1) -------------------
    # 이 shim 은 체크포인트 자체 코드를 끌어다 쓰는데, 그 코드에 세대가 둘 있다.
    #   신(예: localfix_e6t_...): mode=현행명(SPATIAL_MODE_ALIASES 의 값) · _local_token_count 있음
    #   구(예: KTH_FullFT_20260529_2053, KTH_720k_s2only_20260708_1215):
    #        mode=llava_sp_*  · _local_token_count 없음 · LOCAL_BRANCH 참조 0곳
    # 구 코드의 _infer_extra_tokens 는 원래 llava_sp_* 이름으로 분기한다. 별칭을
    # 씌워 주면 "Unsupported spatial_feature_mode: <현행명>" 으로 죽는다.
    # 따라서 별칭 변환은 신 코드일 때만 한다. (구 llava_sp_both -> extra=12, 268)
    _is_new = hasattr(cls, "_local_token_count")
    shim.spatial_feature_mode = alias.get(raw, raw) if _is_new else raw
    shim.sfe_scales = getattr(config, "sfe_scales", None)
    shim.spatial_pool_size = getattr(config, "spatial_pool_size", None)
    shim.spatial_crop_size = getattr(config, "spatial_crop_size", None)
    shim.spatial_crop_stride = getattr(config, "spatial_crop_stride", None)
    # --- 2026-08-31: 구세대 KTH 체크포인트 호환 (수정 2곳 중 2) -------------------
    # _local_token_count 는 신 코드에만 있다. 무조건 바인딩하면 구 체크포인트가
    #   AttributeError: type object 'InternVLChatModel' has no attribute
    #   '_local_token_count'
    # 로 로드 단계에서 죽는다(lmms-eval 이 예외를 삼켜 rc=0 으로 보인다).
    # 구 코드의 llava_sp_* 경로는 이 메서드를 부르지 않으므로 안 넘겨도 된다.
    # 가산 전용 수정 — 신 코드 체크포인트의 동작은 바뀌지 않는다.
    if _is_new:
        shim._local_token_count = cls._local_token_count.__get__(shim)
    vc = config.vision_config
    grid = int((vc.image_size // vc.patch_size) * float(getattr(config, "downsample_ratio", 0.5)))
    return int(cls._infer_extra_tokens(shim, grid))
