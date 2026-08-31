# `pair_match` −14: the multimodal-merge path only

A reproducible ~14-point accuracy gap between two ways of running **the same
checkpoint**, on a 4-way multiple-choice multi-image benchmark. Every input we know how
to compare has been measured identical. This document is a request for review: what is
left?

- **Branch:** `debug/pair-match-minus14` (off `feat/koni-v`, HEAD `25ae4b95`)
- **Build under test:** vLLM `0.1.dev7+g1be9ea959` (recent `main`, DeepSeek-V4/KDA-era code),
  installed into a separate conda env, with a custom model shim registered for
  `InternVLChatModel`.
- **Repro scripts:** [`debug_pair_match/`](debug_pair_match/) — two run scripts (need a GPU)
  and one offline check script (CPU only).

---

## 1. Problem statement

### The checkpoint

InternVL3.5-8B with a custom "KTH" vision module supplied by the checkpoint's own
remote code.

| field | value |
|---|---|
| `config.architectures` | `["InternVLChatModel"]` |
| `config.template` | `internvl2_5` |
| `max_dynamic_patch` | 12 |
| `use_thumbnail` | `True` |
| image tokens / tile | **268** = 256 (`(448/14)^2 * 0.5^2`) + **12 extra spatial tokens** added by the KTH module |
| `LOCAL_BRANCH` | `global_ca` (this is what sets extra = 12) |

### Path A — "wrapper" (the reference; produces the published numbers)

`lmms-eval` + a custom model class (`internvl3_5_KTH.py`). Vision runs with **HF, in the
main process**: the checkpoint's own `model.chat()` builds the prompt string, tiles the
images, runs the vision tower, and splices the image features into
`inputs_embeds`. Only the *decode* goes to vLLM — a **stock vLLM 0.21.0** engine loaded
with the **text-only extracted Qwen3** decoder, fed `prompt_embeds`
(`enable_prompt_embeds=True`, `enforce_eager=True`, `VLLM_BATCH_INVARIANT=1`).

> **PRISMM `pair_match` = 78.65.** Fully deterministic: bit-identical outputs across reruns.

### Path B — "fork" (the serving path; the one that is wrong)

`lmms-eval --model vllm` against this custom vLLM build. Vision runs **inside
EngineCore**: `KTHInternvlChatModel` (`vllm/model_executor/models/kth_internvl.py` +
`_kth/`) subclasses vLLM's `InternVLChatModel`, swaps in the checkpoint's vision tower,
and reports 268 image tokens per tile so the placeholder expansion matches.

> **PRISMM `pair_match` = 63–64.** Also reproducible: 192 questions, two identical
> reruns landing on the same number, ~97% answer agreement between reruns. (After one
> chat-template change both reruns instead landed on 56.77 — that change was reverted;
> it is a separate, understood effect, see §2 "prompt text".)

### Why we believe the fork is the outlier

A third path — **HF only, no vLLM at all** — scores **77.60**, agreeing with the wrapper
(78.65) within noise. So two independent paths agree and the fork does not.

### The gap tracks image count

Answer-agreement (does the same question get the same answer string?) between wrapper
and fork, per benchmark, after rescoring:

| benchmark | answer agreement | images / question |
|---|---|---|
| `mmlongbench_doc` | **20%** | up to 8 |
| `mmmu_val` | 70% | multi |
| **`prismm_bench_pair_match`** | **74%** | **4–5** |
| `dude` | 86% | multi-page |
| `chartqapro` | 83% | 1 |
| `chartqa`, `tablevqa`, `textvqa`, `dtcbench`, `ocrvqa`, `docvqa` | **91–96%** | **1** |

**Single-image benchmarks agree within ±0.6 points after rescoring** (OCRVQA is
identical to two decimals). The disagreement is ordered by the number of images per
question. Whatever is wrong lives in the path that merges *several* images' features
into one sequence.

(Caveat on `mmlongbench_doc`: the 20% there is partly the *wrapper's* fault, not the
fork's — the wrapper answers "Not answerable" 32% of the time on that benchmark vs
12–15% for HF and fork. Do not use `mmlongbench_doc` as evidence in either direction.
`pair_match` is the clean control: 192 questions, 4-way multiple choice, both paths
answer everything.)

---

## 2. Eliminated with measurements — please do not re-suggest these

Each row was measured, not reasoned about.

| hypothesis | how it was killed |
|---|---|
| **Prompt text differs** (conv template vs jinja render) | Rendered prompt strings compared per question: **byte-identical 192/192**. |
| **Prompt token IDs differ** | Full token-ID sequences identical, except a single trailing-newline token for documents that *end* with `<image>`. Real `pair_match` docs end with text, so 0 affected. |
| **Preprocessing / pixel tensors** | Both preprocessing paths run on the same images: `torch.allclose` identical, shapes identical. |
| **Tiling tie-break order** | InternVL's `find_closest_aspect_ratio` picks the *last* tied candidate visited, so the result depends on container iteration order. The harness wraps the sorted list back into a `set`; upstream vLLM iterates the sorted list. They disagree on ~9–14% of real images. **The fork shim deliberately reproduces the harness's `set` order** (`_kth/kth_tiling.py`). And the measurement points the other way anyway: questions where both paths produce the **same** tile counts show a **larger** drop (−15.8) than questions where the tile counts differ (−12.5). |
| **Per-request tile budget `64 // n`** | The reference caps tiles per request: `dyn = max(1, min(max_num, total_max_num // n))`. Implemented in the fork (`KTHInternVLMultiModalProcessor.apply`). For n=4–5 both sides get 12 anyway. Verified end to end: `num_patches` per image = `[1, 3, 4, 3]` on a sample question, equal to the wrapper's tile counts; `pixel_values_flat` shapes match per image. |
| **Placeholder spans / image-token count** | `<IMG_CONTEXT>` runs measured at the token level with deliberately distinct-size images: `268 × [1, 3, 4, 3]` in image order. Wrapper's `tiles × 268` == fork's `get_num_image_tokens` at **every** image count tested (n = 1, 4, 6, 8). |
| **Feature split back to images** | `internvl.py::_process_vision_input` splits image-major by `num_patches × feature_size`, and `feature_size` is taken from the *actual* vision output (268), not from config. Code verified — excerpt in §4. |
| **Vision module / weights** | Same class, loaded from the checkpoint's remote code, in both paths. All 383 params present in the checkpoint; **zero buffers** (so no meta-device leftovers). Features compared on CPU between the two construction styles: **bit-identical, max diff 0.0**. Batch-independent to 1.2e-06. |
| **flash_attn** | Absent in both envs → both fall back to the same naive attention. |
| **Sampling** | `temperature=0` greedy on both. |
| **Async scheduling** | fork `pair_match`: async OFF 63.54 == async ON 63.54. |
| **Engine knobs (fork)** | All within noise (SE ≈ ±3.4pp at n=192): `enable_prefix_caching=False` → 64.06; `max_num_batched_tokens=40960` / no chunked prefill → 64.58; `enforce_eager` → 66.15; `max_num_seqs=1` → 64.06; **all four combined → 63.54**. |
| **`VLLM_BATCH_INVARIANT`** | The wrapper needs it for determinism and it is worth points there: **78.65 with, 74.48 without**. The fork could not enable it until today — two crashes, both now fixed (§2.1). With the flag finally on, the fork is **64.06**. Not the cause. |
| **Image ordering / permutation** | Tried remapping the model's chosen letter under every rotation and reversal of the image order. Identity is best (68% under that scoring); every shift is much worse. Images are not being reordered. |
| **Scoring / task code** | Targets and inputs identical 192/192 between paths; per-sample scores consistent with the recorded responses; `tasks/` trees diff to 0 files. |

### 2.1 The two `VLLM_BATCH_INVARIANT` crashes (fixed, for the record)

Both are artifacts of running vision *inside* EngineCore, which the reference never does.

1. `enable_batch_invariant_mode()` overrides `aten::softmax` / `aten::_softmax` on CUDA
   process-wide. The vision tower's `_naive_attn` does `attn = attn.softmax(dim=-1)`,
   and the batch-invariant kernel's `torch.amax` hits
   `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` on
   these shapes. **Fix:** route vision attention through
   `F.scaled_dot_product_attention` when the flag is on (SDPA never emits the patched
   `aten::softmax`). Applied only when the flag is set.
2. `mean_batch_invariant` returns float32 unconditionally, so the KTH spatial module's
   pooled tensor arrives at a **bf16** `Linear` →
   `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`.
   **Fix:** coerce `Linear` inputs to the weight dtype (no-op when they already match).

Both fixes are in `_kth/kth_vision.py::_make_batch_invariant_safe` (excerpt in §4).

### 2.2 One real bug that was found and fixed — and it only bought 1.04 points

`lmms-eval`'s generic `vllm.py` used to append the text whole and then all images after
it. For a `pair_match` prompt shaped `A) <image> B) <image> C) <image> D) <image>` this
left **8 placeholders for 4 images** in the rendered prompt (4 real image parts, plus
the 4 literal `<image>` strings still in the text). vLLM bound the images to the first
four, so the A/B/C/D correspondence was accidentally right and the rest was literal
noise. Fixed by splicing images at their placeholder positions when the counts match
(the same condition the wrapper's `_build` uses).

> Effect: `pair_match` 63.54 → **64.58**. A genuine bug, and it recovered **1.04 of the
> 14 points.** The rest is unexplained.

---

## 3. The open question

Input token IDs are identical. Pixel tensors are identical. Per-image routing metadata
(`num_patches`, placeholder spans, feature sizes) is identical. Vision weights and
vision outputs are bit-identical. Sampling is greedy on both. Engine knobs are noise.

**What is left that can move a 4-way multiple choice from 78.65 to 63–64, reproducibly,
only on the path where the multimodal merge happens inside the engine?**

Specific things we would like a reviewer to check or rule out:

1. **Known regressions in recent vLLM `main`** (this build is `0.1.dev7`, i.e. after the
   DeepSeek-V4 / KDA merges) touching:
   - the multimodal embedding merge (`merge_multimodal_embeddings` / `get_input_embeddings`
     placeholder scatter),
   - `position_ids` / M-RoPE computation for multimodal placeholder tokens,
   - encoder batching or the encoder cache for InternVL-style models.
2. **`prompt_embeds` vs `token IDs + mm inputs` in V1 EngineCore.** The wrapper feeds
   `prompt_embeds` (which is *not* the mm path at all — it is a text-only engine
   receiving pre-merged embeddings); the fork feeds token IDs plus mm kwargs. Does V1
   treat these two differently with respect to `position_ids`, attention metadata, or KV
   layout in a way that could change results — beyond the embeddings themselves?
3. **Silent truncation or double-application on the merge path.** Is there anywhere that
   merged mm embeddings could
   - bypass or *duplicate* an embedding scale / input layernorm,
   - be silently truncated by the encoder budget (`encoder_budget`,
     `max_num_encoder_input_tokens`) when several images land in one request,
   - or be scattered into the wrong positions when a single request carries 4–5
     independent images with *different* patch counts?

   Note the shape of the evidence: the error grows with image count and is absent at
   n=1. Anything that is per-request-constant would show at n=1 too; anything that
   accumulates per image would not.

---

## 4. Code map and excerpts

Excerpts are quoted verbatim from the **installed** copies — the exact code that
produced the numbers above. Original absolute paths and line numbers are in the comment
at the top of each block.

> Note on this branch: the checkout carries `_kth/koni_tiling.py`, while the installed
> build under measurement has the same code as `_kth/kth_tiling.py`. The file was
> renamed; the excerpts below are from the installed (measured) copies.

### 4.1 The shim: tile budget and image-token count

`vllm/model_executor/models/kth_internvl.py`

```python
# /…/envs/koni_fork/lib/python3.11/site-packages/vllm/model_executor/models/kth_internvl.py:92-127
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


# Tile budget shared across the images of one request. The reference harness applies it
# per request in the KTH wrapper:
#
#     dyn = max(1, min(max_num, total_max_num // n_images))
#
# Upstream vLLM gives *every* image the full `max_dynamic_patch`. For 1-5 images the two
# agree (64 // 5 = 12), so single-image benchmarks are untouched.
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
```

### 4.2 Tiling tie-break order (reproduces the harness, deliberately)

`vllm/model_executor/models/_kth/kth_tiling.py`

```python
# /…/site-packages/vllm/model_executor/models/_kth/kth_tiling.py:41-73
def kth_target_ratios(min_num: int, max_num: int) -> set[tuple[int, int]]:
    """Candidate ratios in the reference harness's iteration order.

    Deliberately returns the ``set`` itself: the tie-break depends on iteration
    order, and this is the order every published number was measured with.
    Tuples of small ints hash deterministically, so this is stable across runs.
    """
    return {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }


def kth_image_to_pixel_values(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    tiles = dynamic_preprocess_internvl(
        image,
        target_ratios=kth_target_ratios(min_num, max_num),
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )
    transform = build_transform(input_size=input_size)
    return torch.stack([transform(t) for t in tiles])
```

### 4.3 Vision tower: batch-invariant compatibility shims

`vllm/model_executor/models/_kth/kth_vision.py`

```python
# /…/site-packages/vllm/model_executor/models/_kth/kth_vision.py:94-173 (abridged)
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
    return n
```

### 4.4 Upstream: where the features are split back per image

`vllm/model_executor/models/internvl.py` — **this is the code the reviewer should look
at hardest.** `feature_size` is taken from the actual vision output (so 268, not 256),
and the split is image-major.

```python
# /…/site-packages/vllm/model_executor/models/internvl.py:755-781
    def _process_vision_input(
        self,
        image_input: InternVLImageInputs | InternVLVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        if (
            image_input["type"] == "image_embeds"
            or image_input["type"] == "video_embeds"
        ):
            return image_input["data"]

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)
```

Note the `len(num_patches) == 1` early return: **the single-image case takes a different
branch** from the multi-image case, and single-image benchmarks are exactly the ones
that agree.

For reference, the base per-tile token count that the shim adds +12 to:

```python
# /…/site-packages/vllm/model_executor/models/internvl.py:366-372
    def get_hf_processor(self, **kwargs: object) -> InternVLProcessor:
        config = self.get_hf_config()
        vision_config = config.vision_config

        image_processor = self.get_image_processor(**kwargs)
        image_size = image_processor.image_size
        patch_size = vision_config.patch_size
        downsample_ratio = config.downsample_ratio
        image_seq_length = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
```

### 4.5 Reference wrapper — prompt build (Path A)

External repo (`lmms-eval-for-KONI`), included here because it is the definition of
"correct".

```python
# lmms_eval/models/simple/internvl3_5_KTH.py:547-575
        def _build(contexts, visuals):
            """Replicate model.chat prompt building. Returns (query, pixel_values)."""
            n = len(visuals)
            processed = []
            if n == 0:
                question = contexts
            else:
                dyn = max(1, min(self.max_num, self.total_max_num // n))
                processed = [
                    _iv3.load_image(v, max_num=dyn).to(torch.bfloat16).to(self._device) for v in visuals
                ]
                # Match parent: keep author-interleaved tags, else prepend one per image.
                question = contexts if contexts.count("<image>") == n else " ".join(["<image>"] * n) + "\n" + contexts
            template = get_conv_template(model.template)
            template.system_message = model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            for p in processed:  # one <image> -> one image's token block, in order
                block = IMG_START + IMG_CTX * model.num_image_token * p.size(0) + IMG_END
                query = query.replace("<image>", block, 1)
            pv = torch.cat(processed, dim=0) if processed else None
            return query, pv
```

`model.num_image_token` is **268** here (the checkpoint's own value, already including
the +12), and `p.size(0)` is *that image's* tile count — so each image gets its own
block length. This is the invariant the fork has to match, and by measurement it does.

### 4.6 Reference wrapper — decode through vLLM `prompt_embeds` (Path A)

```python
# lmms_eval/models/simple/internvl3_5_KTH.py:449-490
        def _vllm_generate(
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            generation_config=None,
            **kwargs,
        ):
            """Drop-in for ``language_model.generate`` using vLLM via prompt_embeds.

            ``model.chat(...)`` always supplies ``inputs_embeds`` (image tokens already
            fused), so we only support that path. Returns a ``(batch, seq)`` LongTensor
            of generated token ids, which ``chat`` decodes with the tokenizer.
            """
            assert inputs_embeds is not None, "internvl3_5_KTH supports the inputs_embeds path only"
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=int(max_new_tokens),
                stop_token_ids=_vllm_stop_ids,
            )

            # Build one prompt-embeds entry per batch row. batch_chat left-pads, so
            # drop padding positions via attention_mask before handing the real
            # prompt embeddings to vLLM.
            prompts = []
            for i in range(inputs_embeds.shape[0]):
                emb = inputs_embeds[i]
                if attention_mask is not None:
                    emb = emb[attention_mask[i].bool()]
                prompts.append({"prompt_embeds": emb.to(torch.bfloat16).contiguous()})

            # One generate() call -> vLLM continuous-batches the whole list and
            # returns outputs in input order.
            outs = vllm_engine.generate(prompts, sampling)
            sequences = [list(o.outputs[0].token_ids) for o in outs]
            ...

        # Route the HF chat path's LLM decode through vLLM.
        self._model.language_model.generate = _vllm_generate
```

**The whole difference between the two paths is here**: Path A hands vLLM a finished
embedding matrix; Path B hands vLLM token IDs plus `pixel_values_flat` and lets the
engine build that matrix. Everything measured on either side of that boundary is
identical.

### 4.7 Fork harness — message assembly (Path B)

```python
# lmms_eval/models/simple/vllm.py:~500-540 (comments translated/abridged)
                    messages = [{"role": "user", "content": []}]
                    _flat = self.flatten(imgs)

                    def _img_part(_i):
                        return {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_i}"},
                        }

                    # Respect <image> placeholder POSITIONS in the text.
                    #
                    # This used to append the whole text and then all images after it.
                    # Identical for single-image tasks, but it destroys the mapping on
                    # author-interleaved multi-image tasks: pair_match is
                    # 'A) <image> B) <image> C) <image> D) <image>', so the literal
                    # <image> strings survived AND four images were appended at the
                    # end -> the model could not tell which one was A.
                    #
                    # Splice only when the counts match exactly (same condition as the
                    # wrapper's internvl3_5_KTH._build).
                    if _flat and contexts.count("<image>") == len(_flat):
                        _parts = contexts.split("<image>")
                        for _k, _seg in enumerate(_parts):
                            # The chat template emits '<image>\n' per image part. If the
                            # next source segment starts with a newline, drop that one --
                            # otherwise every image gains a blank line.
                            #
                            # NOTE: we tried fixing this by removing the '\n' from the
                            # template instead and reverted it: the rendered strings were
                            # 192/192 IDENTICAL yet the score dropped reproducibly
                            # 63.02 -> 56.77 (3 runs). vLLM's llm.chat() image-part
                            # handling depends on this newline. Do not touch the template.
                            if _k > 0 and _seg.startswith("\n"):
                                _seg = _seg[1:]
                            if _seg:
                                messages[0]["content"].append({"type": "text", "text": _seg})
                            if _k < len(_flat):
                                messages[0]["content"].append(_img_part(_flat[_k]))
```

That last NOTE is itself a curiosity worth a reviewer's attention: **a change that left
the rendered prompt byte-identical moved the score by 6 points, reproducibly.** That is
consistent with the merge path being sensitive to something other than the final string
— which is the same suspicion this whole document is about.

---

## 5. Reproduction

See [`debug_pair_match/`](debug_pair_match/):

| script | needs GPU | what it does |
|---|---|---|
| `repro_fork.sh` | yes | Path B: `lmms-eval --model vllm` against this build. Expect `pair_match` ≈ 63–64. |
| `repro_wrapper.sh` | yes | Path A: `lmms-eval --model internvl3_5_KTH`, HF vision + `prompt_embeds` decode. Expect ≈ 78.65. |
| `offline_checks.py` | **no** | The three input-equality checks, on CPU: (a) prompt-string diff, (b) pixel-tensor `allclose`, (c) placeholder spans / `num_patches` via `processor.apply`. |

`offline_checks.py` is the useful one for a reviewer without the checkpoint's GPU
footprint: it re-derives, from scratch, the "the inputs are identical" claims in §2.

---

Live findings log: `RL/eval/CLAUDE.md` §5 in the internal repo.
