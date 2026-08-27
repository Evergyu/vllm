# Serving encoder-modified InternVL (KTH) checkpoints

## What this is

These checkpoints are InternVL with extra spatial vision modules (KTH) added to the
encoder, then SFT'd. `KTHInternvlChatModel` runs the checkpoint's **own** vision tower
and lets vLLM run the Qwen3 decoder on its normal engine — continuous batching, paged
attention, prefix caching, OpenAI-compatible API.

## Serve

```bash
vllm serve /path/to/checkpoint --trust-remote-code
```

`--trust-remote-code` is required: the vision tower class is loaded from the
checkpoint's own `modeling_internvl_chat.py` at runtime.

No `config.json` edit is needed. These checkpoints declare
`architectures: ["InternVLChatModel"]`, and that name routes here; a plain InternVL
checkpoint (no `spatial_feature_mode`) falls through to stock behaviour unchanged.

`LOCAL_BRANCH` selects the cropping branch and **must match training**. It is read by
the checkpoint's own code, and it changes the per-tile token count, so a mismatch
desyncs the image placeholders:

| `LOCAL_BRANCH` | extra tokens/tile (6 `sfe_scales`, grid 16) |
|---|---|
| `center_crop` (default) | 12 |
| `global_ca` | 12 |
| `none` | 6 |
| `hybrid` | 28 |
| `dense_4x4` | 22 |
| `dense_8x8` | 70 |

## Why the modeling code is not vendored

It used to be, and it went stale. The checkpoints renamed `spatial_feature_mode` from
`llava_sp_*` to `koni_*`, replaced the content-aware gate
(`σ(W·pool+b)` → `σ(τ·cos(w, LN(pool))+b)`), and added the `LOCAL_BRANCH` branch.
Against a current checkpoint the vendored copy:

- computed 0 extra tokens per tile instead of 12,
- never built the spatial modules, so their checkpoint weights were dropped,
- and reported every vision parameter as loaded.

The two errors cancelled into a consistent-looking 256 tokens/tile. The model loaded,
ran, and answered with the entire KTH contribution missing — with no error anywhere.

So the class is pulled from the checkpoint at runtime instead. Drift is not possible.
Weight loading is strict: a shape mismatch, an unmatched checkpoint key, or a parameter
left unmaterialized raises rather than silently zero-filling.

## Correctness reference

The reference is HuggingFace `model.chat()`. On the same checkpoint, swapping only the
decoder (HF vision + prompt assembly, vLLM decode) reproduces it:

| bench | HF `model.chat()` | vLLM decode |
|---|---|---|
| ChartQA (n=2,500) | 82.80 | 82.88 |
| OCRVQA (n=3,072) | 53.87 | 53.81 |

Item-level disagreement is symmetric (HF-only-correct 5 vs vLLM-only-correct 7 on
ChartQA), i.e. numerical jitter on borderline items, not a systematic loss.

**This model path additionally reimplements tiling** (vLLM's `InternVLProcessor` rather
than the checkpoint's `dynamic_preprocess`). That axis is not covered by the numbers
above. Verify tile count and pixel values against the HF reference before trusting a
new deployment.

## Note on `async_scheduling`

vLLM's `enable_prompt_embeds` + `async_scheduling` combination had a decode bug where
scattered decode slots kept the previous prefill's image embeddings. **This model does
not use `prompt_embeds`**, so it is unaffected. The fix that matters for it is already
in `_prepare_input_ids`, which refreshes the whole `is_token_ids` mask on the async path.
