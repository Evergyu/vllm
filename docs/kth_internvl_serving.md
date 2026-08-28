# Serving encoder-modified InternVL (KTH) checkpoints

## What this is

These checkpoints are InternVL with extra spatial vision modules (KTH) added to the
encoder, then SFT'd. `KTHInternvlChatModel` runs the checkpoint's **own** vision tower
and lets vLLM run the Qwen3 decoder on its normal engine — continuous batching, paged
attention, prefix caching, OpenAI-compatible API.

## Serve

```bash
export LOCAL_BRANCH=global_ca          # must match training
vllm serve /path/to/checkpoint \
    --trust-remote-code \
    --chat-template examples/template/koni_v.jinja
```

`--chat-template` matters. The checkpoint's own `chat_template.jinja` injects **no**
system message, but `model.chat()` — the reference path — always supplies the KONI-V
identity prompt from `conversation.py`, and the model was SFT'd with it. Without this
flag a request that carries no system message is served in a state the model never saw
in training. The shipped template injects it and steps aside when the caller sends
their own.

`--trust-remote-code` is required: the vision tower class is loaded from the
checkpoint's own `modeling_internvl_chat.py` at runtime.

No `config.json` edit is needed. Both `architectures: ["InternVLChatModel"]` (older
checkpoints) and `["KoniVChatModel"]` (the published KONI-V repos) route here. A plain
InternVL checkpoint — one whose config declares no `spatial_feature_mode` — falls
through to stock behaviour unchanged.

The vision tower comes from the checkpoint's own remote code either way, so the class
names inside it do not matter: `auto_map` is what is followed.

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

### Tiling

Tiling now reproduces the reference harness bit for bit — verified pixel-identical on
12 aspect ratios including the cases where upstream disagrees.

InternVL's `find_closest_aspect_ratio` returns the **last** tied candidate that passes
its area test, so the tile count depends on the iteration order of the candidate
container. Upstream vLLM sorts by block count; the harness passes a `set`. On a
300-image sample they differ on 14.0% of ChartQA and 9.3% of DocVQA images, in both
directions — a 1000x1000 image is 1 tile under the harness and 9 upstream.

Every published KONI-V number was measured with the harness order, so
`_kth/koni_tiling.py` reproduces that order (by building the container the same way,
not by freezing a hash order). If the reference pipeline ever changes, this follows.

## Note on `async_scheduling`

vLLM's `enable_prompt_embeds` + `async_scheduling` combination had a decode bug where
scattered decode slots kept the previous prefill's image embeddings. **This model does
not use `prompt_embeds`**, so it is unaffected. The fix that matters for it is already
in `_prepare_input_ids`, which refreshes the whole `is_token_ids` mask on the async path.
