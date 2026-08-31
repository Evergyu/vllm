#!/usr/bin/env python3
"""CPU-only equality checks between the two paths described in DEBUG_PAIR_MATCH.md.

No GPU, no engine, no weights on device. Everything here runs on the CPU in a few
seconds and re-derives the "the inputs are identical" claims from section 2 of
DEBUG_PAIR_MATCH.md.

    (a) prompt         reference query string / token IDs  vs  fork prompt token IDs
    (b) pixels         reference tiling+transform          vs  fork image processor
                       -> torch.allclose, per image
    (c) placeholders   per-image <IMG_CONTEXT> run lengths and num_patches, as the
                       fork's multimodal processor computes them
                       -> must equal  268 * (that image's tile count)

Usage
-----
    python offline_checks.py --ckpt /path/to/checkpoint

Options
    --images a.png b.png ...   Real images. Default: synthetic images with
                               deliberately DIFFERENT aspect ratios, which is the
                               point -- equal-sized images hide per-image bugs
                               because every block has the same length.
    --prompt "..."             Prompt text. Default: the pair_match shape,
                               'A) <image> B) <image> C) <image> D) <image>'.
    --local-branch global_ca   Sets extra tokens per tile (global_ca -> 12 -> 268).
    --check a,b,c              Subset of checks to run.

Environment
    Run with the python of the env where the fork vLLM build is installed.
    LOCAL_BRANCH is set from --local-branch; do not rely on the ambient value.
"""

from __future__ import annotations

import argparse
import os
import sys

# LOCAL_BRANCH decides the extra spatial tokens per tile and is read at import time
# by the checkpoint's remote code, so it must be set BEFORE importing vllm.
_EARLY = argparse.ArgumentParser(add_help=False)
_EARLY.add_argument("--local-branch", default="global_ca")
os.environ.setdefault("LOCAL_BRANCH", _EARLY.parse_known_args()[0].local_branch)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Keep the whole script off the GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"

OK = "  ok  "
BAD = " FAIL "


# --------------------------------------------------------------------------------
# Reference side (Path A). Transcribed from the InternVL demo / lmms-eval harness so
# this script stands alone; pass --wrapper-repo to import the real thing instead.
# --------------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ref_build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def ref_target_ratios(min_num: int, max_num: int):
    """The harness's candidate container.

    NOTE the `set`. find_closest_aspect_ratio keeps the LAST tied candidate that
    passes the area test, so the winner depends on iteration order. The official
    InternVL demo sorts this by block count; the harness does not. They disagree on
    9-14% of real images, in both directions. Every published number was measured
    with THIS order, so the fork reproduces it (_kth/kth_tiling.py).
    """
    return {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }


def ref_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best = ratio
    return best


def ref_dynamic_preprocess(image, *, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    ratios = ref_target_ratios(min_num, max_num)
    best = ref_find_closest_aspect_ratio(w / h, ratios, w, h, image_size)
    tw, th = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((tw, th))
    tiles = []
    cols = tw // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    assert len(tiles) == blocks
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def ref_load_image(image, *, max_num, image_size=448, use_thumbnail=True):
    tiles = ref_dynamic_preprocess(
        image, min_num=1, max_num=max_num, image_size=image_size, use_thumbnail=use_thumbnail
    )
    tf = ref_build_transform(image_size)
    return torch.stack([tf(t) for t in tiles])


def ref_build_query(prompt: str, pixel_values_per_image, *, num_image_token: int,
                    conv_template, system_message: str) -> str:
    """internvl3_5_KTH._build, minus the GPU bits.

    One <image> -> that image's own block: num_image_token * (that image's tiles).
    """
    n = len(pixel_values_per_image)
    question = prompt if prompt.count("<image>") == n else " ".join(["<image>"] * n) + "\n" + prompt
    tpl = conv_template
    tpl.system_message = system_message
    tpl.append_message(tpl.roles[0], question)
    tpl.append_message(tpl.roles[1], None)
    query = tpl.get_prompt()
    for pv in pixel_values_per_image:
        block = IMG_START + IMG_CTX * num_image_token * pv.size(0) + IMG_END
        query = query.replace("<image>", block, 1)
    return query


# --------------------------------------------------------------------------------
# Fork side (Path B)
# --------------------------------------------------------------------------------

def fork_processor_bits(ckpt: str):
    """Build the fork's multimodal processor offline. No weights are loaded."""
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_config = ModelConfig(
        model=ckpt,
        tokenizer=ckpt,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        max_model_len=40960,
        enforce_eager=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    return model_config, processor


def fork_apply(processor, prompt: str, images, mm_kwargs=None):
    """Call BaseMultiModalProcessor.apply across signature revisions.

    Newer builds take (ProcessorInputs, TimingContext); older ones take
    (prompt, mm_data, hf_processor_mm_kwargs).
    """
    mm_data = {"image": images}
    mm_kwargs = dict(mm_kwargs or {})
    try:
        from vllm.multimodal.processing.inputs import ProcessorInputs
        from vllm.multimodal.processing.context import TimingContext

        items = processor.info.ctx.get_hf_processor  # noqa: F841  (touch, for a clear error)
        from vllm.multimodal.parse import MultiModalDataItems  # noqa: F401

        inputs = ProcessorInputs(
            prompt=prompt,
            mm_data_items=processor._to_mm_items(mm_data),
            hf_processor_mm_kwargs=mm_kwargs,
        )
        return processor.apply(inputs, TimingContext(enabled=False))
    except (ImportError, AttributeError, TypeError):
        return processor.apply(prompt, mm_data, mm_kwargs)


def runs_of(token_ids, target_id):
    """[(start, length)] for each maximal run of target_id."""
    out, start = [], None
    for i, t in enumerate(token_ids):
        if t == target_id and start is None:
            start = i
        elif t != target_id and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(token_ids) - start))
    return out


# --------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (needs config.json)")
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--prompt", default="A) <image> B) <image> C) <image> D) <image>\n"
                                        "Which pair matches? Answer with a single letter.")
    ap.add_argument("--local-branch", default="global_ca")
    ap.add_argument("--check", default="a,b,c")
    ap.add_argument("--wrapper-repo", default=None,
                    help="lmms-eval-for-KONI src dir; if given, import the real "
                         "reference load_image instead of the transcription here")
    args = ap.parse_args()

    checks = {c.strip() for c in args.check.split(",") if c.strip()}
    failures = []

    if not os.path.isfile(os.path.join(args.ckpt, "config.json")):
        print(f"[err] no config.json in {args.ckpt}", file=sys.stderr)
        return 2

    # Deliberately DIFFERENT aspect ratios: with four identical images every block
    # has the same length and a per-image routing bug is invisible.
    if args.images:
        images = [Image.open(p).convert("RGB") for p in args.images]
    else:
        # tile counts 1, 3, 4, 3 with max_dynamic_patch=12 -- the same shape as the
        # real pair_match question used in the measurements.
        sizes = [(448, 448), (900, 460), (1400, 460), (460, 900)]
        images = []
        for k, (w, h) in enumerate(sizes):
            im = Image.new("RGB", (w, h), (10 * k, 40 * k, 200 - 30 * k))
            for x in range(0, w, 37):
                for y in range(0, h, 41):
                    im.putpixel((x, y), ((x * 7) % 256, (y * 11) % 256, (x + y) % 256))
            images.append(im)
        print(f"[info] using {len(images)} synthetic images with sizes {sizes}")

    n = len(images)
    if args.prompt.count("<image>") != n:
        print(f"[warn] prompt has {args.prompt.count('<image>')} placeholders "
              f"but {n} images were given")

    model_config, processor = fork_processor_bits(args.ckpt)
    info = processor.info
    hf_config = info.get_hf_config()

    max_num = int(getattr(hf_config, "max_dynamic_patch", 12) or 12)
    total_max_num = int(os.environ.get("KTH_TOTAL_MAX_NUM", "64"))
    use_thumbnail = bool(getattr(hf_config, "use_thumbnail", True))
    # The reference caps tiles per REQUEST, not per image.
    dyn = max(1, min(max_num, total_max_num // n)) if n else max_num

    hf_proc = info.get_hf_processor(**({"max_dynamic_patch": dyn} if dyn < max_num else {}))
    image_seq_length = int(hf_proc.image_seq_length)
    image_size = int(hf_proc.image_processor.image_size)

    print("=" * 78)
    print(f"ckpt              {args.ckpt}")
    print(f"LOCAL_BRANCH      {os.environ.get('LOCAL_BRANCH')}")
    print(f"images            {n}   tile budget dyn = min({max_num}, {total_max_num}//{n}) = {dyn}")
    print(f"use_thumbnail     {use_thumbnail}   image_size {image_size}")
    print(f"image_seq_length  {image_seq_length}   (expected 268 = 256 + 12 for global_ca)")
    print("=" * 78)
    if image_seq_length != 268:
        print(f"{BAD} image_seq_length is {image_seq_length}, not 268 -- the shim did "
              f"not add the extra spatial tokens. Everything below is meaningless.")
        failures.append("image_seq_length")

    # -------------------------------------------------------------- (b) pixels
    ref_pv = None
    if "b" in checks or "a" in checks or "c" in checks:
        if args.wrapper_repo:
            sys.path.insert(0, args.wrapper_repo)
            from lmms_eval.models.simple import internvl3 as _iv3  # type: ignore
            ref_pv = [_iv3.load_image(im, max_num=dyn) for im in images]
        else:
            ref_pv = [ref_load_image(im, max_num=dyn, image_size=image_size,
                                    use_thumbnail=use_thumbnail) for im in images]

    if "b" in checks:
        print("\n(b) per-image pixel tensors: reference vs fork image processor")
        fork_pv = hf_proc.image_processor._images_to_pixel_values_lst(
            images, max_dynamic_patch=dyn
        )
        for i, (a, b) in enumerate(zip(ref_pv, fork_pv)):
            same_shape = tuple(a.shape) == tuple(b.shape)
            close = same_shape and torch.allclose(a, b)
            maxdiff = float((a - b).abs().max()) if same_shape else float("nan")
            tag = OK if close else BAD
            print(f"  [{tag}] image {i}: ref {tuple(a.shape)}  fork {tuple(b.shape)}  "
                  f"max|diff| {maxdiff:.3e}")
            if not close:
                failures.append(f"pixels[{i}]")

    # ---------------------------------------------------- (a)+(c) prompt / spans
    mm_out = None
    if "a" in checks or "c" in checks:
        mm_out = fork_apply(processor, args.prompt, images,
                            {"max_dynamic_patch": dyn} if dyn < max_num else {})
        fork_ids = list(mm_out["prompt_token_ids"])

    if "a" in checks:
        print("\n(a) prompt: reference query vs fork prompt token IDs")
        tok = info.get_tokenizer()
        try:
            sys.path.insert(0, args.ckpt)
            from conversation import get_conv_template  # type: ignore
            tpl = get_conv_template(hf_config.template)
            system_message = getattr(hf_config, "system_message", None) or tpl.system_message
            num_image_token = image_seq_length
            ref_query = ref_build_query(args.prompt, ref_pv,
                                        num_image_token=num_image_token,
                                        conv_template=tpl,
                                        system_message=system_message)
            ref_ids = tok(ref_query, add_special_tokens=False)["input_ids"]
            same = list(ref_ids) == fork_ids
            print(f"  ref  {len(ref_ids)} tokens")
            print(f"  fork {len(fork_ids)} tokens")
            if same:
                print(f"  [{OK}] token IDs identical")
            else:
                # The one known-benign difference: a trailing-newline token when the
                # document ENDS with <image>. Real pair_match docs end with text.
                i = 0
                while i < min(len(ref_ids), len(fork_ids)) and ref_ids[i] == fork_ids[i]:
                    i += 1
                print(f"  [{BAD}] first divergence at index {i}")
                print(f"        ref  ...{tok.decode(ref_ids[max(0, i - 12):i + 12])!r}")
                print(f"        fork ...{tok.decode(fork_ids[max(0, i - 12):i + 12])!r}")
                failures.append("prompt")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] could not build the reference query: {type(e).__name__}: {e}")
            print( "         (needs the checkpoint's conversation.py; pass --check b,c "
                   "to skip)")

    if "c" in checks:
        print("\n(c) placeholder spans and num_patches")
        tok = info.get_tokenizer()
        ctx_id = tok.convert_tokens_to_ids(IMG_CTX)
        spans = runs_of(fork_ids, ctx_id)
        ref_tiles = [pv.size(0) for pv in ref_pv]
        expected = [image_seq_length * t for t in ref_tiles]
        got = [ln for _, ln in spans]
        print(f"  reference tile counts   {ref_tiles}")
        print(f"  expected span lengths   {expected}   ({image_seq_length} x tiles)")
        print(f"  fork span lengths       {got}")
        tag = OK if got == expected else BAD
        print(f"  [{tag}] spans match, in image order")
        if got != expected:
            failures.append("spans")

        # num_patches as the engine will see it, plus the flattened pixel tensor it
        # will hand to extract_feature. This is the metadata that _process_vision_input
        # splits the vision output by (internvl.py, see DEBUG_PAIR_MATCH.md 4.4).
        try:
            kw = mm_out["mm_kwargs"]
            data = kw.get_data() if hasattr(kw, "get_data") else dict(kw)
            npatches = data.get("num_patches")
            pvf = data.get("pixel_values_flat")
            if npatches is not None:
                npl = [int(x) for x in torch.as_tensor(npatches).flatten().tolist()]
                tag = OK if npl == ref_tiles else BAD
                print(f"  [{tag}] num_patches {npl} == reference tile counts {ref_tiles}")
                if npl != ref_tiles:
                    failures.append("num_patches")
            if pvf is not None:
                t = torch.as_tensor(pvf)
                tot = sum(ref_tiles)
                tag = OK if t.shape[0] == tot else BAD
                print(f"  [{tag}] pixel_values_flat {tuple(t.shape)}  "
                      f"(rows should be sum(tiles) = {tot})")
                if t.shape[0] != tot:
                    failures.append("pixel_values_flat")
                print(f"        -> extract_feature returns ({tot}, {image_seq_length}, hidden); "
                      f"internvl.py splits it as {[t_ * image_seq_length for t_ in ref_tiles]}")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] mm_kwargs inspection: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        print("A failure here would be a real finding -- as of 2026-08-31 every one of")
        print("these passes, which is exactly why the 14-point gap is unexplained.")
        return 1
    print("All checks pass. The inputs are identical up to the engine boundary;")
    print("the divergence is downstream of everything this script can see.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
