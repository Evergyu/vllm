"""Tiling that reproduces the KTH reference pipeline exactly.

## Why this exists

InternVL's ``find_closest_aspect_ratio`` picks, among candidates whose aspect-ratio
distance ties, **the last one visited that satisfies the area test**. The result
therefore depends on the iteration order of the candidate container.

* upstream vLLM (and the official InternVL demo) iterate a list sorted by block count,
  so the largest qualifying tied ratio wins;
* the reference evaluation harness (``lmms-eval``) wraps that sorted list back into a
  ``set`` before passing it, so the winner is whatever the hash order puts last.

Those disagree on real images — 14.0% of a 300-image ChartQA sample and 9.3% of DocVQA
get a different tile count, in both directions (8->2 blocks on some, 2->8 on others).

Every published KTH number — the HF ``model.chat()`` reference included, since it
shares the harness front-end — was produced with the ``set`` ordering. Serving must
reproduce those numbers, so this module reproduces that ordering rather than the
upstream one. It does so by **constructing the container the same way** instead of
freezing a hash order, so if the harness's behaviour ever changes, this follows.

## Throughput

Nothing here touches the engine. This is per-request CPU preprocessing that vLLM
already runs in its async frontend, so continuous batching, paged attention and
prefix caching are unaffected.
"""
from PIL import Image

import torch

from vllm.transformers_utils.processors.internvl import (
    InternVLImageProcessor,
    build_transform,
    dynamic_preprocess_internvl,
    resolve_internvl_min_max_num,
)


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


class KTHInternVLImageProcessor(InternVLImageProcessor):
    """``InternVLImageProcessor`` with the reference harness's tie-break order."""

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> list[torch.Tensor]:
        if min_dynamic_patch is None:
            min_dynamic_patch = self.min_dynamic_patch
        if max_dynamic_patch is None:
            max_dynamic_patch = self.max_dynamic_patch
        if dynamic_image_size is None:
            dynamic_image_size = self.dynamic_image_size

        min_num, max_num = resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # applied inside kth_image_to_pixel_values
        )

        return [
            kth_image_to_pixel_values(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            )
            for image in images
        ]
