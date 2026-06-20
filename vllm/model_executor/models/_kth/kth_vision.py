"""Build the encoder-modified InternVL vision tower for native vLLM use.

The checkpoint's original ``InternVLChatModel`` is instantiated with a stub LLM so
only the vision tower (ViT + mlp1 + the extra spatial modules) is constructed. vLLM
calls its ``extract_feature`` for numerically-exact image features, then runs the
Qwen3 decoder on vLLM's engine. The original modeling is reused verbatim (vendored),
so features match the HuggingFace reference.
"""
import torch
import torch.nn as nn

from .configuration_internvl_chat import InternVLChatConfig
from . import _orig_modeling as _orig


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


def build_kth_vision(config: InternVLChatConfig) -> nn.Module:
    """Build the modified vision tower on the meta device (weights filled by vLLM).

    The returned module owns ``vision_model`` / ``mlp1`` / the spatial modules; its
    ``extract_feature(pixel_values)`` yields ``(num_patches, num_image_token, hidden)``
    including the extra spatial tokens.
    """
    stub = _StubLLM(config.llm_config)
    with torch.device("meta"):
        m = _orig.InternVLChatModel(config, language_model=stub, use_flash_attn=False)
    m.language_model = None  # drop the stub so it is excluded from weight loading
    return m
