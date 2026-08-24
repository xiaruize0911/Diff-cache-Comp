"""PixArt-Sigma loading, tuned for this pod's storage.

`/workspace` is a MooseFS mount that reads at ~25 MB/s, so weights are staged on
local disk (`/root/pixart`) and the 19 GB fp32 T5 lives in tmpfs. Text embeddings
are precomputed once (`scripts/prepare_assets.py`); after that the experiments run
with `text_encoder=None`, which keeps 9.5 GB off the GPU and makes every run
independent of the T5 copy surviving a restart.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

FAST_ROOT = Path(os.environ.get("PIXART_ROOT", "/root/pixart"))
DURABLE_ROOT = Path(os.environ.get("PIXART_DURABLE_ROOT", "/workspace/models/pixart_sigma_512_fp16"))


def _resolve_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    if (FAST_ROOT / "transformer" / "config.json").exists():
        return FAST_ROOT
    return DURABLE_ROOT


def load_pixart_pipeline(
    root: str | Path | None = None,
    dtype: torch.dtype = torch.float16,
    with_text_encoder: bool = False,
):
    """Load the pipeline. Without the text encoder, pass precomputed `prompt_embeds`."""
    from diffusers import PixArtSigmaPipeline

    root = _resolve_root(root)
    kwargs = {"torch_dtype": dtype}
    if not with_text_encoder:
        kwargs["text_encoder"] = None
        kwargs["tokenizer"] = None
    pipeline = PixArtSigmaPipeline.from_pretrained(root, **kwargs)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def load_prompt_embeddings(path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Return `{prompt: kwargs}` ready to splat into the pipeline call."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    out = {}
    for prompt, tensors in payload["prompts"].items():
        out[prompt] = {
            # the pipeline defaults `negative_prompt=""`, which collides with
            # supplying negative embeds directly
            "negative_prompt": None,
            "prompt_embeds": tensors["prompt_embeds"].to(device, dtype),
            "prompt_attention_mask": tensors["prompt_attention_mask"].to(device),
            "negative_prompt_embeds": payload["negative"]["prompt_embeds"].to(device, dtype),
            "negative_prompt_attention_mask": payload["negative"]["prompt_attention_mask"].to(device),
        }
    return out
