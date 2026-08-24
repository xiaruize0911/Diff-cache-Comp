from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer


DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_sd15_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    torch_dtype: torch.dtype = torch.float16,
    local_files_only: bool = True,
) -> StableDiffusionPipeline:
    """Load the minimal FP16 SD1.5 pipeline without duplicate FP32 weights."""
    local_root = Path(os.environ.get("SD15_LOCAL_MODEL_DIR", "/root/sd15-model"))
    required = (
        local_root / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        local_root / "vae" / "diffusion_pytorch_model.fp16.safetensors",
        local_root / "text_encoder" / "model.fp16.safetensors",
    )
    snapshot = (
        local_root
        if all(path.is_file() for path in required)
        else Path(snapshot_download(model_id, local_files_only=local_files_only))
    )
    unet = UNet2DConditionModel.from_pretrained(
        snapshot / "unet", variant="fp16", torch_dtype=torch_dtype, use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae", variant="fp16", torch_dtype=torch_dtype, use_safetensors=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        snapshot / "text_encoder", variant="fp16", torch_dtype=torch_dtype, use_safetensors=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(snapshot / "tokenizer")
    scheduler = DDIMScheduler.from_pretrained(snapshot / "scheduler")
    return StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )
