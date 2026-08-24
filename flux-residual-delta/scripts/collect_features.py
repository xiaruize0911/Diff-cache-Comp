#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from flux_residual_delta.capture import FluxSegmentCapture
from flux_residual_delta.config import load_config
from flux_residual_delta.dataset import ShardWriter, build_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    pipeline = FluxPipeline.from_pretrained(cfg["model"]["model_id"], torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)
    segment = cfg["segment"]
    capture_cfg = cfg["capture"]
    pairs = [tuple(pair) for pair in capture_cfg["anchor_target_steps"]]
    maximum = max(max(pair) for pair in pairs)
    if maximum >= cfg["model"]["num_inference_steps"]:
        raise ValueError("Configured anchor/target step exceeds denoising schedule")
    generator = torch.Generator(device="cpu")
    with ShardWriter(capture_cfg["output_dir"], capture_cfg["shard_size"]) as writer:
        for prompt_index, prompt in enumerate(tqdm(prompts, desc="prompts")):
            generator.manual_seed(int(capture_cfg["seed"]) + prompt_index)
            with FluxSegmentCapture(
                pipeline.transformer,
                segment["start"],
                segment["length"],
                segment["hidden_dim"],
            ) as capture:
                pipeline(
                    prompt,
                    height=cfg["model"]["height"],
                    width=cfg["model"]["width"],
                    num_inference_steps=cfg["model"]["num_inference_steps"],
                    guidance_scale=cfg["model"]["guidance_scale"],
                    generator=generator,
                    output_type="latent",
                )
            if len(capture.features) != cfg["model"]["num_inference_steps"]:
                raise RuntimeError(
                    f"Expected {cfg['model']['num_inference_steps']} steps, captured {len(capture.features)}"
                )
            for anchor_step, target_step in pairs:
                writer.add(
                    build_pair(
                        capture.features[anchor_step],
                        capture.features[target_step],
                        prompt_id=f"prompt-{prompt_index:06d}",
                    )
                )


if __name__ == "__main__":
    main()
