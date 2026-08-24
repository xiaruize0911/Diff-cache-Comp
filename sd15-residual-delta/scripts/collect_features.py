#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from sd15_residual_delta.capture import SD15FeatureCapture
from sd15_residual_delta.config import load_config
from sd15_residual_delta.dataset import ShardWriter, build_pair
from sd15_residual_delta.pipeline import load_sd15_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    pairs = [tuple(pair) for pair in cfg["capture"]["anchor_target_steps"]]
    selected_steps = {step for pair in pairs for step in pair}
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    with ShardWriter(cfg["capture"]["output_dir"], cfg["capture"]["shard_size"]) as writer:
        for prompt_index, prompt in enumerate(tqdm(prompts, desc="prompts")):
            generator = torch.Generator(device="cuda").manual_seed(
                int(cfg["capture"]["seed"]) + prompt_index
            )
            with torch.inference_mode(), SD15FeatureCapture(
                pipeline.unet, selected_steps
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
            for module_name, module_features in capture.features.items():
                for anchor_step, target_step in pairs:
                    if anchor_step not in module_features or target_step not in module_features:
                        raise RuntimeError(
                            f"Missing {module_name} features for pair {(anchor_step, target_step)}"
                        )
                    writer.add(
                        build_pair(
                            module_features[anchor_step],
                            module_features[target_step],
                            prompt_id=f"prompt-{prompt_index:06d}",
                        )
                    )


if __name__ == "__main__":
    main()
