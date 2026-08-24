#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from sd15_residual_delta.config import load_config
from sd15_residual_delta.dataset import ShardWriter
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


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
    selected_steps = {int(step) for step in cfg["capture"]["on_policy_target_steps"]}
    selected_modules = {
        int(module_id) for module_id in cfg["capture"].get("module_ids", range(16))
    }
    anchor_steps = {
        int(step) for step in cfg["capture"].get("anchor_steps", [])
    } or None
    capture_seeds = [
        int(seed)
        for seed in cfg["capture"].get("seeds", [cfg["capture"].get("seed", 1234)])
    ]
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    with ShardWriter(cfg["capture"]["output_dir"], cfg["capture"]["shard_size"]) as writer:
        for prompt_index, prompt in enumerate(tqdm(prompts, desc="on-policy prompts")):
            prompt_id = f"prompt-{prompt_index:06d}"
            items = []

            def observe(**observation) -> None:
                if (
                    observation["step"] not in selected_steps
                    or observation["module_id"] not in selected_modules
                ):
                    return
                hidden = observation["hidden"]
                anchor_hidden = observation["anchor_hidden"]
                anchor_residual = observation["anchor_residual"]
                oracle_residual = observation["oracle_residual"]
                items.append(
                    {
                        "prompt_id": prompt_id,
                        "module_name": observation["module_name"],
                        "module_id": observation["module_id"],
                        "channels": hidden.shape[1],
                        "anchor_step": observation["step"] - observation["horizon"],
                        "target_step": observation["step"],
                        "delta_h": (hidden.float() - anchor_hidden.float()).to(
                            device="cpu", dtype=torch.float16
                        ),
                        "anchor_residual": anchor_residual.to(
                            device="cpu", dtype=torch.float16
                        ),
                        "target_delta_residual": (
                            oracle_residual.float() - anchor_residual.float()
                        ).to(device="cpu", dtype=torch.float16),
                        "timestep": float(observation["timestep"].flatten()[0].item()),
                        "horizon": observation["horizon"],
                    }
                )

            for seed in capture_seeds:
                generator = torch.Generator(device="cuda").manual_seed(
                    seed + prompt_index
                )
                before = len(items)
                with torch.inference_mode(), SD15AttentionRuntime(
                    pipeline.unet,
                    cache_interval=int(cfg["reuse"]["cache_interval"]),
                    anchor_steps=anchor_steps,
                    reuse_observer=observe,
                    reuse_observer_keys={
                        (step, module_id)
                        for step in selected_steps
                        for module_id in selected_modules
                    },
                ):
                    pipeline(
                        prompt,
                        height=cfg["model"]["height"],
                        width=cfg["model"]["width"],
                        num_inference_steps=cfg["model"]["num_inference_steps"],
                        guidance_scale=cfg["model"]["guidance_scale"],
                        generator=generator,
                        output_type="latent",
                    )
                expected_per_seed = len(selected_steps) * len(selected_modules)
                observed = len(items) - before
                if observed != expected_per_seed:
                    raise RuntimeError(
                        f"Expected {expected_per_seed} items for {prompt_id} seed {seed}, got {observed}"
                    )
            for item in items:
                writer.add(item)


if __name__ == "__main__":
    main()
