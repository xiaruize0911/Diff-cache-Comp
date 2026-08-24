#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline

from flux_residual_delta.config import load_config
from flux_residual_delta.runtime import FluxSegmentRuntime
from train_surrogate import make_model


def load_surrogate(cfg: dict, checkpoint: str | None):
    if checkpoint is None:
        return None
    model = make_model(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=("exact", "fixed", "learned"), default="exact")
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache-interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs/runtime")
    args = parser.parse_args()
    if args.mode == "learned" and not args.checkpoint:
        parser.error("--checkpoint is required for learned mode")
    cfg = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = FluxPipeline.from_pretrained(cfg["model"]["model_id"], torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=False)
    surrogate = load_surrogate(cfg, args.checkpoint) if args.mode == "learned" else None
    call = {
        "prompt": args.prompt,
        "height": cfg["model"]["height"],
        "width": cfg["model"]["width"],
        "num_inference_steps": cfg["model"]["num_inference_steps"],
        "guidance_scale": cfg["model"]["guidance_scale"],
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    stats = None
    with torch.inference_mode():
        if args.mode == "exact":
            result = pipeline(**call)
        else:
            segment = cfg["segment"]
            with FluxSegmentRuntime(
                pipeline.transformer,
                segment["start"],
                segment["length"],
                cache_interval=args.cache_interval,
                surrogate=surrogate,
            ) as runtime:
                result = pipeline(**call)
                stats = {
                    "full_steps": runtime.stats.full_steps,
                    "reused_steps": runtime.stats.reused_steps,
                }
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    image_path = output / f"{args.mode}-seed-{args.seed}.png"
    result.images[0].save(image_path)
    report = {
        "mode": args.mode,
        "seed": args.seed,
        "prompt": args.prompt,
        "elapsed_seconds": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "cache_interval": args.cache_interval if args.mode != "exact" else None,
        "segment": cfg["segment"] if args.mode != "exact" else None,
        "runtime": stats,
        "image": str(image_path),
    }
    report_path = image_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
