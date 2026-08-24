#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sd15_residual_delta.config import load_config
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=("exact", "fixed", "learned"), default="exact")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs/runtime")
    args = parser.parse_args()
    if args.mode == "learned" and not args.checkpoint:
        parser.error("--checkpoint is required for learned mode")
    cfg = load_config(args.config)
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    surrogate = None
    if args.mode == "learned":
        surrogate = build_surrogate_bank(cfg).to(device="cuda", dtype=torch.float16).eval()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        surrogate.load_state_dict(state["model"])
    call = {
        "prompt": args.prompt,
        "height": cfg["model"]["height"],
        "width": cfg["model"]["width"],
        "num_inference_steps": cfg["model"]["num_inference_steps"],
        "guidance_scale": cfg["model"]["guidance_scale"],
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
    }
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    stats = None
    with torch.inference_mode():
        if args.mode == "exact":
            result = pipeline(**call)
        else:
            with SD15AttentionRuntime(
                pipeline.unet,
                cache_interval=int(cfg["reuse"]["cache_interval"]),
                surrogate_bank=surrogate,
            ) as runtime:
                result = pipeline(**call)
                stats = vars(runtime.stats)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / f"{args.mode}-seed-{args.seed}.png"
    result.images[0].save(image_path)
    report = {
        "mode": args.mode,
        "elapsed_seconds": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "runtime": stats,
        "image": str(image_path),
    }
    image_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
