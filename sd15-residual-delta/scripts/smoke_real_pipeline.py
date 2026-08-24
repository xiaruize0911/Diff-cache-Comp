#!/usr/bin/env python
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


def run(pipeline, call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = pipeline(**call).images
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


def main() -> None:
    pipeline = load_sd15_pipeline().to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    prompt = "a cinematic photograph of a red fox standing in fresh snow at sunrise"
    base = {
        "prompt": prompt,
        "height": 256,
        "width": 256,
        "num_inference_steps": 4,
        "guidance_scale": 7.5,
        "output_type": "latent",
    }
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        exact, exact_seconds = run(
            pipeline,
            {**base, "generator": torch.Generator("cuda").manual_seed(1234)},
        )
        with SD15AttentionRuntime(pipeline.unet, cache_interval=2) as runtime:
            fixed, fixed_seconds = run(
                pipeline,
                {**base, "generator": torch.Generator("cuda").manual_seed(1234)},
            )
            stats = vars(runtime.stats)
    report = {
        "exact_seconds": exact_seconds,
        "fixed_seconds": fixed_seconds,
        "speedup": exact_seconds / fixed_seconds,
        "latent_mse": torch.mean((exact.float() - fixed.float()).square()).item(),
        "latent_shape": list(exact.shape),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "runtime": stats,
    }
    output = Path("runs/smoke_runtime.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
