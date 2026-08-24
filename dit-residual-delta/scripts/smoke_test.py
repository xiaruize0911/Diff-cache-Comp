#!/usr/bin/env python
"""Minimal GPU sanity check: exact vs fixed-cache generation on one prompt."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, psnr, skimage_ssim
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTBlockRuntime

PROMPT = "a yellow tram passing through a snow-covered European town at blue hour"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pixart_sigma_512.toml")
    parser.add_argument("--output-dir", default="runs/smoke")
    parser.add_argument("--cache-interval", type=int, default=2)
    parser.add_argument("--embeddings", default="data/embeddings/design4.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)["model"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    embeddings = load_prompt_embeddings(args.embeddings)
    prompt_kwargs = embeddings[PROMPT]
    num_blocks = len(pipeline.transformer.transformer_blocks)
    steps = int(cfg["num_inference_steps"])
    common = {
        **prompt_kwargs,
        "height": int(cfg["height"]),
        "width": int(cfg["width"]),
        "num_inference_steps": steps,
        "guidance_scale": float(cfg["guidance_scale"]),
    }

    def run(anchor_steps=None):
        generator = torch.Generator(device="cuda").manual_seed(7101)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            if anchor_steps is None:
                image = pipeline(**common, generator=generator).images[0]
                stats = None
            else:
                with DiTBlockRuntime(pipeline.transformer, cache_interval=999,
                                     anchor_steps=anchor_steps) as runtime:
                    image = pipeline(**common, generator=generator).images[0]
                    stats = vars(runtime.stats)
        torch.cuda.synchronize()
        return image, time.perf_counter() - start, stats

    run()  # warm-up
    exact_image, exact_seconds, _ = run()
    anchors = {s for s in range(steps) if s % args.cache_interval == 0}
    cached_image, cached_seconds, stats = run(anchors)

    exact_image.save(out / "exact.png")
    cached_image.save(out / f"fixed_i{args.cache_interval}.png")
    report = {
        "num_blocks": num_blocks,
        "steps": steps,
        "anchor_steps": sorted(anchors),
        "exact_seconds": exact_seconds,
        "cached_seconds": cached_seconds,
        "speedup": exact_seconds / cached_seconds,
        "psnr_vs_exact": psnr(image_tensor(cached_image), image_tensor(exact_image)),
        "ssim_gaussian_vs_exact": skimage_ssim(cached_image, exact_image, True),
        "runtime_stats": stats,
        "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (out / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
