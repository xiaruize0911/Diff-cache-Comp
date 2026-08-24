#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from sd15_residual_delta.config import load_config
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


def image_tensor(image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def psnr(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    mse = functional.mse_loss(candidate, reference).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    size, sigma = 11, 1.5
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coords.square()) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = candidate.shape[1]
    kernel = kernel_2d.expand(channels, 1, size, size)

    def filter_image(value: torch.Tensor) -> torch.Tensor:
        value = functional.pad(value, (size // 2,) * 4, mode="reflect")
        return functional.conv2d(value, kernel, groups=channels)

    mu_x, mu_y = filter_image(candidate), filter_image(reference)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = filter_image(candidate.square()) - mu_x2
    sigma_y = filter_image(reference.square()) - mu_y2
    sigma_xy = filter_image(candidate * reference) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    surrogate = build_surrogate_bank(cfg).to(device="cuda", dtype=torch.float16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    surrogate.load_state_dict(state["model"])

    common = {
        "prompt": args.prompt,
        "height": cfg["model"]["height"],
        "width": cfg["model"]["width"],
        "num_inference_steps": cfg["model"]["num_inference_steps"],
        "guidance_scale": cfg["model"]["guidance_scale"],
    }
    with torch.inference_mode():
        pipeline(
            **common,
            generator=torch.Generator(device="cuda").manual_seed(args.seed + 10_000),
        )

    images, timings, runtime_stats, peak_vram = {}, {}, {}, {}
    modes = ("exact", "fixed", "learned", "learned_320_640")
    selected_module_ids = {0, 1, 2, 3, 9, 10, 11, 12, 13, 14}
    for mode in modes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        stats = None
        with torch.inference_mode():
            if mode == "exact":
                result = pipeline(**common, generator=generator)
            else:
                with SD15AttentionRuntime(
                    pipeline.unet,
                    cache_interval=int(cfg["reuse"]["cache_interval"]),
                    surrogate_bank=surrogate if mode.startswith("learned") else None,
                    surrogate_module_ids=(
                        selected_module_ids if mode == "learned_320_640" else None
                    ),
                ) as runtime:
                    result = pipeline(**common, generator=generator)
                    stats = vars(runtime.stats)
        torch.cuda.synchronize()
        timings[mode] = time.perf_counter() - start
        runtime_stats[mode] = stats
        peak_vram[mode] = torch.cuda.max_memory_allocated() / 2**30
        images[mode] = result.images[0]
        images[mode].save(output_dir / f"{mode}-seed-{args.seed}.png")

    reference = image_tensor(images["exact"])
    metrics = {}
    for mode in modes[1:]:
        candidate = image_tensor(images[mode])
        metrics[mode] = {
            "mse": functional.mse_loss(candidate, reference).item(),
            "psnr": psnr(candidate, reference),
            "ssim": ssim(candidate, reference),
        }
    fixed_mse = metrics["fixed"]["mse"]
    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "timings_seconds": timings,
        "speedup_vs_exact": {
            mode: timings["exact"] / timings[mode] for mode in modes[1:]
        },
        "peak_vram_gib": peak_vram,
        "runtime": runtime_stats,
        "image_metrics_vs_exact": metrics,
        "mse_improvement_over_fixed": {
            mode: 1.0 - metrics[mode]["mse"] / fixed_mse for mode in modes[2:]
        },
        "selected_module_ids_320_640": sorted(selected_module_ids),
        "checkpoint_step": state.get("step"),
        "checkpoint_val_relative_mse": state.get("val_relative_mse"),
    }
    report_path = output_dir / "runtime_triplet_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
