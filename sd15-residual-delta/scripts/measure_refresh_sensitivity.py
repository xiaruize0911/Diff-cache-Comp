#!/usr/bin/env python3
"""Measure final-image sensitivity to one exact attention refresh at a time."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from evaluate_runtime_triplet import image_tensor, psnr, ssim
from sd15_residual_delta.config import load_config
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime, find_transformer2d_modules


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def generate(pipeline, common: dict, seed: int, refresh_keys):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipeline_args = {
        key: value for key, value in common.items() if key != "cache_interval"
    }
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        if refresh_keys is None:
            result = pipeline(**pipeline_args, generator=generator)
            stats = None
        else:
            with SD15AttentionRuntime(
                pipeline.unet,
                cache_interval=int(common["cache_interval"]),
                oracle_refresh_keys=refresh_keys,
            ) as runtime:
                result = pipeline(**pipeline_args, generator=generator)
                stats = vars(runtime.stats)
    torch.cuda.synchronize()
    return result.images[0], time.perf_counter() - start, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reuse-steps", nargs="+", type=int)
    parser.add_argument("--module-ids", nargs="+", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_interval = int(cfg["reuse"]["cache_interval"])
    num_steps = int(cfg["model"]["num_inference_steps"])
    default_reuse_steps = [step for step in range(num_steps) if step % cache_interval]
    reuse_steps = args.reuse_steps or default_reuse_steps
    if any(step not in default_reuse_steps for step in reuse_steps):
        raise ValueError("--reuse-steps must contain only non-anchor denoising steps")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    targets = find_transformer2d_modules(pipeline.unet)
    module_ids = args.module_ids or list(range(len(targets)))
    if any(module_id < 0 or module_id >= len(targets) for module_id in module_ids):
        raise ValueError("--module-ids contains an unknown module")
    common = {
        "prompt": args.prompt,
        "height": int(cfg["model"]["height"]),
        "width": int(cfg["model"]["width"]),
        "num_inference_steps": num_steps,
        "guidance_scale": float(cfg["model"]["guidance_scale"]),
        "cache_interval": cache_interval,
    }

    # Warm both paths before measuring.
    generate(pipeline, common, args.seed + 100_000, None)
    generate(pipeline, common, args.seed + 100_000, set())
    exact_image, exact_seconds, _ = generate(pipeline, common, args.seed, None)
    fixed_image, fixed_seconds, fixed_stats = generate(pipeline, common, args.seed, set())
    exact_image.save(output_dir / "exact.png")
    fixed_image.save(output_dir / "fixed.png")
    exact_tensor = image_tensor(exact_image)
    fixed_tensor = image_tensor(fixed_image)
    fixed_mse = functional.mse_loss(fixed_tensor, exact_tensor).item()

    interventions = []
    total = len(reuse_steps) * len(module_ids)
    for step in reuse_steps:
        step_records = []
        for module_id in module_ids:
            image, elapsed, stats = generate(
                pipeline, common, args.seed, {(step, module_id)}
            )
            candidate = image_tensor(image)
            mse = functional.mse_loss(candidate, exact_tensor).item()
            record = {
                "step": step,
                "horizon": step % cache_interval,
                "module_id": module_id,
                "module_name": targets[module_id][0],
                "seconds": elapsed,
                "mse_vs_exact": mse,
                "mse_improvement_over_fixed": 1.0 - mse / fixed_mse,
                "psnr_vs_exact": psnr(candidate, exact_tensor),
                "ssim_vs_exact": ssim(candidate, exact_tensor),
                "runtime": stats,
            }
            interventions.append(record)
            step_records.append(record)
        print(
            json.dumps(
                {
                    "completed": len(interventions),
                    "total": total,
                    "step": step,
                    "best_module_id": max(
                        step_records, key=lambda item: item["mse_improvement_over_fixed"]
                    )["module_id"],
                    "best_mse_improvement": max(
                        item["mse_improvement_over_fixed"] for item in step_records
                    ),
                }
            ),
            flush=True,
        )

    by_module_horizon = []
    for module_id in module_ids:
        for horizon in sorted({step % cache_interval for step in reuse_steps}):
            values = [
                item["mse_improvement_over_fixed"]
                for item in interventions
                if item["module_id"] == module_id and item["horizon"] == horizon
            ]
            if values:
                by_module_horizon.append(
                    {
                        "module_id": module_id,
                        "module_name": targets[module_id][0],
                        "horizon": horizon,
                        "num_interventions": len(values),
                        "mse_improvement_over_fixed": numeric_summary(values),
                        "positive_rate": sum(value > 0 for value in values) / len(values),
                    }
                )

    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "reuse_steps": reuse_steps,
        "module_ids": module_ids,
        "fixed": {
            "mse_vs_exact": fixed_mse,
            "psnr_vs_exact": psnr(fixed_tensor, exact_tensor),
            "ssim_vs_exact": ssim(fixed_tensor, exact_tensor),
            "exact_seconds": exact_seconds,
            "fixed_seconds": fixed_seconds,
            "runtime": fixed_stats,
        },
        "interventions": interventions,
        "by_module_horizon": by_module_horizon,
    }
    (output_dir / "refresh_sensitivity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"fixed": report["fixed"], "by_module_horizon": by_module_horizon}, indent=2))


if __name__ == "__main__":
    main()
