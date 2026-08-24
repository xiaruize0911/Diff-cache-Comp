#!/usr/bin/env python3
"""Search explicit SD1.5 attention-cache anchor schedules."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
import lpips
import numpy as np
from skimage.metrics import structural_similarity

from evaluate_runtime_triplet import image_tensor, psnr, ssim
from sd15_residual_delta.config import load_config
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def skimage_ssim(candidate, reference, gaussian_weights: bool) -> float:
    candidate_array = np.asarray(candidate, dtype=np.float32) / 255.0
    reference_array = np.asarray(reference, dtype=np.float32) / 255.0
    options = {
        "channel_axis": 2,
        "data_range": 1.0,
        "gaussian_weights": gaussian_weights,
    }
    if gaussian_weights:
        options.update({"sigma": 1.5, "use_sample_covariance": False})
    return float(structural_similarity(reference_array, candidate_array, **options))


def lpips_alex(metric, candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate = candidate.to(device="cuda", dtype=torch.float32).mul(2).sub(1)
    reference = reference.to(device="cuda", dtype=torch.float32).mul(2).sub(1)
    with torch.inference_mode():
        return float(metric(candidate, reference, normalize=False).item())


def generate(pipeline, common: dict, seed: int, anchor_steps: set[int] | None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        if anchor_steps is None:
            result = pipeline(**common, generator=generator)
            stats = None
        else:
            with SD15AttentionRuntime(
                pipeline.unet,
                cache_interval=999,
                anchor_steps=anchor_steps,
            ) as runtime:
                result = pipeline(**common, generator=generator)
                stats = vars(runtime.stats)
    torch.cuda.synchronize()
    return result.images[0], time.perf_counter() - start, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--schedules-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--target-ssim", type=float, default=0.85)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--limit-schedules", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = [
        line.strip()
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schedules = json.loads(Path(args.schedules_file).read_text(encoding="utf-8"))
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    if args.limit_schedules:
        schedules = schedules[: args.limit_schedules]
    if not prompts or not schedules:
        raise ValueError("prompts and schedules must be non-empty")
    schedule_names = [item["name"] for item in schedules]
    if len(schedule_names) != len(set(schedule_names)):
        raise ValueError("schedule names must be unique")
    num_steps = int(cfg["model"]["num_inference_steps"])
    for item in schedules:
        steps = item["anchor_steps"]
        if steps != sorted(set(steps)) or any(step < 0 or step >= num_steps for step in steps):
            raise ValueError(f"invalid anchor steps for {item['name']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perceptual_metric = lpips.LPIPS(net="alex").to("cuda").eval()
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    common_base = {
        "height": int(cfg["model"]["height"]),
        "width": int(cfg["model"]["width"]),
        "num_inference_steps": num_steps,
        "guidance_scale": float(cfg["model"]["guidance_scale"]),
    }

    warm = {**common_base, "prompt": prompts[0]}
    generate(pipeline, warm, args.seeds[0] + 100_000, None)
    generate(
        pipeline,
        warm,
        args.seeds[0] + 100_000,
        set(schedules[0]["anchor_steps"]),
    )

    cases = []
    total = len(prompts) * len(args.seeds)
    for prompt_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            common = {**common_base, "prompt": prompt}
            exact_image, exact_seconds, _ = generate(pipeline, common, seed, None)
            exact_tensor = image_tensor(exact_image)
            schedule_results = {}
            for schedule in schedules:
                image, elapsed, stats = generate(
                    pipeline, common, seed, set(schedule["anchor_steps"])
                )
                candidate = image_tensor(image)
                schedule_results[schedule["name"]] = {
                    "num_anchor_steps": len(schedule["anchor_steps"]),
                    "anchor_steps": schedule["anchor_steps"],
                    "seconds": elapsed,
                    "speedup_vs_exact": exact_seconds / elapsed,
                    "mse_vs_exact": functional.mse_loss(candidate, exact_tensor).item(),
                    "psnr_vs_exact": psnr(candidate, exact_tensor),
                    "ssim_project_vs_exact": ssim(candidate, exact_tensor),
                    "ssim_skimage_vs_exact": skimage_ssim(image, exact_image, False),
                    "ssim_skimage_gaussian_vs_exact": skimage_ssim(
                        image, exact_image, True
                    ),
                    "lpips_alex_vs_exact": lpips_alex(
                        perceptual_metric, candidate, exact_tensor
                    ),
                    "runtime": stats,
                }
            cases.append(
                {
                    "case": f"prompt-{prompt_index:02d}-seed-{seed}",
                    "prompt": prompt,
                    "seed": seed,
                    "exact_seconds": exact_seconds,
                    "schedules": schedule_results,
                }
            )
            (output_dir / "cases.partial.json").write_text(
                json.dumps(cases, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "completed": len(cases),
                        "total": total,
                        "case": cases[-1]["case"],
                    }
                ),
                flush=True,
            )

    aggregate = {}
    for schedule in schedules:
        name = schedule["name"]
        speedups = [case["schedules"][name]["speedup_vs_exact"] for case in cases]
        mses = [case["schedules"][name]["mse_vs_exact"] for case in cases]
        project_ssims = [
            case["schedules"][name]["ssim_project_vs_exact"] for case in cases
        ]
        skimage_ssims = [
            case["schedules"][name]["ssim_skimage_vs_exact"] for case in cases
        ]
        gaussian_ssims = [
            case["schedules"][name]["ssim_skimage_gaussian_vs_exact"]
            for case in cases
        ]
        lpips_values = [
            case["schedules"][name]["lpips_alex_vs_exact"] for case in cases
        ]
        aggregate[name] = {
            "anchor_steps": schedule["anchor_steps"],
            "num_anchor_steps": len(schedule["anchor_steps"]),
            "cache_ratio": 1.0 - len(schedule["anchor_steps"]) / num_steps,
            "speedup_vs_exact": numeric_summary(speedups),
            "mse_vs_exact": numeric_summary(mses),
            "ssim_project_vs_exact": numeric_summary(project_ssims),
            "ssim_skimage_vs_exact": numeric_summary(skimage_ssims),
            "ssim_skimage_gaussian_vs_exact": numeric_summary(gaussian_ssims),
            "lpips_alex_vs_exact": numeric_summary(lpips_values),
            "passes_speed_gate": statistics.mean(speedups) >= args.min_speedup,
            "passes_ssim_target": statistics.mean(gaussian_ssims) > args.target_ssim,
            "passes_lpips_target": statistics.mean(lpips_values) < 0.125,
        }

    for name, item in aggregate.items():
        item["pareto_optimal_speed_ssim"] = not any(
            other_name != name
            and other["speedup_vs_exact"]["mean"] >= item["speedup_vs_exact"]["mean"]
            and other["ssim_skimage_gaussian_vs_exact"]["mean"]
            >= item["ssim_skimage_gaussian_vs_exact"]["mean"]
            and (
                other["speedup_vs_exact"]["mean"] > item["speedup_vs_exact"]["mean"]
                or other["ssim_skimage_gaussian_vs_exact"]["mean"]
                > item["ssim_skimage_gaussian_vs_exact"]["mean"]
            )
            for other_name, other in aggregate.items()
        )

    eligible = [
        {"name": name, **item}
        for name, item in aggregate.items()
        if item["passes_speed_gate"]
        and item["passes_ssim_target"]
        and item["passes_lpips_target"]
    ]
    report = {
        "prompt_file": str(Path(args.prompt_file)),
        "prompts": prompts,
        "seeds": args.seeds,
        "num_cases": len(cases),
        "min_speedup": args.min_speedup,
        "target_ssim": args.target_ssim,
        "aggregate": aggregate,
        "eligible": eligible,
        "cases": cases,
    }
    (output_dir / "anchor_schedule_search.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "cases.partial.json").unlink(missing_ok=True)
    print(json.dumps({"aggregate": aggregate, "eligible": eligible}, indent=2))


if __name__ == "__main__":
    main()
