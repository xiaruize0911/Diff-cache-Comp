#!/usr/bin/env python3
"""Search which SD1.5 attention modules should remain exact on reuse steps."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as functional

from evaluate_runtime_triplet import image_tensor, psnr, ssim
from search_anchor_schedules import lpips_alex, numeric_summary, skimage_ssim
from sd15_residual_delta.config import load_config
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime, find_transformer2d_modules


def generate(
    pipeline,
    common: dict,
    seed: int,
    anchor_steps: set[int] | None,
    cached_module_ids: set[int] | None = None,
    refresh_keys: set[tuple[int, int]] | None = None,
):
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
                cached_module_ids=cached_module_ids,
                oracle_refresh_keys=refresh_keys,
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
    parser.add_argument("--anchor-steps", nargs="+", type=int, required=True)
    parser.add_argument("--masks-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--target-ssim", type=float, default=0.85)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = [
        line.strip()
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    masks = json.loads(Path(args.masks_file).read_text(encoding="utf-8"))
    if not prompts or not masks:
        raise ValueError("prompts and masks must be non-empty")
    names = [mask["name"] for mask in masks]
    if len(names) != len(set(names)):
        raise ValueError("mask names must be unique")
    anchor_steps = set(args.anchor_steps)
    num_steps = int(cfg["model"]["num_inference_steps"])
    if 0 not in anchor_steps or any(step < 0 or step >= num_steps for step in anchor_steps):
        raise ValueError("anchor steps must contain zero and lie within the denoising schedule")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perceptual_metric = lpips.LPIPS(net="alex").to("cuda").eval()
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    module_names = [name for name, _ in find_transformer2d_modules(pipeline.unet)]
    all_module_ids = set(range(len(module_names)))
    cached_ids = {}
    refresh_keys = {}
    for mask in masks:
        exact_ids = {int(value) for value in mask["exact_module_ids"]}
        if not exact_ids <= all_module_ids:
            raise ValueError(f"invalid module id in {mask['name']}")
        cached_ids[mask["name"]] = all_module_ids - exact_ids
        refresh_keys[mask["name"]] = {
            (int(step), int(module)) for step, module in mask.get("refresh_keys", [])
        }

    common_base = {
        "height": int(cfg["model"]["height"]),
        "width": int(cfg["model"]["width"]),
        "num_inference_steps": num_steps,
        "guidance_scale": float(cfg["model"]["guidance_scale"]),
    }
    warm = {**common_base, "prompt": prompts[0]}
    warm_seed = args.seeds[0] + 100_000
    generate(pipeline, warm, warm_seed, None)
    for name in names:
        generate(
            pipeline,
            warm,
            warm_seed,
            anchor_steps,
            cached_ids[name],
            refresh_keys[name],
        )

    cases = []
    total = len(prompts) * len(args.seeds)
    for prompt_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            common = {**common_base, "prompt": prompt}
            exact_image, exact_seconds, _ = generate(pipeline, common, seed, None)
            exact_tensor = image_tensor(exact_image)
            results = {}
            for name in names:
                image, elapsed, stats = generate(
                    pipeline,
                    common,
                    seed,
                    anchor_steps,
                    cached_ids[name],
                    refresh_keys[name],
                )
                candidate = image_tensor(image)
                results[name] = {
                    "seconds": elapsed,
                    "speedup_vs_exact": exact_seconds / elapsed,
                    "mse_vs_exact": functional.mse_loss(candidate, exact_tensor).item(),
                    "psnr_vs_exact": psnr(candidate, exact_tensor),
                    "ssim_project_vs_exact": ssim(candidate, exact_tensor),
                    "ssim_skimage_vs_exact": skimage_ssim(image, exact_image, False),
                    "ssim_skimage_gaussian_vs_exact": skimage_ssim(image, exact_image, True),
                    "lpips_alex_vs_exact": lpips_alex(perceptual_metric, candidate, exact_tensor),
                    "runtime": stats,
                }
            cases.append(
                {
                    "case": f"prompt-{prompt_index:02d}-seed-{seed}",
                    "prompt": prompt,
                    "seed": seed,
                    "exact_seconds": exact_seconds,
                    "masks": results,
                }
            )
            (output_dir / "cases.partial.json").write_text(
                json.dumps(cases, indent=2), encoding="utf-8"
            )
            print(json.dumps({"completed": len(cases), "total": total, "case": cases[-1]["case"]}), flush=True)

    aggregate = {}
    all_cached_mse = statistics.mean(
        case["masks"]["all_cached"]["mse_vs_exact"] for case in cases
    )
    for mask in masks:
        name = mask["name"]
        values = [case["masks"][name] for case in cases]
        aggregate[name] = {
            "exact_module_ids": mask["exact_module_ids"],
            "cached_module_ids": sorted(cached_ids[name]),
            "refresh_keys": sorted([list(key) for key in refresh_keys[name]]),
            **{
                metric: numeric_summary([value[metric] for value in values])
                for metric in (
                    "speedup_vs_exact",
                    "mse_vs_exact",
                    "psnr_vs_exact",
                    "ssim_project_vs_exact",
                    "ssim_skimage_vs_exact",
                    "ssim_skimage_gaussian_vs_exact",
                    "lpips_alex_vs_exact",
                )
            },
        }
        aggregate[name]["aggregate_mse_improvement_over_all_cached"] = (
            1.0 - aggregate[name]["mse_vs_exact"]["mean"] / all_cached_mse
        )
        aggregate[name]["passes_speed_gate"] = (
            aggregate[name]["speedup_vs_exact"]["mean"] >= args.min_speedup
        )
        aggregate[name]["passes_ssim_target"] = (
            aggregate[name]["ssim_skimage_gaussian_vs_exact"]["mean"] > args.target_ssim
        )
        aggregate[name]["passes_lpips_target"] = (
            aggregate[name]["lpips_alex_vs_exact"]["mean"] < 0.125
        )
        aggregate[name]["eligible"] = all(
            aggregate[name][gate]
            for gate in ("passes_speed_gate", "passes_ssim_target", "passes_lpips_target")
        )

    report = {
        "prompt_file": str(Path(args.prompt_file)),
        "prompts": prompts,
        "seeds": args.seeds,
        "num_cases": len(cases),
        "anchor_steps": sorted(anchor_steps),
        "module_names": module_names,
        "aggregate": aggregate,
        "eligible": [name for name, result in aggregate.items() if result["eligible"]],
        "cases": cases,
    }
    (output_dir / "module_cache_mask_search.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "cases.partial.json").unlink(missing_ok=True)
    print(json.dumps({"aggregate": aggregate, "eligible": report["eligible"]}, indent=2))


if __name__ == "__main__":
    main()
