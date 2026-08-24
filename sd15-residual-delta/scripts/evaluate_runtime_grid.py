#!/usr/bin/env python
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
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


BASE_MODES = ("exact", "fixed", "learned", "learned_320_640")
SELECTED_MODULE_IDS = {0, 1, 2, 3, 9, 10, 11, 12, 13, 14}


def generate(pipeline, surrogate, common, seed: int, mode: str, custom_keys=None):
    cache_interval = common["_cache_interval"]
    pipeline_args = {key: value for key, value in common.items() if not key.startswith("_")}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    stats = None
    with torch.inference_mode():
        if mode == "exact":
            result = pipeline(**pipeline_args, generator=generator)
        else:
            with SD15AttentionRuntime(
                pipeline.unet,
                cache_interval=cache_interval,
                surrogate_bank=surrogate if mode.startswith("learned") else None,
                surrogate_module_ids=(
                    SELECTED_MODULE_IDS if mode == "learned_320_640" else None
                ),
                surrogate_module_horizon_keys=(
                    custom_keys if mode == "learned_custom" else None
                ),
            ) as runtime:
                result = pipeline(**pipeline_args, generator=generator)
                stats = vars(runtime.stats)
    torch.cuda.synchronize()
    return (
        result.images[0],
        time.perf_counter() - start,
        stats,
        torch.cuda.max_memory_allocated() / 2**30,
    )


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--custom-surrogate-keys",
        nargs="*",
        metavar="MODULE:HORIZON",
        help="Add learned_custom mode restricted to module/horizon pairs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = [
        line.strip()
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("prompt file is empty")
    custom_keys = None
    if args.custom_surrogate_keys:
        custom_keys = set()
        for value in args.custom_surrogate_keys:
            module_text, separator, horizon_text = value.partition(":")
            if not separator:
                raise ValueError(f"invalid custom surrogate key: {value!r}")
            custom_keys.add((int(module_text), int(horizon_text)))
    modes = BASE_MODES + (("learned_custom",) if custom_keys else ())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    surrogate = build_surrogate_bank(cfg).to(device="cuda", dtype=torch.float16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    surrogate.load_state_dict(state["model"])

    base = {
        "height": cfg["model"]["height"],
        "width": cfg["model"]["width"],
        "num_inference_steps": cfg["model"]["num_inference_steps"],
        "guidance_scale": cfg["model"]["guidance_scale"],
        "_cache_interval": int(cfg["reuse"]["cache_interval"]),
    }

    # Warm every execution path so the first measured case does not pay one-time
    # CUDA kernel and allocator costs.
    warm = {**base, "prompt": prompts[0]}
    for mode in modes:
        generate(
            pipeline, surrogate, warm, args.seeds[0] + 100_000, mode, custom_keys
        )

    cases = []
    for prompt_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            case_name = f"prompt-{prompt_index:02d}-seed-{seed}"
            case_dir = output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            common = {**base, "prompt": prompt}
            images, timings, runtime_stats, peak_vram = {}, {}, {}, {}
            for mode in modes:
                image, elapsed, stats, peak = generate(
                    pipeline, surrogate, common, seed, mode, custom_keys
                )
                images[mode] = image
                timings[mode] = elapsed
                runtime_stats[mode] = stats
                peak_vram[mode] = peak
                image.save(case_dir / f"{mode}.png")

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
            case = {
                "case": case_name,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "seed": seed,
                "timings_seconds": timings,
                "speedup_vs_exact": {
                    mode: timings["exact"] / timings[mode] for mode in modes[1:]
                },
                "peak_vram_gib": peak_vram,
                "runtime": runtime_stats,
                "image_metrics_vs_exact": metrics,
                "mse_improvement_over_fixed": {
                    mode: 1.0 - metrics[mode]["mse"] / fixed_mse
                    for mode in modes[2:]
                },
            }
            write_json(case_dir / "metrics.json", case)
            cases.append(case)
            write_json(output_dir / "cases.partial.json", cases)
            print(
                json.dumps(
                    {
                        "completed": len(cases),
                        "total": len(prompts) * len(args.seeds),
                        "case": case_name,
                        "mse_improvement_over_fixed": case[
                            "mse_improvement_over_fixed"
                        ],
                    }
                ),
                flush=True,
            )

    aggregate = {}
    for mode in modes[1:]:
        aggregate[mode] = {
            "speedup_vs_exact": numeric_summary(
                [case["speedup_vs_exact"][mode] for case in cases]
            ),
            "mse": numeric_summary(
                [case["image_metrics_vs_exact"][mode]["mse"] for case in cases]
            ),
            "psnr": numeric_summary(
                [case["image_metrics_vs_exact"][mode]["psnr"] for case in cases]
            ),
            "ssim": numeric_summary(
                [case["image_metrics_vs_exact"][mode]["ssim"] for case in cases]
            ),
        }
        if mode.startswith("learned"):
            improvements = [
                case["mse_improvement_over_fixed"][mode] for case in cases
            ]
            aggregate[mode]["mse_improvement_over_fixed"] = numeric_summary(
                improvements
            )
            aggregate[mode]["mse_win_rate_vs_fixed"] = sum(
                value > 0 for value in improvements
            ) / len(improvements)
            aggregate[mode]["ssim_win_rate_vs_fixed"] = sum(
                case["image_metrics_vs_exact"][mode]["ssim"]
                >= case["image_metrics_vs_exact"]["fixed"]["ssim"]
                for case in cases
            ) / len(cases)

    report = {
        "prompt_file": str(Path(args.prompt_file)),
        "prompts": prompts,
        "seeds": args.seeds,
        "num_cases": len(cases),
        "selected_module_ids_320_640": sorted(SELECTED_MODULE_IDS),
        "custom_surrogate_module_horizon_keys": (
            sorted([list(key) for key in custom_keys]) if custom_keys else None
        ),
        "checkpoint_step": state.get("step"),
        "checkpoint_val_relative_mse": state.get("val_relative_mse"),
        "aggregate": aggregate,
        "cases": cases,
    }
    write_json(output_dir / "runtime_grid_metrics.json", report)
    (output_dir / "cases.partial.json").unlink(missing_ok=True)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
