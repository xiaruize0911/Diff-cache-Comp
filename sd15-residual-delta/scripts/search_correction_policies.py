#!/usr/bin/env python3
"""Search sparse residual-delta correction policies on one anchor schedule."""

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
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


def parse_assignment(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def policy_keys(policy: dict) -> set[tuple[int, int]]:
    if "keys" in policy:
        return {(int(module), int(horizon)) for module, horizon in policy["keys"]}
    return {
        (int(module), int(horizon))
        for module in policy["module_ids"]
        for horizon in policy["horizons"]
    }


def policy_step_keys(policy: dict) -> set[tuple[int, int]] | None:
    if "step_keys" not in policy:
        return None
    return {(int(step), int(module)) for module, step in policy["step_keys"]}


def generate(
    pipeline,
    common: dict,
    seed: int,
    anchor_steps: set[int] | None,
    surrogate=None,
    keys: set[tuple[int, int]] | None = None,
    step_keys: set[tuple[int, int]] | None = None,
    scale: float = 1.0,
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
                surrogate_bank=surrogate,
                surrogate_module_horizon_keys=keys,
                surrogate_step_module_keys=step_keys,
                surrogate_scale=scale,
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
    parser.add_argument("--checkpoint", action="append", type=parse_assignment, required=True)
    parser.add_argument("--policies-file", required=True)
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
    policies = json.loads(Path(args.policies_file).read_text(encoding="utf-8"))
    if not prompts or not policies:
        raise ValueError("prompts and policies must be non-empty")
    anchor_steps = set(args.anchor_steps)
    num_steps = int(cfg["model"]["num_inference_steps"])
    if 0 not in anchor_steps or any(step < 0 or step >= num_steps for step in anchor_steps):
        raise ValueError("anchor steps must contain zero and lie within the denoising schedule")

    policy_map = {
        policy["name"]: {
            "keys": policy_keys(policy),
            "step_keys": policy_step_keys(policy),
            "scale": float(policy.get("scale", 1.0)),
        }
        for policy in policies
    }
    if len(policy_map) != len(policies):
        raise ValueError("policy names must be unique")
    checkpoint_paths = dict(args.checkpoint)
    if len(checkpoint_paths) != len(args.checkpoint):
        raise ValueError("checkpoint labels must be unique")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perceptual_metric = lpips.LPIPS(net="alex").to("cuda").eval()
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    surrogates = {}
    checkpoint_metadata = {}
    for label, checkpoint_path in checkpoint_paths.items():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        surrogate = build_surrogate_bank(cfg).to(device="cuda", dtype=torch.float16).eval()
        surrogate.load_state_dict(state["model"])
        surrogates[label] = surrogate
        checkpoint_metadata[label] = {
            "path": str(checkpoint_path),
            "step": state.get("step"),
            "val_relative_mse": state.get("val_relative_mse"),
        }

    variants = {
        f"{checkpoint_label}__{policy_name}": (
            surrogates[checkpoint_label],
            spec["keys"],
            spec["step_keys"],
            spec["scale"],
        )
        for checkpoint_label in checkpoint_paths
        for policy_name, spec in policy_map.items()
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
    generate(pipeline, warm, warm_seed, anchor_steps)
    for surrogate, keys, step_keys, scale in variants.values():
        generate(
            pipeline, warm, warm_seed, anchor_steps, surrogate, keys, step_keys, scale
        )

    cases = []
    total = len(prompts) * len(args.seeds)
    for prompt_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            common = {**common_base, "prompt": prompt}
            exact_image, exact_seconds, _ = generate(pipeline, common, seed, None)
            exact_tensor = image_tensor(exact_image)
            results = {}
            for name, (surrogate, keys, step_keys, scale) in {
                "fixed": (None, None, None, 1.0),
                **variants,
            }.items():
                image, elapsed, stats = generate(
                    pipeline,
                    common,
                    seed,
                    anchor_steps,
                    surrogate,
                    keys,
                    step_keys,
                    scale,
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
                    "variants": results,
                }
            )
            (output_dir / "cases.partial.json").write_text(
                json.dumps(cases, indent=2), encoding="utf-8"
            )
            print(json.dumps({"completed": len(cases), "total": total, "case": cases[-1]["case"]}), flush=True)

    aggregate = {}
    for name in ("fixed", *variants):
        values = [case["variants"][name] for case in cases]
        aggregate[name] = {
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
        }
        aggregate[name].update(
            {
                "passes_speed_gate": aggregate[name]["speedup_vs_exact"]["mean"] >= args.min_speedup,
                "passes_ssim_target": aggregate[name]["ssim_skimage_gaussian_vs_exact"]["mean"] > args.target_ssim,
                "passes_lpips_target": aggregate[name]["lpips_alex_vs_exact"]["mean"] < 0.125,
            }
        )
        aggregate[name]["eligible"] = all(
            aggregate[name][gate]
            for gate in ("passes_speed_gate", "passes_ssim_target", "passes_lpips_target")
        )
        if name != "fixed":
            fixed_mses = [case["variants"]["fixed"]["mse_vs_exact"] for case in cases]
            candidate_mses = [case["variants"][name]["mse_vs_exact"] for case in cases]
            aggregate[name]["aggregate_mse_improvement_over_fixed"] = (
                1.0 - statistics.mean(candidate_mses) / statistics.mean(fixed_mses)
            )

    report = {
        "prompt_file": str(Path(args.prompt_file)),
        "prompts": prompts,
        "seeds": args.seeds,
        "num_cases": len(cases),
        "anchor_steps": sorted(anchor_steps),
        "checkpoints": checkpoint_metadata,
        "policies": {
            name: {
                "keys": sorted([list(key) for key in spec["keys"]]),
                "step_keys": (
                    sorted([[module, step] for step, module in spec["step_keys"]])
                    if spec["step_keys"] is not None
                    else None
                ),
                "scale": spec["scale"],
            }
            for name, spec in policy_map.items()
        },
        "aggregate": aggregate,
        "eligible": [name for name, result in aggregate.items() if result["eligible"]],
        "cases": cases,
    }
    (output_dir / "correction_policy_search.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "cases.partial.json").unlink(missing_ok=True)
    print(json.dumps({"aggregate": aggregate, "eligible": report["eligible"]}, indent=2))


if __name__ == "__main__":
    main()
