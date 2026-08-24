#!/usr/bin/env python
"""Upper bound for any corrector: replace the cached residual with the TRUE residual.

`oracle_refresh_keys` runs the real attention module at the given (step, module)
positions, so image quality measured here is exactly what a *perfect* predictor
at those positions would achieve. If this ceiling is low, no amount of predictor
capacity or training data can help -- the approach itself is capped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as functional

from evaluate_runtime_triplet import image_tensor, psnr, ssim
from search_anchor_schedules import lpips_alex, numeric_summary, skimage_ssim
from sd15_residual_delta.config import load_config
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


def generate(pipeline, common, seed, anchor_steps, oracle_refresh_keys=None,
             forced_cache_keys=None):
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
                oracle_refresh_keys=oracle_refresh_keys,
                forced_cache_keys=forced_cache_keys,
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
    parser.add_argument("--variants-file", required=True,
                        help='JSON: [{"name":..., "module_ids":[...]}] -> oracle at those modules, all reuse steps')
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    anchor_steps = set(args.anchor_steps)
    num_steps = int(cfg["model"]["num_inference_steps"])
    reuse_steps = [s for s in range(num_steps) if s not in anchor_steps]

    specs = json.loads(Path(args.variants_file).read_text())
    exact_anchor_steps = sorted(anchor_steps)
    variants = {}
    for spec in specs:
        # restore true residual at reuse-step slots (quality up, speed down)
        rsteps = spec.get("steps", reuse_steps)
        oracle = {
            (int(s), int(m))
            for m in spec.get("module_ids", spec.get("exact_module_ids", []))
            for s in rsteps
        } or None
        # cache at otherwise-exact slots (speed up, quality down)
        csteps = spec.get("cache_steps", exact_anchor_steps)
        forced = {
            (int(s), int(m)) for m in spec.get("cache_module_ids", []) for s in csteps
        } or None
        variants[spec["name"]] = {"oracle": oracle, "forced": forced}

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    perceptual = lpips.LPIPS(net="alex").to("cuda").eval()
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    common_base = {
        "height": int(cfg["model"]["height"]), "width": int(cfg["model"]["width"]),
        "num_inference_steps": num_steps,
        "guidance_scale": float(cfg["model"]["guidance_scale"]),
    }
    warm = {**common_base, "prompt": prompts[0]}
    generate(pipeline, warm, args.seeds[0] + 100_000, None)
    generate(pipeline, warm, args.seeds[0] + 100_000, anchor_steps)

    cases = []
    for pi, prompt in enumerate(prompts):
        for seed in args.seeds:
            common = {**common_base, "prompt": prompt}
            exact_image, exact_seconds, _ = generate(pipeline, common, seed, None)
            exact_tensor = image_tensor(exact_image)
            results = {}
            for name, spec in {"fixed": {"oracle": None, "forced": None}, **variants}.items():
                image, elapsed, stats = generate(
                    pipeline, common, seed, anchor_steps,
                    spec["oracle"], spec["forced"],
                )
                cand = image_tensor(image)
                results[name] = {
                    "seconds": float(elapsed),
                    "speedup_vs_exact": float(exact_seconds / elapsed),
                    "mse_vs_exact": float(functional.mse_loss(cand, exact_tensor).item()),
                    "psnr_vs_exact": float(psnr(cand, exact_tensor)),
                    "ssim_skimage_gaussian_vs_exact": float(skimage_ssim(image, exact_image, True)),
                    "lpips_alex_vs_exact": float(lpips_alex(perceptual, cand, exact_tensor)),
                    "runtime": stats,
                }
            cases.append({"case": f"prompt-{pi:02d}-seed-{seed}", "prompt": prompt,
                          "seed": seed, "exact_seconds": exact_seconds, "variants": results})
            (out / "cases.partial.json").write_text(json.dumps(cases, indent=2))
            print(json.dumps({"completed": len(cases)}), flush=True)

    names = ["fixed", *variants]
    aggregate = {
        n: {
            m: numeric_summary([float(c["variants"][n][m]) for c in cases])
            for m in ("speedup_vs_exact", "mse_vs_exact", "psnr_vs_exact",
                      "ssim_skimage_gaussian_vs_exact", "lpips_alex_vs_exact")
        } for n in names
    }
    payload = {"prompt_file": args.prompt_file, "seeds": args.seeds,
               "anchor_steps": sorted(anchor_steps), "reuse_steps": reuse_steps,
               "variants": {k: {kk: (sorted(vv) if vv else None) for kk, vv in v.items()}
                            for k, v in variants.items()},
               "aggregate": aggregate, "cases": cases}
    (out / "oracle_ceiling.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(aggregate, indent=1))


if __name__ == "__main__":
    main()
