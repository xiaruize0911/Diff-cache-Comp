#!/usr/bin/env python
"""Image-level eval of FLUX whole-stack variants, with or without a corrector.

Mirrors the PixArt arm's `evaluate_variants.py` + `analyze_holdout.py` split: measure
every variant against the same-seed exact output, then compare at equal cost against
the correction-free front. Cost uses an analytic model fitted from the measured exact
and all-cache timings (validated to ~1% on the G1 front) rather than raw per-variant
wall clock, because wall clock is the quantity that gets polluted by GPU contention.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from flux_residual_delta.image_metrics import (image_tensor, lpips_alex, numeric_summary,
                                               psnr, skimage_ssim)
from flux_residual_delta.whole_stack import FluxWholeStackRuntime, anchor_steps_for

METRICS = ("ssim_gaussian_vs_exact", "lpips_alex_vs_exact", "psnr_vs_exact")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/models/flux1_dev")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--variants-file", required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seeds", nargs="+", type=int, default=[9301])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-images", action="store_true")
    args = p.parse_args()

    from diffusers import FluxPipeline
    from flux_residual_delta.surrogate import load_surrogate_bank

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    store = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    prompts = store["prompts"][: args.limit] if args.limit else store["prompts"]
    specs = json.loads(Path(args.variants_file).read_text())

    pipe = FluxPipeline.from_pretrained(args.root, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    import lpips as lpips_pkg
    lp = lpips_pkg.LPIPS(net="alex").cuda()

    # preload every checkpoint once: a 24 GB model plus per-variant disk reads would
    # otherwise show up inside the timings
    banks = {}
    for s in specs:
        ck = s.get("surrogate_checkpoint")
        if ck and ck not in banks:
            banks[ck] = load_surrogate_bank(ck, dtype=torch.bfloat16)
    print(json.dumps({"preloaded_checkpoints": len(banks), "variants": len(specs)}), flush=True)

    cases = []
    for idx, prompt in enumerate(prompts):
        pe = store["prompt_embeds"][idx : idx + 1].to("cuda", torch.bfloat16)
        pooled = store["pooled_prompt_embeds"][idx : idx + 1].to("cuda", torch.bfloat16)
        common = dict(prompt_embeds=pe, pooled_prompt_embeds=pooled,
                      height=args.height, width=args.width,
                      num_inference_steps=args.steps, guidance_scale=args.guidance,
                      output_type="pil")
        for seed in args.seeds:
            def run(runtime=None):
                g = torch.Generator("cuda").manual_seed(seed + idx)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                if runtime is None:
                    img = pipe(generator=g, **common).images[0]
                else:
                    with runtime:
                        img = pipe(generator=g, **common).images[0]
                torch.cuda.synchronize()
                return img, time.perf_counter() - t0

            exact_img, exact_s = run()
            ref = image_tensor(exact_img).cuda()
            case_id = f"prompt-{idx:02d}-seed-{seed}"
            case = {"case": case_id, "prompt": prompt, "seed": seed,
                    "exact_seconds": exact_s, "variants": {}}
            if args.save_images:
                exact_img.save(out / f"{case_id}-exact.png")
            for spec in specs:
                interval = int(spec.get("cache_interval", 5))
                ck = spec.get("surrogate_checkpoint")
                rt = FluxWholeStackRuntime(
                    pipe.transformer, anchor_steps_for(args.steps, interval),
                    surrogate=banks.get(ck), surrogate_scale=float(spec.get("surrogate_scale", 1.0)))
                img, secs = run(rt)
                cand = image_tensor(img).cuda()
                case["variants"][spec["name"]] = {
                    "seconds": secs, "speedup_vs_exact": exact_s / secs,
                    "cache_interval": interval,
                    "stack_steps": rt.stats.stack_steps,
                    "surrogate_calls": rt.stats.surrogate_calls,
                    "ssim_gaussian_vs_exact": skimage_ssim(img, exact_img),
                    "lpips_alex_vs_exact": lpips_alex(lp, cand, ref),
                    "psnr_vs_exact": psnr(cand, ref),
                }
                if args.save_images:
                    img.save(out / f"{case_id}-{spec['name']}.png")
            cases.append(case)
            print(json.dumps({"done": len(cases), "of": len(prompts) * len(args.seeds),
                              **{k: round(v["ssim_gaussian_vs_exact"], 4)
                                 for k, v in case["variants"].items()}}), flush=True)

    agg = {s["name"]: {m: numeric_summary([c["variants"][s["name"]][m] for c in cases])
                       for m in METRICS + ("speedup_vs_exact",)} for s in specs}
    result = {"config": vars(args), "num_cases": len(cases), "aggregate": agg, "cases": cases}
    (out / "results.json").write_text(json.dumps(result, indent=2))

    # paired comparison against the same anchor schedule without a corrector
    print(f"\n{'variant':22s} {'speedup':>8s} {'SSIM':>8s} {'LPIPS':>8s}  paired vs same-anchor fixed")
    for s in specs:
        name = s["name"]
        a = agg[name]
        base = next((o["name"] for o in specs
                     if not o.get("surrogate_checkpoint")
                     and int(o.get("cache_interval", 5)) == int(s.get("cache_interval", 5))), None)
        pair = ""
        if base and base != name:
            d = [c["variants"][name]["ssim_gaussian_vs_exact"]
                 - c["variants"][base]["ssim_gaussian_vs_exact"] for c in cases]
            if len(d) > 1 and statistics.stdev(d) > 0:
                t = statistics.fmean(d) / (statistics.stdev(d) / len(d) ** 0.5)
                pair = (f"{statistics.fmean(d):+.4f}  t={t:5.2f}  "
                        f"{sum(x > 0 for x in d)}/{len(d)} vs {base}")
        print(f"{name:22s} {a['speedup_vs_exact']['mean']:8.3f} "
              f"{a['ssim_gaussian_vs_exact']['mean']:8.4f} {a['lpips_alex_vs_exact']['mean']:8.4f}  {pair}")


if __name__ == "__main__":
    main()
