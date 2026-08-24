#!/usr/bin/env python
"""FLUX gates G0/G1: all-cache ceiling, and the correction-free speed/quality front.

Gate order follows the PixArt arm (docs/dit_pivot_plan.md): measure how much of the
runtime the block stack actually is, then map the fixed-cache Pareto front, and only
then consider training a corrector. Deliberately NOT gated on residual MSE -- the
PixArt arm showed that metric inverts the ranking.

G0 falls out of the same runtime: a variant whose only anchor is step 0 skips the
stack on every other step, so exact/that ratio IS the all-cache ceiling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flux_residual_delta.image_metrics import (image_tensor, lpips_alex, numeric_summary,
                                               psnr, skimage_ssim)
from flux_residual_delta.whole_stack import FluxWholeStackRuntime, anchor_steps_for


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/models/flux1_dev")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--intervals", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=9301)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-images", action="store_true")
    args = p.parse_args()

    from diffusers import FluxPipeline
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    store = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    prompts = store["prompts"][: args.limit] if args.limit else store["prompts"]

    pipe = FluxPipeline.from_pretrained(args.root, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    import lpips as lpips_pkg
    lp = lpips_pkg.LPIPS(net="alex").cuda()

    # only step 0 computes the stack -> the all-cache ceiling (G0)
    variants = [("all_cache", args.steps)] + [(f"fixed_i{i}", i) for i in args.intervals]
    cases = []
    for idx, prompt in enumerate(prompts):
        pe = store["prompt_embeds"][idx : idx + 1].to("cuda", torch.bfloat16)
        pooled = store["pooled_prompt_embeds"][idx : idx + 1].to("cuda", torch.bfloat16)
        common = dict(prompt_embeds=pe, pooled_prompt_embeds=pooled,
                      height=args.height, width=args.width,
                      num_inference_steps=args.steps, guidance_scale=args.guidance,
                      output_type="pil")

        def run(runtime=None):
            g = torch.Generator("cuda").manual_seed(args.seed + idx)
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
        case = {"index": idx, "prompt": prompt, "exact_seconds": exact_s, "variants": {}}
        if args.save_images:
            exact_img.save(out / f"prompt-{idx:02d}-exact.png")
        for name, interval in variants:
            rt = FluxWholeStackRuntime(pipe.transformer, anchor_steps_for(args.steps, interval))
            img, secs = run(rt)
            cand = image_tensor(img).cuda()
            case["variants"][name] = {
                "seconds": secs, "speedup_vs_exact": exact_s / secs,
                "ssim_gaussian_vs_exact": skimage_ssim(img, exact_img),
                "lpips_alex_vs_exact": lpips_alex(lp, cand, ref),
                "psnr_vs_exact": psnr(cand, ref),
                "stack_steps": rt.stats.stack_steps, "reused_steps": rt.stats.reused_steps,
            }
            if args.save_images:
                img.save(out / f"prompt-{idx:02d}-{name}.png")
        cases.append(case)
        print(json.dumps({"done": idx + 1, "of": len(prompts),
                          "exact_s": round(exact_s, 2),
                          **{k: round(v["ssim_gaussian_vs_exact"], 4)
                             for k, v in case["variants"].items()}}), flush=True)

    METRICS = ("speedup_vs_exact", "ssim_gaussian_vs_exact",
               "lpips_alex_vs_exact", "psnr_vs_exact")
    agg = {name: {m: numeric_summary([c["variants"][name][m] for c in cases])
                  for m in METRICS}
           for name, _ in variants}
    result = {"config": vars(args), "num_cases": len(cases), "aggregate": agg, "cases": cases}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {m: round(v[m]["mean"], 4) for m in v} for k, v in agg.items()}, indent=2))


if __name__ == "__main__":
    main()
