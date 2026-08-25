#!/usr/bin/env python
"""Test the inversion under an adaptive schedule, without the schedule confound.

The paper calls this "the single most informative experiment we did not run", and
explains why its own attempt does not count: under a content-adaptive policy the
refresh decision reads the hidden state that the corrector modifies, so every arm
chooses a DIFFERENT schedule and therefore a different compute budget. The
scale-invariant arm took 3.26 refreshes against the standard arm's 3.49 and scored
lower, which is plausibly the cost gap alone.

The fix it names is a schedule whose decisions are independent of the corrector's
output. This builds that: phase one runs a plain-cache rollout per (prompt, seed)
with the TeaCache-style accumulated-relative-change indicator and FREEZES the anchor
steps it chooses; phase two replays that frozen schedule for every arm. All arms then
see identical anchors and identical compute, and the only difference is the loss the
corrector was trained with.

The mechanism's prediction is directional: equalising staleness flattens the per-site
target-energy profile, which is the imbalance the scale-invariant objective corrects,
so the inv-minus-std gap should SHRINK relative to a uniform schedule.
"""
from __future__ import annotations

import argparse, json, statistics as st, math
from pathlib import Path

import lpips
import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, lpips_alex, numeric_summary, psnr, skimage_ssim
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTSegmentRuntime, uniform_segments
from dit_residual_delta.surrogate import load_surrogate_bank

_BANKS: dict[str, object] = {}


def bank_for(path):
    if path is None:
        return None
    if path not in _BANKS:
        _BANKS[path] = load_surrogate_bank(path)
    return _BANKS[path]


def run(pipeline, kwargs, seed, *, anchors=None, threshold=None, checkpoint=None,
        scale=1.0, record=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    chosen: list[int] = []
    with torch.inference_mode():
        runtime = DiTSegmentRuntime(
            pipeline.transformer,
            segments=uniform_segments(len(pipeline.transformer.transformer_blocks), 1),
            anchor_steps=anchors, adaptive_threshold=threshold,
            surrogate_bank=bank_for(checkpoint), surrogate_scale=scale)
        with runtime:
            handle = None
            if record:
                def after(module, a, kw, out, rt=runtime, box=chosen):
                    if rt._full_step:
                        box.append(rt._step_index - 1)
                    return out
                handle = pipeline.transformer.register_forward_hook(after, with_kwargs=True)
            try:
                result = pipeline(**kwargs, generator=generator)
            finally:
                if handle is not None:
                    handle.remove()
    return result.images[0], chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--threshold", type=float, required=True,
                    help="adaptive threshold, calibrate to match the uniform refresh count")
    ap.add_argument("--uniform-interval", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--std-checkpoints", nargs="+", required=True)
    ap.add_argument("--inv-checkpoints", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)["model"]
    steps = int(cfg["num_inference_steps"])
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "num_inference_steps": steps, "guidance_scale": float(cfg["guidance_scale"])}
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    embeddings = load_prompt_embeddings(args.embeddings)
    perceptual = lpips.LPIPS(net="alex").to("cuda").eval()
    uniform = set(range(0, steps, args.uniform_interval))

    arms = [("plain", None)]
    arms += [(f"std{i}", c) for i, c in enumerate(args.std_checkpoints)]
    arms += [(f"inv{i}", c) for i, c in enumerate(args.inv_checkpoints)]

    cases, refresh_counts = [], []
    for pi, prompt in enumerate(prompts):
        for seed in args.seeds:
            kwargs = {**common, **embeddings[prompt]}
            exact_image, _ = run(pipeline, kwargs, seed, anchors=set(range(steps)))
            ref = image_tensor(exact_image)
            # phase 1: freeze the schedule the indicator picks on a corrector-free rollout
            _, frozen = run(pipeline, kwargs, seed, threshold=args.threshold, record=True)
            frozen = set(frozen)
            refresh_counts.append(len(frozen))
            row = {"case": f"prompt-{pi:02d}-seed-{seed}", "seed": seed,
                   "frozen_anchors": sorted(frozen), "n_refresh": len(frozen),
                   "variants": {}}
            # phase 2: every arm replays the SAME anchors, uniform and frozen
            for sched_name, anchors in (("uniform", uniform), ("frozen", frozen)):
                for arm, ckpt in arms:
                    image, _ = run(pipeline, kwargs, seed, anchors=anchors,
                                   checkpoint=ckpt, scale=args.scale if ckpt else 1.0)
                    cand = image_tensor(image)
                    row["variants"][f"{sched_name}_{arm}"] = {
                        "ssim": skimage_ssim(image, exact_image),
                        "lpips": lpips_alex(perceptual, cand, ref),
                        "psnr": psnr(cand, ref)}
            cases.append(row)
            print(json.dumps({"completed": len(cases), "n_refresh": len(frozen)}), flush=True)

    def arm_mean(sched, prefix, metric="ssim"):
        vals = []
        for c in cases:
            per_seed = [c["variants"][k][metric] for k in c["variants"]
                        if k.startswith(f"{sched}_{prefix}")]
            vals.append(sum(per_seed) / len(per_seed))
        return vals

    out = {"threshold": args.threshold, "uniform_interval": args.uniform_interval,
           "scale": args.scale, "steps": steps,
           "refresh_count": {"mean": sum(refresh_counts) / len(refresh_counts),
                             "uniform": len(uniform)},
           "gap": {}, "cases": cases}
    for sched in ("uniform", "frozen"):
        inv, std = arm_mean(sched, "inv"), arm_mean(sched, "std")
        dif = [a - b for a, b in zip(inv, std)]
        mu, sd = st.mean(dif), st.stdev(dif)
        out["gap"][sched] = {"inv_mean": sum(inv) / len(inv), "std_mean": sum(std) / len(std),
                             "inv_minus_std": mu, "t": mu / (sd / math.sqrt(len(dif))),
                             "wins": sum(1 for v in dif if v > 0), "n": len(dif)}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
