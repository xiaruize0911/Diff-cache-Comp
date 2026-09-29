#!/usr/bin/env python
"""Evaluate K=1 correctors under a non-uniform, corrector-independent schedule.

Same two-phase protocol as eval_frozen_adaptive.py: phase one picks refresh steps
with the TeaCache-style accumulated-relative-change indicator on a plain-cache
rollout and FREEZES them per (prompt, seed); phase two replays those anchors, and
the uniform i=5 anchors, for every arm. Arms therefore share anchors and compute.

Unlike eval_frozen_adaptive.py, each arm carries its own injection strength
(`--arm name=checkpoint:sigma`), so a residual-trained and a trajectory-trained
corrector can each run at the sigma selected for it on validation.
"""
from __future__ import annotations

import argparse, json, math, statistics as st
from pathlib import Path

import lpips

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, lpips_alex, psnr, skimage_ssim
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings

from eval_frozen_adaptive import run


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mu, sd = st.mean(d), st.stdev(d)
    return {"mean": mu, "t": mu / (sd / math.sqrt(len(d))), "wins": sum(v > 0 for v in d),
            "n": len(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--uniform-interval", type=int, default=5)
    ap.add_argument("--arm", action="append", required=True,
                    help="name=checkpoint:sigma, repeatable")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    arms = [("plain", None, 1.0)]
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        ckpt, sigma = rest.rsplit(":", 1)
        arms.append((name, ckpt, float(sigma)))

    cfg = load_config(args.config)["model"]
    steps = int(cfg["num_inference_steps"])
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "num_inference_steps": steps, "guidance_scale": float(cfg["guidance_scale"])}
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    embeddings = load_prompt_embeddings(args.embeddings)
    perceptual = lpips.LPIPS(net="alex").to("cuda").eval()
    uniform = set(range(0, steps, args.uniform_interval))

    cases = []
    for pi, prompt in enumerate(prompts):
        for seed in args.seeds:
            kwargs = {**common, **embeddings[prompt]}
            exact_image, _ = run(pipeline, kwargs, seed, anchors=set(range(steps)))
            ref = image_tensor(exact_image)
            _, frozen = run(pipeline, kwargs, seed, threshold=args.threshold, record=True)
            frozen = set(frozen)
            row = {"case": f"prompt-{pi:02d}-seed-{seed}", "seed": seed,
                   "frozen_anchors": sorted(frozen), "n_refresh": len(frozen),
                   "max_tau": max(t - max(a for a in frozen if a <= t) for t in range(steps)),
                   "variants": {}}
            for sched, anchors in (("uniform", uniform), ("frozen", frozen)):
                for name, ckpt, sigma in arms:
                    image, _ = run(pipeline, kwargs, seed, anchors=anchors,
                                   checkpoint=ckpt, scale=sigma)
                    cand = image_tensor(image)
                    row["variants"][f"{sched}_{name}"] = {
                        "ssim": skimage_ssim(image, exact_image),
                        "lpips": lpips_alex(perceptual, cand, ref),
                        "psnr": psnr(cand, ref)}
            cases.append(row)
            print(json.dumps({"completed": len(cases), "n_refresh": len(frozen)}), flush=True)

    def col(key, metric="ssim"):
        return [c["variants"][key][metric] for c in cases]

    out = {"threshold": args.threshold, "uniform_interval": args.uniform_interval,
           "steps": steps, "arms": [{"name": n, "checkpoint": c, "sigma": s} for n, c, s in arms],
           "refresh_count_mean": st.mean(c["n_refresh"] for c in cases),
           "max_tau_range": [min(c["max_tau"] for c in cases), max(c["max_tau"] for c in cases)],
           "summary": {}, "cases": cases}
    names = [n for n, _, _ in arms]
    for sched in ("uniform", "frozen"):
        s = out["summary"][sched] = {}
        for n in names:
            s[n] = {m: st.mean(col(f"{sched}_{n}", m)) for m in ("ssim", "lpips", "psnr")}
            if n != "plain":
                s[n]["vs_plain_ssim"] = paired(col(f"{sched}_{n}"), col(f"{sched}_plain"))
        for i, a in enumerate(names[1:], 1):
            for b in names[i + 1:]:
                s[f"{b}_minus_{a}_ssim"] = paired(col(f"{sched}_{b}"), col(f"{sched}_{a}"))
    out["plain_frozen_minus_uniform_ssim"] = paired(col("frozen_plain"), col("uniform_plain"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
