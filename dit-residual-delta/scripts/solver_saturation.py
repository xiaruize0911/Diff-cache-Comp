#!/usr/bin/env python
"""Does the backbone saturate at low step counts, and is that the solver's doing?

The report concluded that beyond a crossover simply lowering the step count beats
caching plus correction. That was measured with the shipped DPMSolverMultistep --
a high-order solver built for few-step sampling -- so the exact-few-steps baseline
may be unusually strong here and the conclusion may be a property of the solver
rather than of caching. This measures the exact-model quality curve against step
count under two solvers, scoring every point against ONE high-step reference from
its own solver so the curves are internally comparable.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from diffusers import DDIMScheduler

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, lpips_alex, skimage_ssim
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--embeddings", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--limit", type=int, default=12)
ap.add_argument("--seed", type=int, default=5101)
ap.add_argument("--steps", nargs="+", type=int, default=[5, 10, 15, 20, 30, 50])
ap.add_argument("--reference-steps", type=int, default=100)
a = ap.parse_args()

import lpips as lpips_pkg
import ImageReward as RM
perceptual = lpips_pkg.LPIPS(net="alex").to("cuda").eval()
reward = RM.load("ImageReward-v1.0", device="cuda")

cfg = load_config(a.config)["model"]
pipe = load_pixart_pipeline(with_text_encoder=False).to("cuda")
shipped = pipe.scheduler
ddim = DDIMScheduler.from_config(shipped.config)
prompts = [l.strip() for l in Path(a.prompt_file).read_text().splitlines() if l.strip()][: a.limit]
emb = load_prompt_embeddings(a.embeddings)
common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
          "guidance_scale": float(cfg["guidance_scale"])}
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

def gen(prompt, steps):
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    with torch.inference_mode():
        return pipe(**emb[prompt], **common, num_inference_steps=steps, generator=g).images[0]

results = {}
for tag, sched in (("dpmsolver", shipped), ("ddim", ddim)):
    pipe.scheduler = sched
    rows = {s: {"ssim": [], "lpips": [], "reward": []} for s in a.steps}
    ref_reward = []
    for pi, p in enumerate(prompts):
        ref = gen(p, a.reference_steps)
        rp = out / f"{tag}_p{pi}_ref.png"; ref.save(rp)
        ref_reward.append(reward.score(p, str(rp)))
        rt = image_tensor(ref)
        for s in a.steps:
            img = gen(p, s)
            ip = out / f"{tag}_p{pi}_s{s}.png"; img.save(ip)
            rows[s]["ssim"].append(skimage_ssim(img, ref, True))
            rows[s]["lpips"].append(lpips_alex(perceptual, image_tensor(img), rt))
            rows[s]["reward"].append(reward.score(p, str(ip)))
        print(json.dumps({"solver": tag, "prompt_done": pi + 1}), flush=True)
    results[tag] = {
        "reference_reward": sum(ref_reward) / len(ref_reward),
        "steps": {str(s): {k: sum(v) / len(v) for k, v in m.items()} for s, m in rows.items()},
    }
(out / "saturation.json").write_text(json.dumps(
    {"reference_steps": a.reference_steps, "n_prompts": len(prompts),
     "seed": a.seed, "results": results}, indent=2))
print(json.dumps({"written": str(out / "saturation.json")}))
