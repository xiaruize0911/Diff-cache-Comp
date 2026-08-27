#!/usr/bin/env python
"""Per-step deviation from the exact trajectory, and its distribution across images.

The report has residual-space error and end-of-rollout image metrics but nothing in
between, so the shape of the error over the denoising trajectory has never been
looked at. That shape is where the tau dependence lives: error should grow inside an
anchor interval and reset where the cache refreshes, and how much of it a corrector
removes at each tau is exactly what "does residual accuracy matter" is about.

This pipeline's __call__ has no callback_on_step_end, so the denoising loop from
train_latent_objective.py is reused (imported, not reimplemented, so the two stay
in step). Error is measured on the latent, relative: ||x_cached - x_exact|| /
||x_exact|| at every step, averaged over prompts.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.surrogate import load_surrogate_bank
from dit_residual_delta.variants import build_variant, runtime_for_variant

_spec = importlib.util.spec_from_file_location(
    "_lat", str(Path(__file__).parent / "train_latent_objective.py"))
_lat = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_lat)
rollout = _lat.rollout

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--embeddings", required=True)
ap.add_argument("--variants-file", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--limit", type=int, default=8)
ap.add_argument("--seed", type=int, default=5101)
a = ap.parse_args()

cfg = load_config(a.config)["model"]
steps = int(cfg["num_inference_steps"])
pipe = load_pixart_pipeline(with_text_encoder=False).to("cuda")
prompts = [l.strip() for l in Path(a.prompt_file).read_text().splitlines() if l.strip()][: a.limit]
emb = load_prompt_embeddings(a.embeddings)
specs = json.loads(Path(a.variants_file).read_text())
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

_B = {}
def bank_for(spec):
    p = spec.get("surrogate_checkpoint")
    if p is None: return None
    key = (p, spec.get("deploy_dtype", "fp16"))
    if key not in _B:
        _B[key] = load_surrogate_bank(
            p, dtype=torch.float16 if key[1] == "fp16" else torch.float32)
    return _B[key]

curves = {s["name"]: [] for s in specs}
with torch.no_grad():
    for pi, prompt in enumerate(prompts):
        g = torch.Generator(device="cuda").manual_seed(a.seed + pi)
        ref = rollout(pipe, emb[prompt], steps, cfg, g)
        for spec in specs:
            g = torch.Generator(device="cuda").manual_seed(a.seed + pi)
            v = build_variant(spec, steps, len(pipe.transformer.transformer_blocks))
            with runtime_for_variant(pipe.transformer, v, bank_for(spec)):
                got = rollout(pipe, emb[prompt], steps, cfg, g,
                              anchor_every=spec.get("cache_interval"))
            curves[spec["name"]].append([
                float((c.float() - r.float()).norm() / r.float().norm())
                for c, r in zip(got, ref)])
        print(json.dumps({"prompt_done": pi + 1}), flush=True)

data = {k: np.array(v) for k, v in curves.items()}
(out / "step_error.json").write_text(json.dumps(
    {k: v.tolist() for k, v in data.items()}, indent=1))

anchors = sorted({i for i in range(steps)
                  if i % int(specs[0].get("cache_interval", 5)) == 0})
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
for name, arr in data.items():
    m = arr.mean(0)
    ax[0].plot(range(len(m)), m, marker="o", ms=2.6, lw=1.1, label=name)
    ax[0].fill_between(range(len(m)), arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                       alpha=0.10, linewidth=0)
for x in anchors:
    ax[0].axvline(x, color="0.75", lw=0.5, ls=":", zorder=0)
ax[0].set_xlabel("denoising step  (dotted = cache refresh)")
ax[0].set_ylabel("$\\|x_{\\rm cached}-x_{\\rm exact}\\|\\,/\\,\\|x_{\\rm exact}\\|$")
ax[0].set_title("latent error along the trajectory (mean $\\pm$ 1 sd, %d prompts)"
                % len(prompts), fontsize=8.5)
ax[0].legend(fontsize=6.4, ncol=2); ax[0].grid(alpha=0.25, lw=0.4)

finals = {k: v[:, -1] for k, v in data.items()}
ax[1].hist(list(finals.values()), bins=10, label=list(finals.keys()))
ax[1].set_xlabel("final-step relative latent error")
ax[1].set_ylabel("images")
ax[1].set_title("distribution at the end of the rollout", fontsize=8.5)
ax[1].legend(fontsize=6.4); ax[1].grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.5)
fig.savefig(out / "step_error.png", dpi=185, bbox_inches="tight")
print(json.dumps({"written": str(out / "step_error.png"),
                  "final_mean": {k: round(float(v.mean()), 4) for k, v in finals.items()}}, indent=1))
