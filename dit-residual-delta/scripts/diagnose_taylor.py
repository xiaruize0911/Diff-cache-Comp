#!/usr/bin/env python
"""Does Taylor extrapolation of the cached residual reduce residual error?

TaylorSeer replaces verbatim reuse with a finite-difference forecast of the cached
quantity. Whether that helps depends on how smooth the residual trajectory is at
the step count and solver in use -- a claim measured on 50-step FLUX need not hold
on 20-step PixArt with DPMSolver. This measures it directly.

The rollout is plain order-0 caching, so every forecast order is scored on the
SAME trajectory (no confound from each order steering itself elsewhere). At each
reuse slot we already know the true whole-stack residual from the observer; the
refresh history is snapshotted independently. rel_mse = 1 is verbatim reuse by
construction, so any order that scores above 1 is worse than reusing.
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTSegmentRuntime, taylor_forecast

ORDERS = (0, 1, 2, 3)


class Probe:
    def __init__(self):
        self.history: list[tuple[int, torch.Tensor]] = []
        self.rows = []

    def snapshot(self, step: int, residual: torch.Tensor) -> None:
        self.history.insert(0, (int(step), residual.detach().clone()))
        del self.history[max(ORDERS) + 1:]

    def __call__(self, *, block_id, step, horizon, timestep, hidden,
                 anchor_hidden, anchor_residual, true_residual):
        target = (true_residual - anchor_residual).float()
        row = {"step": int(step), "horizon": int(horizon),
               "target_energy": float((target ** 2).sum()),
               "target_rms": float(target.pow(2).mean().sqrt())}
        for order in ORDERS:
            if order == 0 or len(self.history) < order + 1:
                pred = torch.zeros_like(target)
            else:
                pred = (taylor_forecast(self.history, int(step), order).float()
                        - anchor_residual.float())
            row[f"err_o{order}"] = float(((target - pred) ** 2).sum())
            row[f"pred_rms_o{order}"] = float(pred.pow(2).mean().sqrt())
        self.rows.append(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--cache-interval", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    embeds = load_prompt_embeddings(args.embeddings)
    missing = [p for p in prompts if p not in embeds]
    if missing:
        raise SystemExit(f"missing embeddings for {len(missing)} prompt(s): {missing[:2]}")
    steps = int(model_cfg["num_inference_steps"])
    anchors = set(range(0, steps, args.cache_interval))
    common_base = {"height": int(model_cfg["height"]), "width": int(model_cfg["width"]),
                   "num_inference_steps": steps,
                   "guidance_scale": float(model_cfg["guidance_scale"])}

    probe = Probe()
    for prompt in prompts:
        for seed in args.seeds:
            probe.history.clear()
            gen = torch.Generator(device="cuda").manual_seed(seed)
            common = {**common_base, **embeds[prompt]}
            with torch.inference_mode():
                with DiTSegmentRuntime(pipeline.transformer, anchor_steps=anchors,
                                       reuse_observer=probe, taylor_order=0) as rt:
                    def after(module, a, kw, out, rt=rt, probe=probe):
                        if rt._full_step and rt.caches[0].residual is not None:
                            probe.snapshot(rt._step_index - 1, rt.caches[0].residual)
                        return out
                    h = pipeline.transformer.register_forward_hook(after, with_kwargs=True)
                    try:
                        pipeline(**common, generator=gen)
                    finally:
                        h.remove()

    out = {"cache_interval": args.cache_interval, "anchors": sorted(anchors),
           "n_slots": len(probe.rows), "orders": {}}
    tot = sum(r["target_energy"] for r in probe.rows)
    for order in ORDERS:
        err = sum(r[f"err_o{order}"] for r in probe.rows)
        per_h = defaultdict(lambda: [0.0, 0.0])
        for r in probe.rows:
            per_h[r["horizon"]][0] += r[f"err_o{order}"]
            per_h[r["horizon"]][1] += r["target_energy"]
        out["orders"][str(order)] = {
            "rel_mse": err / tot,
            "per_horizon": {str(k): v[0] / v[1] for k, v in sorted(per_h.items())},
            "pred_rms_mean": sum(r[f"pred_rms_o{order}"] for r in probe.rows) / len(probe.rows),
        }
    out["target_rms_by_horizon"] = {
        str(h): sum(r["target_rms"] for r in probe.rows if r["horizon"] == h)
                / max(1, sum(1 for r in probe.rows if r["horizon"] == h))
        for h in sorted({r["horizon"] for r in probe.rows})}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["rel_mse"] for k, v in out["orders"].items()}, indent=1))


if __name__ == "__main__":
    main()
