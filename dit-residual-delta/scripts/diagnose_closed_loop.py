#!/usr/bin/env python
"""Why does an accurate C fail when deployed?

Runs a rollout with C active AND an observer that computes the true residual at
every reuse slot. That gives C's error measured on the trajectory C itself
created -- the closed-loop quantity -- next to the open-loop number the trainer
reports. It also tracks the norms of the states so a feedback blow-up shows up
directly as growth across steps.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTBlockRuntime
from dit_residual_delta.surrogate import load_surrogate_bank


class ClosedLoopProbe:
    def __init__(self, bank, scale: float):
        self.bank = bank
        self.scale = scale
        self.rows = []

    def __call__(self, *, block_id, step, horizon, timestep, hidden,
                 anchor_hidden, anchor_residual, true_residual):
        delta_h = hidden - anchor_hidden
        target = true_residual - anchor_residual
        if self.bank is None:
            prediction = torch.zeros_like(target)
        else:
            prediction = self.bank(delta_h, anchor_residual, anchor_hidden,
                                   timestep, horizon, block_id) * self.scale
        target32, prediction32 = target.float(), prediction.float()
        self.rows.append({
            "step": int(step), "block": int(block_id), "horizon": int(horizon),
            "err": float(((target32 - prediction32) ** 2).sum()),
            "target_energy": float((target32 ** 2).sum()),
            "pred_rms": float(prediction32.pow(2).mean().sqrt()),
            "target_rms": float(target32.pow(2).mean().sqrt()),
            "delta_h_rms": float(delta_h.float().pow(2).mean().sqrt()),
            "hidden_rms": float(hidden.float().pow(2).mean().sqrt()),
        })


def summarize(rows):
    by_step = defaultdict(lambda: [0.0, 0.0, [], [], [], []])
    total = [0.0, 0.0]
    for r in rows:
        slot = by_step[r["step"]]
        slot[0] += r["err"]; slot[1] += r["target_energy"]
        slot[2].append(r["pred_rms"]); slot[3].append(r["target_rms"])
        slot[4].append(r["delta_h_rms"]); slot[5].append(r["hidden_rms"])
        total[0] += r["err"]; total[1] += r["target_energy"]
    out = []
    for step in sorted(by_step):
        e, t, pr, tr, dh, hs = by_step[step]
        out.append({"step": step, "rel_mse": e / t,
                    "pred_rms": sum(pr) / len(pr), "target_rms": sum(tr) / len(tr),
                    "delta_h_rms": sum(dh) / len(dh), "hidden_rms": sum(hs) / len(hs)})
    return {"rel_mse_overall": total[0] / total[1], "per_step": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pixart_sigma_512.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embeddings", default="data/embeddings/design4.pt")
    parser.add_argument("--prompt-file", default="data/prompts/design4.txt")
    parser.add_argument("--anchor-steps", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--open-loop", action="store_true",
                        help="C predicts but is NOT applied: measures on-policy accuracy "
                             "on the same trajectory the capture saw")
    parser.add_argument("--dump-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=8201)
    parser.add_argument("--output", default="runs/closed_loop_diagnosis.json")
    args = parser.parse_args()

    cfg = load_config(args.config)["model"]
    prompt = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()][0]
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    embeddings = load_prompt_embeddings(args.embeddings)
    bank = load_surrogate_bank(args.checkpoint)
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "num_inference_steps": int(cfg["num_inference_steps"]),
              "guidance_scale": float(cfg["guidance_scale"])}

    report = {"checkpoint": args.checkpoint, "anchor_steps": args.anchor_steps, "scales": {}}
    for scale in args.scales:
        probe = ClosedLoopProbe(bank if scale > 0 else None, scale)
        apply_scale = 0.0 if args.open_loop else scale
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.inference_mode():
            with DiTBlockRuntime(pipeline.transformer, cache_interval=999,
                                 anchor_steps=set(args.anchor_steps),
                                 surrogate_bank=bank if apply_scale > 0 else None,
                                 surrogate_scale=apply_scale,
                                 reuse_observer=probe):
                pipeline(**common, **embeddings[prompt], generator=generator)
        if args.dump_rows:
            for row in probe.rows[: args.dump_rows]:
                print("   row " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                              for k, v in row.items()}), flush=True)
        summary = summarize(probe.rows)
        report["scales"][str(scale)] = summary
        print(f"scale={scale}: closed-loop rel_mse = {summary['rel_mse_overall']:.4f}", flush=True)
        for row in summary["per_step"]:
            print(f"   step {row['step']:2d}  rel_mse={row['rel_mse']:8.4f} "
                  f"target_rms={row['target_rms']:.4f} pred_rms={row['pred_rms']:.4f} "
                  f"dh_rms={row['delta_h_rms']:.4f} hidden_rms={row['hidden_rms']:.3f}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
