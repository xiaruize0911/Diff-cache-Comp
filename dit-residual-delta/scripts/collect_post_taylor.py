#!/usr/bin/env python
"""Capture the residual a Taylor forecast LEAVES BEHIND, on the Taylor trajectory.

The naive stack of Taylor plus our corrector already beats Taylor alone (+0.0110
SSIM, t=5.2 at 50 steps), but it double-counts: the corrector was trained to
predict R_true - R_anchor while the runtime has already injected the forecast, so
the linear part is applied twice and the whole thing has to be damped to sigma=0.25
to survive. The fix is to train on what the forecast does not already explain.

Two things this gets right that a naive re-use of collect_features.py would not:

  * The rollout runs WITH taylor_order=1, so the states are the ones the combined
    method actually visits. Capturing on a plain-cache trajectory would train the
    corrector for a trajectory it never sees.
  * The target is R_true - R_taylor, computed from the same cache history the
    runtime forecasts from, so it is the forecast the runtime will actually inject
    rather than a re-derivation that might differ.

The corrector's INPUTS are unchanged -- at deployment the runtime still hands it
`cache.residual`, not the forecast -- so the trained checkpoint drops into the
existing runtime with no modification, as `taylor_order=1` plus
`surrogate_checkpoint`.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch

from dit_residual_delta.capture import SlotRecorder
from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTSegmentRuntime, taylor_forecast, uniform_segments


class PostTaylorRecorder(SlotRecorder):
    """SlotRecorder whose `delta_residual` field holds R_true - R_taylor.

    The parent computes `true_residual - anchor_residual`, so passing
    `anchor_residual + (R_true - R_taylor)` as the true residual makes the stored
    target exactly the post-forecast remainder while every other field keeps its
    usual meaning.
    """

    def __init__(self, order: int, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.runtime = None
        self.n_no_history = 0

    def __call__(self, *, block_id, step, horizon, timestep, hidden,
                 anchor_hidden, anchor_residual, true_residual) -> None:
        cache = self.runtime.caches[int(block_id)]
        if cache.history and len(cache.history) > self.order:
            forecast = taylor_forecast(cache.history, int(step), self.order)
        else:
            # Not enough refresh points yet: the runtime injects the plain cache
            # here too, so the remainder is the ordinary delta. Counted, not hidden.
            forecast = anchor_residual
            self.n_no_history += 1
        remainder = true_residual - forecast
        super().__call__(block_id=block_id, step=step, horizon=horizon,
                         timestep=timestep, hidden=hidden, anchor_hidden=anchor_hidden,
                         anchor_residual=anchor_residual,
                         true_residual=anchor_residual + remainder)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--anchor-steps", nargs="+", type=int, required=True)
    ap.add_argument("--taylor-order", type=int, default=1)
    ap.add_argument("--num-segments", type=int, default=1)
    ap.add_argument("--tokens-per-slot", type=int, default=48)
    ap.add_argument("--shard-size", type=int, default=16)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)["model"]
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    embeddings = load_prompt_embeddings(args.embeddings)
    missing = [p for p in prompts if p not in embeddings]
    if missing:
        raise SystemExit(f"missing embeddings for {len(missing)} prompt(s)")
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "num_inference_steps": int(cfg["num_inference_steps"]),
              "guidance_scale": float(cfg["guidance_scale"])}
    anchors = set(args.anchor_steps)
    depth = len(pipeline.transformer.transformer_blocks)

    recorder = PostTaylorRecorder(order=args.taylor_order,
                                 tokens_per_slot=args.tokens_per_slot)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {"shards": [], "anchor_steps": sorted(anchors),
                "num_segments": args.num_segments, "taylor_order": args.taylor_order,
                "target": "R_true - R_taylor", "trajectory": "taylor",
                "tokens_per_slot": args.tokens_per_slot,
                "prompt_file": args.prompt_file, "seeds": args.seeds}
    shard, in_shard, start = 0, 0, time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for seed in args.seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed + pi)
            with torch.inference_mode():
                runtime = DiTSegmentRuntime(
                    pipeline.transformer,
                    segments=uniform_segments(depth, args.num_segments),
                    anchor_steps=anchors, reuse_observer=recorder,
                    taylor_order=args.taylor_order)
                recorder.runtime = runtime
                with runtime:
                    pipeline(**common, **embeddings[prompt], generator=generator)
            in_shard += 1
        if in_shard >= args.shard_size or pi == len(prompts) - 1:
            payload = recorder.to_payload()
            torch.save(payload, out / f"shard_{shard:04d}.pt")
            print(json.dumps({"shard": shard, "slots": payload["delta_h"].shape[0],
                              "seconds": round(time.perf_counter() - start)}), flush=True)
            manifest["shards"].append(f"shard_{shard:04d}.pt")
            recorder.reset(); shard += 1; in_shard = 0
    manifest["slots_without_history"] = recorder.n_no_history
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"shards": shard, "slots_without_history": recorder.n_no_history}))


if __name__ == "__main__":
    main()
