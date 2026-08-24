#!/usr/bin/env python
"""Fit Block Caching's per-block scale-shift correction.

Wimbauer et al. train scale and shift by distillation: run the cached student,
take the uncached teacher's features as targets, and optimise only scale/shift
with everything else frozen. Our on-policy capture already IS that setup -- the
observer records the true residual at the states the cached rollout visits -- so
fitting to `anchor_residual + delta_residual` on captured slots is their objective
localised to one block.

Selection metric is the same relative residual MSE the learned corrector reports,
so the two are directly comparable: rel_mse = 1 is verbatim reuse by construction.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch

from dit_residual_delta.dataset import SlotDataset
from dit_residual_delta.surrogate import ScaleShiftCorrector


def evaluate(model, dataset, batch_size=512):
    model.eval()
    err = tot = 0.0
    per_block: dict[int, list[float]] = {}
    with torch.no_grad():
        for batch in dataset.iter_batches(batch_size):
            pred = model(batch["delta_h"], batch["anchor_residual"], batch["anchor_hidden"],
                         batch["timestep"], batch["horizon"], batch["block_id"].long())
            target = batch["delta_residual"]
            e = ((target - pred) ** 2).sum(dim=(1, 2))
            t = (target ** 2).sum(dim=(1, 2))
            err += float(e.sum()); tot += float(t.sum())
            ids = batch["block_id"].long()
            for v in ids.unique().tolist():
                m = ids == v
                slot = per_block.setdefault(int(v), [0.0, 0.0])
                slot[0] += float(e[m].sum()); slot[1] += float(t[m].sum())
    model.train()
    return {"rel_mse": err / tot,
            "per_block": {str(k): v[0] / v[1] for k, v in sorted(per_block.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--num-blocks", type=int, default=28)
    ap.add_argument("--conditioning-dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-slots", type=int, default=48)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--validate-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train = SlotDataset(args.train_dir)
    val = SlotDataset(args.val_dir)
    print(json.dumps({"train_slots": train.num_slots, "val_slots": val.num_slots,
                      "dim": train.dim}), flush=True)
    model = ScaleShiftCorrector(dim=train.dim, conditioning_dim=args.conditioning_dim,
                                num_blocks=args.num_blocks).cuda()
    print(f"parameters: {model.parameter_count()/1e6:.2f}M", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    history, best = [], {"rel_mse": float("inf"), "step": -1}
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        for group in optimiser.param_groups:
            group["lr"] = args.learning_rate * min(1.0, step / max(1, args.warmup))
        index = torch.randint(0, train.num_slots, (args.batch_slots,), device=train.device)
        batch = train.batch(index)
        pred = model(batch["delta_h"], batch["anchor_residual"], batch["anchor_hidden"],
                     batch["timestep"], batch["horizon"], batch["block_id"].long())
        # Plain MSE on the residual delta: Block Caching's distillation loss has no
        # per-site reweighting, and using ours would change the method under test.
        loss = ((pred - batch["delta_residual"]) ** 2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(model, val)
            history.append({"step": step, "loss": float(loss),
                            "val_rel_mse": metrics["rel_mse"]})
            print(json.dumps({"step": step, "loss": round(float(loss), 6),
                              "val_rel_mse": round(metrics["rel_mse"], 5),
                              "seconds": round(time.perf_counter() - start)}), flush=True)
            if metrics["rel_mse"] < best["rel_mse"]:
                best = {"rel_mse": metrics["rel_mse"], "step": step, "metrics": metrics}
                torch.save({"kind": "scale_shift",
                            "config": {"dim": train.dim,
                                       "conditioning_dim": args.conditioning_dim,
                                       "num_blocks": args.num_blocks},
                            "model": model.state_dict(),
                            # identity scaling: this corrector consumes raw residuals
                            "scale": {"delta_h": 1.0, "anchor_hidden": 1.0,
                                      "anchor_residual": 1.0, "delta_residual": 1.0},
                            "step": step, "val_rel_mse": metrics["rel_mse"]},
                           out / "best.pt")
    (out / "train_report.json").write_text(json.dumps(
        {"args": vars(args), "parameters": model.parameter_count(),
         "best": best, "history": history}, indent=1))
    print(json.dumps({"best_rel_mse": round(best["rel_mse"], 5), "best_step": best["step"]}))


if __name__ == "__main__":
    main()
