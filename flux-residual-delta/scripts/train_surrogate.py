#!/usr/bin/env python
"""Train the shared residual-delta corrector C.

Selection metric is relative residual MSE on a prompt-disjoint validation split:

    rel_mse = sum ||delta_R - delta_R_hat||^2 / sum ||delta_R||^2

rel_mse = 1 is "predict nothing" (plain fixed cache). The blend measurement in
docs/dit_gate_report_2026-08-21.md maps rel_mse to recovered image quality:
0.0625 -> ~26% of the gap, 0.01 -> ~51%. That is the bar, not a 20% improvement.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from flux_residual_delta.dataset import SlotDataset
from flux_residual_delta.surrogate import BlockResidualDeltaSurrogate, SurrogateConfig


def evaluate(model, dataset, scale, batch_size=512):
    model.eval()
    err = tot = 0.0
    per_block: dict[int, list[float]] = {}
    per_horizon: dict[int, list[float]] = {}
    with torch.no_grad():
        for batch in dataset.iter_batches(batch_size):
            pred = model(batch["delta_h"] / scale["delta_h"],
                         batch["anchor_residual"] / scale["anchor_residual"],
                         batch["anchor_hidden"] / scale["anchor_hidden"],
                         batch["timestep"], batch["horizon"], batch["block_id"])
            pred = pred * scale["delta_residual"]
            target = batch["delta_residual"]
            e = ((target - pred) ** 2).sum(dim=(1, 2))
            t = (target ** 2).sum(dim=(1, 2))
            err += float(e.sum()); tot += float(t.sum())
            for key, store in (("block_id", per_block), ("horizon", per_horizon)):
                values = batch[key].long() if key == "block_id" else batch[key].long()
                for v in values.unique().tolist():
                    mask = values == v
                    slot = store.setdefault(int(v), [0.0, 0.0])
                    slot[0] += float(e[mask].sum()); slot[1] += float(t[mask].sum())
    model.train()
    return {
        "rel_mse": err / tot,
        "per_block": {str(k): v[0] / v[1] for k, v in sorted(per_block.items())},
        "per_horizon": {str(k): v[0] / v[1] for k, v in sorted(per_horizon.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="/dev/shm/dit-features/train")
    parser.add_argument("--val-dir", default="/dev/shm/dit-features/val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mix-tokens", action="store_true")
    parser.add_argument("--linear-rank", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=28)
    parser.add_argument("--no-anchor-hidden", action="store_true")
    parser.add_argument("--no-anchor-residual", action="store_true")
    parser.add_argument("--no-delta-h", action="store_true")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-slots", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--normalize-loss", action="store_true",
                        help="per-slot scale-invariant loss; without it the single global\n                             delta_residual scale makes the gradient magnitude-weighted, so\n                             long horizons (3.3x the rms of horizon 1) dominate training")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train = SlotDataset(args.train_dir)
    val = SlotDataset(args.val_dir)
    scale = train.stats()
    print(json.dumps({"train_slots": train.num_slots, "val_slots": val.num_slots,
                      "tokens": train.tokens, "rms": {k: round(v, 5) for k, v in scale.items()}}), flush=True)

    cfg = SurrogateConfig(dim=train.dim, width=args.width, depth=args.depth,
                          mix_tokens=args.mix_tokens, linear_rank=args.linear_rank,
                          num_blocks=args.num_blocks,
                          use_delta_h=not args.no_delta_h,
                          use_anchor_residual=not args.no_anchor_residual,
                          use_anchor_hidden=not args.no_anchor_hidden)
    model = BlockResidualDeltaSurrogate(cfg).cuda()
    print(f"parameters: {model.parameter_count()/1e6:.2f}M", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    history, best = [], {"rel_mse": float("inf"), "step": -1}
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_at(step)
        index = torch.randint(0, train.num_slots, (args.batch_slots,), device=train.device)
        batch = train.batch(index)
        pred = model(batch["delta_h"] / scale["delta_h"],
                     batch["anchor_residual"] / scale["anchor_residual"],
                     batch["anchor_hidden"] / scale["anchor_hidden"],
                     batch["timestep"], batch["horizon"], batch["block_id"])
        target = batch["delta_residual"] / scale["delta_residual"]
        if args.normalize_loss:
            # weight each slot by its own target magnitude: optimises relative error,
            # which is what the selection metric measures per horizon
            slot_rms = target.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-3)
            loss = ((pred - target) / slot_rms).pow(2).mean()
        else:
            loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(model, val, scale)
            history.append({"step": step, "loss": float(loss), **{"val_rel_mse": metrics["rel_mse"]}})
            print(json.dumps({"step": step, "loss": round(float(loss), 5),
                              "val_rel_mse": round(metrics["rel_mse"], 5),
                              "seconds": round(time.perf_counter() - start)}), flush=True)
            if metrics["rel_mse"] < best["rel_mse"]:
                best = {"rel_mse": metrics["rel_mse"], "step": step, "metrics": metrics}
                torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                            "scale": scale, "step": step,
                            "val_rel_mse": metrics["rel_mse"]}, out / "best.pt")
    report = {"args": vars(args), "config": cfg.__dict__, "scale": scale,
              "parameters": model.parameter_count(), "history": history, "best": best}
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"best_val_rel_mse": best["rel_mse"], "best_step": best["step"]}, indent=1))


if __name__ == "__main__":
    main()
