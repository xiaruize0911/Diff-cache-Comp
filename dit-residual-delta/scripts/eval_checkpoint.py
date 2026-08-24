#!/usr/bin/env python
"""Score one trained corrector on an arbitrary slot split.

`train_surrogate.py` only ever measures C on the split it trained against. The
DAgger question is different: it needs the *same* C measured on two distributions
-- the fixed-cache trajectory it was trained on, and the trajectory it produces
itself. Same metric as training (relative residual MSE, using the checkpoint's
own normalisation scales), so numbers are comparable to `train_report.json`.
"""
from __future__ import annotations

import argparse
import json
import statistics

import torch

from dit_residual_delta.dataset import SlotDataset
from dit_residual_delta.surrogate import BlockResidualDeltaSurrogate, SurrogateConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scale = payload["scale"]
    model = BlockResidualDeltaSurrogate(SurrogateConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    model = model.cuda().float().eval()

    data = SlotDataset(args.val_dir)
    err = tot = 0.0
    per_horizon: dict[int, list[float]] = {}
    # per-slot relative error, kept alongside the energy-weighted ratio: rel_mse
    # sums errors and targets separately, so slots with large ||delta_R|| dominate
    # it (horizon 4 carries ~11x the energy of horizon 1). A per-slot statistic
    # weights every injection site equally, which is closer to what deployment
    # experiences -- each site feeds the same propagation chain regardless of size.
    slot_rel: list[float] = []
    # "predict nothing" reference: rel_mse == 1 by construction, kept as a guard
    # against a silently mis-scaled checkpoint reading better than fixed reuse.
    with torch.no_grad():
        for batch in data.iter_batches(args.batch):
            pred = model(batch["delta_h"] / scale["delta_h"],
                         batch["anchor_residual"] / scale["anchor_residual"],
                         batch["anchor_hidden"] / scale["anchor_hidden"],
                         batch["timestep"], batch["horizon"], batch["block_id"])
            pred = pred * scale["delta_residual"]
            target = batch["delta_residual"]
            e = ((target - pred) ** 2).sum(dim=(1, 2))
            t = (target ** 2).sum(dim=(1, 2))
            err += float(e.sum()); tot += float(t.sum())
            slot_rel.extend((e / t.clamp_min(1e-12)).sqrt().tolist())
            horizons = batch["horizon"].long()
            for v in horizons.unique().tolist():
                mask = horizons == v
                slot = per_horizon.setdefault(int(v), [0.0, 0.0])
                slot[0] += float(e[mask].sum()); slot[1] += float(t[mask].sum())

    print(json.dumps({
        "label": args.label or args.val_dir,
        "checkpoint": args.checkpoint,
        "checkpoint_train_val_rel_mse": payload.get("val_rel_mse"),
        "val_dir": args.val_dir,
        "slots": data.num_slots,
        "rel_mse": err / tot,
        "per_slot_rel_error": {
            "median": statistics.median(slot_rel),
            "mean": statistics.fmean(slot_rel),
            "p90": sorted(slot_rel)[int(0.9 * (len(slot_rel) - 1))],
        },
        "per_horizon": {str(k): v[0] / v[1] for k, v in sorted(per_horizon.items())},
    }, indent=2))


if __name__ == "__main__":
    main()
