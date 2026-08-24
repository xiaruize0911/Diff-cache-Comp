#!/usr/bin/env python
"""How much of delta_R is linearly predictable? A ceiling for local correctors.

For small delta_h, delta_R ~= J(h_anchor) . delta_h, so a per-(block, horizon)
ridge-regressed linear map is the first-order truth. If that closed-form fit
cannot approach the rel_mse target that the blend curve demands, no small
nonlinear head on the same inputs will either, and the training run is not worth
starting. Several feature sets are compared to see what carries the signal.

Splits are held on the GPU; normal equations are accumulated once per group and
reused across ridge strengths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dit_residual_delta.dataset import SlotDataset

FEATURES = {
    "dh": ("delta_h",),
    "dh+anchorR": ("delta_h", "anchor_residual"),
    "dh+anchorH": ("delta_h", "anchor_hidden"),
    "dh+anchorR+anchorH": ("delta_h", "anchor_residual", "anchor_hidden"),
}


def design(dataset, index, keys, bias=True):
    parts = [dataset.data[k][index].reshape(-1, dataset.dim).float() for k in keys]
    x = torch.cat(parts, dim=-1)
    if bias:
        x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=-1)
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="/dev/shm/dit-features/train")
    parser.add_argument("--val-dir", default="/dev/shm/dit-features/val")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURES))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1e-6, 1e-4, 1e-2, 1e-1])
    parser.add_argument("--group", choices=["block_horizon", "block", "global"], default="block_horizon")
    parser.add_argument("--max-train-slots", type=int, default=4000)
    parser.add_argument("--output", default="runs/linear_probe.json")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False   # normal equations need the precision
    train = SlotDataset(args.train_dir)
    val = SlotDataset(args.val_dir)
    print(f"train slots {train.num_slots}, val slots {val.num_slots}, dim {train.dim}", flush=True)

    def grouping(dataset):
        if args.group == "global":
            return {(-1, -1): torch.arange(dataset.num_slots, device=dataset.device)}
        blocks = dataset.data["block_id"].long()
        if args.group == "block":
            return {(int(b), -1): (blocks == b).nonzero(as_tuple=True)[0]
                    for b in blocks.unique().tolist()}
        horizons = dataset.data["horizon"].long()
        out = {}
        for b in blocks.unique().tolist():
            for h in horizons.unique().tolist():
                idx = ((blocks == b) & (horizons == h)).nonzero(as_tuple=True)[0]
                if idx.numel():
                    out[(int(b), int(h))] = idx
        return out

    train_groups, val_groups = grouping(train), grouping(val)
    print(f"groups: {len(train_groups)}", flush=True)
    report = {"group": args.group, "results": {}}

    for feature_set in args.feature_sets:
        keys = FEATURES[feature_set]
        totals = {lam: [0.0, 0.0] for lam in args.lambdas}
        per_block = {lam: {} for lam in args.lambdas}
        for key, tr_index in train_groups.items():
            va_index = val_groups.get(key)
            if va_index is None or va_index.numel() == 0:
                continue
            if tr_index.numel() > args.max_train_slots:
                sel = torch.randperm(tr_index.numel(), device=tr_index.device)[: args.max_train_slots]
                tr_index = tr_index[sel]
            x = design(train, tr_index, keys)
            y = train.data["delta_residual"][tr_index].reshape(-1, train.dim).float()
            xtx = x.T @ x
            xty = x.T @ y
            eye = torch.eye(xtx.shape[0], device=x.device)
            mean_diagonal = xtx.diagonal().mean()
            xv = design(val, va_index, keys)
            yv = val.data["delta_residual"][va_index].reshape(-1, val.dim).float()
            for lam in args.lambdas:
                weight = torch.linalg.solve(xtx + lam * mean_diagonal * eye, xty)
                err = float(((yv - xv @ weight) ** 2).sum())
                tot = float((yv ** 2).sum())
                totals[lam][0] += err; totals[lam][1] += tot
                slot = per_block[lam].setdefault(key[0], [0.0, 0.0])
                slot[0] += err; slot[1] += tot
            del x, y, xtx, xty, xv, yv
        best = None
        for lam in args.lambdas:
            score = totals[lam][0] / totals[lam][1]
            print(f"  {feature_set:22} lam={lam:<8} val rel_mse={score:.4f}", flush=True)
            if best is None or score < best["rel_mse"]:
                best = {"lam": lam, "rel_mse": score,
                        "per_block": {str(k): v[0] / v[1] for k, v in sorted(per_block[lam].items())}}
        report["results"][feature_set] = best
        print(f"{feature_set}: BEST val rel_mse = {best['rel_mse']:.4f} (lam={best['lam']})", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: round(v["rel_mse"], 4) for k, v in report["results"].items()}, indent=1))


if __name__ == "__main__":
    main()
