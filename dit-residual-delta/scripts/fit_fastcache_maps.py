#!/usr/bin/env python
"""Fit FastCache's per-block linear stand-in W_l, b_l in closed form.

FastCache replaces a skipped block by h_out = W_l h + b_l. The paper calls these
"learnable" without giving a fitting procedure, so we solve them by ridge least
squares on on-policy captured slots -- the most favourable instantiation, since a
closed-form optimum bounds any weakly-trained head on the same inputs.

We fit the residual rather than the output, R ~= h A_b + bias_b, so the deployed
map is W_l = I + A_b. Fitting the residual keeps the target centred and makes the
reported rel-MSE directly comparable with every other family in this repo, where
rel-MSE = 1 is verbatim reuse.

Ridge strength is chosen per block on a held-out slice of the same capture.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch

from dit_residual_delta.dataset import SlotDataset


def load_exact_shards(directory):
    """Shards written by collect_exact_blocks.py: hidden / residual directly."""
    parts = sorted(Path(directory).glob("shard_*.pt"))
    if not parts:
        raise SystemExit(f"no shards in {directory}")
    chunks = [torch.load(p, map_location="cpu") for p in parts]
    return {k: torch.cat([c[k] for c in chunks]) for k in ("hidden", "residual", "block_id", "step")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False        # normal equations need the precision
    if (Path(args.train_dir) / "manifest.json").exists() and json.loads(
            (Path(args.train_dir) / "manifest.json").read_text()).get("trajectory") == "exact":
        raw = load_exact_shards(args.train_dir)
        dim = raw["hidden"].shape[-1]

        class _Exact:
            num_slots = raw["hidden"].shape[0]
            tokens = raw["hidden"].shape[1]
            data = {"anchor_hidden": raw["hidden"], "delta_h": torch.zeros(1),
                    "anchor_residual": raw["residual"], "delta_residual": torch.zeros(1),
                    "block_id": raw["block_id"]}
        ds = _Exact()
        ds.dim = dim
        exact_mode = True
    else:
        ds = SlotDataset(args.train_dir)
        dim = ds.dim
        exact_mode = False
    block_id = ds.data["block_id"].long()
    blocks = sorted(set(block_id.tolist()))
    print(f"slots {ds.num_slots}, tokens {ds.tokens}, dim {dim}, blocks {len(blocks)}", flush=True)

    report, maps = {}, {}
    for b in blocks:
        index = (block_id == b).nonzero(as_tuple=True)[0]
        cut = int(len(index) * (1 - args.holdout_fraction))
        for tag, sl in (("fit", index[:cut]), ("val", index[cut:])):
            if exact_mode:
                H = ds.data["anchor_hidden"][sl].float().reshape(-1, dim).cuda()
                R = ds.data["anchor_residual"][sl].float().reshape(-1, dim).cuda()
            else:
                H = (ds.data["anchor_hidden"][sl].float() + ds.data["delta_h"][sl].float()
                     ).reshape(-1, dim).cuda()
                R = (ds.data["anchor_residual"][sl].float() + ds.data["delta_residual"][sl].float()
                     ).reshape(-1, dim).cuda()
            ones = torch.ones(H.shape[0], 1, device="cuda")
            X = torch.cat([H, ones], dim=-1)
            if tag == "fit":
                XtX, XtY = X.T @ X, X.T @ R
                n_fit = X.shape[0]
            else:
                Xv, Rv = X, R
        best = None
        for lam in args.lambdas:
            ridge = XtX + lam * torch.eye(dim + 1, device="cuda")
            sol = torch.linalg.solve(ridge, XtY)
            pred = Xv @ sol
            # rel-MSE against verbatim reuse is not defined here (the target is the
            # full residual, not its delta), so we report the plain relative error.
            rel = float(((Rv - pred) ** 2).sum() / (Rv ** 2).sum())
            if best is None or rel < best[0]:
                best = (rel, lam, sol)
        rel, lam, sol = best
        maps[str(b)] = {"weight": sol[:dim].contiguous().cpu(),
                        "bias": sol[dim].contiguous().cpu()}
        report[str(b)] = {"rel_error": rel, "lambda": lam, "fit_rows": n_fit,
                          "val_rows": int(Xv.shape[0])}
        print(json.dumps({"block": b, "rel_error": round(rel, 5), "lambda": lam}), flush=True)
        del XtX, XtY, Xv, Rv

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"maps": maps, "dim": dim, "report": report}, out)
    mean = sum(v["rel_error"] for v in report.values()) / len(report)
    Path(str(out) + ".json").write_text(json.dumps(
        {"mean_rel_error": mean, "per_block": report}, indent=1))
    print(json.dumps({"written": str(out), "mean_rel_error": round(mean, 5)}))


if __name__ == "__main__":
    main()
