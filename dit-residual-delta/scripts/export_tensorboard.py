#!/usr/bin/env python
"""Export every `runs/*/train_report.json` history into TensorBoard event files.

Training here logs to JSON, not to a live writer, so this is a faithful re-emission
of what was recorded -- nothing is recomputed or interpolated. Two scalars per run:
`train/loss` and `val/rel_mse`. `val/rel_mse_best_so_far` is added because the
checkpoint actually deployed is the arg-min over the logged points, so the running
minimum is the curve that corresponds to what was evaluated downstream.

Each run also gets its hparams (recorded in the report's `args`) so runs can be
filtered in the HPARAMS tab.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="runs")
ap.add_argument("--output-dir", default="runs/tensorboard")
a = ap.parse_args()

out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
reports = sorted(Path(a.runs_dir).glob("*/train_report.json"))
n_ok = n_skip = 0
summary = []
for rp in reports:
    rep = json.loads(rp.read_text())
    hist = rep.get("history") or []
    if not hist:
        n_skip += 1
        continue
    name = rp.parent.name
    w = SummaryWriter(str(out / name))
    best = float("inf")
    for h in hist:
        step = int(h["step"])
        if "loss" in h:
            w.add_scalar("train/loss", float(h["loss"]), step)
        v = float(h["val_rel_mse"])
        w.add_scalar("val/rel_mse", v, step)
        best = min(best, v)
        w.add_scalar("val/rel_mse_best_so_far", best, step)
    args = rep.get("args") or {}
    hp = {k: v for k, v in args.items() if isinstance(v, (int, float, str, bool))}
    if hp:
        try:
            w.add_hparams(hp, {"hparam/best_val_rel_mse": rep["best"]["rel_mse"],
                               "hparam/best_step": float(rep["best"]["step"])})
        except Exception as e:      # hparams are a convenience, not the data
            print(json.dumps({"run": name, "hparams_skipped": str(e)}))
    w.close()
    n_ok += 1
    tot = hist[-1]["step"]
    summary.append({"run": name, "steps": tot, "best_step": rep["best"]["step"],
                    "best_val_rel_mse": rep["best"]["rel_mse"],
                    "truncated": rep["best"]["step"] == tot,
                    "best_at_first_eval": rep["best"]["step"] == hist[0]["step"]})
(out / "index.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"exported": n_ok, "skipped_no_history": n_skip,
                  "truncated": sum(s["truncated"] for s in summary),
                  "best_at_first_eval": sum(s["best_at_first_eval"] for s in summary),
                  "output": str(out)}, indent=1))
