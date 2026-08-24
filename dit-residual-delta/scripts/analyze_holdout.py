#!/usr/bin/env python
"""Turn a holdout `results.json` into the two comparisons that decide anything.

1. Equal-cost vs the correction-free frontier. The frontier is the fixed_i*
   family interpolated on the FLOPs axis. FLOPs, not wall clock: `evaluate_
   variants.py` timings are contended whenever anything else touches the GPU
   (the 2026-08-22 holdout run recorded exact at 1.93 s against 1.015 s clean),
   while FLOPs are deterministic. Speed claims belong to `benchmark_speed.py`.
2. Paired per-case delta against the *same anchor schedule* without a corrector,
   which is the comparison that isolates what C contributes.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = ("ssim_gaussian_vs_exact", "lpips_alex_vs_exact", "psnr_vs_exact")


def interpolate(points: list[tuple[float, float]], x: float) -> float | None:
    """Linear interpolation of the frontier; None outside its span (never extrapolate)."""
    points = sorted(points)
    if x < points[0][0] or x > points[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def paired(cases: list[dict], variant: str, baseline: str, metric: str) -> dict | None:
    deltas = [c["variants"][variant][metric] - c["variants"][baseline][metric]
              for c in cases if variant in c["variants"] and baseline in c["variants"]]
    if len(deltas) < 2:
        return None
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    n = len(deltas)
    t = mean / (sd / n ** 0.5) if sd > 0 else float("inf")
    return {"mean": mean, "t": t, "n": n, "wins": sum(d > 0 for d in deltas)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--flops", default="runs/flops.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    flops_table = json.loads(Path(args.flops).read_text())
    exact_flops = flops_table["exact"]["flops"]
    cases = data["cases"]
    agg = data["aggregate"]

    # every corrector here shares one architecture, so cost depends only on the
    # anchor interval: reuse steps x one C call each
    def flops_of(name: str) -> float | None:
        if name in flops_table:
            return flops_table[name]["flops"]
        for interval in (2, 3, 4, 5):
            if f"_i{interval}_" in name or name == f"fixed_i{interval}":
                key = f"fixed_i{interval}" if name.startswith("fixed") else f"K1_i{interval}_s075"
                if key in flops_table:
                    return flops_table[key]["flops"]
        return None

    frontier = {m: [(flops_of(n), agg[n][m]["mean"]) for n in agg if n.startswith("fixed_i")]
                for m in METRICS}

    rows = []
    for name in agg:
        f = flops_of(name)
        if f is None:
            print(f"!! no FLOPs for {name}")
            continue
        interval = next((i for i in (2, 3, 4, 5)
                         if f"_i{i}_" in name or name == f"fixed_i{i}"), None)
        row = {"name": name, "tflop": f / 1e12, "flops_speedup": exact_flops / f,
               "interval": interval}
        for m in METRICS:
            row[m] = agg[name][m]["mean"]
            ref = interpolate([p for p in frontier[m] if p[0]], f)
            row[m + "_vs_frontier"] = None if ref is None else agg[name][m]["mean"] - ref
        base = f"fixed_i{interval}"
        if not name.startswith("fixed") and base in agg:
            row["paired_vs_same_anchor"] = {
                m: paired(cases, name, base, m) for m in METRICS[:2]}
        rows.append(row)

    rows.sort(key=lambda r: -r["flops_speedup"])
    print(f"{'variant':22s} {'TFLOP':>7s} {'xFLOPs':>7s} {'SSIM':>7s} {'dFront':>8s} "
          f"{'LPIPS':>7s} {'dFront':>8s} {'dSSIM|t|wins vs same-anchor fixed':>34s}")
    for r in rows:
        d = r.get("paired_vs_same_anchor") or {}
        ps = d.get("ssim_gaussian_vs_exact")
        pair = (f"{ps['mean']:+.4f}  t={ps['t']:5.2f}  {ps['wins']}/{ps['n']}"
                if ps else "")
        def fmt(v, w=8, p=4):
            return f"{v:+.{p}f}".rjust(w) if v is not None else "n/a".rjust(w)
        print(f"{r['name']:22s} {r['tflop']:7.2f} {r['flops_speedup']:7.3f} "
              f"{r['ssim_gaussian_vs_exact']:7.4f} {fmt(r['ssim_gaussian_vs_exact_vs_frontier'])} "
              f"{r['lpips_alex_vs_exact']:7.4f} {fmt(r['lpips_alex_vs_exact_vs_frontier'])} "
              f"{pair:>34s}")

    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
