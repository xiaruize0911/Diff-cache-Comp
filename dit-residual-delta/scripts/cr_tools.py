#!/usr/bin/env python
"""Helpers for the camera-ready rerun (see scripts/camera_ready.sh).

Protocol this enforces, because the submitted version violated it:
  * every corrector's injection strength sigma is chosen on val24 x 3 seeds, over ONE
    grid shared by all correctors (the submission searched the residual arms at K=14
    and K=28 over two values on one seed, and the trajectory arms over nine on the
    very cases it then reported);
  * every reported number comes from test24 x 3 seeds, which selection never sees.

Subcommands
  sweep   write the val-sweep variants file for a list of correctors
  select  pick each corrector's sigma = argmax mean SSIM on the sweep results
  test    write the test variants file for a group of arms at their selected sigma
"""
from __future__ import annotations

import argparse, json, statistics as st
from pathlib import Path

GRID = [0.05, 0.1, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]


def base(K: int) -> dict:
    # K=28 is evaluated with the per-block runtime, as in the original configs
    spec = {"steps": 20, "cache_interval": 5}
    if K != 28:
        spec["num_segments"] = K
    return spec


def corrector(name: str) -> tuple[str, int]:
    """'m_k7' -> ('runs/m_k7/best.pt', 7)"""
    return f"runs/{name}/best.pt", int(name.rsplit("_k", 1)[1])


def cmd_sweep(a):
    specs = [{"name": "cache_i5", **base(1)}] if a.with_cache else []
    for name in a.correctors:
        ckpt, K = corrector(name)
        for s in GRID:
            specs.append({**base(K), "name": f"{name}@{s:g}",
                          "surrogate_checkpoint": ckpt, "surrogate_scale": s})
    Path(a.out).write_text(json.dumps(specs, indent=1))
    print(f"{len(specs)} variants -> {a.out}")


def cmd_select(a):
    means: dict[str, dict[float, float]] = {}
    for f in a.results:
        d = json.loads(Path(f).read_text())
        for name, agg in d["aggregate"].items():
            if "@" not in name:
                continue
            c, s = name.split("@")
            means.setdefault(c, {})[float(s)] = agg["ssim_gaussian_vs_exact"]["mean"]
    out = {}
    for c, curve in sorted(means.items()):
        best = max(curve, key=curve.get)
        missing = sorted(set(GRID) - set(curve))
        if missing:
            raise SystemExit(f"{c}: sweep incomplete, missing sigma {missing}")
        out[c] = {"sigma": best, "val_ssim": curve[best],
                  "at_grid_edge": best in (GRID[0], GRID[-1]),
                  "curve": {f"{s:g}": curve[s] for s in sorted(curve)}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for c, v in out.items():
        print(c, v["sigma"], round(v["val_ssim"], 4), "EDGE" if v["at_grid_edge"] else "")


def cmd_test(a):
    sig = json.loads(Path(a.sigma).read_text())
    specs = []
    if a.group == "table1":
        specs += [{"name": "cache_i5", **base(1)},
                  {"name": "taylor1", **base(1), "taylor_order": 1},
                  {"name": "taylor2", **base(1), "taylor_order": 2},
                  {"name": "blockcache", **base(1), "surrogate_checkpoint": "runs/bc_k1/best.pt",
                   "surrogate_scale": 1.0}]
        names = ["m_k1", "traj_k1"]
    elif a.group == "fp16k1":
        names = ["ctrl_k1"]
    else:  # per-K arms deployed in fp32
        names = [n for n in sig if not n.endswith("_k1")]
    for n in names:
        ckpt, K = corrector(n)
        specs.append({**base(K), "name": n, "surrogate_checkpoint": ckpt,
                      "surrogate_scale": sig[n]["sigma"]})
    Path(a.out).write_text(json.dumps(specs, indent=1))
    print(f"{len(specs)} variants -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep"); p.add_argument("--out", required=True)
    p.add_argument("--with-cache", action="store_true", help="add the verbatim-cache baseline")
    p.add_argument("correctors", nargs="+")
    p = sub.add_parser("select"); p.add_argument("--out", required=True)
    p.add_argument("results", nargs="+")
    p = sub.add_parser("test"); p.add_argument("--sigma", required=True)
    p.add_argument("--group", choices=["table1", "fp16k1", "perk"], required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sweep": cmd_sweep, "select": cmd_select, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
