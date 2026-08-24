#!/usr/bin/env python
"""Generate oracle-ceiling variant specs.

`per-block`  -- one variant per block, true residual restored at every reuse step
                for that block alone. Gives the DiT analogue of the SD1.5
                per-module sensitivity ranking.
`top-k`      -- reads a per-block results.json and emits cumulative top-K variants,
                so the ceiling can be read as a function of the fraction of
                cache slots that a *perfect* corrector would have to fix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["per-block", "top-k"], required=True)
    parser.add_argument("--num-blocks", type=int, default=28)
    parser.add_argument("--anchor-steps", nargs="+", type=int, required=True)
    parser.add_argument("--sensitivity-results", help="results.json from the per-block run")
    parser.add_argument("--metric", default="ssim_gaussian_vs_exact")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 7, 14])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    anchors = sorted(set(args.anchor_steps))
    if args.mode == "per-block":
        specs = [
            {"name": f"oracle_b{b:02d}", "anchor_steps": anchors, "oracle_block_ids": [b]}
            for b in range(args.num_blocks)
        ]
    else:
        payload = json.loads(Path(args.sensitivity_results).read_text())
        aggregate = payload["aggregate"]
        fixed = aggregate["fixed"][args.metric]["mean"] if "fixed" in aggregate else None
        scored = []
        for name, metrics in aggregate.items():
            if not name.startswith("oracle_b"):
                continue
            block = int(name.split("oracle_b")[1])
            gain = metrics[args.metric]["mean"] - (fixed if fixed is not None else 0.0)
            scored.append((gain, block))
        scored.sort(reverse=True)
        ranking = [b for _, b in scored]
        specs = [{"name": "fixed", "anchor_steps": anchors}]
        for k in args.k:
            specs.append({
                "name": f"oracle_top{k}",
                "anchor_steps": anchors,
                "oracle_block_ids": sorted(ranking[:k]),
            })
        specs.append({"name": "oracle_all", "anchor_steps": anchors,
                      "oracle_block_ids": list(range(args.num_blocks))})
        print(json.dumps({"ranking_best_first": ranking}, indent=1))

    Path(args.output).write_text(json.dumps(specs, indent=2))
    print(f"wrote {len(specs)} variants -> {args.output}")


if __name__ == "__main__":
    main()
