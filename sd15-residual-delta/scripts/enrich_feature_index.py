#!/usr/bin/env python3
"""Add step and horizon metadata to indexes produced by older collectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dir")
    args = parser.parse_args()
    root = Path(args.feature_dir)
    index_path = root / "index.json"
    records = json.loads(index_path.read_text(encoding="utf-8"))
    cached_name = None
    cached_shard = None
    for record in records:
        if record["shard"] != cached_name:
            cached_name = record["shard"]
            cached_shard = torch.load(
                root / cached_name, map_location="cpu", weights_only=False
            )
        item = cached_shard[int(record["offset"])]
        record["target_step"] = int(item["target_step"])
        record["horizon"] = int(item["horizon"])
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(index_path)
    print(json.dumps({"records": len(records), "index": str(index_path)}))


if __name__ == "__main__":
    main()
