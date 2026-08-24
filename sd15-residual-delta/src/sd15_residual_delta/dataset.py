from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .capture import ModuleStepFeature


def build_pair(anchor: ModuleStepFeature, target: ModuleStepFeature, prompt_id: str) -> dict:
    if anchor.module_name != target.module_name or anchor.hidden.shape != target.hidden.shape:
        raise ValueError("Anchor and target module features are incompatible")
    return {
        "prompt_id": prompt_id,
        "module_name": anchor.module_name,
        "module_id": anchor.module_id,
        "channels": anchor.hidden.shape[1],
        "anchor_step": anchor.step,
        "target_step": target.step,
        "delta_h": (target.hidden.float() - anchor.hidden.float()).to(torch.float16),
        "anchor_residual": anchor.residual,
        "target_delta_residual": (
            target.residual.float() - anchor.residual.float()
        ).to(torch.float16),
        "timestep": target.timestep,
        "horizon": target.step - anchor.step,
    }


class ShardWriter:
    def __init__(self, output_dir: str | Path, shard_size: int = 8) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buffer: list[dict] = []
        self.index: list[dict] = []
        self.shard_index = 0

    def add(self, item: dict) -> None:
        self.buffer.append(item)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        filename = f"shard-{self.shard_index:06d}.pt"
        torch.save(self.buffer, self.output_dir / filename)
        for offset, item in enumerate(self.buffer):
            self.index.append(
                {
                    "shard": filename,
                    "offset": offset,
                    "prompt_id": item["prompt_id"],
                    "module_id": item["module_id"],
                    "channels": item["channels"],
                    "target_step": item["target_step"],
                    "horizon": item["horizon"],
                }
            )
        self.buffer = []
        self.shard_index += 1

    def close(self) -> None:
        self.flush()
        (self.output_dir / "index.json").write_text(
            json.dumps(self.index, indent=2), encoding="utf-8"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()


class ResidualPairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        seed: int = 2026,
        target_steps: set[int] | None = None,
    ) -> None:
        self.root = Path(root)
        records = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        prompt_ids = sorted({record["prompt_id"] for record in records})
        random.Random(seed).shuffle(prompt_ids)
        train_end, val_end = int(len(prompt_ids) * 0.8), int(len(prompt_ids) * 0.9)
        selected = {
            "train": set(prompt_ids[:train_end]),
            "val": set(prompt_ids[train_end:val_end]),
            "test": set(prompt_ids[val_end:]),
        }[split]
        self.records = [
            record
            for record in records
            if record["prompt_id"] in selected
            and (
                target_steps is None
                or int(record.get("target_step", -1)) in target_steps
            )
        ]
        self._cached_name: str | None = None
        self._cached_shard: list[dict] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if self._cached_name != record["shard"]:
            self._cached_shard = torch.load(
                self.root / record["shard"], map_location="cpu", weights_only=False
            )
            self._cached_name = record["shard"]
        assert self._cached_shard is not None
        return self._cached_shard[record["offset"]]


def identity_collate(items: list[dict]) -> dict:
    if len(items) != 1:
        raise ValueError("Variable-resolution residual training requires micro-batch size 1")
    return items[0]
