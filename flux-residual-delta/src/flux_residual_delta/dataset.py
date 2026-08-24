"""In-memory slot dataset. Shards live in tmpfs; the whole split fits on the GPU."""
from __future__ import annotations

from pathlib import Path

import torch

FEATURE_KEYS = ("delta_h", "anchor_hidden", "anchor_residual", "delta_residual")
META_KEYS = ("block_id", "step", "horizon", "timestep")


class SlotDataset:
    def __init__(self, directory: str | Path, device: str = "cuda", limit_shards: int | None = None):
        directory = Path(directory)
        shards = sorted(directory.glob("shard_*.pt"))
        if limit_shards:
            shards = shards[:limit_shards]
        if not shards:
            raise SystemExit(f"no shards in {directory}")
        parts: dict[str, list[torch.Tensor]] = {k: [] for k in FEATURE_KEYS + META_KEYS}
        for shard in shards:
            payload = torch.load(shard, map_location="cpu", weights_only=True)
            for k in FEATURE_KEYS + META_KEYS:
                parts[k].append(payload[k])
        self.data = {k: torch.cat(v).to(device) for k, v in parts.items()}
        self.device = device
        self.num_slots = self.data["delta_h"].shape[0]
        self.tokens = self.data["delta_h"].shape[1]
        self.dim = self.data["delta_h"].shape[2]

    def stats(self, chunk: int = 512) -> dict[str, float]:
        """Exact per-feature rms, used to normalise inputs and target in training.

        This was a strided subsample (`[::num_slots // 2000]`), which aliased
        against the slot ordering: horizons repeat with period 4 (reuse steps
        1,2,3,4,6,7,8,9,... at interval 5), so a 5376-slot split gives stride 2
        and only horizons 1 and 3 are ever sampled. Feature magnitude grows with
        horizon, so every scale came out ~22% low -- delta_h 0.0575 against a
        true 0.0704 -- and differently for splits of different sizes, which made
        scales incomparable across runs. Exact accumulation costs one pass.
        """
        out = {}
        for k in FEATURE_KEYS:
            data = self.data[k]
            total = 0.0
            for start in range(0, self.num_slots, chunk):
                total += float(data[start : start + chunk].float().pow(2).sum())
            out[k] = (total / data.numel()) ** 0.5
        return out

    def batch(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "delta_h": self.data["delta_h"][index].float(),
            "anchor_hidden": self.data["anchor_hidden"][index].float(),
            "anchor_residual": self.data["anchor_residual"][index].float(),
            "delta_residual": self.data["delta_residual"][index].float(),
            "timestep": self.data["timestep"][index],
            "horizon": self.data["horizon"][index].float(),
            "block_id": self.data["block_id"][index].long(),
        }

    def iter_batches(self, batch_size: int):
        for start in range(0, self.num_slots, batch_size):
            yield self.batch(torch.arange(start, min(start + batch_size, self.num_slots),
                                          device=self.device))
