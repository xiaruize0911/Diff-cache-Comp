"""In-memory slot dataset.

Banks were held entirely on the GPU, which capped a split at what fits in VRAM --
about 50k slots at 48 tokens on a 46 GB card. Since the supervision here is free
(the target is the exact residual, produced by the same forward pass that produces
the input, so there is nothing to annotate), sample count is worth scaling well
past that. Passing `store="cpu"` keeps the bank in host RAM, pinned, and stages
each batch to `compute_device`: a 48-slot batch is 21 MB, so at ~40 steps/s that
is under 1 GB/s over a link that does more than ten times that. Default behaviour
is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import torch

FEATURE_KEYS = ("delta_h", "anchor_hidden", "anchor_residual", "delta_residual")
META_KEYS = ("block_id", "step", "horizon", "timestep")


class SlotDataset:
    def __init__(self, directory: str | Path, device: str = "cuda",
                 limit_shards: int | None = None, store: str | None = None):
        directory = Path(directory)
        shards = sorted(directory.glob("shard_*.pt"))
        if limit_shards:
            shards = shards[:limit_shards]
        if not shards:
            raise SystemExit(f"no shards in {directory}")
        store = store or device
        # Shards are memory-mapped and copied into one preallocated tensor per key on
        # the store device. The previous torch.cat over fully loaded shards held the
        # bank in host RAM twice (~40 GB for a 50k-slot split), which a 50 GB pod
        # cannot survive. Same order as torch.cat, so the bank is bit-identical.
        payloads = [torch.load(s, map_location="cpu", weights_only=True, mmap=True)
                    for s in shards]
        total = sum(p["delta_h"].shape[0] for p in payloads)
        self.data = {}
        for k in FEATURE_KEYS + META_KEYS:
            first = payloads[0][k]
            dest = torch.empty((total, *first.shape[1:]), dtype=first.dtype, device=store)
            offset = 0
            for p in payloads:
                n = p[k].shape[0]
                dest[offset:offset + n].copy_(p[k])
                offset += n
            self.data[k] = dest
        del payloads
        if store == "cpu" and device != "cpu":
            # pin once so every later staging copy is async-capable
            self.data = {k: v.pin_memory() for k, v in self.data.items()}
        self.store = store
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
                block = data[start : start + chunk]
                if self.store != self.device:
                    block = block.to(self.device, non_blocking=True)
                total += float(block.float().pow(2).sum())
            out[k] = (total / data.numel()) ** 0.5
        return out

    def batch(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
        # gather on whichever device holds the bank, then stage the (small) batch
        idx = index.to(self.store)
        raw = {k: self.data[k][idx] for k in FEATURE_KEYS + META_KEYS}
        if self.store != self.device:
            raw = {k: v.to(self.device, non_blocking=True) for k, v in raw.items()}
        return {
            "delta_h": raw["delta_h"].float(),
            "anchor_hidden": raw["anchor_hidden"].float(),
            "anchor_residual": raw["anchor_residual"].float(),
            "delta_residual": raw["delta_residual"].float(),
            "timestep": raw["timestep"],
            "horizon": raw["horizon"].float(),
            "block_id": raw["block_id"].long(),
        }

    def iter_batches(self, batch_size: int):
        for start in range(0, self.num_slots, batch_size):
            yield self.batch(torch.arange(start, min(start + batch_size, self.num_slots),
                                          device=self.store))
