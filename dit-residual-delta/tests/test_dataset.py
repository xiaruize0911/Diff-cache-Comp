"""Slot statistics must not alias against the horizon ordering."""
from __future__ import annotations

import torch

from dit_residual_delta.dataset import FEATURE_KEYS, META_KEYS, SlotDataset


def _write_shard(path, num_slots, tokens=4, dim=8, first_horizon=1):
    """Slots in capture order: horizon cycles 1,2,3,4 with magnitude growing in it."""
    horizons = [(first_horizon + i - 1) % 4 + 1 for i in range(num_slots)]
    payload = {}
    for key in FEATURE_KEYS:
        rows = [torch.full((tokens, dim), float(h), dtype=torch.float16) for h in horizons]
        payload[key] = torch.stack(rows)
    payload["block_id"] = torch.zeros(num_slots, dtype=torch.int16)
    payload["step"] = torch.arange(num_slots, dtype=torch.int16)
    payload["horizon"] = torch.tensor(horizons, dtype=torch.int16)
    payload["timestep"] = torch.zeros(num_slots, dtype=torch.float32)
    torch.save(payload, path)


def test_stats_covers_every_horizon(tmp_path):
    # 5376 slots is the real split size; the old strided sampler used stride 2
    # there and saw only horizons 1 and 3, biasing every scale ~22% low.
    for shard in range(4):
        _write_shard(tmp_path / f"shard_{shard:04d}.pt", 336, first_horizon=1)
    data = SlotDataset(tmp_path, device="cpu")
    assert data.num_slots == 1344

    exact = (sum(h * h for h in (1, 2, 3, 4)) / 4) ** 0.5      # 2.7386
    aliased = (sum(h * h for h in (1, 3)) / 2) ** 0.5          # 2.2361
    for key in FEATURE_KEYS:
        assert abs(data.stats()[key] - exact) < 1e-3, key
        assert abs(data.stats()[key] - aliased) > 0.4, key


def test_stats_matches_bruteforce_on_odd_slot_count(tmp_path):
    _write_shard(tmp_path / "shard_0000.pt", 37)
    _write_shard(tmp_path / "shard_0001.pt", 13, first_horizon=3)
    data = SlotDataset(tmp_path, device="cpu")
    stats = data.stats(chunk=8)   # chunking must not change the result
    for key in FEATURE_KEYS:
        reference = float(data.data[key].float().pow(2).mean().sqrt())
        assert abs(stats[key] - reference) < 1e-4, key


def test_meta_keys_survive_concatenation(tmp_path):
    _write_shard(tmp_path / "shard_0000.pt", 8)
    data = SlotDataset(tmp_path, device="cpu")
    for key in FEATURE_KEYS + META_KEYS:
        assert data.data[key].shape[0] == 8, key
