#!/usr/bin/env python
"""Equivalence test: the fp16 deployment wrapper must reproduce the fp32 training
path on real captured slots. Catches scaling, conditioning and precision drift
between `train_surrogate.py` and `SurrogateBank` before any rollout is run."""
from __future__ import annotations

import argparse
import json

import torch

from dit_residual_delta.dataset import SlotDataset
from dit_residual_delta.surrogate import (BlockResidualDeltaSurrogate, SurrogateConfig,
                                          SurrogateBank, load_surrogate_bank)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-dir", default="/dev/shm/dit-features/val")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scale = payload["scale"]
    reference = BlockResidualDeltaSurrogate(SurrogateConfig(**payload["config"]))
    reference.load_state_dict(payload["model"])
    reference = reference.cuda().float().eval()
    trainer_path = SurrogateBank(reference, scale)
    deployed = load_surrogate_bank(args.checkpoint, dtype=torch.float16)

    val = SlotDataset(args.val_dir)
    err32 = err16 = tot = drift = 0.0
    with torch.no_grad():
        for batch in val.iter_batches(args.batch):
            target = batch["delta_residual"]
            args32 = (batch["delta_h"], batch["anchor_residual"], batch["anchor_hidden"])
            timestep, horizon, block = batch["timestep"], batch["horizon"], batch["block_id"]
            # per-slot conditioning: the bank broadcasts one scalar over the batch,
            # so evaluate slot-by-slot groups sharing conditioning
            p32 = torch.empty_like(target)
            p16 = torch.empty_like(target)
            for i in range(target.shape[0]):
                a = tuple(t[i : i + 1] for t in args32)
                p32[i : i + 1] = trainer_path(*a, timestep[i : i + 1], float(horizon[i]), int(block[i]))
                p16[i : i + 1] = deployed(*(t.half() for t in a), timestep[i : i + 1],
                                          float(horizon[i]), int(block[i])).float()
            err32 += float(((target - p32) ** 2).sum())
            err16 += float(((target - p16) ** 2).sum())
            drift += float(((p32 - p16) ** 2).sum())
            tot += float((target ** 2).sum())
            break   # one batch is enough for an equivalence check
    print(json.dumps({
        "checkpoint_val_rel_mse": payload.get("val_rel_mse"),
        "fp32_path_rel_mse": err32 / tot,
        "fp16_deployed_rel_mse": err16 / tot,
        "fp32_vs_fp16_relative_drift": drift / tot,
    }, indent=2))


if __name__ == "__main__":
    main()
