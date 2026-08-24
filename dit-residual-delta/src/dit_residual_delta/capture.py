"""On-policy residual-delta capture.

At every reuse slot of a cached rollout, the observer runs the real block once to
get the true residual, while the rollout keeps propagating the cached value. The
recorded states are therefore exactly the states the corrector meets at
deployment -- the exposure-bias trap that the SD1.5 arm hit with teacher-forced
capture is avoided by construction.

Tokens are subsampled per slot. A token-wise corrector treats tokens as i.i.d.
samples, so 32-64 tokens from each of the 2x1024 CFG token positions is a cheap,
unbiased sample; a full slot would be 4.7 MB and 448 slots per image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class SlotRecorder:
    tokens_per_slot: int = 48
    generator: torch.Generator | None = None
    delta_h: list[Tensor] = field(default_factory=list)
    anchor_hidden: list[Tensor] = field(default_factory=list)
    anchor_residual: list[Tensor] = field(default_factory=list)
    delta_residual: list[Tensor] = field(default_factory=list)
    block_id: list[int] = field(default_factory=list)
    step: list[int] = field(default_factory=list)
    horizon: list[int] = field(default_factory=list)
    timestep: list[float] = field(default_factory=list)

    def __call__(self, *, block_id, step, horizon, timestep, hidden,
                 anchor_hidden, anchor_residual, true_residual) -> None:
        flat_delta_h = (hidden - anchor_hidden).reshape(-1, hidden.shape[-1])
        flat_anchor_h = anchor_hidden.reshape(-1, anchor_hidden.shape[-1])
        flat_anchor = anchor_residual.reshape(-1, anchor_residual.shape[-1])
        flat_delta_r = (true_residual - anchor_residual).reshape(-1, true_residual.shape[-1])
        count = min(self.tokens_per_slot, flat_delta_h.shape[0])
        index = torch.randperm(flat_delta_h.shape[0], device=flat_delta_h.device,
                               generator=self.generator)[:count]
        self.delta_h.append(flat_delta_h[index].to("cpu", torch.float16))
        self.anchor_hidden.append(flat_anchor_h[index].to("cpu", torch.float16))
        self.anchor_residual.append(flat_anchor[index].to("cpu", torch.float16))
        self.delta_residual.append(flat_delta_r[index].to("cpu", torch.float16))
        self.block_id.append(int(block_id))
        self.step.append(int(step))
        self.horizon.append(int(horizon))
        value = timestep
        if isinstance(value, Tensor):
            value = float(value.flatten()[0].item())
        self.timestep.append(float(value if value is not None else 0.0))

    def to_payload(self) -> dict[str, Tensor]:
        return {
            "delta_h": torch.stack(self.delta_h),                 # (slots, tokens, dim)
            "anchor_hidden": torch.stack(self.anchor_hidden),
            "anchor_residual": torch.stack(self.anchor_residual),
            "delta_residual": torch.stack(self.delta_residual),
            "block_id": torch.tensor(self.block_id, dtype=torch.int16),
            "step": torch.tensor(self.step, dtype=torch.int16),
            "horizon": torch.tensor(self.horizon, dtype=torch.int16),
            "timestep": torch.tensor(self.timestep, dtype=torch.float32),
        }

    def save(self, path: str | Path) -> dict[str, int]:
        payload = self.to_payload()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return {"slots": payload["delta_h"].shape[0],
                "tokens": payload["delta_h"].shape[1],
                "dim": payload["delta_h"].shape[2]}

    def reset(self) -> None:
        self.delta_h.clear(); self.anchor_hidden.clear(); self.anchor_residual.clear()
        self.delta_residual.clear()
        self.block_id.clear(); self.step.clear(); self.horizon.clear(); self.timestep.clear()
