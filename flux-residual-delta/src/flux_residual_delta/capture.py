from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def _first_hidden_tensor(value: Any, hidden_dim: int) -> Tensor:
    if isinstance(value, Tensor) and value.ndim == 3 and value.shape[-1] == hidden_dim:
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_hidden_tensor(item, hidden_dim)
            except LookupError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_hidden_tensor(item, hidden_dim)
            except LookupError:
                pass
    raise LookupError(f"No [batch, tokens, {hidden_dim}] tensor found")


@dataclass
class StepFeature:
    call_index: int
    timestep: float
    hidden: Tensor
    residual: Tensor


class FluxSegmentCapture:
    """Capture exact input/residual pairs for a contiguous single-block segment."""

    def __init__(self, transformer: nn.Module, start: int, length: int, hidden_dim: int = 3072):
        blocks = transformer.single_transformer_blocks
        if start < 0 or length <= 0 or start + length > len(blocks):
            raise ValueError(f"Invalid segment [{start}, {start + length}) for {len(blocks)} blocks")
        self.transformer = transformer
        self.start = start
        self.length = length
        self.hidden_dim = hidden_dim
        self.features: list[StepFeature] = []
        self._input: Tensor | None = None
        self._timestep = float("nan")
        self._handles: list[Any] = []

    def _root_pre_hook(self, module, args, kwargs):
        value = kwargs.get("timestep")
        if isinstance(value, Tensor):
            self._timestep = float(value.detach().flatten()[0].cpu())

    def _segment_pre_hook(self, module, args, kwargs):
        source = kwargs if kwargs else args
        hidden = _first_hidden_tensor(source, self.hidden_dim)
        self._input = hidden.detach().to(device="cpu", dtype=torch.bfloat16, copy=True)

    def _segment_post_hook(self, module, args, kwargs, output):
        if self._input is None:
            raise RuntimeError("Segment output observed without segment input")
        hidden_out = _first_hidden_tensor(output, self.hidden_dim)
        hidden_out = hidden_out.detach().to(device="cpu", dtype=torch.bfloat16, copy=True)
        residual = hidden_out.float().sub(self._input.float()).to(torch.bfloat16)
        self.features.append(
            StepFeature(len(self.features), self._timestep, self._input, residual)
        )
        self._input = None

    def __enter__(self):
        blocks = self.transformer.single_transformer_blocks
        self._handles = [
            self.transformer.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True),
            blocks[self.start].register_forward_pre_hook(self._segment_pre_hook, with_kwargs=True),
            blocks[self.start + self.length - 1].register_forward_hook(
                self._segment_post_hook, with_kwargs=True
            ),
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
