from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class RuntimeStats:
    full_steps: int = 0
    reused_steps: int = 0


class _StartBlock(nn.Module):
    def __init__(self, owner: "FluxSegmentRuntime", original: nn.Module, is_last: bool) -> None:
        super().__init__()
        self.owner = owner
        self.original = original
        self.is_last = is_last

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ) -> Tensor:
        if self.owner.should_run_full(hidden_states):
            self.owner._anchor_input_candidate = hidden_states.detach()
            output = self.original(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if self.is_last:
                self.owner.finish_anchor(output)
            return output
        return self.owner.approximate(hidden_states)


class _FollowingBlock(nn.Module):
    def __init__(self, owner: "FluxSegmentRuntime", original: nn.Module, is_last: bool) -> None:
        super().__init__()
        self.owner = owner
        self.original = original
        self.is_last = is_last

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ) -> Tensor:
        if not self.owner._full_step:
            return hidden_states
        output = self.original(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if self.is_last:
            self.owner.finish_anchor(output)
        return output


class FluxSegmentRuntime:
    """Temporarily replace a FLUX single-block segment during denoising.

    Every ``cache_interval`` calls the exact segment is executed and its input
    and residual become the new anchor. Intermediate calls execute either fixed
    residual reuse (``surrogate=None``) or the learned residual-delta model.
    All original modules are restored on context exit.
    """

    def __init__(
        self,
        transformer: nn.Module,
        start: int,
        length: int,
        cache_interval: int,
        surrogate: nn.Module | None = None,
    ) -> None:
        if cache_interval < 1:
            raise ValueError("cache_interval must be at least one")
        blocks = transformer.single_transformer_blocks
        if start < 0 or length <= 0 or start + length > len(blocks):
            raise ValueError(f"Invalid segment [{start}, {start + length}) for {len(blocks)} blocks")
        self.transformer = transformer
        self.start = start
        self.length = length
        self.cache_interval = cache_interval
        self.surrogate = surrogate
        self.stats = RuntimeStats()
        self._originals: list[nn.Module] = []
        self._handles: list[Any] = []
        self._step_index = 0
        self._anchor_step = -1
        self._current_timestep: Tensor | None = None
        self._full_step = True
        self._anchor_input_candidate: Tensor | None = None
        self._anchor_hidden: Tensor | None = None
        self._anchor_residual: Tensor | None = None

    def _root_pre_hook(self, module, args, kwargs):
        timestep = kwargs.get("timestep")
        self._current_timestep = timestep.detach() if isinstance(timestep, Tensor) else None
        self._full_step = (
            self._anchor_hidden is None
            or self._step_index - self._anchor_step >= self.cache_interval
        )

    def _root_post_hook(self, module, args, kwargs, output):
        if self._full_step:
            self.stats.full_steps += 1
        else:
            self.stats.reused_steps += 1
        self._step_index += 1

    def should_run_full(self, hidden_states: Tensor) -> bool:
        if self._anchor_hidden is not None and self._anchor_hidden.shape != hidden_states.shape:
            self._full_step = True
        return self._full_step

    def finish_anchor(self, output: Tensor) -> None:
        if self._anchor_input_candidate is None:
            raise RuntimeError("Anchor segment ended without a captured input")
        self._anchor_hidden = self._anchor_input_candidate.detach()
        self._anchor_residual = output.detach() - self._anchor_hidden
        self._anchor_step = self._step_index
        self._anchor_input_candidate = None

    def approximate(self, hidden_states: Tensor) -> Tensor:
        if self._anchor_hidden is None or self._anchor_residual is None:
            raise RuntimeError("Cannot reuse before an exact anchor step")
        delta = torch.zeros_like(hidden_states)
        if self.surrogate is not None:
            timestep = self._current_timestep
            if timestep is None:
                timestep = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
            timestep = timestep.to(device=hidden_states.device, dtype=torch.float32).flatten()
            if timestep.numel() == 1 and hidden_states.shape[0] > 1:
                timestep = timestep.expand(hidden_states.shape[0])
            horizon = torch.full_like(timestep, float(self._step_index - self._anchor_step))
            delta = self.surrogate(
                hidden_states - self._anchor_hidden,
                self._anchor_residual,
                timestep,
                horizon,
            )
        return hidden_states + self._anchor_residual + delta

    def reset(self) -> None:
        self.stats = RuntimeStats()
        self._step_index = 0
        self._anchor_step = -1
        self._current_timestep = None
        self._full_step = True
        self._anchor_input_candidate = None
        self._anchor_hidden = None
        self._anchor_residual = None

    def __enter__(self):
        self.reset()
        blocks = self.transformer.single_transformer_blocks
        self._originals = list(blocks[self.start : self.start + self.length])
        for offset, original in enumerate(self._originals):
            is_last = offset == self.length - 1
            wrapper = (
                _StartBlock(self, original, is_last)
                if offset == 0
                else _FollowingBlock(self, original, is_last)
            )
            blocks[self.start + offset] = wrapper
        self._handles = [
            self.transformer.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True),
            self.transformer.register_forward_hook(self._root_post_hook, with_kwargs=True),
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        blocks = self.transformer.single_transformer_blocks
        for offset, original in enumerate(self._originals):
            blocks[self.start + offset] = original
        for handle in self._handles:
            handle.remove()
        self._originals.clear()
        self._handles.clear()
