"""Whole-stack residual reuse for FLUX, the granularity the PixArt arm validated.

The PixArt arm swept injection granularity K and found only K=1 -- the entire block
stack cached as ONE unit, one correction per step -- survives deployment; per-block
injection turns additive error accumulation into multiplicative and destroys the image
(docs/corrector_report_2026-08-22.md). So this port starts at K=1 rather than the
segment-of-single-blocks design in `runtime.py`.

FLUX's forward makes this cheap. With both block lists empty, the concat/slice
arithmetic around the single-stream loop is an identity:

    h = x_embedder(latent)
    (no dual blocks)                  -> hidden = h, encoder = e
    hidden = cat([e, h])
    (no single blocks)
    hidden = hidden[:, e_len:]        -> exactly h
    norm_out(hidden, temb) -> proj_out

so a reuse step is "empty the lists, add the cached residual before norm_out". That
skips the per-block Python dispatch entirely, which on PixArt was worth 10.5%.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class FluxRuntimeStats:
    transformer_calls: int = 0
    stack_steps: int = 0        # steps that ran the real 57 blocks
    reused_steps: int = 0
    surrogate_calls: int = 0


@dataclass
class _Cache:
    hidden: Tensor | None = None
    residual: Tensor | None = None


class FluxWholeStackRuntime:
    """Context manager: caches the whole 57-block stack's residual on the image stream."""

    def __init__(self, transformer: nn.Module, anchor_steps: set[int],
                 surrogate=None, surrogate_scale: float = 1.0, reuse_observer=None):
        self.transformer = transformer
        self.anchor_steps = set(anchor_steps)
        self.surrogate = surrogate
        self.surrogate_scale = float(surrogate_scale)
        # With an observer set, reuse steps run the REAL stack to obtain the true
        # residual, but propagation still uses the cached value -- so the recorded
        # states are the ones the corrector meets at deployment. Costs a full stack
        # pass per reuse step, i.e. collection runs at roughly exact speed.
        self.reuse_observer = reuse_observer
        self.stats = FluxRuntimeStats()
        self.cache = _Cache()
        self._dual = transformer.transformer_blocks
        self._single = transformer.single_transformer_blocks
        self._empty = nn.ModuleList()
        self._step = -1
        self._anchor_step = 0
        self._h_in: Tensor | None = None
        self._timestep = None
        self._handles: list = []

    # ---- hooks -------------------------------------------------------------
    def _root_pre(self, module, args, kwargs):
        self._step += 1
        self.stats.transformer_calls += 1
        full = self._step in self.anchor_steps or self.cache.residual is None
        if full:
            self._anchor_step = self._step
            self.stats.stack_steps += 1
            self.transformer.transformer_blocks = self._dual
            self.transformer.single_transformer_blocks = self._single
        else:
            self.stats.reused_steps += 1
            if self.reuse_observer is not None:
                # keep the real blocks: we need ground truth at this state
                self.transformer.transformer_blocks = self._dual
                self.transformer.single_transformer_blocks = self._single
            else:
                # empty lists => block loops become no-ops, identity through concat/slice
                self.transformer.transformer_blocks = self._empty
                self.transformer.single_transformer_blocks = self._empty
        self._timestep = kwargs.get("timestep", args[2] if len(args) > 2 else None)
        return None

    def _embed_post(self, module, args, output):
        self._h_in = output
        return None

    def _norm_pre(self, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        full = self._step in self.anchor_steps or self.cache.residual is None
        if full:
            self.cache.hidden = self._h_in.detach()
            self.cache.residual = (hidden - self._h_in).detach()
            return None
        if self.reuse_observer is not None:
            # blocks ran for real, so `hidden` is the true stack output at this state
            true_residual = (hidden - self._h_in).detach()
            self.reuse_observer(
                block_id=0, step=self._step,
                horizon=int(self._step - self._anchor_step),
                timestep=self._timestep,
                hidden=self._h_in.detach(),
                anchor_hidden=self.cache.hidden,
                anchor_residual=self.cache.residual,
                true_residual=true_residual,
            )
            hidden = self._h_in            # propagate the cached branch, not the truth
        delta = 0.0
        if self.surrogate is not None:
            delta = self.surrogate(
                hidden - self.cache.hidden, self.cache.residual, self.cache.hidden,
                self._timestep, int(self._step - self._anchor_step), 0,
            ) * self.surrogate_scale
            self.stats.surrogate_calls += 1
        new_hidden = hidden + self.cache.residual + delta
        if args:
            return (new_hidden,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = new_hidden
        return args, kwargs

    # ---- lifecycle ---------------------------------------------------------
    def __enter__(self):
        t = self.transformer
        self._handles = [
            t.register_forward_pre_hook(self._root_pre, with_kwargs=True),
            t.x_embedder.register_forward_hook(self._embed_post),
            t.norm_out.register_forward_pre_hook(self._norm_pre, with_kwargs=True),
        ]
        self._step = -1
        self.cache = _Cache()
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.transformer.transformer_blocks = self._dual
        self.transformer.single_transformer_blocks = self._single
        return False


def anchor_steps_for(num_steps: int, interval: int) -> set[int]:
    return {s for s in range(num_steps) if s % interval == 0}
