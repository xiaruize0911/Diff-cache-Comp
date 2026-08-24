"""Block-level residual reuse for DiT backbones (PixArt-Sigma first).

Mirrors `sd15_residual_delta.runtime` so the two backbones are measured with the
same instrument. The cached unit here is one `BasicTransformerBlock` of the DiT
stack -- the DiT analogue of SD1.5's `Transformer2DModel` attention module. Input
and output share a shape, so at denoising step `t`:

    R_t = Block(h_t) - h_t
    h_out_hat = h_t + R_anchor + delta_R_hat

`delta_R_hat` is zero for plain fixed reuse. `oracle_refresh_keys` replaces the
cached residual with the true one at selected (step, block) slots; that is the
upper bound any corrector can reach at those slots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


def taylor_forecast(history: list[tuple[int, Tensor]], step: int, order: int) -> Tensor:
    """Newton-divided-difference forecast of a cached quantity at `step`.

    Order 0 returns the most recent refresh value, i.e. verbatim reuse. Higher
    orders extrapolate the polynomial through the last `order + 1` refresh points,
    which is the mechanism TaylorSeer uses in place of reuse. Divided differences
    are formed in float32: the stored residuals are fp16 and the leading
    differences cancel heavily. Non-uniform refresh spacing is handled by
    construction, so this also works under an adaptive schedule.
    """
    order = min(order, len(history) - 1)
    newest = history[0][1]
    if order <= 0:
        return newest
    steps = [float(history[j][0]) for j in range(order + 1)]
    table = [history[j][1].float() for j in range(order + 1)]
    out = table[0].clone()
    for level in range(1, order + 1):
        table = [
            (table[j] - table[j + 1]) / (steps[j] - steps[j + level])
            for j in range(len(table) - 1)
        ]
        basis = 1.0
        for j in range(level):
            basis *= step - steps[j]
        out += table[0] * basis
    return out.to(newest.dtype)


def find_transformer_blocks(transformer: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return the ordered `[(name, block)]` of the DiT block stack."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("Model has no `transformer_blocks` stack")
    return [(f"transformer_blocks.{i}", block) for i, block in enumerate(blocks)]


@dataclass
class BlockCache:
    hidden: Tensor | None = None
    residual: Tensor | None = None
    # (step, residual) at past refresh points, newest first. Only populated when a
    # Taylor forecast is requested; plain reuse needs nothing but `residual`.
    history: list[tuple[int, Tensor]] = field(default_factory=list)


@dataclass
class RuntimeStats:
    full_steps: int = 0
    reused_steps: int = 0
    exact_block_calls: int = 0
    reused_block_calls: int = 0
    surrogate_block_calls: int = 0
    oracle_refresh_block_calls: int = 0
    observer_block_calls: int = 0


class _CachedBlock(nn.Module):
    def __init__(self, owner: "DiTBlockRuntime", name: str, block_id: int, original: nn.Module):
        super().__init__()
        self.owner = owner
        self.name = name
        self.block_id = block_id
        self.original = original

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        owner = self.owner
        cache = owner.caches[self.name]
        cache_this_block = owner.cached_block_ids is None or self.block_id in owner.cached_block_ids
        force_full = cache_this_block and (
            cache.hidden is None or cache.hidden.shape != hidden_states.shape
        )
        # Cache at a slot that the schedule calls exact (speed up, quality down).
        forced_cache = (
            cache_this_block
            and not force_full
            and owner.forced_cache_keys is not None
            and (owner._step_index, self.block_id) in owner.forced_cache_keys
        )
        if (owner._full_step and not forced_cache) or force_full or not cache_this_block:
            output = self.original(hidden_states, *args, **kwargs)
            if cache_this_block:
                cache.hidden = hidden_states.detach()
                cache.residual = output.detach() - cache.hidden
                record_refresh(owner, cache, owner._step_index)
            owner.stats.exact_block_calls += 1
            return output

        assert cache.hidden is not None and cache.residual is not None
        oracle_refresh = (
            owner.oracle_refresh_keys is not None
            and (owner._step_index, self.block_id) in owner.oracle_refresh_keys
        )
        if oracle_refresh:
            output = self.original(hidden_states, *args, **kwargs)
            owner.stats.exact_block_calls += 1
            owner.stats.oracle_refresh_block_calls += 1
            if owner.oracle_blend >= 1.0:
                return output
            # Partial oracle: a corrector that recovers a fraction `blend` of the
            # true residual delta. blend=0 is plain reuse, blend=1 is exact. This
            # measures how accurate a corrector must be, independent of how one
            # would be built. It is a fidelity probe only -- it runs the real
            # block, so it makes no speed claim.
            true_residual = output - hidden_states
            blended = cache.residual + owner.oracle_blend * (true_residual - cache.residual)
            return hidden_states + blended

        if owner.reuse_observer is not None and (
            owner.reuse_observer_keys is None
            or (owner._step_index, self.block_id) in owner.reuse_observer_keys
        ):
            # On-policy capture: the true residual at the *cached* trajectory's
            # state. Propagation below still uses the cache, so `hidden_states`
            # stays on the distribution the corrector will actually see.
            true_residual = self.original(hidden_states, *args, **kwargs).detach() - hidden_states.detach()
            owner.stats.observer_block_calls += 1
            owner.reuse_observer(
                block_id=self.block_id,
                step=owner._step_index,
                horizon=int(owner._step_index - owner._anchor_step),
                timestep=owner.current_timestep,
                hidden=hidden_states.detach(),
                anchor_hidden=cache.hidden,
                anchor_residual=cache.residual,
                true_residual=true_residual,
            )

        delta_residual = torch.zeros_like(hidden_states)
        if owner.surrogate_bank is not None and (
            owner.surrogate_block_ids is None or self.block_id in owner.surrogate_block_ids
        ):
            horizon = int(owner._step_index - owner._anchor_step)
            delta_residual = owner.surrogate_bank(
                hidden_states - cache.hidden,
                cache.residual,
                cache.hidden,
                owner.current_timestep,
                horizon,
                self.block_id,
            ) * owner.surrogate_scale
            owner.stats.surrogate_block_calls += 1

        owner.stats.reused_block_calls += 1
        return hidden_states + cached_base(owner, cache, owner._step_index) + delta_residual


class DiTBlockRuntime:
    """Reuse DiT block residuals between exact anchor steps."""

    def __init__(
        self,
        transformer: nn.Module,
        cache_interval: int = 3,
        anchor_steps: set[int] | None = None,
        cached_block_ids: set[int] | None = None,
        oracle_refresh_keys: set[tuple[int, int]] | None = None,
        oracle_blend: float = 1.0,
        forced_cache_keys: set[tuple[int, int]] | None = None,
        surrogate_bank: nn.Module | None = None,
        surrogate_block_ids: set[int] | None = None,
        surrogate_scale: float = 1.0,
        reuse_observer: Any | None = None,
        reuse_observer_keys: set[tuple[int, int]] | None = None,
        step_offset: int = 0,
        taylor_order: int = 0,
    ) -> None:
        if cache_interval < 1:
            raise ValueError("cache_interval must be at least one")
        self.transformer = transformer
        self.cache_interval = cache_interval
        self.anchor_steps = anchor_steps
        self.cached_block_ids = cached_block_ids
        self.oracle_refresh_keys = oracle_refresh_keys
        self.oracle_blend = float(oracle_blend)
        self.forced_cache_keys = forced_cache_keys
        self.surrogate_bank = surrogate_bank
        self.surrogate_block_ids = surrogate_block_ids
        self.surrogate_scale = float(surrogate_scale)
        self.reuse_observer = reuse_observer
        self.reuse_observer_keys = reuse_observer_keys
        self.taylor_order = int(taylor_order)
        self.targets = find_transformer_blocks(transformer)
        self.caches = {name: BlockCache() for name, _ in self.targets}
        self.stats = RuntimeStats()
        self.current_timestep: Tensor | None = None
        self._step_index = int(step_offset)
        self._anchor_step = -1
        self._full_step = True
        self._handles: list[Any] = []

    @property
    def num_blocks(self) -> int:
        return len(self.targets)

    def _root_pre_hook(self, module, args, kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        if isinstance(timestep, Tensor):
            self.current_timestep = timestep.detach()
        elif timestep is not None:
            self.current_timestep = torch.as_tensor(timestep)
        if self.anchor_steps is None:
            self._full_step = (
                self._anchor_step < 0
                or self._step_index - self._anchor_step >= self.cache_interval
            )
        else:
            self._full_step = self._anchor_step < 0 or self._step_index in self.anchor_steps

    def _root_post_hook(self, module, args, kwargs, output):
        if self._full_step:
            self._anchor_step = self._step_index
            self.stats.full_steps += 1
        else:
            self.stats.reused_steps += 1
        self._step_index += 1

    def __enter__(self):
        stack = self.transformer.transformer_blocks
        for block_id, (name, original) in enumerate(self.targets):
            stack[block_id] = _CachedBlock(self, name, block_id, original)
        self._handles = [
            self.transformer.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True),
            self.transformer.register_forward_hook(self._root_post_hook, with_kwargs=True),
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        stack = self.transformer.transformer_blocks
        for block_id, (_, original) in enumerate(self.targets):
            stack[block_id] = original
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def uniform_segments(num_blocks: int, count: int) -> list[tuple[int, int]]:
    """Split `num_blocks` into `count` contiguous, near-equal segments."""
    edges = [round(i * num_blocks / count) for i in range(count + 1)]
    return [(edges[i], edges[i + 1]) for i in range(count)]


class _SegmentPassthrough(nn.Module):
    """Identity stand-in for blocks the segment head already accounted for."""

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        return hidden_states


class _SegmentHead(nn.Module):
    def __init__(self, owner: "DiTSegmentRuntime", blocks: list[nn.Module], segment_id: int = 0):
        super().__init__()
        self.owner = owner
        self.blocks = nn.ModuleList(blocks)
        self.segment_id = segment_id

    def _run(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states, *args, **kwargs)
        return hidden_states

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        owner = self.owner
        cache = owner.caches[self.segment_id]
        force_full = cache.hidden is None or cache.hidden.shape != hidden_states.shape
        if owner._full_step or force_full:
            output = self._run(hidden_states, *args, **kwargs)
            cache.hidden = hidden_states.detach()
            cache.residual = output.detach() - cache.hidden
            record_refresh(owner, cache, owner._step_index)
            owner.stats.exact_block_calls += len(self.blocks)
            return output

        if owner.oracle_blend is not None:
            true_output = self._run(hidden_states, *args, **kwargs)
            owner.stats.oracle_refresh_block_calls += 1
            if owner.oracle_blend >= 1.0:
                return true_output
            true_residual = true_output - hidden_states
            blended = cache.residual + owner.oracle_blend * (true_residual - cache.residual)
            return hidden_states + blended

        if owner.reuse_observer is not None:
            true_residual = self._run(hidden_states, *args, **kwargs).detach() - hidden_states.detach()
            owner.stats.observer_block_calls += 1
            owner.reuse_observer(
                block_id=self.segment_id,
                step=owner._step_index,
                horizon=int(owner._step_index - owner._anchor_step),
                timestep=owner.current_timestep,
                hidden=hidden_states.detach(),
                anchor_hidden=cache.hidden,
                anchor_residual=cache.residual,
                true_residual=true_residual,
            )

        delta_residual = torch.zeros_like(hidden_states)
        if owner.surrogate_bank is not None:
            delta_residual = owner.surrogate_bank(
                hidden_states - cache.hidden,
                cache.residual,
                cache.hidden,
                owner.current_timestep,
                int(owner._step_index - owner._anchor_step),
                self.segment_id,
            ) * owner.surrogate_scale
            owner.stats.surrogate_block_calls += 1
        owner.stats.reused_block_calls += 1
        return hidden_states + cached_base(owner, cache, owner._step_index) + delta_residual


class DiTSegmentRuntime:
    """Cache a contiguous run of DiT blocks as ONE unit: one correction per step.

    The per-block runtime injects a correction 28 times per reuse step, and the
    closed-loop diagnosis shows why that fails: with fixed cache the per-block
    error is state-independent, so deviations accumulate additively, but a
    *working* corrector makes each block's output track the true block, which
    turns the error recursion into delta_{b+1} = (I + J_b) delta_b + eps_b --
    multiplicative through the depth of the stack. Treating the segment as one
    unit leaves a single injection point per step and removes that depth-wise
    amplification entirely.
    """

    def __init__(
        self,
        transformer: nn.Module,
        segment: tuple[int, int] | None = None,
        segments: list[tuple[int, int]] | None = None,
        anchor_steps: set[int] | None = None,
        cache_interval: int = 3,
        surrogate_bank: nn.Module | None = None,
        surrogate_scale: float = 1.0,
        reuse_observer: Any | None = None,
        oracle_blend: float | None = None,
        step_offset: int = 0,
        adaptive_threshold: float | None = None,
        taylor_order: int = 0,
    ) -> None:
        self.transformer = transformer
        blocks = list(transformer.transformer_blocks)
        if segments is not None:
            self.segments = [(int(a), int(b)) for a, b in segments]
        elif segment is not None:
            self.segments = [(int(segment[0]), int(segment[1]))]
        else:
            self.segments = [(0, len(blocks))]
        previous_end = -1
        for start, end in self.segments:
            if not (0 <= start < end <= len(blocks)) or start < previous_end:
                raise ValueError(f"bad segments {self.segments} for {len(blocks)} blocks")
            previous_end = end
        self.segment = self.segments[0]
        self.cache_interval = cache_interval
        self.anchor_steps = anchor_steps
        self.surrogate_bank = surrogate_bank
        self.surrogate_scale = float(surrogate_scale)
        self.reuse_observer = reuse_observer
        self.oracle_blend = oracle_blend
        self.taylor_order = int(taylor_order)
        self.caches = [BlockCache() for _ in self.segments]
        self.stats = RuntimeStats()
        self.current_timestep: Tensor | None = None
        self._step_index = int(step_offset)
        self._anchor_step = -1
        self._full_step = True
        self._handles: list[Any] = []
        self._originals = blocks
        # TeaCache-style adaptive scheduling: refresh when the accumulated relative
        # change of the transformer input since the last refresh crosses a threshold,
        # instead of on a fixed interval. This reimplements the *mechanism* of
        # timestep-aware adaptive caching, not the published method: we do not have
        # their model-specific fitted rescaling polynomial, so the threshold is simply
        # calibrated to match a target number of refreshes per image.
        self.adaptive_threshold = adaptive_threshold
        self._adaptive_ref: Tensor | None = None
        self._adaptive_accum = 0.0

    _root_post_hook = DiTBlockRuntime._root_post_hook

    def _root_pre_hook(self, module, args, kwargs):
        DiTBlockRuntime._root_pre_hook(self, module, args, kwargs)
        if self.adaptive_threshold is None:
            return
        x = kwargs.get("hidden_states")
        if x is None and args:
            x = args[0]
        if not isinstance(x, Tensor):
            self._full_step = True
            return
        cur = x.detach().float()
        if self._adaptive_ref is None:
            self._full_step, self._adaptive_accum = True, 0.0
        else:
            denom = self._adaptive_ref.abs().mean().clamp_min(1e-8)
            self._adaptive_accum += float((cur - self._adaptive_ref).abs().mean() / denom)
            if self._adaptive_accum >= self.adaptive_threshold:
                self._full_step, self._adaptive_accum = True, 0.0
            else:
                self._full_step = False
        self._adaptive_ref = cur

    def __enter__(self):
        stack = self.transformer.transformer_blocks
        for segment_id, (start, end) in enumerate(self.segments):
            stack[start] = _SegmentHead(self, self._originals[start:end], segment_id)
            for i in range(start + 1, end):
                stack[i] = _SegmentPassthrough()
        self._handles = [
            self.transformer.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True),
            self.transformer.register_forward_hook(self._root_post_hook, with_kwargs=True),
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        stack = self.transformer.transformer_blocks
        for i, original in enumerate(self._originals):
            stack[i] = original
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def record_refresh(owner, cache: BlockCache, step: int) -> None:
    """Push this refresh onto the cache history, keeping only what the order needs."""
    order = getattr(owner, "taylor_order", 0)
    if order <= 0:
        return
    cache.history.insert(0, (int(step), cache.residual))
    del cache.history[order + 1:]


def cached_base(owner, cache: BlockCache, step: int) -> Tensor:
    """The quantity reuse injects: verbatim cache, or its Taylor forecast."""
    order = getattr(owner, "taylor_order", 0)
    if order <= 0 or not cache.history:
        return cache.residual
    return taylor_forecast(cache.history, step, order)


# ---------------------------------------------------------------------------
# FastCache (arXiv 2505.20353), reimplemented as a mechanism.
#
# Two things the paper leaves under-specified, and what we do about them:
#
# 1. It calls W_l, b_l "learnable" but gives no fitting procedure. We solve them
#    in closed form by ridge least squares on captured per-block slots, which is
#    the most favourable instantiation available -- a closed-form optimum on
#    on-policy data, not a weakly-trained head.
# 2. Its skip test, delta^2 <= chi2_{ND,1-alpha}/(ND), is vacuous at these
#    dimensions: ND = 1024*1152 puts the bound at delta <= 1.0011, while the
#    observed relative change is far below 1, so every block at every step would
#    be skipped. Rather than implement something degenerate we treat the bound as
#    a free threshold and calibrate it on validation to a target compute budget,
#    exactly as we do for the adaptive-schedule baseline.
#
# Not implemented: the spatial token-selection / token-merging module. This
# runtime decides at block granularity over all tokens, which is the granularity
# our other families use, so the comparison stays on one axis.
# ---------------------------------------------------------------------------


class _FastCacheBlock(nn.Module):
    def __init__(self, owner: "FastCacheRuntime", name: str, block_id: int, original: nn.Module):
        super().__init__()
        self.owner = owner
        self.name = name
        self.block_id = block_id
        self.original = original

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        owner = self.owner
        previous = owner.previous_input.get(self.name)
        skip = False
        if previous is not None and previous[0].shape == hidden_states.shape:
            # One fp32-accumulated norm over the difference, no cast copies and no
            # recomputation of the reference norm: the delta test runs 28x per step,
            # so an allocation here shows up directly in wall clock.
            reference, reference_norm = previous
            difference = torch.linalg.vector_norm(
                hidden_states.sub(reference), dtype=torch.float32)
            delta = float(difference / reference_norm)
            owner.deltas.append(delta)
            skip = delta <= owner.threshold and self.block_id in owner.linear_maps
        owner.previous_input[self.name] = (
            hidden_states.detach(),
            torch.linalg.vector_norm(hidden_states.detach(), dtype=torch.float32).clamp_min(1e-12),
        )
        if not skip:
            owner.stats.exact_block_calls += 1
            return self.original(hidden_states, *args, **kwargs)
        weight, bias = owner.linear_maps[self.block_id]
        residual = torch.addmm(bias, hidden_states.reshape(-1, hidden_states.shape[-1]), weight)
        owner.stats.reused_block_calls += 1
        return hidden_states + residual.reshape(hidden_states.shape)


class FastCacheRuntime:
    """Per-block, per-step adaptive skipping with a linear stand-in on skip.

    No anchor schedule: the decision is taken independently for every (step,
    block) from the relative change of that block's own input since the previous
    step. `linear_maps[b] = (A_b, bias_b)` predicts the block's residual from its
    current input, so the skipped output is `h + A_b h + bias_b` -- FastCache's
    `W_l h + b_l` with `W_l = I + A_b`.
    """

    def __init__(self, transformer: nn.Module, linear_maps: dict[int, tuple[Tensor, Tensor]],
                 threshold: float, step_offset: int = 0) -> None:
        self.transformer = transformer
        self.linear_maps = linear_maps
        self.threshold = float(threshold)
        self.targets = find_transformer_blocks(transformer)
        self.previous_input: dict[str, tuple[Tensor, Tensor]] = {}
        self.deltas: list[float] = []
        self.stats = RuntimeStats()
        self.current_timestep: Tensor | None = None
        self._step_index = int(step_offset)
        self._full_step = True
        self._handles: list[Any] = []

    def _root_pre_hook(self, module, args, kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        if isinstance(timestep, Tensor):
            self.current_timestep = timestep.detach()

    def _root_post_hook(self, module, args, kwargs, output):
        self.stats.full_steps += 1
        self._step_index += 1

    def __enter__(self):
        stack = self.transformer.transformer_blocks
        for block_id, (name, original) in enumerate(self.targets):
            stack[block_id] = _FastCacheBlock(self, name, block_id, original)
        self._handles = [
            self.transformer.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True),
            self.transformer.register_forward_hook(self._root_post_hook, with_kwargs=True),
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        stack = self.transformer.transformer_blocks
        for block_id, (_, original) in enumerate(self.targets):
            stack[block_id] = original
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.previous_input.clear()


def load_fastcache_maps(path: str, device="cuda", dtype=torch.float16):
    """Load the ridge-fitted per-block maps written by fit_fastcache_maps.py."""
    payload = torch.load(path, map_location="cpu")
    maps = {}
    for key, entry in payload["maps"].items():
        weight = entry["weight"].to(device=device, dtype=dtype)
        bias = entry["bias"].to(device=device, dtype=dtype)
        maps[int(key)] = (weight, bias)
    return maps
