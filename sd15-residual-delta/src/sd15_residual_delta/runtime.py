from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from torch import Tensor, nn


def _sample(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    raise TypeError(f"Unsupported Transformer2DModel output: {type(output)!r}")


def _repack(sample: Tensor, return_dict: bool) -> Any:
    return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


def find_transformer2d_modules(unet: nn.Module) -> list[tuple[str, Transformer2DModel]]:
    return [(name, module) for name, module in unet.named_modules() if isinstance(module, Transformer2DModel)]


def _parent_and_key(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    parent = root
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    return parent, pieces[-1]


def _set_child(parent: nn.Module, key: str, value: nn.Module) -> None:
    if key.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


@dataclass
class ModuleCache:
    hidden: Tensor | None = None
    residual: Tensor | None = None


@dataclass
class RuntimeStats:
    full_steps: int = 0
    reused_steps: int = 0
    exact_module_calls: int = 0
    reused_module_calls: int = 0
    surrogate_module_calls: int = 0
    oracle_module_calls: int = 0
    oracle_refresh_module_calls: int = 0


class _CachedTransformer2D(nn.Module):
    def __init__(self, owner: "SD15AttentionRuntime", name: str, module_id: int, original: nn.Module):
        super().__init__()
        self.owner = owner
        self.name = name
        self.module_id = module_id
        self.original = original

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        return_dict = kwargs.get("return_dict", True)
        cache = self.owner.caches[self.name]
        cache_this_module = (
            self.owner.cached_module_ids is None
            or self.module_id in self.owner.cached_module_ids
        )
        force_full = cache_this_module and (
            cache.hidden is None or cache.hidden.shape != hidden_states.shape
        )
        # Cache this module even though the step itself is exact. Note that
        # `_anchor_step` still advances on a full step, so `current_horizon`
        # reads 0 here; that is harmless for plain reuse (hidden + residual)
        # but would mis-condition a surrogate, so do not combine the two.
        forced_cache = (
            cache_this_module
            and not force_full
            and self.owner.forced_cache_keys is not None
            and (self.owner._step_index, self.module_id)
            in self.owner.forced_cache_keys
        )
        if (self.owner._full_step and not forced_cache) or force_full or not cache_this_module:
            output = self.original(hidden_states, *args, **kwargs)
            if cache_this_module:
                sample = _sample(output)
                cache.hidden = hidden_states.detach()
                cache.residual = sample.detach() - cache.hidden
            self.owner.stats.exact_module_calls += 1
            return output
        assert cache.hidden is not None and cache.residual is not None
        oracle_refresh = (
            self.owner.oracle_refresh_keys is not None
            and (self.owner._step_index, self.module_id)
            in self.owner.oracle_refresh_keys
        )
        if oracle_refresh:
            output = self.original(hidden_states, *args, **kwargs)
            self.owner.stats.exact_module_calls += 1
            self.owner.stats.oracle_refresh_module_calls += 1
            return output
        delta_residual = torch.zeros_like(hidden_states)
        current_horizon = int(self.owner._step_index - self.owner._anchor_step)
        use_surrogate = self.owner.surrogate_bank is not None and (
            self.owner.surrogate_module_ids is None
            or self.module_id in self.owner.surrogate_module_ids
        ) and (
            self.owner.surrogate_module_horizon_keys is None
            or (self.module_id, current_horizon)
            in self.owner.surrogate_module_horizon_keys
        ) and (
            self.owner.surrogate_step_module_keys is None
            or (self.owner._step_index, self.module_id)
            in self.owner.surrogate_step_module_keys
        )
        observe_reuse = self.owner.reuse_observer is not None and (
            self.owner.reuse_observer_keys is None
            or (self.owner._step_index, self.module_id)
            in self.owner.reuse_observer_keys
        )
        if use_surrogate or observe_reuse:
            batch = hidden_states.shape[0]
            timestep = self.owner.current_timestep
            if timestep is None:
                timestep = torch.zeros(batch, device=hidden_states.device)
            timestep = timestep.to(hidden_states.device, torch.float32).flatten()
            if timestep.numel() == 1 and batch > 1:
                timestep = timestep.expand(batch)
            horizon = torch.full_like(timestep, float(current_horizon))
            module_id = torch.full((batch,), self.module_id, device=hidden_states.device, dtype=torch.long)
        if observe_reuse:
            oracle_output = self.original(hidden_states, *args, **kwargs)
            oracle_residual = _sample(oracle_output).detach() - hidden_states.detach()
            self.owner.stats.oracle_module_calls += 1
            self.owner.reuse_observer(
                module_name=self.name,
                module_id=self.module_id,
                step=self.owner._step_index,
                hidden=hidden_states.detach(),
                anchor_hidden=cache.hidden,
                anchor_residual=cache.residual,
                oracle_residual=oracle_residual,
                timestep=timestep.detach(),
                horizon=int(self.owner._step_index - self.owner._anchor_step),
            )
        if use_surrogate:
            delta_h = hidden_states - cache.hidden
            delta_residual = self.owner.surrogate_bank(
                delta_h,
                cache.residual,
                timestep,
                horizon,
                module_id,
            )
            delta_residual = delta_residual * self.owner.surrogate_scale
            self.owner.stats.surrogate_module_calls += 1
            if self.owner.surrogate_observer is not None:
                self.owner.surrogate_observer(
                    module_name=self.name,
                    module_id=self.module_id,
                    step=self.owner._step_index,
                    horizon=int(self.owner._step_index - self.owner._anchor_step),
                    delta_h=delta_h.detach(),
                    anchor_hidden=cache.hidden,
                    anchor_residual=cache.residual,
                    predicted_delta_residual=delta_residual.detach(),
                )
        self.owner.stats.reused_module_calls += 1
        return _repack(hidden_states + cache.residual + delta_residual, return_dict)


class SD15AttentionRuntime:
    """Replace SD1.5 Transformer2D attention calls between exact anchor steps."""

    def __init__(
        self,
        unet: nn.Module,
        cache_interval: int = 3,
        surrogate_bank: nn.Module | None = None,
        surrogate_module_ids: set[int] | None = None,
        surrogate_module_horizon_keys: set[tuple[int, int]] | None = None,
        surrogate_step_module_keys: set[tuple[int, int]] | None = None,
        surrogate_scale: float = 1.0,
        reuse_observer: Any | None = None,
        reuse_observer_keys: set[tuple[int, int]] | None = None,
        surrogate_observer: Any | None = None,
        oracle_refresh_keys: set[tuple[int, int]] | None = None,
        anchor_steps: set[int] | None = None,
        cached_module_ids: set[int] | None = None,
        forced_cache_keys: set[tuple[int, int]] | None = None,
        step_offset: int = 0,
    ) -> None:
        if cache_interval < 1:
            raise ValueError("cache_interval must be at least one")
        self.unet = unet
        self.cache_interval = cache_interval
        self.surrogate_bank = surrogate_bank
        self.surrogate_module_ids = surrogate_module_ids
        self.surrogate_module_horizon_keys = surrogate_module_horizon_keys
        self.surrogate_step_module_keys = surrogate_step_module_keys
        self.surrogate_scale = float(surrogate_scale)
        self.reuse_observer = reuse_observer
        self.reuse_observer_keys = reuse_observer_keys
        self.surrogate_observer = surrogate_observer
        self.oracle_refresh_keys = oracle_refresh_keys
        self.forced_cache_keys = forced_cache_keys
        self.anchor_steps = anchor_steps
        self.cached_module_ids = cached_module_ids
        self.targets = find_transformer2d_modules(unet)
        if not self.targets:
            raise ValueError("No Transformer2DModel modules found")
        self.caches = {name: ModuleCache() for name, _ in self.targets}
        self.stats = RuntimeStats()
        self.current_timestep: Tensor | None = None
        self._step_index = int(step_offset)
        self._anchor_step = -1
        self._full_step = True
        self._handles: list[Any] = []

    def _root_pre_hook(self, module, args, kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        self.current_timestep = timestep.detach() if isinstance(timestep, Tensor) else torch.as_tensor(timestep)
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
        for module_id, (name, original) in enumerate(self.targets):
            parent, key = _parent_and_key(self.unet, name)
            _set_child(parent, key, _CachedTransformer2D(self, name, module_id, original))
        self._handles = [
            self.unet.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True),
            self.unet.register_forward_hook(self._root_post_hook, with_kwargs=True),
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for name, original in self.targets:
            parent, key = _parent_and_key(self.unet, name)
            _set_child(parent, key, original)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
