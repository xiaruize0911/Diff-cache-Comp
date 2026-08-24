from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .runtime import find_transformer2d_modules


def _sample(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    raise TypeError(f"Unsupported output type: {type(output)!r}")


@dataclass
class ModuleStepFeature:
    module_name: str
    module_id: int
    step: int
    timestep: float
    hidden: Tensor
    residual: Tensor


class SD15FeatureCapture:
    """Capture selected exact attention-module inputs and residuals."""

    def __init__(self, unet: nn.Module, selected_steps: set[int]) -> None:
        self.unet = unet
        self.targets = find_transformer2d_modules(unet)
        self.selected_steps = selected_steps
        self.features: dict[str, dict[int, ModuleStepFeature]] = {
            name: {} for name, _ in self.targets
        }
        self._inputs: dict[str, Tensor] = {}
        self._step = 0
        self._timestep = float("nan")
        self._handles: list[Any] = []

    def _root_pre(self, module, args, kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        if isinstance(timestep, Tensor):
            self._timestep = float(timestep.detach().flatten()[0].cpu())
        elif timestep is not None:
            self._timestep = float(timestep)

    def _root_post(self, module, args, kwargs, output):
        self._step += 1

    def _module_pre(self, name: str):
        def hook(module, args, kwargs):
            if self._step not in self.selected_steps:
                return
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0]
            self._inputs[name] = hidden.detach().to("cpu", torch.float16, copy=True)

        return hook

    def _module_post(self, name: str, module_id: int):
        def hook(module, args, kwargs, output):
            if self._step not in self.selected_steps:
                return
            hidden = self._inputs.pop(name)
            sample = _sample(output).detach().to("cpu", torch.float16, copy=True)
            residual = (sample.float() - hidden.float()).to(torch.float16)
            self.features[name][self._step] = ModuleStepFeature(
                module_name=name,
                module_id=module_id,
                step=self._step,
                timestep=self._timestep,
                hidden=hidden,
                residual=residual,
            )

        return hook

    def __enter__(self):
        self._handles = [
            self.unet.register_forward_pre_hook(self._root_pre, with_kwargs=True),
            self.unet.register_forward_hook(self._root_post, with_kwargs=True),
        ]
        for module_id, (name, module) in enumerate(self.targets):
            self._handles.append(
                module.register_forward_pre_hook(self._module_pre(name), with_kwargs=True)
            )
            self._handles.append(
                module.register_forward_hook(self._module_post(name, module_id), with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
