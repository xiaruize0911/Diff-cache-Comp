#!/usr/bin/env python
from __future__ import annotations

import torch

from flux_residual_delta.metrics import relative_mse
from flux_residual_delta.surrogate import ResidualDeltaSurrogate, SurrogateConfig


def main() -> None:
    cfg = SurrogateConfig(hidden_dim=64, width=32, depth=1, heads=4, conditioning_dim=32)
    model = ResidualDeltaSurrogate(cfg)
    delta_h = torch.randn(2, 16, 64)
    anchor = torch.randn_like(delta_h)
    target = torch.randn_like(delta_h)
    timestep = torch.tensor([0.8, 0.4])
    horizon = torch.tensor([2.0, 4.0])
    prediction = model(delta_h, anchor, timestep, horizon)
    loss = relative_mse(prediction, target).mean()
    loss.backward()
    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
    print(f"smoke test passed: shape={tuple(prediction.shape)}, loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
