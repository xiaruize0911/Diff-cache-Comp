from __future__ import annotations

import torch
from torch import Tensor


def relative_mse(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=(-2, -1))
    scale = target.float().square().mean(dim=(-2, -1)).clamp_min(eps)
    return error / scale


def cosine_similarity(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    return torch.nn.functional.cosine_similarity(prediction, target, dim=1, eps=eps)
