import torch
from torch import Tensor


def relative_mse(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    error = (prediction.float() - target.float()).square().flatten(1).mean(1)
    scale = target.float().square().flatten(1).mean(1).clamp_min(eps)
    return error / scale


def cosine_similarity(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(
        prediction.float().flatten(1), target.float().flatten(1), dim=1
    )
