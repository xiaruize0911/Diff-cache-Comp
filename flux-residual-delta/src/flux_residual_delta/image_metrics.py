from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor


def image_tensor(image) -> Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda")


def psnr(candidate: Tensor, reference: Tensor) -> float:
    mse = functional.mse_loss(candidate, reference).item()
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def skimage_ssim(candidate_image, reference_image, gaussian: bool = True) -> float:
    from skimage.metrics import structural_similarity

    a = np.asarray(candidate_image).astype(np.float64) / 255.0
    b = np.asarray(reference_image).astype(np.float64) / 255.0
    return float(
        structural_similarity(
            b, a, channel_axis=2, data_range=1.0,
            gaussian_weights=gaussian, sigma=1.5, use_sample_covariance=not gaussian,
        )
    )


def lpips_alex(model, candidate: Tensor, reference: Tensor) -> float:
    with torch.no_grad():
        return float(model(candidate * 2 - 1, reference * 2 - 1).item())


def numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }
