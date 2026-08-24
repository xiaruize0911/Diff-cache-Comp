"""Correctness of the distributional metrics we report on COCO.

The FID case is the important one: at the sample size we can afford (~2000 images) the
2048-dim Inception covariance is rank-deficient and FID between two draws from the SAME
distribution reads in the hundreds. That is why the paper reports KID -- whose estimator
is unbiased -- as the primary distributional statistic and uses FID only comparatively
against a shared reference.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest


def _load_metrics():
    """Load kid()/frechet() from the eval script without its GPU-only imports."""
    src = open("scripts/eval_fid_coco.py").read()
    from scipy import linalg
    ns = {"np": np, "linalg": linalg}
    exec(src[src.index("def kid("): src.index("class _Feats")], ns)
    return ns["kid"], ns["frechet"]


kid, frechet = _load_metrics()
D = 2048


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(0)
    return (rng.normal(size=(2000, D)), rng.normal(size=(2000, D)))


def test_kid_is_unbiased_on_identical_distributions(samples):
    a, b = samples
    value, subset_sd = kid(a, b)
    assert abs(value) < 3 * subset_sd, f"KID {value:.2e} should straddle 0 (sd {subset_sd:.1e})"


def test_kid_grows_quadratically_with_mean_shift(samples):
    a, b = samples
    small, _ = kid(a + 0.05, b)
    large, _ = kid(a + 0.15, b)
    assert small > 0 and large > small
    # polynomial-kernel MMD^2 scales with the squared shift: (0.15/0.05)^2 = 9
    assert 7.5 < large / small < 10.5, f"ratio {large/small:.2f} not ~9"


def test_fid_is_severely_biased_at_this_sample_size(samples):
    """Documents *why* we do not report absolute FID: same distribution, huge FID."""
    a, b = samples
    value = frechet(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))
    assert value > 100, (
        f"expected large small-sample bias, got {value:.1f}; if this ever drops near 0 "
        "the sample size or feature dimension changed and the paper's caveat needs revisiting")
