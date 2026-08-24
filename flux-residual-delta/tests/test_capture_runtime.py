import torch
from diffusers import FluxTransformer2DModel

from flux_residual_delta.capture import FluxSegmentCapture
from flux_residual_delta.runtime import FluxSegmentRuntime


def tiny_flux():
    return FluxTransformer2DModel(
        num_layers=1,
        num_single_layers=3,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        in_channels=8,
        axes_dims_rope=(4, 6, 6),
    ).eval()


def inputs(timestep=0.5):
    return {
        "hidden_states": torch.randn(1, 4, 8),
        "encoder_hidden_states": torch.randn(1, 3, 32),
        "pooled_projections": torch.randn(1, 16),
        "timestep": torch.tensor([timestep]),
        "img_ids": torch.zeros(4, 3),
        "txt_ids": torch.zeros(3, 3),
        "return_dict": False,
    }


def test_capture_observes_unified_tokens():
    model = tiny_flux()
    with torch.inference_mode(), FluxSegmentCapture(model, 1, 2, hidden_dim=32) as capture:
        model(**inputs())
    feature = capture.features[0]
    assert feature.hidden.shape == (1, 7, 32)
    assert feature.residual.shape == feature.hidden.shape
    assert feature.timestep == 0.5


def test_runtime_runs_anchor_then_fixed_reuse_and_restores_modules():
    model = tiny_flux()
    originals = list(model.single_transformer_blocks)
    with torch.inference_mode(), FluxSegmentRuntime(
        model, start=1, length=2, cache_interval=2, surrogate=None
    ) as runtime:
        model(**inputs(0.8))
        model(**inputs(0.6))
        model(**inputs(0.4))
        assert runtime.stats.full_steps == 2
        assert runtime.stats.reused_steps == 1
    assert all(current is original for current, original in zip(model.single_transformer_blocks, originals))
