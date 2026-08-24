"""Corrector invariants, carried over from the segment-design tests to the K=1 model.

Same three properties as before (shape, zero-init == plain fixed reuse, gradient
reaches the input projection); only the class and call signature changed when the
validated PixArt corrector replaced the segment scaffold.
"""
import torch

from flux_residual_delta.surrogate import BlockResidualDeltaSurrogate, SurrogateConfig


def tiny_config(**overrides):
    values = dict(dim=32, width=16, depth=1, conditioning_dim=16, num_blocks=1)
    values.update(overrides)
    return SurrogateConfig(**values)


def _inputs(batch=2, tokens=7, dim=32):
    return dict(
        delta_h=torch.randn(batch, tokens, dim),
        anchor_residual=torch.randn(batch, tokens, dim),
        anchor_hidden=torch.randn(batch, tokens, dim),
        timestep=torch.full((batch,), 500.0),
        horizon=torch.ones(batch),
        block_id=torch.zeros(batch, dtype=torch.long),
    )


def test_forward_shape_and_zero_initialisation():
    model = BlockResidualDeltaSurrogate(tiny_config())
    args = _inputs()
    prediction = model(**args)
    assert prediction.shape == args["delta_h"].shape
    # untrained must be EXACTLY plain fixed reuse, or deployment silently changes
    assert torch.count_nonzero(prediction) == 0


def test_gradient_reaches_input_projection_after_output_update():
    model = BlockResidualDeltaSurrogate(tiny_config())
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.add_(torch.randn_like(p) * 0.01)
    prediction = model(**_inputs(batch=1, tokens=5))
    prediction.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients reached any parameter"
    assert any(g.abs().sum() > 0 for g in grads)


def test_low_rank_linear_path_changes_output():
    """The per-block low-rank path was what made the PixArt corrector learnable."""
    args = _inputs()
    plain = BlockResidualDeltaSurrogate(tiny_config(linear_rank=0))
    ranked = BlockResidualDeltaSurrogate(tiny_config(linear_rank=8))
    assert ranked.parameter_count() > plain.parameter_count()
