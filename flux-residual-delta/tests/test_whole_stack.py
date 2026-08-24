"""Whole-stack (K=1) runtime semantics on a stub transformer -- no weights, no GPU.

The design rests on one arithmetic claim: with both block lists empty, FLUX's
`cat([encoder, hidden]) -> single blocks -> slice` sequence returns the post-
x_embedder hidden state *exactly*, so a reuse step is just "add the cached
residual". If that ever stops holding, every FLUX number is silently wrong.
"""
import torch
from torch import nn

from flux_residual_delta.whole_stack import FluxWholeStackRuntime, anchor_steps_for


class _Dual(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **kw):
        return encoder_hidden_states + 0.5, hidden_states * 1.1 + 0.3


class _Single(nn.Module):
    def forward(self, hidden_states, **kw):
        return hidden_states * 1.05 - 0.1


class _NormOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, hidden_states, temb=None):
        self.seen.append(hidden_states.clone())
        return hidden_states


class _StubFlux(nn.Module):
    """Mirrors diffusers' FluxTransformer2DModel.forward control flow."""

    def __init__(self, n_dual=2, n_single=3):
        super().__init__()
        self.x_embedder = nn.Identity()
        self.transformer_blocks = nn.ModuleList([_Dual() for _ in range(n_dual)])
        self.single_transformer_blocks = nn.ModuleList([_Single() for _ in range(n_single)])
        self.norm_out = _NormOut()
        self.proj_out = nn.Identity()

    def forward(self, hidden_states, encoder_hidden_states, timestep=None):
        h = self.x_embedder(hidden_states)
        e = encoder_hidden_states
        for block in self.transformer_blocks:
            e, h = block(hidden_states=h, encoder_hidden_states=e)
        h = torch.cat([e, h], dim=1)
        for block in self.single_transformer_blocks:
            h = block(hidden_states=h)
        h = h[:, e.shape[1]:, ...]
        h = self.norm_out(h, None)
        return self.proj_out(h)


def _run(model, h, e, step_timestep=1.0):
    return model(hidden_states=h, encoder_hidden_states=e, timestep=step_timestep)


def test_empty_block_lists_are_an_identity():
    """The core arithmetic claim, checked directly."""
    model = _StubFlux()
    h = torch.randn(1, 6, 4)
    e = torch.randn(1, 3, 4)
    model.transformer_blocks = nn.ModuleList()
    model.single_transformer_blocks = nn.ModuleList()
    _run(model, h, e)
    torch.testing.assert_close(model.norm_out.seen[-1], h)


def test_reuse_step_equals_hidden_plus_cached_residual():
    model = _StubFlux()
    e = torch.randn(1, 3, 4)
    h0 = torch.randn(1, 6, 4)
    h1 = torch.randn(1, 6, 4)
    with FluxWholeStackRuntime(model, anchor_steps={0}) as rt:
        _run(model, h0, e)                      # step 0: anchor, real stack
        anchor_residual = rt.cache.residual.clone()
        real_out_step0 = model.norm_out.seen[-1]
        torch.testing.assert_close(real_out_step0 - h0, anchor_residual)
        _run(model, h1, e)                      # step 1: reuse
        torch.testing.assert_close(model.norm_out.seen[-1], h1 + anchor_residual)
    assert rt.stats.stack_steps == 1 and rt.stats.reused_steps == 1


def test_observer_records_truth_but_propagates_cache():
    model = _StubFlux()
    e = torch.randn(1, 3, 4)
    h0, h1 = torch.randn(1, 6, 4), torch.randn(1, 6, 4)
    records = []

    def observer(**kw):
        records.append(kw)

    with FluxWholeStackRuntime(model, anchor_steps={0}, reuse_observer=observer) as rt:
        _run(model, h0, e)
        cached = rt.cache.residual.clone()
        _run(model, h1, e)
        # propagation must use the cached branch, not the truth it just measured
        torch.testing.assert_close(model.norm_out.seen[-1], h1 + cached)

    assert len(records) == 1
    rec = records[0]
    assert rec["horizon"] == 1 and rec["step"] == 1
    # the recorded target is the TRUE residual at this state, computed with real blocks
    model2 = _StubFlux()
    _run(model2, h1, e)
    torch.testing.assert_close(rec["true_residual"], model2.norm_out.seen[-1] - h1)


def test_anchor_steps_helper():
    assert anchor_steps_for(20, 5) == {0, 5, 10, 15}
    assert anchor_steps_for(20, 20) == {0}
