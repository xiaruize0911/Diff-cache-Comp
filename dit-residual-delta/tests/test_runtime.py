from __future__ import annotations

import torch
from torch import nn

from dit_residual_delta.runtime import DiTBlockRuntime, find_transformer_blocks


class ToyBlock(nn.Module):
    """Input-shape-preserving block, so `out - in` is a residual like a DiT block."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.linear = nn.Linear(dim, dim)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn(dim, dim, generator=generator) * 0.05)
            self.linear.bias.zero_()
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return hidden_states + torch.tanh(self.linear(hidden_states))


class ToyDiT(nn.Module):
    def __init__(self, dim: int = 8, depth: int = 4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(ToyBlock(dim, seed=i) for i in range(depth))

    def forward(self, hidden_states, timestep=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, timestep=timestep)
        return hidden_states


def rollout(model, runtime=None, steps=6, dim=8):
    torch.manual_seed(0)
    state = torch.randn(2, 5, dim)
    outputs = []
    for step in range(steps):
        state = model(state + 0.01 * step, timestep=torch.tensor([float(step)]))
        outputs.append(state.clone())
    return outputs


def test_finds_blocks():
    model = ToyDiT()
    assert [name for name, _ in find_transformer_blocks(model)] == [
        f"transformer_blocks.{i}" for i in range(4)
    ]


def test_all_anchor_steps_reproduce_exact():
    steps = 6
    reference = rollout(ToyDiT())
    model = ToyDiT()
    with DiTBlockRuntime(model, anchor_steps=set(range(steps))) as runtime:
        cached = rollout(model)
    assert runtime.stats.reused_block_calls == 0
    assert runtime.stats.full_steps == steps
    for a, b in zip(reference, cached):
        assert torch.allclose(a, b, atol=1e-6)


def test_reuse_counts_and_restoration():
    steps, depth = 6, 4
    model = ToyDiT(depth=depth)
    anchors = {0, 3}
    with DiTBlockRuntime(model, anchor_steps=anchors) as runtime:
        rollout(model, steps=steps)
        assert all(isinstance(b, nn.Module) for b in model.transformer_blocks)
    assert runtime.stats.full_steps == len(anchors)
    assert runtime.stats.reused_steps == steps - len(anchors)
    assert runtime.stats.exact_block_calls == len(anchors) * depth
    assert runtime.stats.reused_block_calls == (steps - len(anchors)) * depth
    # blocks are put back after the context exits
    assert [type(b).__name__ for b in model.transformer_blocks] == ["ToyBlock"] * depth


def test_oracle_refresh_runs_the_real_block():
    steps, depth = 6, 4
    model = ToyDiT(depth=depth)
    anchors = {0, 3}
    reuse_steps = [s for s in range(steps) if s not in anchors]
    oracle_keys = {(s, 1) for s in reuse_steps}
    with DiTBlockRuntime(model, anchor_steps=anchors, oracle_refresh_keys=oracle_keys) as runtime:
        rollout(model, steps=steps)
    assert runtime.stats.oracle_refresh_block_calls == len(oracle_keys)
    assert runtime.stats.exact_block_calls == len(anchors) * depth + len(oracle_keys)
    assert runtime.stats.reused_block_calls == (steps - len(anchors)) * depth - len(oracle_keys)


def test_full_oracle_equals_exact():
    """Oracle at every reuse slot must reproduce the exact rollout bit-for-bit."""
    steps, depth = 6, 4
    reference = rollout(ToyDiT(depth=depth), steps=steps)
    model = ToyDiT(depth=depth)
    anchors = {0}
    oracle_keys = {(s, b) for s in range(steps) if s not in anchors for b in range(depth)}
    with DiTBlockRuntime(model, anchor_steps=anchors, oracle_refresh_keys=oracle_keys) as runtime:
        got = rollout(model, steps=steps)
    assert runtime.stats.reused_block_calls == 0
    for a, b in zip(reference, got):
        assert torch.equal(a, b)


def test_forced_cache_turns_an_anchor_slot_into_reuse():
    steps, depth = 6, 4
    model = ToyDiT(depth=depth)
    anchors = {0, 2, 4}
    forced = {(4, 0)}
    with DiTBlockRuntime(model, anchor_steps=anchors, forced_cache_keys=forced) as runtime:
        rollout(model, steps=steps)
    assert runtime.stats.exact_block_calls == len(anchors) * depth - len(forced)
    assert runtime.stats.reused_block_calls == (steps - len(anchors)) * depth + len(forced)


def test_cached_block_ids_leaves_other_blocks_exact():
    steps, depth = 6, 4
    model = ToyDiT(depth=depth)
    anchors = {0, 3}
    with DiTBlockRuntime(model, anchor_steps=anchors, cached_block_ids={2}) as runtime:
        rollout(model, steps=steps)
    reuse_steps = steps - len(anchors)
    assert runtime.stats.reused_block_calls == reuse_steps
    assert runtime.stats.exact_block_calls == steps * depth - reuse_steps


def test_segment_runtime_all_anchors_matches_exact():
    from dit_residual_delta.runtime import DiTSegmentRuntime
    steps, depth = 6, 4
    reference = rollout(ToyDiT(depth=depth), steps=steps)
    model = ToyDiT(depth=depth)
    with DiTSegmentRuntime(model, anchor_steps=set(range(steps))) as runtime:
        got = rollout(model, steps=steps)
    assert runtime.stats.reused_block_calls == 0
    for a, b in zip(reference, got):
        assert torch.allclose(a, b, atol=1e-6)


def test_segment_runtime_full_oracle_matches_exact():
    from dit_residual_delta.runtime import DiTSegmentRuntime
    steps, depth = 6, 4
    reference = rollout(ToyDiT(depth=depth), steps=steps)
    model = ToyDiT(depth=depth)
    with DiTSegmentRuntime(model, anchor_steps={0}, oracle_blend=1.0) as runtime:
        got = rollout(model, steps=steps)
    assert runtime.stats.oracle_refresh_block_calls == steps - 1
    for a, b in zip(reference, got):
        assert torch.equal(a, b)


def test_segment_runtime_one_injection_per_reuse_step():
    from dit_residual_delta.runtime import DiTSegmentRuntime
    steps, depth = 6, 4
    model = ToyDiT(depth=depth)
    anchors = {0, 3}
    with DiTSegmentRuntime(model, anchor_steps=anchors) as runtime:
        rollout(model, steps=steps)
    assert runtime.stats.reused_block_calls == steps - len(anchors)
    assert runtime.stats.exact_block_calls == len(anchors) * depth
    assert [type(b).__name__ for b in model.transformer_blocks] == ["ToyBlock"] * depth


def test_segment_runtime_partial_segment_leaves_others_exact():
    from dit_residual_delta.runtime import DiTSegmentRuntime
    steps, depth = 6, 4
    model = ToyDiT(depth=depth)
    with DiTSegmentRuntime(model, segment=(1, 3), anchor_steps={0, 3}) as runtime:
        rollout(model, steps=steps)
    # blocks 0 and 3 run every step; blocks 1-2 only on anchor steps
    assert runtime.stats.exact_block_calls == 2 * 2
    assert model.transformer_blocks[0].calls == steps
    assert model.transformer_blocks[3].calls == steps
    assert model.transformer_blocks[1].calls == 2


def test_multi_segment_fixed_cache_matches_single_segment():
    """Any contiguous partition gives the same fixed-cache output: both equal
    h_t + sum of the anchor-step block residuals."""
    from dit_residual_delta.runtime import DiTSegmentRuntime, uniform_segments
    steps, depth = 6, 4
    anchors = {0, 3}
    outputs = []
    for count in (1, 2, 4):
        model = ToyDiT(depth=depth)
        with DiTSegmentRuntime(model, segments=uniform_segments(depth, count),
                               anchor_steps=anchors):
            outputs.append(rollout(model, steps=steps))
    for a, b in zip(outputs[0], outputs[1]):
        assert torch.allclose(a, b, atol=1e-6)
    for a, b in zip(outputs[0], outputs[2]):
        assert torch.allclose(a, b, atol=1e-6)


def test_multi_segment_injects_once_per_segment_per_reuse_step():
    from dit_residual_delta.runtime import DiTSegmentRuntime, uniform_segments
    steps, depth = 6, 4
    model = ToyDiT(depth=depth)
    anchors = {0, 3}
    with DiTSegmentRuntime(model, segments=uniform_segments(depth, 2),
                           anchor_steps=anchors) as runtime:
        rollout(model, steps=steps)
    assert runtime.stats.reused_block_calls == (steps - len(anchors)) * 2
