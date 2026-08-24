import torch
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from torch import nn

from sd15_residual_delta.capture import SD15FeatureCapture
from sd15_residual_delta.dataset import ShardWriter, ResidualPairDataset, build_pair
from sd15_residual_delta.runtime import SD15AttentionRuntime, find_transformer2d_modules
from sd15_residual_delta.surrogate import AttentionResidualDeltaSurrogate, SurrogateConfig


class TinyAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Transformer2DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=16,
            num_layers=1,
            cross_attention_dim=12,
            norm_num_groups=4,
        )

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.attn(
            sample,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class TinyTwoAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Transformer2DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=16,
            num_layers=1,
            cross_attention_dim=12,
            norm_num_groups=4,
        )
        self.attn2 = Transformer2DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=16,
            num_layers=1,
            cross_attention_dim=12,
            norm_num_groups=4,
        )

    def forward(self, sample, timestep, encoder_hidden_states):
        hidden = self.attn1(
            sample,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        return self.attn2(
            hidden,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def test_surrogate_shape_and_zero_initialization():
    model = AttentionResidualDeltaSurrogate(
        SurrogateConfig(channels=16, width=8, depth=1, conditioning_dim=16, num_modules=2)
    )
    hidden = torch.randn(2, 16, 8, 8)
    output = model(
        hidden,
        torch.randn_like(hidden),
        torch.ones(2),
        torch.ones(2),
        torch.tensor([0, 1]),
    )
    assert output.shape == hidden.shape
    assert torch.count_nonzero(output) == 0


def test_global_surrogate_shape_and_zero_initialization():
    model = AttentionResidualDeltaSurrogate(
        SurrogateConfig(
            channels=16,
            width=8,
            depth=1,
            conditioning_dim=16,
            num_modules=2,
            global_attention=True,
            global_attention_heads=2,
        )
    )
    hidden = torch.randn(2, 16, 4, 4)
    output = model(
        hidden,
        torch.randn_like(hidden),
        torch.ones(2),
        torch.ones(2),
        torch.tensor([0, 1]),
    )
    assert output.shape == hidden.shape
    assert torch.count_nonzero(output) == 0


def test_surrogate_condition_dtype_matches_half_precision_model():
    if not torch.cuda.is_available():
        return
    model = AttentionResidualDeltaSurrogate(
        SurrogateConfig(channels=16, width=8, depth=1, conditioning_dim=16, num_modules=2)
    ).to(device="cuda", dtype=torch.float16)
    hidden = torch.randn(2, 16, 4, 4, device="cuda", dtype=torch.float16)
    output = model(
        hidden,
        torch.randn_like(hidden),
        torch.ones(2, device="cuda"),
        torch.ones(2, device="cuda"),
        torch.tensor([0, 1], device="cuda"),
    )
    assert output.dtype == torch.float16


def test_runtime_anchor_reuse_and_restoration():
    unet = TinyAttentionUNet().eval()
    original = unet.attn
    assert len(find_transformer2d_modules(unet)) == 1
    encoder = torch.randn(1, 3, 12)
    with torch.inference_mode(), SD15AttentionRuntime(unet, cache_interval=2) as runtime:
        for step in range(3):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.full_steps == 2
        assert runtime.stats.reused_steps == 1
        assert runtime.stats.exact_module_calls == 2
        assert runtime.stats.reused_module_calls == 1
    assert unet.attn is original


def test_runtime_on_policy_observer_gets_oracle_reuse_target():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    observations = []
    with torch.inference_mode(), SD15AttentionRuntime(
        unet, cache_interval=2, reuse_observer=lambda **item: observations.append(item)
    ) as runtime:
        for step in range(3):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.oracle_module_calls == 1
    assert len(observations) == 1
    assert observations[0]["step"] == 1
    assert observations[0]["horizon"] == 1
    assert observations[0]["oracle_residual"].shape == (1, 16, 4, 4)


def test_runtime_on_policy_observer_keys_limit_oracle_calls():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    observations = []
    with torch.inference_mode(), SD15AttentionRuntime(
        unet,
        cache_interval=4,
        reuse_observer=lambda **item: observations.append(item),
        reuse_observer_keys={(2, 0)},
    ) as runtime:
        for step in range(4):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.oracle_module_calls == 1
        assert runtime.stats.reused_module_calls == 3
    assert len(observations) == 1
    assert observations[0]["step"] == 2
    assert observations[0]["horizon"] == 2


def test_runtime_surrogate_observer_gets_prediction_without_oracle_call():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    observations = []
    surrogate = AttentionResidualDeltaSurrogate(
        SurrogateConfig(
            channels=16, width=8, depth=1, conditioning_dim=16, num_modules=1
        )
    ).eval()
    with torch.inference_mode(), SD15AttentionRuntime(
        unet,
        cache_interval=2,
        surrogate_bank=surrogate,
        surrogate_observer=lambda **item: observations.append(item),
    ) as runtime:
        for step in range(3):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.surrogate_module_calls == 1
        assert runtime.stats.oracle_module_calls == 0
    assert len(observations) == 1
    assert observations[0]["step"] == 1
    assert observations[0]["horizon"] == 1
    assert observations[0]["delta_h"].shape == (1, 16, 4, 4)
    assert observations[0]["anchor_hidden"].shape == (1, 16, 4, 4)
    assert observations[0]["predicted_delta_residual"].shape == (1, 16, 4, 4)


def test_runtime_surrogate_step_gate_limits_calls():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    surrogate = AttentionResidualDeltaSurrogate(
        SurrogateConfig(
            channels=16, width=8, depth=1, conditioning_dim=16, num_modules=1
        )
    ).eval()
    with torch.inference_mode(), SD15AttentionRuntime(
        unet,
        cache_interval=4,
        surrogate_bank=surrogate,
        surrogate_step_module_keys={(2, 0)},
        surrogate_scale=0.5,
    ) as runtime:
        for step in range(4):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.surrogate_module_calls == 1
        assert runtime.stats.reused_module_calls == 3


def test_runtime_step_offset_uses_global_gate_coordinates():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    surrogate = AttentionResidualDeltaSurrogate(
        SurrogateConfig(
            channels=16, width=8, depth=1, conditioning_dim=16, num_modules=1
        )
    ).eval()
    with torch.inference_mode(), SD15AttentionRuntime(
        unet,
        cache_interval=4,
        surrogate_bank=surrogate,
        surrogate_step_module_keys={(14, 0)},
        step_offset=12,
    ) as runtime:
        for step in range(12, 15):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.surrogate_module_calls == 1


def test_runtime_single_oracle_refresh_returns_exact_module_output():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    anchor_input = torch.randn(1, 16, 4, 4)
    reuse_input = torch.randn(1, 16, 4, 4)
    with torch.inference_mode():
        expected = unet(reuse_input, torch.tensor(1), encoder)
        with SD15AttentionRuntime(
            unet, cache_interval=2, oracle_refresh_keys={(1, 0)}
        ) as runtime:
            unet(anchor_input, torch.tensor(0), encoder)
            actual = unet(reuse_input, torch.tensor(1), encoder)
        assert torch.allclose(actual, expected)
        assert runtime.stats.full_steps == 1
        assert runtime.stats.reused_steps == 1
        assert runtime.stats.exact_module_calls == 2
        assert runtime.stats.reused_module_calls == 0
        assert runtime.stats.oracle_refresh_module_calls == 1


def test_runtime_module_horizon_gate_limits_surrogate_calls():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    surrogate = AttentionResidualDeltaSurrogate(
        SurrogateConfig(
            channels=16, width=8, depth=1, conditioning_dim=16, num_modules=1
        )
    ).eval()
    with torch.inference_mode(), SD15AttentionRuntime(
        unet,
        cache_interval=3,
        surrogate_bank=surrogate,
        surrogate_module_horizon_keys={(0, 2)},
    ) as runtime:
        for step in range(3):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.full_steps == 1
        assert runtime.stats.reused_steps == 2
        assert runtime.stats.reused_module_calls == 2
        assert runtime.stats.surrogate_module_calls == 1


def test_runtime_explicit_anchor_schedule_overrides_periodic_interval():
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    with torch.inference_mode(), SD15AttentionRuntime(
        unet,
        cache_interval=99,
        anchor_steps={0, 1, 4},
    ) as runtime:
        for step in range(6):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.full_steps == 3
        assert runtime.stats.reused_steps == 3
        assert runtime.stats.exact_module_calls == 3
        assert runtime.stats.reused_module_calls == 3


def test_runtime_only_reuses_selected_cache_modules():
    unet = TinyTwoAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    with torch.inference_mode(), SD15AttentionRuntime(
        unet,
        cache_interval=2,
        cached_module_ids={0},
    ) as runtime:
        for step in range(3):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
        assert runtime.stats.full_steps == 2
        assert runtime.stats.reused_steps == 1
        assert runtime.stats.exact_module_calls == 5
        assert runtime.stats.reused_module_calls == 1


def test_selected_step_capture_and_dataset(tmp_path):
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    with torch.inference_mode(), SD15FeatureCapture(unet, {0, 2}) as capture:
        for step in range(3):
            unet(torch.randn(1, 16, 4, 4), torch.tensor(step), encoder)
    module_name = next(iter(capture.features))
    assert set(capture.features[module_name]) == {0, 2}
    pair = build_pair(
        capture.features[module_name][0], capture.features[module_name][2], "prompt-000000"
    )
    with ShardWriter(tmp_path, shard_size=1) as writer:
        writer.add(pair)
    test_set = ResidualPairDataset(tmp_path, "test", seed=0)
    assert len(test_set) == 1
    assert test_set[0]["delta_h"].shape == (1, 16, 4, 4)


def test_runtime_forced_cache_reuses_at_an_otherwise_exact_step():
    """forced_cache_keys is the mirror of oracle_refresh_keys: it caches a
    module at a step that the schedule marks exact, trading quality for speed."""
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    first = torch.randn(1, 16, 4, 4)
    second = torch.randn(1, 16, 4, 4)
    with torch.inference_mode():
        # anchor_steps={0,1} makes both steps exact; force module 0 to cache at step 1
        with SD15AttentionRuntime(
            unet, cache_interval=999, anchor_steps={0, 1},
            forced_cache_keys={(1, 0)},
        ) as runtime:
            unet(first, torch.tensor(0), encoder)
            unet(second, torch.tensor(1), encoder)
        assert runtime.stats.full_steps == 2
        assert runtime.stats.exact_module_calls == 1
        assert runtime.stats.reused_module_calls == 1


def test_runtime_forced_cache_cannot_fire_without_a_populated_cache():
    """At step 0 there is no cache yet, so forced_cache must fall back to exact."""
    unet = TinyAttentionUNet().eval()
    encoder = torch.randn(1, 3, 12)
    with torch.inference_mode():
        with SD15AttentionRuntime(
            unet, cache_interval=999, anchor_steps={0},
            forced_cache_keys={(0, 0)},
        ) as runtime:
            unet(torch.randn(1, 16, 4, 4), torch.tensor(0), encoder)
        assert runtime.stats.exact_module_calls == 1
        assert runtime.stats.reused_module_calls == 0
