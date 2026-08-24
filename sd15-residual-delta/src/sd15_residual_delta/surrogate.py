from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SurrogateConfig:
    channels: int
    width: int
    depth: int = 2
    conditioning_dim: int = 128
    num_modules: int = 16
    use_delta_h: bool = True
    use_anchor_residual: bool = True
    global_attention: bool = False
    global_attention_heads: int = 4


class ScalarFourierEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("embedding dimension must be even")
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, value: Tensor) -> Tensor:
        phase = value.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.cat((phase.cos(), phase.sin()), dim=-1)


class ConditionedConvBlock(nn.Module):
    def __init__(self, width: int, conditioning_dim: int) -> None:
        super().__init__()
        groups = math.gcd(width, 32)
        self.norm = nn.GroupNorm(groups, width, affine=False)
        self.modulation = nn.Linear(conditioning_dim, width * 2)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width * 2, 1)
        self.output = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        value = self.norm(hidden)
        value = value * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        value = self.pointwise(self.depthwise(value))
        value, gate = value.chunk(2, dim=1)
        return hidden + self.output(torch.nn.functional.silu(value) * torch.sigmoid(gate))


class ConditionedGlobalAttentionBlock(nn.Module):
    """Cheap global mixing at the surrogate bottleneck for low-resolution features."""

    def __init__(self, width: int, conditioning_dim: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Surrogate width must be divisible by attention heads")
        self.norm = nn.GroupNorm(math.gcd(width, 32), width, affine=False)
        self.modulation = nn.Linear(conditioning_dim, width * 2)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        value = self.norm(hidden)
        value = value * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        batch, channels, height, width = value.shape
        tokens = value.flatten(2).transpose(1, 2)
        tokens = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        value = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return hidden + self.output(value)


class AttentionResidualDeltaSurrogate(nn.Module):
    def __init__(self, cfg: SurrogateConfig) -> None:
        super().__init__()
        if not (cfg.use_delta_h or cfg.use_anchor_residual):
            raise ValueError("At least one feature source must be enabled")
        self.cfg = cfg
        self.delta_projection = nn.Conv2d(cfg.channels, cfg.width, 1, bias=False)
        self.anchor_projection = nn.Conv2d(cfg.channels, cfg.width, 1, bias=False)
        self.fusion = nn.Conv2d(cfg.width * 2, cfg.width, 1)
        self.scalar_embedding = ScalarFourierEmbedding(cfg.conditioning_dim // 2)
        self.condition = nn.Sequential(
            nn.Linear(cfg.conditioning_dim, cfg.conditioning_dim),
            nn.SiLU(),
            nn.Linear(cfg.conditioning_dim, cfg.conditioning_dim),
        )
        self.module_embedding = nn.Embedding(cfg.num_modules, cfg.conditioning_dim)
        self.blocks = nn.ModuleList(
            [ConditionedConvBlock(cfg.width, cfg.conditioning_dim) for _ in range(cfg.depth)]
        )
        self.global_block = (
            ConditionedGlobalAttentionBlock(
                cfg.width, cfg.conditioning_dim, cfg.global_attention_heads
            )
            if cfg.global_attention
            else None
        )
        self.final_norm = nn.GroupNorm(math.gcd(cfg.width, 32), cfg.width)
        self.output = nn.Conv2d(cfg.width, cfg.channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        delta_h: Tensor,
        anchor_residual: Tensor,
        timestep: Tensor,
        horizon: Tensor,
        module_id: Tensor,
    ) -> Tensor:
        delta = self.delta_projection(delta_h) if self.cfg.use_delta_h else torch.zeros_like(
            self.anchor_projection(anchor_residual)
        )
        anchor = (
            self.anchor_projection(anchor_residual)
            if self.cfg.use_anchor_residual
            else torch.zeros_like(delta)
        )
        hidden = self.fusion(torch.cat((delta, anchor), dim=1))
        scalar_condition = torch.cat(
            (self.scalar_embedding(timestep), self.scalar_embedding(horizon)), dim=-1
        ).to(hidden.dtype)
        condition = self.condition(scalar_condition) + self.module_embedding(module_id)
        for block in self.blocks:
            hidden = block(hidden, condition)
        if self.global_block is not None:
            hidden = self.global_block(hidden, condition)
        return self.output(torch.nn.functional.silu(self.final_norm(hidden)))


class SharedSurrogateBank(nn.Module):
    def __init__(
        self,
        widths: dict[int, int],
        num_modules: int,
        depth: int = 2,
        conditioning_dim: int = 128,
        use_delta_h: bool = True,
        use_anchor_residual: bool = True,
        global_attention_channels: tuple[int, ...] = (),
        global_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleDict(
            {
                str(channels): AttentionResidualDeltaSurrogate(
                    SurrogateConfig(
                        channels=channels,
                        width=width,
                        depth=depth,
                        conditioning_dim=conditioning_dim,
                        num_modules=num_modules,
                        use_delta_h=use_delta_h,
                        use_anchor_residual=use_anchor_residual,
                        global_attention=channels in global_attention_channels,
                        global_attention_heads=global_attention_heads,
                    )
                )
                for channels, width in widths.items()
            }
        )

    def forward(
        self,
        delta_h: Tensor,
        anchor_residual: Tensor,
        timestep: Tensor,
        horizon: Tensor,
        module_id: Tensor,
    ) -> Tensor:
        key = str(delta_h.shape[1])
        if key not in self.models:
            raise KeyError(f"No surrogate configured for {delta_h.shape[1]} channels")
        return self.models[key](delta_h, anchor_residual, timestep, horizon, module_id)
