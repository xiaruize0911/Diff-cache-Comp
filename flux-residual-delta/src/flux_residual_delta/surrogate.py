"""Token-wise residual-delta corrector C for a DiT block stack.

    delta_R_hat = C(delta_h, R_anchor, h_anchor | timestep, horizon, block_id)

One shared model covers all 28 blocks; block identity, horizon and timestep enter
through AdaLN-style modulation, mirroring the SD1.5 arm's shared surrogate so the
two backbones are compared at equal model structure. The output projection is
zero-initialised, so an untrained C is exactly fixed reuse.

Token-wise by design: a DiT block mixes tokens, but a corrector that also mixed
tokens would cost a large fraction of the block it replaces. `mix_tokens` adds one
cheap global attention layer for the ablation that tests whether that matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SurrogateConfig:
    dim: int = 1152
    width: int = 256
    depth: int = 3
    conditioning_dim: int = 128
    num_blocks: int = 28
    use_delta_h: bool = True
    use_anchor_residual: bool = True
    use_anchor_hidden: bool = True
    mix_tokens: bool = False
    attention_heads: int = 4
    linear_rank: int = 0        # per-block low-rank linear path; 0 disables it


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
        # Always fp32: timestep phases reach ~1000, where fp16 spacing is 0.5, so
        # cos/sin of an fp16 phase is unrelated to the fp32 value the model was
        # trained on. Only the result is cast back to the module dtype.
        frequencies = self.frequencies.float()
        phase = value.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat((phase.cos(), phase.sin()), dim=-1)


class ConditionedMLPBlock(nn.Module):
    def __init__(self, width: int, conditioning_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(conditioning_dim, width * 2)
        self.up = nn.Linear(width, width * 4)
        self.down = nn.Linear(width * 2, width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        value = self.norm(hidden) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        value, gate = self.up(value).chunk(2, dim=-1)
        return hidden + self.down(torch.nn.functional.silu(value) * torch.sigmoid(gate))


class ConditionedAttentionBlock(nn.Module):
    def __init__(self, width: int, conditioning_dim: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(conditioning_dim, width * 2)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        value = self.norm(hidden) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        value = self.attention(value, value, value, need_weights=False)[0]
        return hidden + self.output(value)


class PerBlockLowRankLinear(nn.Module):
    """delta_R += (x @ A_b) @ B_b, one low-rank map per block.

    The ridge probe shows delta_R is largely a linear function of the inputs with
    block-specific coefficients -- a first-order Jacobian effect. A shared MLP has
    to spend capacity re-deriving that per block; giving it an explicit per-block
    linear path is far cheaper than widening the MLP until it can.
    """

    def __init__(self, in_dim: int, out_dim: int, num_blocks: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Parameter(torch.randn(num_blocks, in_dim, rank) / math.sqrt(in_dim))
        self.up = nn.Parameter(torch.zeros(num_blocks, rank, out_dim))

    def forward(self, features: Tensor, block_id: Tensor) -> Tensor:
        down = self.down[block_id]                      # (batch, in_dim, rank)
        up = self.up[block_id]                          # (batch, rank, out_dim)
        return torch.bmm(torch.bmm(features, down), up)


class BlockResidualDeltaSurrogate(nn.Module):
    def __init__(self, cfg: SurrogateConfig) -> None:
        super().__init__()
        if not (cfg.use_delta_h or cfg.use_anchor_residual or cfg.use_anchor_hidden):
            raise ValueError("at least one input feature must be enabled")
        self.cfg = cfg
        self.delta_projection = nn.Linear(cfg.dim, cfg.width, bias=False) if cfg.use_delta_h else None
        self.anchor_residual_projection = (
            nn.Linear(cfg.dim, cfg.width, bias=False) if cfg.use_anchor_residual else None
        )
        self.anchor_hidden_projection = (
            nn.Linear(cfg.dim, cfg.width, bias=False) if cfg.use_anchor_hidden else None
        )
        self.timestep_embedding = ScalarFourierEmbedding(cfg.conditioning_dim)
        self.horizon_embedding = ScalarFourierEmbedding(cfg.conditioning_dim)
        self.block_embedding = nn.Embedding(cfg.num_blocks, cfg.conditioning_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(cfg.conditioning_dim * 3, cfg.conditioning_dim),
            nn.SiLU(),
            nn.Linear(cfg.conditioning_dim, cfg.conditioning_dim),
        )
        layers: list[nn.Module] = []
        for i in range(cfg.depth):
            layers.append(ConditionedMLPBlock(cfg.width, cfg.conditioning_dim))
            if cfg.mix_tokens and i == cfg.depth // 2:
                layers.append(ConditionedAttentionBlock(cfg.width, cfg.conditioning_dim, cfg.attention_heads))
        self.layers = nn.ModuleList(layers)
        self.output_norm = nn.LayerNorm(cfg.width, elementwise_affine=False)
        self.output = nn.Linear(cfg.width, cfg.dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.linear_path = None
        if cfg.linear_rank > 0:
            active = sum((cfg.use_delta_h, cfg.use_anchor_residual, cfg.use_anchor_hidden))
            self.linear_path = PerBlockLowRankLinear(
                cfg.dim * active, cfg.dim, cfg.num_blocks, cfg.linear_rank)

    def condition(self, timestep: Tensor, horizon: Tensor, block_id: Tensor) -> Tensor:
        dtype = self.block_embedding.weight.dtype
        parts = [
            self.timestep_embedding(timestep).to(dtype),
            self.horizon_embedding(horizon).to(dtype),
            self.block_embedding(block_id.long()),
        ]
        return self.condition_mlp(torch.cat(parts, dim=-1))

    def forward(self, delta_h: Tensor, anchor_residual: Tensor, anchor_hidden: Tensor,
                timestep: Tensor, horizon: Tensor, block_id: Tensor) -> Tensor:
        """All feature tensors are `(batch, tokens, dim)`; conditioning is `(batch,)`."""
        hidden = None
        for tensor, projection in (
            (delta_h, self.delta_projection),
            (anchor_residual, self.anchor_residual_projection),
            (anchor_hidden, self.anchor_hidden_projection),
        ):
            if projection is None:
                continue
            value = projection(tensor)
            hidden = value if hidden is None else hidden + value
        condition = self.condition(timestep, horizon, block_id)
        for layer in self.layers:
            hidden = layer(hidden, condition)
        out = self.output(self.output_norm(hidden))
        if self.linear_path is not None:
            features = torch.cat(
                [t for t, p in ((delta_h, self.delta_projection),
                                (anchor_residual, self.anchor_residual_projection),
                                (anchor_hidden, self.anchor_hidden_projection)) if p is not None],
                dim=-1)
            out = out + self.linear_path(features, block_id.long())
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SurrogateBank(nn.Module):
    """Deployment wrapper: applies the training-time input/output scaling and
    broadcasts scalar conditioning across the CFG batch."""

    def __init__(self, model: BlockResidualDeltaSurrogate, scale: dict[str, float]) -> None:
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, delta_h: Tensor, anchor_residual: Tensor, anchor_hidden: Tensor,
                timestep, horizon, block_id) -> Tensor:
        batch = delta_h.shape[0]
        device = delta_h.device
        if isinstance(timestep, Tensor):
            step_value = timestep.to(device, torch.float32).flatten()
            if step_value.numel() == 1:
                step_value = step_value.expand(batch)
            elif step_value.numel() != batch:
                step_value = step_value[:batch]
        else:
            step_value = torch.full((batch,), float(timestep or 0.0), device=device)
        horizon_value = torch.full((batch,), float(horizon), device=device)
        block_value = torch.full((batch,), int(block_id), device=device, dtype=torch.long)
        prediction = self.model(
            delta_h / self.scale["delta_h"],
            anchor_residual / self.scale["anchor_residual"],
            anchor_hidden / self.scale["anchor_hidden"],
            step_value, horizon_value, block_value,
        )
        return prediction * self.scale["delta_residual"]


def load_surrogate_bank(path, device: str = "cuda", dtype: torch.dtype = torch.float16) -> "SurrogateBank":
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SurrogateConfig(**payload["config"])
    model = BlockResidualDeltaSurrogate(cfg)
    model.load_state_dict(payload["model"])
    bank = SurrogateBank(model, payload["scale"]).to(device=device, dtype=dtype).eval()
    bank.checkpoint_info = {"step": payload.get("step"), "val_rel_mse": payload.get("val_rel_mse")}
    return bank
