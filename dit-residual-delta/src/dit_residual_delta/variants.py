"""One place where a variant JSON entry becomes a runtime.

The cost scripts (count_flops, benchmark_speed) and the quality scripts
(evaluate_variants, eval_fid_coco) MUST read a spec the same way. They did not:
the cost side defaulted a missing `num_segments` to 1 (whole stack, one
correction per step) while the quality side read the same spec as per-block
caching (K=28, 28 corrections per step). A `K28_*` entry that omitted the key
was therefore evaluated at K=28 and priced at K=1 -- 13.3 vs 22.1 TFLOP/image.

The rule, fixed here once: `segment` or `num_segments` selects the segment
runtime; a spec with neither means per-block caching.
"""
from __future__ import annotations

from typing import Any

from .runtime import DiTBlockRuntime, DiTSegmentRuntime, uniform_segments


def resolve_anchor_steps(spec: dict, num_steps: int) -> set[int]:
    # `"exact": true` used to fall through to the interval-2 default below, so the
    # "exact_20" arm of runs/same_steps was verbatim caching at i=2, not the exact
    # model. An explicit exact spec now refreshes every step.
    if spec.get("exact"):
        if "anchor_steps" in spec or "cache_interval" in spec:
            raise ValueError(f"exact spec {spec['name']!r} must not also set a schedule")
        return set(range(num_steps))
    if "anchor_steps" in spec:
        return {int(s) for s in spec["anchor_steps"]}
    interval = int(spec.get("cache_interval", 2))
    return {s for s in range(num_steps) if s % interval == 0}


def build_variant(spec: dict, num_steps: int, num_blocks: int) -> dict:
    anchors = resolve_anchor_steps(spec, num_steps)
    reuse_steps = [s for s in range(num_steps) if s not in anchors]
    oracle_steps = [int(s) for s in spec.get("oracle_steps", reuse_steps)]
    oracle_blocks = [int(b) for b in spec.get("oracle_block_ids", [])]
    oracle = {(s, b) for b in oracle_blocks for s in oracle_steps} or None
    forced = {
        (int(s), int(b))
        for b in spec.get("forced_cache_block_ids", [])
        for s in spec.get("forced_cache_steps", sorted(anchors))
    } or None
    cached_blocks = spec.get("cached_block_ids")
    surrogate_blocks = spec.get("surrogate_block_ids")
    segment = spec.get("segment")
    num_segments = spec.get("num_segments")
    return {
        "segment": tuple(int(v) for v in segment) if segment else None,
        "num_segments": int(num_segments) if num_segments else None,
        "surrogate_checkpoint": spec.get("surrogate_checkpoint"),
        "surrogate_scale": float(spec.get("surrogate_scale", 1.0)),
        "adaptive_threshold": (float(spec["adaptive_threshold"])
                               if spec.get("adaptive_threshold") is not None else None),
        "taylor_order": int(spec.get("taylor_order", 0)),
        "surrogate_block_ids": {int(b) for b in surrogate_blocks} if surrogate_blocks else None,
        "name": spec["name"],
        "oracle_blend": float(spec.get("oracle_blend", 1.0)),
        "oracle_blend_or_none": (float(spec["oracle_blend"]) if "oracle_blend" in spec
                                 else (1.0 if spec.get("segment_oracle") else None)),
        "anchor_steps": anchors,
        "reuse_steps": reuse_steps,
        "oracle_refresh_keys": oracle,
        "forced_cache_keys": forced,
        "cached_block_ids": {int(b) for b in cached_blocks} if cached_blocks else None,
        "num_oracle_slots": 0 if oracle is None else len(oracle),
        "num_reuse_slots": len(reuse_steps) * num_blocks,
    }


def uses_segment_runtime(variant: dict) -> bool:
    """Segment runtime only when the variant asks for it explicitly."""
    return variant["segment"] is not None or variant["num_segments"] is not None


def runtime_for_variant(transformer, variant: dict, surrogate_bank: Any | None = None):
    """Build (but do not enter) the runtime a normalised variant describes."""
    if uses_segment_runtime(variant):
        depth = len(transformer.transformer_blocks)
        return DiTSegmentRuntime(
            transformer,
            segment=variant["segment"] if variant["num_segments"] is None else None,
            segments=(uniform_segments(depth, variant["num_segments"])
                      if variant["num_segments"] else None),
            anchor_steps=variant["anchor_steps"],
            surrogate_bank=surrogate_bank,
            surrogate_scale=variant["surrogate_scale"],
            oracle_blend=variant["oracle_blend_or_none"],
            adaptive_threshold=variant["adaptive_threshold"],
            taylor_order=variant["taylor_order"],
        )
    return DiTBlockRuntime(
        transformer,
        cache_interval=999,
        anchor_steps=variant["anchor_steps"],
        cached_block_ids=variant["cached_block_ids"],
        oracle_refresh_keys=variant["oracle_refresh_keys"],
        oracle_blend=variant["oracle_blend"],
        forced_cache_keys=variant["forced_cache_keys"],
        surrogate_bank=surrogate_bank,
        surrogate_block_ids=variant["surrogate_block_ids"],
        surrogate_scale=variant["surrogate_scale"],
        taylor_order=variant["taylor_order"],
    )


def runtime_for_spec(transformer, spec: dict, num_steps: int, surrogate_bank: Any | None = None):
    """`runtime_for_variant` straight from a raw JSON spec entry."""
    num_blocks = len(transformer.transformer_blocks)
    return runtime_for_variant(transformer, build_variant(spec, num_steps, num_blocks),
                               surrogate_bank)
