"""FLUX residual-delta surrogate research package.

`whole_stack` (K=1, the whole 57-block stack as one cached unit) is the validated
design carried over from the PixArt arm. `runtime.FluxSegmentRuntime` is the older
segment-of-single-blocks scaffold, kept for reference but superseded: the PixArt
granularity sweep showed per-block/segment injection multiplies error along depth.
"""

from .surrogate import BlockResidualDeltaSurrogate, SurrogateBank, SurrogateConfig
from .whole_stack import FluxWholeStackRuntime, anchor_steps_for

__all__ = [
    "FluxWholeStackRuntime",
    "anchor_steps_for",
    "BlockResidualDeltaSurrogate",
    "SurrogateBank",
    "SurrogateConfig",
]
