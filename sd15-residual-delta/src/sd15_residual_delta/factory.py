from __future__ import annotations

from .surrogate import SharedSurrogateBank


def build_surrogate_bank(config: dict, num_modules: int = 16) -> SharedSurrogateBank:
    section = config["surrogate"]
    return SharedSurrogateBank(
        widths={
            320: int(section["width_320"]),
            640: int(section["width_640"]),
            1280: int(section["width_1280"]),
        },
        num_modules=num_modules,
        depth=int(section["depth"]),
        conditioning_dim=int(section["conditioning_dim"]),
        use_delta_h=bool(section["use_delta_h"]),
        use_anchor_residual=bool(section["use_anchor_residual"]),
        global_attention_channels=tuple(
            int(value) for value in section.get("global_attention_channels", [])
        ),
        global_attention_heads=int(section.get("global_attention_heads", 4)),
    )
