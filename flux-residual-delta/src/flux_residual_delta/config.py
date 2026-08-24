from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )


def dataclass_from_section(cls, section: dict[str, Any]):
    names = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in section.items() if key in names})
