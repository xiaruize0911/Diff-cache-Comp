from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))
