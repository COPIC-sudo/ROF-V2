from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml


def expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables and ~ in config strings."""
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, list):
        return [expand_env_vars(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(expand_env_vars(item) for item in obj)
    if isinstance(obj, dict):
        return {key: expand_env_vars(value) for key, value in obj.items()}
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return expand_env_vars(cfg)
