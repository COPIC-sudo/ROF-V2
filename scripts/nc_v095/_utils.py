from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def repo_root() -> Path:
    return Path.cwd()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def output_dir(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["project"]["output_dir"])
    if not out.is_absolute():
        out = repo_root() / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "EMPTY"}]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_blockers(out_dir: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = out_dir / ("BLOCKERS_V095_ENVSWITCH.csv" if "envswitch" in out_dir.name.lower() else "BLOCKERS_V095.csv")
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = pd.read_csv(path).to_dict("records")
    merged = {(str(r.get("category")), str(r.get("item"))): r for r in existing}
    for row in rows:
        row = dict(row)
        row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        merged[(str(row.get("category")), str(row.get("item")))] = row
    write_csv(path, list(merged.values()))


def sha256(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if max_bytes is not None and total + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - total]
            if chunk:
                h.update(chunk)
                total += len(chunk)
            if max_bytes is not None and total >= max_bytes:
                break
    return h.hexdigest()


def count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            n = sum(1 for _ in f)
        return max(n - 1, 0)
    except Exception:
        return None


def package_status(name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"package": name}
    try:
        mod = importlib.import_module(name)
        row["status"] = "ok"
        row["version"] = getattr(mod, "__version__", "UNKNOWN")
    except Exception as exc:
        row["status"] = "missing"
        row["version"] = ""
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def detect_torch_gpu() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if available else 0
        name = torch.cuda.get_device_name(0) if available else ""
        return {
            "backend": "torch",
            "gpu_available": available,
            "device_count": device_count,
            "device_name": name,
            "torch_version": getattr(torch, "__version__", "UNKNOWN"),
        }
    except Exception as exc:
        return {
            "backend": "torch",
            "gpu_available": False,
            "device_count": 0,
            "device_name": "",
            "torch_version": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def bool_str(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def environment_row() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "cpu_count": os.cpu_count(),
    }


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_first_existing(paths: Iterable[str | Path]) -> Path | None:
    for path in paths:
        p = resolve_path(path)
        if p.exists():
            return p
    return None
