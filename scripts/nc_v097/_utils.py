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


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def output_dir(cfg: dict[str, Any]) -> Path:
    out = resolve_path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


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


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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
            return max(sum(1 for _ in f) - 1, 0)
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


def environment_row() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "cpu_count": os.cpu_count(),
    }


def stable_unit_float(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)


def seed_for(*parts: Any) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8], 16)


def detect_gpu() -> dict[str, Any]:
    row: dict[str, Any] = {
        "decision": "CPU_USED",
        "reason": "No validated GPU parity path exists for the new v0.9.7 aligned feature generator.",
    }
    try:
        import torch

        row.update({
            "torch_import": "ok",
            "torch_version": getattr(torch, "__version__", "UNKNOWN"),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "torch_cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        })
    except Exception as exc:
        row.update({"torch_import": "missing_or_error", "torch_error": f"{type(exc).__name__}: {exc}"})
    for pkg in ["cupy", "numba"]:
        try:
            importlib.import_module(pkg)
            row[f"{pkg}_import"] = "ok"
        except Exception as exc:
            row[f"{pkg}_import"] = "missing_or_error"
            row[f"{pkg}_error"] = f"{type(exc).__name__}: {exc}"
    return row
