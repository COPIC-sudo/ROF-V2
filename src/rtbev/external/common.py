from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(expand_env(v) for v in obj)
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    return obj


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return expand_env(yaml.safe_load(f) or {})


def sha256_file(path: str | Path, max_bytes: int | None = None) -> str:
    p = Path(path)
    h = hashlib.sha256()
    total = 0
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if max_bytes is not None and total + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - total]
            if chunk:
                h.update(chunk)
                total += len(chunk)
            if max_bytes is not None and total >= max_bytes:
                break
    return h.hexdigest()


def config_hash(config_path: str | Path) -> str:
    return sha256_file(config_path)


def require_work_dir(cfg: dict[str, Any]) -> Path:
    work_dir = (cfg.get("project") or {}).get("work_dir") or os.environ.get("ROF_WORK_DIR")
    if not work_dir:
        raise ValueError("project.work_dir missing and ROF_WORK_DIR is not set")
    return Path(str(work_dir)).expanduser()


def resolve_input_path(value: str | Path | None, cfg: dict[str, Any] | None = None) -> Path | None:
    if value is None or str(value) == "":
        return None
    p = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def experiment_out_dir(cfg: dict[str, Any], default_name: str) -> Path:
    work_dir = require_work_dir(cfg)
    out_value = (cfg.get("project") or {}).get("output_dir")
    if out_value:
        p = Path(str(out_value)).expanduser()
        if not p.is_absolute():
            p = work_dir / p
    else:
        p = work_dir / "results" / default_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_csv_rows(path: str | Path) -> int | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows)
    fields: list[str] = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        data = [{"status": "EMPTY"}]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def add_config_hash(rows: Iterable[dict[str, Any]], cfg_hash: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r.setdefault("config_hash", cfg_hash)
        out.append(r)
    return out


def artifact_manifest_rows(config_path: str | Path, outputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    cfg_hash = config_hash(config_path)
    rows: list[dict[str, Any]] = []
    for value in outputs:
        p = Path(value)
        rows.append(
            {
                "artifact": p.name,
                "path": str(p),
                "exists": bool(p.exists()),
                "sha256": sha256_file(p) if p.exists() and p.is_file() else "",
                "rows": count_csv_rows(p) if p.suffix.lower() == ".csv" and p.exists() else "",
                "config_hash": cfg_hash,
            }
        )
    return rows


def run_manifest(config_path: str | Path, cfg: dict[str, Any], outputs: Iterable[str | Path]) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "config_path": str(config_path),
        "config_hash": config_hash(config_path),
        "work_dir": str(require_work_dir(cfg)),
        "outputs": artifact_manifest_rows(config_path, outputs),
    }


def read_csv_required(path: str | Path, name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    return pd.read_csv(p)
