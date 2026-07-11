from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from .utils import ensure_dir


def write_gzip_pickle(path: str | Path, obj) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with gzip.open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_gzip_pickle(path: str | Path):
    p = Path(path)
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def find_tfrecord_files(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    files = sorted(split_dir.glob("*.tfrecord*"))
    if not files:
        raise FileNotFoundError(f"no tfrecord files found under: {split_dir}")
    return files
