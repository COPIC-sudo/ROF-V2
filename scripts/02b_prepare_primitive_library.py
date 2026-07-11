from __future__ import annotations
import argparse
from pathlib import Path

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.tube.rt_library import build_primitive_library_from_stable_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stable-runs-dir", default=None, help="包含 stable_runs_mu*_v*.csv 的目录；默认用 tube.primitive_manifest_dir")
    ap.add_argument("--raw-root", default=None, help="如果 stable_runs 中的 run 路径失效，用这个目录按 basename 回找 raw run CSV")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work_dir = Path(cfg["project"]["work_dir"])
    out_dir = ensure_dir(work_dir / "tube_library")
    stable_dir = Path(args.stable_runs_dir or str(cfg["tube"].get("primitive_manifest_dir", cfg["tube"].get("existing_manifest_dir", ""))))
    raw_root = Path(args.raw_root) if args.raw_root else None
    build_primitive_library_from_stable_runs(stable_dir, out_dir, cfg, raw_root=raw_root)
    print(f"[primitive] wrote primitive_mu*_v*.npz under: {out_dir}")

if __name__ == "__main__":
    main()
