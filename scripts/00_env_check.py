from __future__ import annotations
import argparse
from pathlib import Path
import sys

from _bootstrap import ROOT
from rtbev.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print("=" * 72)
    print("ROF Emergency Pipeline 环境检查")
    print("=" * 72)
    print("[1] Python", sys.version)
    print("[2] 关键依赖")
    for mod in ["numpy", "pandas", "yaml", "matplotlib", "shapely", "scipy", "sklearn", "google.protobuf"]:
        try:
            __import__(mod)
            print(f"  - {mod}: OK")
        except Exception as e:
            print(f"  - {mod}: FAIL ({e})")
    print("[3] Waymo protobuf stubs")
    try:
        from waymo_open_dataset.protos import scenario_pb2  # noqa
        print("  - waymo_open_dataset.protos.scenario_pb2: OK")
    except Exception as e:
        print(f"  - scenario_pb2: FAIL ({e})")
    print("[4] 数据路径")
    root = Path(cfg["dataset"]["scenario_root"])
    print(f"  - scenario_root = {root}")
    print(f"  - exists = {root.exists()}")
    for split in ["validation", "training"]:
        p = root / split
        print(f"  - {split}: {p.exists()} ({p})")
    print("[5] 工作目录")
    work = Path(cfg["project"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    print(f"  - work_dir = {work}")
    print("[6] 旧 tube manifest")
    md = Path(str(cfg["tube"].get("existing_manifest_dir", "")))
    if md.exists():
        print(f"  - tube_layered count = {len(list(md.glob('tube_layered_*.json')))}")
        print(f"  - stable_runs count  = {len(list(md.glob('stable_runs_mu*_v*.csv')))}")
    else:
        print("  - not found; fallback 可用于调试")
    print("[7] Torch / GPU")
    try:
        import torch
        print(f"  - torch = {torch.__version__}")
        print(f"  - cuda available = {torch.cuda.is_available()}")
        print(f"  - cuda device count = {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"  - torch not available: {e}")
    print("=" * 72)

if __name__ == "__main__":
    main()
