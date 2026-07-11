from __future__ import annotations
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.tube.rt_library import (
    import_existing_layered_manifests,
    build_fallback_union_library,
    build_primitive_library_from_stable_runs,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", required=True, choices=["import_existing", "fallback"])
    ap.add_argument("--no-primitives", action="store_true", help="只准备 union tube，不准备 primitive/MSR 库")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["project"]["work_dir"])
    out_dir = ensure_dir(work_dir / "tube_library")

    par_path = str(cfg["tube"].get("optional_vehicle_par_path", "")).strip()
    if args.mode == "import_existing":
        src = str(cfg["tube"].get("existing_manifest_dir", "")).strip()
        if not src:
            raise SystemExit("existing_manifest_dir 为空，请先在 configs/default.yaml 中填写")
        src_path = Path(src)
        import_existing_layered_manifests(src_path, out_dir)
        print(f"[tube] imported layered manifests from: {src}")
        if not args.no_primitives:
            raw_root_s = str(cfg["tube"].get("existing_runs_root", "")).strip()
            raw_root = Path(raw_root_s) if raw_root_s else None
            try:
                build_primitive_library_from_stable_runs(src_path, out_dir, cfg, raw_root=raw_root)
            except Exception as e:
                print(f"[tube] stable primitive import failed: {e}")
                print("[tube] 已保留正式 tube_union；但 MSR 将不可用。若只是测试流程，可改用 --mode fallback。若写正式论文，请提供 stable_runs 对应的 raw run CSV 路径。")
    else:
        build_fallback_union_library(cfg, out_dir, Path(par_path) if par_path else None)
        print(f"[tube] fallback union + primitive library built under: {out_dir}")


if __name__ == "__main__":
    main()
