from __future__ import annotations
import argparse
from pathlib import Path

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import read_gzip_pickle, ensure_dir
from rtbev.tube.rt_library import TubeLibrary
from rtbev.visualize import render_sample_overlay, render_tensor_maps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work_dir = Path(cfg["project"]["work_dir"])
    vis_dir = ensure_dir(work_dir / "visualizations")
    sample = read_gzip_pickle(work_dir / "samples" / f"{args.sample_id}.pkl.gz")
    lib = TubeLibrary.from_workdir(work_dir / "tube_library")
    out_png = vis_dir / f"{args.sample_id}_overlay.png"
    render_sample_overlay(sample, lib, cfg, out_png, device=args.device or cfg["runtime"].get("device", "cpu"))
    print(f"[viz] wrote {out_png}")
    tensor_path = work_dir / "bev_tensors" / f"{args.sample_id}.npz"
    if tensor_path.exists():
        map_png = vis_dir / f"{args.sample_id}_maps.png"
        render_tensor_maps(tensor_path, cfg, map_png, title=f"{args.sample_id} maps")
        print(f"[viz] wrote {map_png}")

if __name__ == "__main__":
    main()
