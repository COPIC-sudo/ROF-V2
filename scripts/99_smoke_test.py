from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir, write_gzip_pickle
from rtbev.labels import assign_label
from rtbev.pipeline import sample_to_bev_tensor_and_features
from rtbev.tube.rt_library import (
    PrimitiveLibrary,
    TubeLibrary,
    build_fallback_union_and_primitive_library,
)


def _run_smoke(work_dir: Path) -> None:
    """Run a compact synthetic end-to-end pipeline in ``work_dir``."""
    cfg = load_config(ROOT / "configs" / "default.yaml")
    # Keep the smoke test small; the full paper library is built separately.
    cfg["tube"]["speed_bins_kph"] = [0, 30, 60]
    cfg["tube"]["mu_bins"] = [0.5]
    cfg["dataset"]["max_future_steps"] = min(
        int(cfg["dataset"].get("max_future_steps", 20)), 10
    )
    cfg.setdefault("metrics", {})["max_msr_primitives"] = 40
    cfg.setdefault("bev", {})["resolution_m"] = 1.0
    cfg["project"]["work_dir"] = str(work_dir.resolve())
    ensure_dir(work_dir)
    build_fallback_union_and_primitive_library(
        cfg, ensure_dir(work_dir / "tube_library"), None
    )

    T = int(cfg["dataset"]["max_future_steps"]) + 1
    dt = float(cfg["labels"]["dt_s"])
    times = np.arange(T, dtype=np.float32) * dt

    current_xy = np.array([[0.0, 0.0], [18.0, 1.5], [24.0, -2.0]], dtype=np.float32)
    current_vel = np.array([[15.0, 0.0], [8.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    current_heading = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    current_size = np.array([[4.8, 1.9], [4.6, 1.8], [4.6, 1.8]], dtype=np.float32)

    future_xy = np.zeros((3, T, 2), dtype=np.float32)
    future_vel = np.zeros((3, T, 2), dtype=np.float32)
    future_heading = np.zeros((3, T), dtype=np.float32)
    future_valid = np.ones((3, T), dtype=bool)
    for i in range(3):
        for k in range(T):
            future_xy[i, k] = current_xy[i] + current_vel[i] * times[k]
            future_vel[i, k] = current_vel[i]
            future_heading[i, k] = current_heading[i]
    future_xy[1, :, 1] -= np.linspace(0, 1.4, T)

    sample = {
        "sample_id": "smoke_demo_0001",
        "scenario_id": "smoke_demo_0001",
        "ego_track_id": 0,
        "ego_index": 0,
        "agent_ids": np.array([0, 1, 2], dtype=np.int64),
        "agent_types": np.array([1, 1, 1], dtype=np.int64),
        "agent_count": 3,
        "times_s": times,
        "current_xy": current_xy,
        "current_vel_xy": current_vel,
        "current_heading": current_heading,
        "current_size_lw": current_size,
        "future_xy": future_xy,
        "future_vel_xy": future_vel,
        "future_heading": future_heading,
        "future_valid": future_valid,
        "map_lane_centerlines": [
            np.array([[-20, -3.5], [80, -3.5]], dtype=np.float32),
            np.array([[-20, 0.0], [80, 0.0]], dtype=np.float32),
            np.array([[-20, 3.5], [80, 3.5]], dtype=np.float32),
        ],
        "map_crosswalks": [],
        "map_driveways": [],
    }
    sample["label"] = assign_label(sample, cfg)
    write_gzip_pickle(ensure_dir(work_dir / "samples") / "smoke_demo_0001.pkl.gz", sample)

    lib = TubeLibrary.from_workdir(work_dir / "tube_library")
    prim_lib = PrimitiveLibrary.from_workdir(work_dir / "tube_library")
    tensors, feats = sample_to_bev_tensor_and_features(
        sample, lib, cfg, device="cpu", primitive_lib=prim_lib
    )
    assert tensors["ego_rt"].ndim == 3
    assert tensors["others_rt"].shape == tensors["ego_rt"].shape
    assert "rcr" in feats and "msr" in feats and "redi_full" in feats
    assert "asr_slice_final" in feats and "asr_cum_final" in feats
    assert "redi_actionability" in feats
    print("smoke test passed")
    print(
        {
            k: feats.get(k)
            for k in [
                "rcr",
                "rfr",
                "rfr_drv",
                "tfrc_s",
                "msr",
                "asr_slice_final",
                "asr_cum_final",
                "ttad_s",
                "collapse_rate_max_per_s",
                "early_blocking_ratio",
                "redi_full",
                "redi_actionability",
                "redi_actionability_delta",
            ]
        }
    )


def main() -> None:
    configured = os.environ.get("ROF_SMOKE_WORK_DIR")
    if configured:
        _run_smoke(Path(configured))
        return
    with tempfile.TemporaryDirectory(prefix="rof_actionability_smoke_") as tmp:
        _run_smoke(Path(tmp))


if __name__ == "__main__":
    main()
