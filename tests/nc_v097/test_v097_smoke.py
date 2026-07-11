from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "nc_v097"))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_libraries() -> None:
    mod = load_module(ROOT / "scripts" / "nc_v097" / "01_generate_aligned_features.py")
    assert len(mod.action_library("base7")) == 7
    assert len(mod.action_library("extended")) == 15
    assert {a["name"] for a in mod.action_library("base7")} <= {a["name"] for a in mod.action_library("extended")}


def test_variant_config_horizon_buffer() -> None:
    mod = load_module(ROOT / "scripts" / "nc_v097" / "01_generate_aligned_features.py")
    cfg = {"labels": {"dt_s": 0.1}, "dataset": {}, "tube": {}, "runtime": {}, "bev": {}}
    out = mod.make_variant_cfg(cfg, horizon_s=4.0, lane_buffer_m=2.0)
    assert out["dataset"]["max_future_steps"] == 40
    assert out["tube"]["horizon_s"] == 4.0
    assert out["tube"]["query_dt_s"] == 0.1
    assert out["bev"]["lane_buffer_m"] == 2.0


def test_temporal_field_contract() -> None:
    gen = load_module(ROOT / "scripts" / "nc_v097" / "01_generate_aligned_features.py")
    eval_mod = load_module(ROOT / "scripts" / "nc_v097" / "02_aligned_oof_eval.py")
    required = {
        "ttad_s",
        "time_to_first_conflict_s",
        "early_blocking_ratio",
        "collapse_rate_max_per_s",
        "collapse_rate_mean_per_s",
    }
    assert required <= set(gen.TEMPORAL_FIELDS)
    assert required <= set(eval_mod.TEMPORAL_FIELDS)


if __name__ == "__main__":
    test_action_libraries()
    test_variant_config_horizon_buffer()
    test_temporal_field_contract()
    print("v097_smoke_ok")
