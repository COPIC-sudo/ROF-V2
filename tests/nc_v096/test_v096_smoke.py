from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(path: Path):
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_moderate_label_rule() -> None:
    mod = load_module(ROOT / "scripts" / "nc_v096" / "01_generate_design_variant_labels.py")
    assert mod.label_moderate(1.0, 1.0) == 0
    assert mod.label_moderate(0.5, 0.5) == 1
    assert mod.label_moderate(0.25, 1.0) == 2
    assert mod.label_moderate(1.0, 0.0) == 3


def test_action_libraries() -> None:
    mod = load_module(ROOT / "scripts" / "nc_v096" / "01_generate_design_variant_labels.py")
    base = mod.action_library("base7")
    ext = mod.action_library("extended")
    assert len(base) == 7
    assert len(ext) > len(base)
    assert [a["name"] for a in base][:7] == [a["name"] for a in ext][:7]


def test_config_variants_exist() -> None:
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "nc_v096" / "nc_v096_endpoint_design_robustness.yaml").read_text(encoding="utf-8"))
    variants = cfg["actionability_labels"]["variants"]
    ids = {v["variant_id"] for v in variants}
    assert "reference_h3_b3_base7_skip" in ids
    assert "future_h3_b3_base7_cvfallback" in ids
    assert len(variants) == 7


if __name__ == "__main__":
    test_moderate_label_rule()
    test_action_libraries()
    test_config_variants_exist()
    print("tests/nc_v096 fallback smoke passed")
