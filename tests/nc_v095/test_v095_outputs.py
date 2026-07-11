#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def test_config_points_to_v095_namespace() -> None:
    cfg = yaml.safe_load(Path("configs/nc_v095/nc_v095_p0_extension.yaml").read_text(encoding="utf-8"))
    assert cfg["project"]["output_dir"] == "results/nc_v095_p0_extension"
    assert "pipeline.py" not in "\n".join(Path("configs/nc_v095/nc_v095_p0_extension.yaml").read_text(encoding="utf-8").splitlines())


def test_secondary_bootstrap_schema_if_present() -> None:
    path = Path("results/nc_v095_p0_extension/waymo_secondary_context_bootstrap_v095.csv")
    if not path.exists():
        return
    df = pd.read_csv(path, nrows=5)
    required = {
        "endpoint",
        "model",
        "seed",
        "comparison",
        "metric",
        "baseline_point",
        "enhanced_point",
        "delta",
        "ci_low",
        "ci_high",
        "bootstrap_unit",
    }
    assert required.issubset(set(df.columns))


def main() -> None:
    test_config_points_to_v095_namespace()
    test_secondary_bootstrap_schema_if_present()
    print("tests/nc_v095/test_v095_outputs.py PASS")


if __name__ == "__main__":
    main()
