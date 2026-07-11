#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rtbev.baselines.feature_sets import feature_lineage_rows, strict_non_action_current_cv_columns
from rtbev.baselines.oof import grouped_oof_predictions, oof_metrics
from rtbev.external.common import (
    add_config_hash,
    artifact_manifest_rows,
    config_hash,
    experiment_out_dir,
    load_yaml_config,
    resolve_input_path,
    run_manifest,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v111/nc_v111_decoupling_audit.yaml")
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=None, choices=["rf", "logreg", None])
    return parser.parse_args()


def _variant_table_specs(cfg: dict[str, Any], kind: str, fallback_path: Path | None) -> list[dict[str, str]]:
    specs = list((cfg.get("inputs") or {}).get(f"{kind}_variants", []) or [])
    if specs:
        return [{"variant_id": str(s["variant_id"]), "path": str(s["path"])} for s in specs]
    if fallback_path:
        return [{"variant_id": "reference_h3_b3_base7", "path": str(fallback_path)}]
    return []


def _read_variant_tables(specs: list[dict[str, str]], cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for spec in specs:
        path = resolve_input_path(spec["path"], cfg)
        if path and path.exists():
            tables[str(spec["variant_id"])] = pd.read_csv(path)
    return tables


def _label_col(df: pd.DataFrame) -> str:
    for col in ["actionability_label_id", "label_id", "y", "y_true"]:
        if col in df.columns:
            return col
    raise ValueError("label table requires actionability_label_id, label_id, y, or y_true")


def _positive_labels(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    col = _label_col(out)
    value = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    if col in ["y", "y_true"]:
        out["_y"] = value.astype(int)
    else:
        out["_y"] = (value >= 2).astype(int)
    return out


def _merge(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    f = feature_df.copy()
    l = _positive_labels(label_df)
    f["sample_id"] = f["sample_id"].astype(str)
    l["sample_id"] = l["sample_id"].astype(str)
    keep = [c for c in ["sample_id", "scenario_id", "segment_id", "_y"] if c in l.columns]
    out = f.merge(l[keep], on="sample_id", how="inner", suffixes=("", "_label"))
    if "scenario_id" not in out.columns and "scenario_id_label" in out.columns:
        out["scenario_id"] = out["scenario_id_label"]
    if "scenario_id" not in out.columns:
        out["scenario_id"] = out["sample_id"]
    return out


def _report(matrix: pd.DataFrame, lineage: pd.DataFrame) -> str:
    off = matrix[matrix.get("diagonal", False) == False] if not matrix.empty and "diagonal" in matrix else pd.DataFrame()
    lines = [
        "# v111 Decoupling Audit Report",
        "",
        "Feature set is strict_non_action_current_cv: current-state and CV features only; candidate-action-derived features are excluded.",
        "",
        f"- feature lineage rows: {len(lineage)}",
        f"- label-feature matrix rows: {len(matrix)}",
        f"- off-diagonal transfer rows: {len(off)}",
        "",
        "## Best Off-diagonal Rows",
        "",
    ]
    if off.empty:
        lines.append("- no off-diagonal rows")
    else:
        for _, row in off.sort_values("AUPRC", ascending=False).head(10).iterrows():
            lines.append(f"- label={row['label_variant']} feature={row['feature_variant']} AUPRC={row.get('AUPRC'):.6g} Recall@5%FPR={row.get('Recall@5%FPR'):.6g}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    out_dir = experiment_out_dir(cfg, "nc_v111_decoupling_audit")
    eval_cfg = cfg.get("evaluation", {})
    fallback_features = resolve_input_path(args.features_csv or (cfg.get("inputs") or {}).get("features_csv"), cfg)
    fallback_labels = resolve_input_path(args.labels_csv or (cfg.get("inputs") or {}).get("labels_csv"), cfg)
    feature_tables = _read_variant_tables(_variant_table_specs(cfg, "feature", fallback_features), cfg)
    label_tables = _read_variant_tables(_variant_table_specs(cfg, "label", fallback_labels), cfg)
    if not feature_tables or not label_tables:
        raise FileNotFoundError("decoupling audit requires at least one feature table and one label table")
    n_folds = int(args.n_folds if args.n_folds is not None else eval_cfg.get("outer_folds", 5))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("scenario_hash_seed", 42))
    model = str(args.model or eval_cfg.get("model", "rf"))
    cfg_hash = config_hash(args.config)

    all_feature_names = set()
    for df in feature_tables.values():
        all_feature_names.update(df.columns.astype(str))
    lineage_rows = feature_lineage_rows(all_feature_names)
    lineage_df = pd.DataFrame(lineage_rows)

    matrix_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    for label_id, labels in label_tables.items():
        for feature_id, features in feature_tables.items():
            merged = _merge(features, labels)
            cols = strict_non_action_current_cv_columns(merged)
            if not cols:
                matrix_rows.append({"label_variant": label_id, "feature_variant": feature_id, "status": "NO_STRICT_FEATURES"})
                continue
            pred = grouped_oof_predictions(merged, cols, "_y", group_col="scenario_id", n_folds=n_folds, seed=seed, model=model)
            metrics = oof_metrics(pred)
            matrix_rows.append(
                {
                    "label_variant": label_id,
                    "feature_variant": feature_id,
                    "diagonal": bool(label_id == feature_id),
                    "feature_set": "strict_non_action_current_cv",
                    "model": model,
                    "group_col": "scenario_id",
                    "n_features": len(cols),
                    "features": ";".join(cols),
                    "status": "OK" if not pred.empty else "NO_VALID_OOF_FOLDS",
                    **metrics,
                }
            )
            for group_col in ["scenario_id", "segment_id"]:
                if group_col not in merged.columns:
                    continue
                pred_g = grouped_oof_predictions(merged, cols, "_y", group_col=group_col, n_folds=n_folds, seed=seed, model=model)
                m = oof_metrics(pred_g)
                grouped_rows.append(
                    {
                        "label_variant": label_id,
                        "feature_variant": feature_id,
                        "feature_set": "strict_non_action_current_cv",
                        "group_col": group_col,
                        "model": model,
                        "n_features": len(cols),
                        **m,
                    }
                )

    lineage_path = out_dir / "feature_lineage_v111.csv"
    matrix_path = out_dir / "label_feature_mismatch_matrix.csv"
    non_action_path = out_dir / "non_action_feature_oof_metrics.csv"
    grouped_path = out_dir / "grouped_oof_metrics.csv"
    report_path = out_dir / "v111_decoupling_report.md"
    manifest_path = out_dir / "artifact_manifest.csv"
    run_path = out_dir / "run_manifest.json"

    write_csv(lineage_path, add_config_hash(lineage_rows, cfg_hash))
    write_csv(matrix_path, add_config_hash(matrix_rows, cfg_hash))
    write_csv(non_action_path, add_config_hash(matrix_rows, cfg_hash))
    write_csv(grouped_path, add_config_hash(grouped_rows, cfg_hash))
    matrix_df = pd.DataFrame(matrix_rows)
    report_path.write_text(_report(matrix_df, lineage_df), encoding="utf-8")
    outputs = [lineage_path, matrix_path, non_action_path, grouped_path, report_path]
    write_csv(manifest_path, artifact_manifest_rows(args.config, outputs))
    write_json(run_path, run_manifest(args.config, cfg, [*outputs, manifest_path]))
    print(f"[v111-decoupling] matrix_rows={len(matrix_rows)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
