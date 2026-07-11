from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir


CONTEXT_FEATURE_COLS = [
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "agent_count",
    "redi_actionability",
    "ttad_s",
    "time_to_first_conflict_s",
    "early_blocking_ratio",
    "collapse_rate_max_per_s",
    "asr_cum_final",
    "asr_slice_final",
    "comfort_asr",
    "emergency_asr",
]

OUTPUT_COLS = [
    "case_category",
    "task",
    "sample_id",
    "scenario_id",
    "original_label_id",
    "original_label_name",
    "actionability_label_id",
    "actionability_label_name",
    "baseline_group",
    "enhanced_group",
    "baseline_score",
    "enhanced_score",
    "score_delta",
    "y_true",
    "baseline_alert_at_5fpr",
    "enhanced_alert_at_5fpr",
    "baseline_alert_at_1fpr",
    "enhanced_alert_at_1fpr",
    "current_min_distance_m",
    "current_ttc_s",
    "ego_speed_kph",
    "agent_count",
    "redi_actionability",
    "ttad_s",
    "time_to_first_conflict_s",
    "early_blocking_ratio",
    "collapse_rate_max_per_s",
    "asr_cum_final",
    "asr_slice_final",
    "comfort_asr",
    "emergency_asr",
    "baseline_threshold_at_5fpr",
    "enhanced_threshold_at_5fpr",
    "baseline_threshold_at_1fpr",
    "enhanced_threshold_at_1fpr",
]


def _parse_csv_arg(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"CSV must include sample_id: {path}; columns={list(df.columns)}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def _coalesce(df: pd.DataFrame, cols: list[str], default="") -> pd.Series:
    out = pd.Series(default, index=df.index)
    for col in cols:
        if col in df.columns:
            s = df[col]
            out = out.where(out.astype(str).ne(str(default)), s)
            out = out.where(out.notna(), s)
    return out


def _threshold_at_fpr(y_true: pd.Series, score: pd.Series, fpr: float) -> float:
    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).to_numpy()
    s = pd.to_numeric(score, errors="coerce").to_numpy(float)
    ok = np.isfinite(s)
    neg = s[ok & (y == 0)]
    if len(neg) == 0:
        return np.nan
    return float(np.quantile(neg, 1.0 - float(fpr)))


def _prediction_group_col(pred: pd.DataFrame, feature_column: str | None = None) -> str:
    if feature_column:
        if feature_column not in pred.columns:
            raise ValueError(f"predictions missing requested feature column {feature_column}; columns={list(pred.columns)}")
        return feature_column
    for col in ["group", "feature_group", "feature_set"]:
        if col in pred.columns:
            return col
    raise ValueError(f"predictions must include group, feature_group, or feature_set; columns={list(pred.columns)}")


def _prepare_context(
    features_csv: str,
    proximity_labels_csv: str,
    actionability_labels_csv: str,
) -> pd.DataFrame:
    features = _read_csv(features_csv)
    feat_cols = ["sample_id"] + [c for c in ["scenario_id", "label_id", "label_name"] + CONTEXT_FEATURE_COLS if c in features.columns]
    ctx = features[feat_cols].copy()
    rename = {}
    if "scenario_id" in ctx.columns:
        rename["scenario_id"] = "feature_scenario_id"
    if "label_id" in ctx.columns:
        rename["label_id"] = "feature_label_id"
    if "label_name" in ctx.columns:
        rename["label_name"] = "feature_label_name"
    ctx = ctx.rename(columns=rename)

    prox = _read_csv(proximity_labels_csv)
    prox_cols = ["sample_id"] + [c for c in ["scenario_id", "label_id", "label_name"] if c in prox.columns]
    prox = prox[prox_cols].copy().rename(
        columns={
            "scenario_id": "proximity_scenario_id",
            "label_id": "proximity_label_id",
            "label_name": "proximity_label_name",
        }
    )
    ctx = ctx.merge(prox, on="sample_id", how="left")

    act = _read_csv(actionability_labels_csv)
    act_cols = ["sample_id"] + [
        c
        for c in [
            "scenario_id",
            "original_label_id",
            "original_label_name",
            "actionability_label_id",
            "actionability_label_name",
        ]
        if c in act.columns
    ]
    act = act[act_cols].copy().rename(columns={"scenario_id": "actionability_scenario_id"})
    ctx = ctx.merge(act, on="sample_id", how="left")

    ctx["scenario_id_context"] = _coalesce(
        ctx,
        ["actionability_scenario_id", "proximity_scenario_id", "feature_scenario_id", "sample_id"],
        default="",
    ).astype(str)
    if "original_label_id" not in ctx.columns:
        ctx["original_label_id"] = np.nan
    ctx["original_label_id"] = pd.to_numeric(
        ctx["original_label_id"].where(ctx["original_label_id"].notna(), ctx.get("proximity_label_id")),
        errors="coerce",
    )
    if ctx["original_label_id"].isna().any() and "feature_label_id" in ctx.columns:
        ctx["original_label_id"] = ctx["original_label_id"].where(
            ctx["original_label_id"].notna(),
            pd.to_numeric(ctx["feature_label_id"], errors="coerce"),
        )
    if "original_label_name" not in ctx.columns:
        ctx["original_label_name"] = ""
    if "proximity_label_name" in ctx.columns:
        ctx["original_label_name"] = ctx["original_label_name"].where(
            ctx["original_label_name"].astype(str).ne(""),
            ctx["proximity_label_name"],
        )
    keep = ["sample_id", "scenario_id_context", "original_label_id", "original_label_name", "actionability_label_id", "actionability_label_name"]
    keep += [c for c in CONTEXT_FEATURE_COLS if c in ctx.columns]
    return ctx[keep].drop_duplicates("sample_id").copy()


def _prepare_task_predictions(
    pred: pd.DataFrame,
    task: str,
    baseline_group: str,
    enhanced_group: str,
    feature_column: str | None,
) -> pd.DataFrame:
    group_col = _prediction_group_col(pred, feature_column)
    required = ["sample_id", "task", "y_true", "score"]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"predictions missing required columns {missing}; columns={list(pred.columns)}")
    pred = pred.copy()
    pred["sample_id"] = pred["sample_id"].astype(str)
    if "scenario_id" not in pred.columns:
        pred["scenario_id"] = pred["sample_id"]
    pred["scenario_id"] = pred["scenario_id"].fillna(pred["sample_id"]).astype(str)
    pred["task"] = pred["task"].astype(str)
    pred[group_col] = pred[group_col].astype(str)
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
    pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce").astype(int)
    sub = pred[pred["task"].eq(task)].copy()
    base = sub[sub[group_col].eq(baseline_group)].copy()
    enh = sub[sub[group_col].eq(enhanced_group)].copy()
    if base.empty:
        raise ValueError(f"no baseline predictions for task={task}, group={baseline_group}")
    if enh.empty:
        raise ValueError(f"no enhanced predictions for task={task}, group={enhanced_group}")
    if "model" in base.columns and "model" in enh.columns:
        common_models = sorted(set(base["model"].astype(str)).intersection(set(enh["model"].astype(str))))
        if common_models:
            model = common_models[0]
            if len(common_models) > 1:
                print(f"[case-selection] multiple models for task={task}; using model={model}")
            base = base[base["model"].astype(str).eq(model)]
            enh = enh[enh["model"].astype(str).eq(model)]
    base = base[["sample_id", "scenario_id", "y_true", "score"]].rename(columns={"score": "baseline_score"})
    enh = enh[["sample_id", "y_true", "score"]].rename(columns={"score": "enhanced_score", "y_true": "y_true_enhanced"})
    merged = base.merge(enh, on="sample_id", how="inner")
    if merged.empty:
        raise ValueError(f"empty prediction merge for task={task}")
    if (merged["y_true"].astype(int) != merged["y_true_enhanced"].astype(int)).any():
        raise ValueError(f"y_true mismatch between baseline/enhanced for task={task}")
    merged = merged.drop(columns=["y_true_enhanced"])
    merged["task"] = task
    merged["baseline_group"] = baseline_group
    merged["enhanced_group"] = enhanced_group
    merged["score_delta"] = merged["enhanced_score"] - merged["baseline_score"]
    return merged


def _add_threshold_flags(df: pd.DataFrame, fpr_level: float, also_fpr1: bool) -> pd.DataFrame:
    out = df.copy()
    b5 = _threshold_at_fpr(out["y_true"], out["baseline_score"], fpr_level)
    e5 = _threshold_at_fpr(out["y_true"], out["enhanced_score"], fpr_level)
    out["baseline_threshold_at_5fpr"] = b5
    out["enhanced_threshold_at_5fpr"] = e5
    out["baseline_alert_at_5fpr"] = out["baseline_score"] >= b5 if np.isfinite(b5) else False
    out["enhanced_alert_at_5fpr"] = out["enhanced_score"] >= e5 if np.isfinite(e5) else False
    if also_fpr1:
        b1 = _threshold_at_fpr(out["y_true"], out["baseline_score"], 0.01)
        e1 = _threshold_at_fpr(out["y_true"], out["enhanced_score"], 0.01)
        out["baseline_threshold_at_1fpr"] = b1
        out["enhanced_threshold_at_1fpr"] = e1
        out["baseline_alert_at_1fpr"] = out["baseline_score"] >= b1 if np.isfinite(b1) else False
        out["enhanced_alert_at_1fpr"] = out["enhanced_score"] >= e1 if np.isfinite(e1) else False
    else:
        out["baseline_threshold_at_1fpr"] = np.nan
        out["enhanced_threshold_at_1fpr"] = np.nan
        out["baseline_alert_at_1fpr"] = False
        out["enhanced_alert_at_1fpr"] = False
    return out


def _select_top(df: pd.DataFrame, category: str, mask: pd.Series, sort_col: str, top_k: int, ascending: bool = False) -> pd.DataFrame:
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(sort_col, ascending=ascending).head(int(top_k)).copy()
    sub.insert(0, "case_category", category)
    return sub


def _select_cases_for_task(df: pd.DataFrame, top_k: int, also_fpr1: bool) -> list[pd.DataFrame]:
    out = []
    y_pos = df["y_true"].astype(int).eq(1)
    y_neg = ~y_pos
    out.append(_select_top(
        df,
        "recovered_positive_at_5fpr",
        y_pos & (~df["baseline_alert_at_5fpr"]) & df["enhanced_alert_at_5fpr"],
        "score_delta",
        top_k,
        ascending=False,
    ))
    if also_fpr1:
        out.append(_select_top(
            df,
            "recovered_positive_at_1fpr",
            y_pos & (~df["baseline_alert_at_1fpr"]) & df["enhanced_alert_at_1fpr"],
            "score_delta",
            top_k,
            ascending=False,
        ))
    pos_df = df.loc[y_pos].copy()
    if not pos_df.empty:
        b_low = float(pos_df["baseline_score"].quantile(0.50))
        e_high = float(pos_df["enhanced_score"].quantile(0.75))
        mask_c = y_pos & (df["baseline_score"] <= b_low) & (df["enhanced_score"] >= e_high)
        if not mask_c.any():
            mask_c = y_pos
        out.append(_select_top(df, "baseline_missed_enhanced_high_score", mask_c, "score_delta", top_k, ascending=False))
    out.append(_select_top(
        df,
        "enhanced_false_positive_at_5fpr",
        y_neg & (~df["baseline_alert_at_5fpr"]) & df["enhanced_alert_at_5fpr"],
        "enhanced_score",
        top_k,
        ascending=False,
    ))
    tmp = df.copy()
    tmp["baseline_minus_enhanced_score"] = tmp["baseline_score"] - tmp["enhanced_score"]
    out.append(_select_top(
        tmp,
        "baseline_false_positive_fixed",
        y_neg & df["baseline_alert_at_5fpr"] & (~df["enhanced_alert_at_5fpr"]),
        "baseline_minus_enhanced_score",
        top_k,
        ascending=False,
    ))
    orig = pd.to_numeric(df["original_label_id"], errors="coerce")
    act = pd.to_numeric(df["actionability_label_id"], errors="coerce")
    out.append(_select_top(
        df,
        "original_safe_but_actionability_critical",
        orig.isin([0, 1]) & (act >= 2),
        "enhanced_score",
        top_k,
        ascending=False,
    ))
    out.append(_select_top(
        df,
        "original_warning_emergency_but_high_actionability",
        (orig >= 2) & (act == 0),
        "baseline_score",
        top_k,
        ascending=False,
    ))
    if str(df["task"].iloc[0]) == "actionability_infeasible":
        out.append(_select_top(
            df,
            "infeasible_true_positive",
            y_pos & df["enhanced_alert_at_5fpr"],
            "enhanced_score",
            top_k,
            ascending=False,
        ))
        out.append(_select_top(
            df,
            "infeasible_missed",
            y_pos & (~df["enhanced_alert_at_5fpr"]),
            "enhanced_score",
            top_k,
            ascending=True,
        ))
    return [x for x in out if x is not None and not x.empty]


def _summary(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(columns=[
            "task",
            "case_category",
            "count",
            "original_label_id_distribution",
            "actionability_label_id_distribution",
            "current_min_distance_m_median",
            "current_ttc_s_median",
            "enhanced_score_median",
            "score_delta_median",
        ])
    rows = []
    for (task, category), sub in cases.groupby(["task", "case_category"], dropna=False):
        orig_dist = pd.to_numeric(sub["original_label_id"], errors="coerce").value_counts().sort_index()
        act_dist = pd.to_numeric(sub["actionability_label_id"], errors="coerce").value_counts().sort_index()
        rows.append({
            "task": task,
            "case_category": category,
            "count": int(len(sub)),
            "original_label_id_distribution": ";".join(f"{int(k)}:{int(v)}" for k, v in orig_dist.items()),
            "actionability_label_id_distribution": ";".join(f"{int(k)}:{int(v)}" for k, v in act_dist.items()),
            "current_min_distance_m_median": float(pd.to_numeric(sub["current_min_distance_m"], errors="coerce").median()),
            "current_ttc_s_median": float(pd.to_numeric(sub["current_ttc_s"], errors="coerce").median()),
            "enhanced_score_median": float(pd.to_numeric(sub["enhanced_score"], errors="coerce").median()),
            "score_delta_median": float(pd.to_numeric(sub["score_delta"], errors="coerce").median()),
        })
    return pd.DataFrame(rows).sort_values(["task", "case_category"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Select interpretable actionability cases from existing feature-group predictions.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--proximity-labels-csv", required=True)
    ap.add_argument("--actionability-labels-csv", required=True)
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--out-name", default="actionability_case_selection")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tasks", default="actionability_critical,actionability_infeasible")
    ap.add_argument("--baseline-group", default="strong_baseline_cv")
    ap.add_argument("--enhanced-group", default="strong_baseline_cv_actionability_no_direct_ratios")
    ap.add_argument("--feature-column", default=None)
    ap.add_argument("--fpr-level", type=float, default=0.05)
    ap.add_argument("--also-select-fpr1", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out_dir = ensure_dir(work / "results" / "nc_actionability_cases" / args.out_name)
    context = _prepare_context(args.features_csv, args.proximity_labels_csv, args.actionability_labels_csv)
    pred = pd.read_csv(args.predictions_csv)
    selected = []
    for task in _parse_csv_arg(args.tasks):
        task_df = _prepare_task_predictions(pred, task, args.baseline_group, args.enhanced_group, args.feature_column)
        task_df = _add_threshold_flags(task_df, float(args.fpr_level), bool(args.also_select_fpr1))
        task_df = task_df.merge(context, on="sample_id", how="left")
        task_df["scenario_id"] = task_df["scenario_id"].where(
            task_df["scenario_id"].notna(),
            task_df["scenario_id_context"],
        )
        task_df["scenario_id"] = task_df["scenario_id"].fillna(task_df["sample_id"]).astype(str)
        selected.extend(_select_cases_for_task(task_df, int(args.top_k), bool(args.also_select_fpr1)))
    cases = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=OUTPUT_COLS)
    for col in OUTPUT_COLS:
        if col not in cases.columns:
            cases[col] = np.nan
    cases = cases[OUTPUT_COLS].copy()
    summary = _summary(cases)
    by_category = (
        cases.groupby(["case_category", "task"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["case_category", "task"])
        if not cases.empty
        else pd.DataFrame(columns=["case_category", "task", "count"])
    )
    cases.to_csv(out_dir / "selected_cases_all.csv", index=False)
    summary.to_csv(out_dir / "selected_cases_summary.csv", index=False)
    by_category.to_csv(out_dir / "selected_cases_by_category.csv", index=False)
    print(f"[case-selection] wrote {out_dir}")
    print(f"[case-selection] selected_rows={len(cases)}")


if __name__ == "__main__":
    main()
