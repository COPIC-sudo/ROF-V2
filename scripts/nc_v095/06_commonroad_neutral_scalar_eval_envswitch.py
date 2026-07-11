#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import _bootstrap  # noqa: F401,E402

from rtbev.config import load_config
from rtbev.pipeline import sample_to_bev_tensor_and_features
from rtbev.tube.rt_library import PrimitiveLibrary, TubeLibrary

from _utils import append_blockers, load_yaml, output_dir, resolve_path, write_csv


SCORES = [
    "distance_inverse",
    "TTC_inverse",
    "ROF_v2_composite",
    "ROF_v2_no_asr_composite",
    "temporal_composite",
]

COMPARISONS = [
    ("ROF_v2_composite", "distance_inverse"),
    ("ROF_v2_composite", "TTC_inverse"),
    ("ROF_v2_no_asr_composite", "distance_inverse"),
    ("ROF_v2_no_asr_composite", "TTC_inverse"),
    ("temporal_composite", "distance_inverse"),
    ("temporal_composite", "TTC_inverse"),
]

METRICS = ["auroc", "auprc", "recall_at_5pct_fpr"]


def read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def clear_resolved_commonroad_scalar_blocker(out_dir: Path) -> None:
    path = out_dir / ("BLOCKERS_V095_ENVSWITCH.csv" if "envswitch" in out_dir.name.lower() else "BLOCKERS_V095.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    if {"category", "item"}.issubset(df.columns):
        keep = ~(
            df["category"].astype(str).eq("commonroad_neutral_confirmation")
            & df["item"].astype(str).eq("neutral_scalar_features")
        )
        write_csv(path, df[keep].to_dict("records"))


def numeric_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return ""
    return ""


def sample_path(row: pd.Series, samples_dir: Path) -> Path:
    explicit = str(row.get("json_gz_path", ""))
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    return samples_dir / f"{row['sample_id']}.json.gz"


def extract_features(cfg_path: Path, out_dir: Path, sample_out_name: str, planner_out_name: str, device: str) -> tuple[Path, Path]:
    cfg = load_config(str(cfg_path))
    cfg.setdefault("runtime", {})["device"] = device
    work_dir = Path(cfg["project"]["work_dir"])
    manifest_path = work_dir / "results" / "commonroad_samples" / sample_out_name / "commonroad_dynamic_ego_samples_manifest.csv"
    samples_dir = work_dir / "results" / "commonroad_samples" / sample_out_name / "samples_json_gz"
    feature_path = out_dir / "commonroad_neutral_scalar_scores.csv"
    feature_failures_path = out_dir / "commonroad_neutral_scalar_feature_failures.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"neutral sample manifest missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.get("export_status", "ok").fillna("ok").astype(str) == "ok"].copy()
    lib = TubeLibrary.from_workdir(work_dir / "tube_library")
    prim_lib = PrimitiveLibrary.from_workdir(work_dir / "tube_library")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        data = row._asdict()
        sid = str(data.get("sample_id", ""))
        t0 = time.perf_counter()
        out: dict[str, Any] = {
            "sample_id": sid,
            "commonroad_scenario_id": data.get("commonroad_scenario_id", ""),
            "scenario_id": data.get("commonroad_scenario_id", sid),
            "status": "failed",
            "error": "",
        }
        try:
            sample = read_json_gz(sample_path(pd.Series(data), samples_dir))
            _, feats = sample_to_bev_tensor_and_features(
                sample,
                lib,
                cfg,
                device=device,
                primitive_lib=prim_lib,
                return_tensors=False,
            )
            out["status"] = "ok"
            out["runtime_s"] = time.perf_counter() - t0
            for key, value in feats.items():
                v = numeric_scalar(value)
                if v != "":
                    out[key] = v
            for key in ["agent_count", "lanelet_count", "current_min_distance_m", "current_ttc_s", "ego_speed_mps"]:
                if key not in out and key in data:
                    out[key] = data.get(key, "")
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["runtime_s"] = time.perf_counter() - t0
            failures.append({"sample_id": sid, "error": out["error"]})
        rows.append(out)
        if idx == 1 or idx % 100 == 0 or idx == len(manifest):
            print(f"[neutral-features] {idx}/{len(manifest)} ok={sum(r.get('status') == 'ok' for r in rows)} failed={len(failures)}")
    pd.DataFrame(rows).to_csv(feature_path, index=False)
    pd.DataFrame(failures).to_csv(feature_failures_path, index=False)
    return feature_path, feature_failures_path


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def norm_clip(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    finite = np.isfinite(arr.to_numpy())
    out = pd.Series(np.nan, index=arr.index, dtype=float)
    if finite.sum() == 0:
        return out
    lo = float(np.nanpercentile(arr.to_numpy()[finite], 1))
    hi = float(np.nanpercentile(arr.to_numpy()[finite], 99))
    if hi <= lo:
        out.loc[finite] = 0.0
    else:
        out.loc[finite] = np.clip((arr.to_numpy()[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    distance = num(out.get("current_min_distance_m", pd.Series(index=out.index, dtype=float)))
    out["distance_inverse"] = -distance
    ttc = num(out.get("current_ttc_s", pd.Series(index=out.index, dtype=float)))
    finite_nonnegative = ttc[(ttc >= 0.0) & np.isfinite(ttc)]
    safe_max = float(finite_nonnegative.max() + 10.0) if len(finite_nonnegative) else 999.0
    out["TTC_inverse"] = -ttc.where(ttc >= 0.0, safe_max)
    redi = num(out.get("redi_actionability", pd.Series(index=out.index, dtype=float)))
    asr_cum = num(out.get("asr_cum_final", pd.Series(index=out.index, dtype=float)))
    msr = num(out.get("msr", pd.Series(index=out.index, dtype=float)))
    out["ASR_cum_inverse"] = 1.0 - asr_cum.where(np.isfinite(asr_cum), msr)
    asr_slice = num(out.get("asr_slice_final", pd.Series(index=out.index, dtype=float)))
    out["ASR_slice_inverse"] = 1.0 - asr_slice
    ttad = num(out.get("ttad_s", pd.Series(index=out.index, dtype=float)))
    finite_ttad = ttad[np.isfinite(ttad)]
    ttad_safe_max = float(finite_ttad.max() + 1.0) if len(finite_ttad) else 999.0
    out["TTAD_inverse"] = -ttad.where(np.isfinite(ttad), ttad_safe_max)
    collapse = num(out.get("collapse_rate_max_per_s", pd.Series(index=out.index, dtype=float)))
    out["collapse_rate"] = collapse
    out["ROF_v2_composite"] = pd.concat(
        [norm_clip(redi), norm_clip(out["ASR_cum_inverse"]), norm_clip(out["TTAD_inverse"]), norm_clip(out["collapse_rate"])],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["ROF_v2_no_asr_composite"] = pd.concat(
        [norm_clip(redi), norm_clip(out["TTAD_inverse"]), norm_clip(out["collapse_rate"])],
        axis=1,
    ).mean(axis=1, skipna=True)
    temporal = [norm_clip(out["TTAD_inverse"]), norm_clip(out["collapse_rate"])]
    if "time_to_first_conflict_s" in out.columns:
        first = num(out["time_to_first_conflict_s"])
        finite = first[np.isfinite(first)]
        first_safe = float(finite.max() + 1.0) if len(finite) else 999.0
        temporal.append(norm_clip(-first.where(np.isfinite(first), first_safe)))
    if "early_blocking_ratio" in out.columns:
        temporal.append(norm_clip(num(out["early_blocking_ratio"])))
    out["temporal_composite"] = pd.concat(temporal, axis=1).mean(axis=1, skipna=True)
    return out


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reason = df.get("planner_failure_reason", pd.Series("", index=df.index)).fillna("").astype(str)
    failure = num(df["planner_failure"]).fillna(0).astype(int)
    frames = {}
    all_frame = df.copy()
    all_frame["_y"] = (failure == 1).astype(int)
    frames["planner_failure_all"] = all_frame
    known = df[~((failure == 1) & (reason == "unknown"))].copy()
    known_reason = known.get("planner_failure_reason", pd.Series("", index=known.index)).fillna("").astype(str)
    known_failure = num(known["planner_failure"]).fillna(0).astype(int)
    known["_y"] = ((known_failure == 1) & (known_reason != "unknown")).astype(int)
    frames["planner_failure_known"] = known
    return frames


def metric_value(y: np.ndarray, score: np.ndarray, metric: str, threshold: float | None = None) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    if metric == "auroc":
        return float(roc_auc_score(y, score))
    if metric == "auprc":
        return float(average_precision_score(y, score))
    if metric == "recall_at_5pct_fpr":
        if threshold is None:
            neg = score[y == 0]
            threshold = float(np.quantile(neg, 0.95))
        pos = y == 1
        return float(np.mean(score[pos] >= threshold)) if np.any(pos) else np.nan
    raise ValueError(metric)


def scenario_hash(value: str, seed: int = 42) -> float:
    h = hashlib.sha1(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12 - 1)


def calibration_thresholds(frame: pd.DataFrame, score_names: list[str]) -> dict[str, float]:
    scenario = frame["commonroad_scenario_id"].astype(str)
    cal = scenario.map(lambda x: scenario_hash(x) < 0.35)
    if cal.nunique() < 2:
        cal = pd.Series(np.arange(len(frame)) % 3 == 0, index=frame.index)
    thresholds = {}
    for score in score_names:
        neg = pd.to_numeric(frame.loc[cal & (frame["_y"] == 0), score], errors="coerce")
        neg = neg[np.isfinite(neg)]
        thresholds[score] = float(np.quantile(neg, 0.95)) if len(neg) else np.nan
    return thresholds


def bootstrap_task(task: str, frame: pd.DataFrame, thresholds: dict[str, float], n_bootstrap: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    scenario = frame["commonroad_scenario_id"].astype(str).to_numpy()
    uniq = np.unique(scenario)
    group_indices = [np.where(scenario == s)[0] for s in uniq]
    y = frame["_y"].to_numpy(int)
    score_arrays = {s: pd.to_numeric(frame[s], errors="coerce").to_numpy(float) for s in SCORES}
    point_rows = []
    delta_rows = []
    boot_points = {(s, m): [] for s in SCORES for m in METRICS}
    boot_deltas = {(e, b, m): [] for e, b in COMPARISONS for m in METRICS}
    for _ in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        for score_name in SCORES:
            for metric in METRICS:
                val = metric_value(yy, score_arrays[score_name][idx], metric, thresholds.get(score_name) if metric == "recall_at_5pct_fpr" else None)
                if np.isfinite(val):
                    boot_points[(score_name, metric)].append(val)
        for enhanced, baseline in COMPARISONS:
            for metric in METRICS:
                ev = metric_value(yy, score_arrays[enhanced][idx], metric, thresholds.get(enhanced) if metric == "recall_at_5pct_fpr" else None)
                bv = metric_value(yy, score_arrays[baseline][idx], metric, thresholds.get(baseline) if metric == "recall_at_5pct_fpr" else None)
                if np.isfinite(ev) and np.isfinite(bv):
                    boot_deltas[(enhanced, baseline, metric)].append(ev - bv)
    for score_name in SCORES:
        for metric in METRICS:
            arr = np.asarray(boot_points[(score_name, metric)], dtype=float)
            point = metric_value(y, score_arrays[score_name], metric, thresholds.get(score_name) if metric == "recall_at_5pct_fpr" else None)
            point_rows.append(
                {
                    "task": task,
                    "score": score_name,
                    "metric": metric,
                    "point": point,
                    "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                    "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                    "n_bootstrap_valid": int(len(arr)),
                    "n_samples": int(len(frame)),
                    "positive_count": int(y.sum()),
                    "positive_rate": float(np.mean(y)),
                    "threshold_at_5pct_fpr": thresholds.get(score_name, np.nan) if metric == "recall_at_5pct_fpr" else np.nan,
                }
            )
    for enhanced, baseline in COMPARISONS:
        for metric in METRICS:
            arr = np.asarray(boot_deltas[(enhanced, baseline, metric)], dtype=float)
            ev = metric_value(y, score_arrays[enhanced], metric, thresholds.get(enhanced) if metric == "recall_at_5pct_fpr" else None)
            bv = metric_value(y, score_arrays[baseline], metric, thresholds.get(baseline) if metric == "recall_at_5pct_fpr" else None)
            delta = ev - bv if np.isfinite(ev) and np.isfinite(bv) else np.nan
            delta_rows.append(
                {
                    "task": task,
                    "enhanced_score": enhanced,
                    "baseline_score": baseline,
                    "metric": metric,
                    "enhanced_point": ev,
                    "baseline_point": bv,
                    "delta": delta,
                    "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                    "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                    "P_delta_gt_0": float(np.mean(arr > 0)) if len(arr) else np.nan,
                    "n_bootstrap_valid": int(len(arr)),
                }
            )
    return point_rows, delta_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nc_v095/nc_v095_p0_extension_envswitch.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-feature-extraction", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = output_dir(cfg)
    base_cfg = resolve_path(cfg["inputs"]["base_config"])
    work_cfg = load_yaml(base_cfg)
    work_dir = Path(str((work_cfg.get("project") or {})["work_dir"]))
    sample_out_name = str(cfg["commonroad"].get("neutral_sample_out_name", "commonroad_neutral_envswitch"))
    planner_out_name = str(cfg["commonroad"].get("neutral_planner_out_name", "neutral_envswitch"))
    labels_path = out_dir / "commonroad_neutral_planner_labels.csv"
    if not labels_path.exists():
        source = work_dir / "results" / "commonroad_planner_feasibility" / planner_out_name / f"commonroad_lattice_planner_labels_{planner_out_name}.csv"
        if source.exists():
            pd.read_csv(source).to_csv(labels_path, index=False)
        else:
            raise FileNotFoundError(f"neutral planner labels missing: {labels_path}; source={source}")
    feature_path = out_dir / "commonroad_neutral_scalar_scores.csv"
    if not args.skip_feature_extraction or not feature_path.exists():
        feature_path, _ = extract_features(base_cfg, out_dir, sample_out_name, planner_out_name, args.device)
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(labels_path)
    features = features[features.get("status", "ok").fillna("ok").astype(str) == "ok"].copy()
    merged = features.merge(labels, on="sample_id", suffixes=("", "_planner"), how="inner")
    if "commonroad_scenario_id" not in merged.columns and "commonroad_scenario_id_planner" in merged.columns:
        merged["commonroad_scenario_id"] = merged["commonroad_scenario_id_planner"]
    merged = add_scores(merged)
    merged.to_csv(out_dir / "commonroad_neutral_scalar_scores.csv", index=False)
    frames = split_frames(merged)
    jobs = []
    for task, frame in frames.items():
        thresholds = calibration_thresholds(frame, SCORES)
        seed = int(hashlib.sha1(task.encode("utf-8")).hexdigest()[:8], 16)
        jobs.append(delayed(bootstrap_task)(task, frame, thresholds, int(args.n_bootstrap), seed))
    results = Parallel(n_jobs=int(args.n_jobs))(jobs)
    point_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for points, deltas in results:
        point_rows.extend(points)
        delta_rows.extend(deltas)
    write_csv(out_dir / "commonroad_neutral_metrics.csv", point_rows)
    write_csv(out_dir / "commonroad_neutral_deltas.csv", delta_rows)
    write_csv(out_dir / "commonroad_neutral_scenario_bootstrap.csv", point_rows + delta_rows)
    taxonomy = labels.get("planner_failure_reason", pd.Series(["missing"] * len(labels))).fillna("missing").astype(str).value_counts()
    write_csv(out_dir / "commonroad_neutral_failure_taxonomy.csv", [{"failure_reason": k, "count": int(v), "fraction": float(v / max(len(labels), 1))} for k, v in taxonomy.items()])
    known_deltas = pd.DataFrame(delta_rows)
    known_auprc = known_deltas[(known_deltas["task"] == "planner_failure_known") & (known_deltas["metric"] == "auprc")]
    supported = known_auprc[pd.to_numeric(known_auprc["ci_low"], errors="coerce") > 0]
    fail_count = int(pd.to_numeric(labels.get("planner_failure", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    known_count = int(((pd.to_numeric(labels.get("planner_failure", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int) == 1) & (labels.get("planner_failure_reason", pd.Series("", index=labels.index)).fillna("").astype(str) != "unknown")).sum())
    status = "PASS" if len(supported) >= 2 and known_count >= 20 else ("PASS_WITH_NARROW_WORDING" if known_count > 0 else "FAIL")
    lines = [
        "# CommonRoad Neutral Claim Gate (EnvSwitch)",
        "",
        f"Status: `{status}`",
        "",
        f"Neutral planner labels: {len(labels)} rows.",
        f"All planner failures: {fail_count}; known planner failures: {known_count}.",
        f"Known-failure AUPRC deltas with CI_low > 0: {len(supported)}.",
        "",
        "Primary endpoint excludes unknown/parser/numerical/software failures from positives.",
        "Thresholds use a deterministic scenario-hash calibration split for nominal 5% FPR.",
    ]
    (out_dir / "commonroad_neutral_claim_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    clear_resolved_commonroad_scalar_blocker(out_dir)
    if status != "PASS":
        append_blockers(
            out_dir,
            [
                {
                    "category": "commonroad_neutral_confirmation",
                    "item": "neutral_scalar_claim_strength",
                    "status": status,
                    "details": f"known_failures={known_count}; supported_known_auprc_deltas={len(supported)}",
                    "resume_command": "Inspect commonroad_neutral_metrics.csv and commonroad_neutral_deltas.csv; increase neutral cohort or refine endpoint only with explicit scientific approval.",
                }
            ],
        )
    print(f"[commonroad-neutral-scalar] labels={len(labels)} merged={len(merged)} known_failures={known_count} status={status}")


if __name__ == "__main__":
    main()
