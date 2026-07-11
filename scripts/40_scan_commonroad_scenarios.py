#!/usr/bin/env python
"""Scan CommonRoad XML scenarios and write a scenario manifest.

This script is intentionally independent of the ROF pipeline. It does not
import waymo_open_dataset, torch, or rtbev.pipeline.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from commonroad.common.file_reader import CommonRoadFileReader
except Exception:  # pragma: no cover - handled at runtime
    CommonRoadFileReader = None


FIELDNAMES = [
    "scenario_id",
    "xml_path",
    "xml_rel_path",
    "parse_status",
    "parse_error",
    "file_size_mb",
    "dt",
    "lanelet_count",
    "dynamic_obstacle_count",
    "static_obstacle_count",
    "planning_problem_count",
    "obstacle_type_counts",
    "max_time_step",
    "tags",
    "source_hint",
    "scenario_family",
    "difficulty_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--commonroad-root", required=True)
    parser.add_argument("--out-name", default="commonroad_scenario_manifest.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-xml-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


def _load_work_dir(config_path: str) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = _expand(yaml.safe_load(f) or {})
    work_dir = (cfg.get("project") or {}).get("work_dir") or os.environ.get("ROF_WORK_DIR")
    if not work_dir:
        raise ValueError("project.work_dir missing in config and ROF_WORK_DIR is not set")
    return Path(str(work_dir))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _short_error(exc: BaseException, limit: int = 500) -> str:
    msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return msg[:limit]


def _scenario_family(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    if "-" in stem:
        return stem.split("-", 1)[0]
    return path.parent.name


def _source_hint(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix().lower()
    tokens: List[str] = []
    country_tokens = ["deu", "usa", "zam", "chn", "rus", "esp", "fra", "ita", "gbr"]
    road_tokens = [
        "intersection",
        "highway",
        "freeway",
        "merging",
        "merge",
        "ramp",
        "urban",
        "city",
        "crossing",
        "handcrafted",
        "lanechange",
        "us101",
        "i80",
    ]
    for token in country_tokens + road_tokens:
        if token in rel:
            tokens.append(token.upper() if token in country_tokens else token)
    return "|".join(dict.fromkeys(tokens)) if tokens else "unknown"


def _keyword_bonus(path: Path, root: Path) -> float:
    rel = path.relative_to(root).as_posix().lower()
    bonus = 0.0
    if any(k in rel for k in ["intersection", "urban", "crossing", "city"]):
        bonus += 10.0
    if any(k in rel for k in ["merging", "merge", "ramp"]):
        bonus += 8.0
    if any(k in rel for k in ["highway", "freeway", "us101", "i80"]):
        bonus += 6.0
    if any(k in rel for k in ["obstacle", "blocked", "critical", "dangerous"]):
        bonus += 10.0
    return bonus


def _difficulty(row: Dict[str, Any], path: Path, root: Path) -> float:
    return (
        _safe_float(row.get("dynamic_obstacle_count"), 0.0) * 2.0
        + _safe_float(row.get("lanelet_count"), 0.0) * 0.2
        + _safe_float(row.get("planning_problem_count"), 0.0) * 2.0
        + _safe_float(row.get("max_time_step"), 0.0) * 0.01
        + _keyword_bonus(path, root)
    )


def _obstacle_type_counts(obstacles: Iterable[Any]) -> str:
    counts: Counter[str] = Counter()
    for obs in obstacles:
        typ = getattr(obs, "obstacle_type", "unknown")
        name = getattr(typ, "value", str(typ))
        counts[str(name)] += 1
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def _planning_problem_count(pp_set: Any) -> int:
    if pp_set is None:
        return 0
    if hasattr(pp_set, "planning_problem_dict"):
        return len(pp_set.planning_problem_dict)
    if hasattr(pp_set, "planning_problem_set"):
        return len(pp_set.planning_problem_set)
    try:
        return len(pp_set)
    except Exception:
        return 0


def _max_prediction_time(dynamic_obstacles: Iterable[Any]) -> int:
    max_t = 0
    for obs in dynamic_obstacles:
        pred = getattr(obs, "prediction", None)
        if pred is None:
            continue
        traj = getattr(pred, "trajectory", None)
        if traj is not None:
            for attr in ("final_state", "initial_state"):
                state = getattr(traj, attr, None)
                max_t = max(max_t, _safe_int(getattr(state, "time_step", 0), 0))
            states = getattr(traj, "state_list", None)
            if states:
                for state in states:
                    max_t = max(max_t, _safe_int(getattr(state, "time_step", 0), 0))
        occs = getattr(pred, "occupancy_set", None)
        if occs:
            for occ in occs:
                max_t = max(max_t, _safe_int(getattr(occ, "time_step", 0), 0))
    return max_t


def _parse_commonroad_io(path: Path) -> Dict[str, Any]:
    if CommonRoadFileReader is None:
        raise ImportError("commonroad-io CommonRoadFileReader is not available")
    scenario, planning_problem_set = CommonRoadFileReader(str(path)).open()
    lanelet_network = getattr(scenario, "lanelet_network", None)
    lanelets = getattr(lanelet_network, "lanelets", []) if lanelet_network is not None else []
    dynamic_obstacles = list(getattr(scenario, "dynamic_obstacles", []) or [])
    static_obstacles = list(getattr(scenario, "static_obstacles", []) or [])
    tags = getattr(scenario, "tags", "")
    if isinstance(tags, (set, list, tuple)):
        tags = ";".join(sorted(str(t) for t in tags))
    all_obstacles = dynamic_obstacles + static_obstacles
    return {
        "scenario_id": str(getattr(scenario, "scenario_id", path.stem)),
        "dt": getattr(scenario, "dt", ""),
        "lanelet_count": len(lanelets),
        "dynamic_obstacle_count": len(dynamic_obstacles),
        "static_obstacle_count": len(static_obstacles),
        "planning_problem_count": _planning_problem_count(planning_problem_set),
        "obstacle_type_counts": _obstacle_type_counts(all_obstacles),
        "max_time_step": _max_prediction_time(dynamic_obstacles),
        "tags": str(tags or ""),
    }


def _iter_elements(root: ET.Element, local_name: str) -> Iterable[ET.Element]:
    for elem in root.iter():
        if _local_name(elem.tag) == local_name:
            yield elem


def _text_of_first(elem: ET.Element, names: List[str]) -> str:
    for child in elem.iter():
        if _local_name(child.tag) in names and child.text:
            return child.text.strip()
    return ""


def _xml_max_time(root: ET.Element) -> int:
    max_t = 0
    for elem in root.iter():
        lname = _local_name(elem.tag).lower()
        if lname in {"timestep", "time_step"}:
            if elem.text:
                max_t = max(max_t, _safe_int(elem.text.strip(), 0))
            for value in elem.attrib.values():
                max_t = max(max_t, _safe_int(value, 0))
        if lname == "time":
            for child in elem:
                if _local_name(child.tag).lower() == "exact" and child.text:
                    max_t = max(max_t, _safe_int(child.text.strip(), 0))
    return max_t


def _parse_xml_fallback(path: Path) -> Dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    dynamic = list(_iter_elements(root, "dynamicObstacle"))
    static = list(_iter_elements(root, "staticObstacle"))
    obstacle_counts: Counter[str] = Counter()
    for obs in dynamic + static:
        typ = _text_of_first(obs, ["type"]) or "unknown"
        obstacle_counts[typ] += 1
    tag_values: List[str] = []
    for tag_elem in _iter_elements(root, "tags"):
        for child in tag_elem:
            if child.text:
                tag_values.append(child.text.strip())
    return {
        "scenario_id": root.attrib.get("benchmarkID") or root.attrib.get("commonRoadVersion") or path.stem,
        "dt": root.attrib.get("timeStepSize", ""),
        "lanelet_count": len(list(_iter_elements(root, "lanelet"))),
        "dynamic_obstacle_count": len(dynamic),
        "static_obstacle_count": len(static),
        "planning_problem_count": len(list(_iter_elements(root, "planningProblem"))),
        "obstacle_type_counts": ";".join(f"{k}:{v}" for k, v in sorted(obstacle_counts.items())),
        "max_time_step": _xml_max_time(root),
        "tags": ";".join(sorted(set(tag_values))),
    }


def _scan_one(path: Path, root: Path, allow_xml_fallback: bool) -> Dict[str, Any]:
    row: Dict[str, Any] = {name: "" for name in FIELDNAMES}
    row.update(
        {
            "scenario_id": path.stem,
            "xml_path": str(path),
            "xml_rel_path": path.relative_to(root).as_posix(),
            "file_size_mb": f"{path.stat().st_size / (1024 * 1024):.6f}",
            "source_hint": _source_hint(path, root),
            "scenario_family": _scenario_family(path),
        }
    )
    commonroad_error = ""
    try:
        parsed = _parse_commonroad_io(path)
        row.update(parsed)
        row["parse_status"] = "ok"
        row["parse_error"] = ""
    except Exception as exc:
        commonroad_error = _short_error(exc)
        if allow_xml_fallback:
            try:
                parsed = _parse_xml_fallback(path)
                row.update(parsed)
                row["parse_status"] = "ok"
                row["parse_error"] = f"commonroad_io_failed_fallback_ok: {commonroad_error}"
            except Exception as fallback_exc:
                row["parse_status"] = "failed"
                row["parse_error"] = f"commonroad_io: {commonroad_error}; xml_fallback: {_short_error(fallback_exc)}"
        else:
            row["parse_status"] = "failed"
            row["parse_error"] = commonroad_error
    row["difficulty_score"] = f"{_difficulty(row, path, root):.6f}"
    return row


def _describe(values: List[float]) -> Dict[str, str]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {k: "" for k in ["count", "mean", "std", "min", "p25", "p50", "p75", "max"]}
    def q(frac: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = frac * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        w = pos - lo
        return vals[lo] * (1 - w) + vals[hi] * w
    return {
        "count": str(len(vals)),
        "mean": f"{statistics.fmean(vals):.6f}",
        "std": f"{statistics.pstdev(vals):.6f}" if len(vals) > 1 else "0.000000",
        "min": f"{vals[0]:.6f}",
        "p25": f"{q(0.25):.6f}",
        "p50": f"{q(0.50):.6f}",
        "p75": f"{q(0.75):.6f}",
        "max": f"{vals[-1]:.6f}",
    }


def _write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    ok_rows = [r for r in rows if r.get("parse_status") == "ok"]
    failed_rows = [r for r in rows if r.get("parse_status") != "ok"]
    summary_fields = [
        "section",
        "metric",
        "value",
        "count",
        "mean",
        "std",
        "min",
        "p25",
        "p50",
        "p75",
        "max",
        "scenario_id",
        "difficulty_score",
        "xml_rel_path",
    ]
    out: List[Dict[str, Any]] = []
    out.extend(
        [
            {"section": "counts", "metric": "total_xml", "value": len(rows)},
            {"section": "counts", "metric": "ok_count", "value": len(ok_rows)},
            {"section": "counts", "metric": "failed_count", "value": len(failed_rows)},
        ]
    )
    for metric in ["dynamic_obstacle_count", "lanelet_count", "planning_problem_count"]:
        desc = _describe([_safe_float(r.get(metric), None) for r in ok_rows])
        rec = {"section": "describe", "metric": metric, "value": ""}
        rec.update(desc)
        out.append(rec)
    source_counts = Counter(str(r.get("source_hint") or "unknown") for r in ok_rows)
    for source, count in source_counts.most_common():
        out.append({"section": "source_hint_distribution", "metric": "source_hint", "value": source, "count": count})
    top = sorted(ok_rows, key=lambda r: _safe_float(r.get("difficulty_score"), 0.0), reverse=True)[:20]
    for r in top:
        out.append(
            {
                "section": "top20_difficulty",
                "metric": "difficulty_score",
                "value": "",
                "scenario_id": r.get("scenario_id"),
                "difficulty_score": r.get("difficulty_score"),
                "xml_rel_path": r.get("xml_rel_path"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for rec in out:
            writer.writerow({name: rec.get(name, "") for name in summary_fields})


def main() -> None:
    args = parse_args()
    commonroad_root = Path(args.commonroad_root)
    if not commonroad_root.exists():
        raise FileNotFoundError(f"CommonRoad root not found: {commonroad_root}")
    work_dir = _load_work_dir(args.config)
    out_dir = work_dir / "results" / "commonroad_pilot_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.out_name
    summary_path = out_dir / "commonroad_scenario_scan_summary.csv"

    xml_files = sorted(commonroad_root.rglob("*.xml"))
    if args.limit is not None:
        xml_files = xml_files[: args.limit]
    print(f"[scan] root={commonroad_root}")
    print(f"[scan] xml_count={len(xml_files)}")
    print(f"[scan] commonroad_io_available={CommonRoadFileReader is not None}")

    rows: List[Dict[str, Any]] = []
    for idx, xml_path in enumerate(xml_files, start=1):
        rows.append(_scan_one(xml_path, commonroad_root, args.allow_xml_fallback))
        if idx % 100 == 0 or idx == len(xml_files):
            ok = sum(1 for r in rows if r.get("parse_status") == "ok")
            failed = len(rows) - ok
            print(f"[scan] processed={idx}/{len(xml_files)} ok={ok} failed={failed}", flush=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, summary_path)
    ok_count = sum(1 for r in rows if r.get("parse_status") == "ok")
    print(f"[done] manifest={manifest_path}")
    print(f"[done] summary={summary_path}")
    print(f"[done] total={len(rows)} ok={ok_count} failed={len(rows) - ok_count}")


if __name__ == "__main__":
    main()
