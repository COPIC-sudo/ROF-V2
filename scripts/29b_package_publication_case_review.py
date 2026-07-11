#!/usr/bin/env python
"""Package publication actionability case panels for human review.

This script only reads existing panel outputs and metadata, then writes a
review bundle with QA files and a zip archive. It does not train models,
regenerate labels/features, or modify pipeline.py.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rtbev.config import load_config
from rtbev.io_utils import ensure_dir


SAMPLE_ORDER = [
    "33e40c2133dc7ed8",
    "ab06a686a52a3fac",
    "2aedd6aa278acd2c",
    "17140261fe2db703",
    "cddf5c8291665dd7",
    "3b007871ad02fbca",
]

CONTACT_TITLES = {
    "33e40c2133dc7ed8": "A. Recovered critical state",
    "ab06a686a52a3fac": "B. Partial actionability depletion",
    "2aedd6aa278acd2c": "C. Proximity warning with high actionability",
    "17140261fe2db703": "D. Proximity warning with high actionability, alternative",
    "cddf5c8291665dd7": "E. Baseline false positive corrected",
    "3b007871ad02fbca": "F. Proximity caution but critical actionability",
}

RECOMMENDED_USE = {
    "33e40c2133dc7ed8": "main_recovered_positive",
    "ab06a686a52a3fac": "supplement_recovered_positive_partial",
    "2aedd6aa278acd2c": "main_proximity_warning_high_actionability",
    "17140261fe2db703": "supplement_proximity_warning_high_actionability",
    "cddf5c8291665dd7": "main_or_supplement_baseline_false_positive_fixed",
    "3b007871ad02fbca": "supplement_concept_proximity_caution_but_critical",
}

FIGURE_PRIORITY = {
    "33e40c2133dc7ed8": 1,
    "2aedd6aa278acd2c": 1,
    "cddf5c8291665dd7": 2,
    "ab06a686a52a3fac": 2,
    "17140261fe2db703": 2,
    "3b007871ad02fbca": 3,
}

REQUIRED_METADATA_FIELDS = [
    "sample_id",
    "intended_use",
    "original_label",
    "actionability_label",
    "distance",
    "TTC",
    "baseline_score",
    "enhanced_score",
    "score_delta",
    "reason_for_selection",
    "output_png",
    "output_pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--out-name", default="publication_case_review_bundle_v1")
    return parser.parse_args()


def _config_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _work_dir(cfg: Dict[str, Any]) -> Path:
    work = _config_get(cfg, "project.work_dir")
    if not work:
        raise ValueError("config project.work_dir is missing")
    return Path(str(work))


def _safe_path(value: Any, panel_dir: Path, fallback: Optional[Path] = None) -> Path:
    if pd.notna(value) and str(value).strip():
        p = Path(str(value))
        if not p.is_absolute():
            p = panel_dir / p
        return p
    if fallback is not None:
        return fallback
    return Path("")


def _infer_panel_file(panel_dir: Path, sample_id: str, suffix: str) -> Optional[Path]:
    matches = sorted(panel_dir.glob(f"*{sample_id}*{suffix}"))
    return matches[0] if matches else None


def _caveat_for(sample_id: str, intended_use: str) -> str:
    if sample_id == "3b007871ad02fbca":
        return "concept/supplement only; not model-success evidence"
    if intended_use.startswith("recovered"):
        return "check whether interaction is not merely distance-proximity"
    if "proximity_warning_high_actionability" in intended_use:
        return "check feasible actions are visible"
    if "baseline_false_positive_fixed" in intended_use:
        return "check actionability reduction is visually plausible"
    return "manual review required"


def _build_review_table(meta: pd.DataFrame, panel_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    present_samples = set(meta["sample_id"].astype(str)) if "sample_id" in meta.columns else set()
    ordered = [sid for sid in SAMPLE_ORDER if sid in present_samples]
    ordered += sorted(present_samples - set(ordered))

    for sample_id in ordered:
        rec = meta[meta["sample_id"].astype(str) == sample_id].iloc[0]
        inferred_png = _infer_panel_file(panel_dir, sample_id, ".png")
        inferred_pdf = _infer_panel_file(panel_dir, sample_id, ".pdf")
        png_path = _safe_path(rec.get("output_png"), panel_dir, inferred_png)
        pdf_path = _safe_path(rec.get("output_pdf"), panel_dir, inferred_pdf)
        intended = str(rec.get("intended_use", ""))
        rows.append(
            {
                "sample_id": sample_id,
                "intended_use": intended,
                "original_label": rec.get("original_label", ""),
                "actionability_label": rec.get("actionability_label", ""),
                "distance": rec.get("distance", np.nan),
                "TTC": rec.get("TTC", np.nan),
                "baseline_score": rec.get("baseline_score", np.nan),
                "enhanced_score": rec.get("enhanced_score", np.nan),
                "score_delta": rec.get("score_delta", np.nan),
                "reason_for_selection": rec.get("reason_for_selection", ""),
                "png_exists": bool(png_path.exists()),
                "pdf_exists": bool(pdf_path.exists()),
                "recommended_use": RECOMMENDED_USE.get(sample_id, "manual_review"),
                "caveat": _caveat_for(sample_id, intended),
                "figure_priority": FIGURE_PRIORITY.get(sample_id, 99),
                "rollout_feasibility_mode": "no_map_collision_only; map drawn as background only",
                "output_png": str(png_path),
                "output_pdf": str(pdf_path),
            }
        )
    return pd.DataFrame(rows)


def _write_caption(out_path: Path, review_df: pd.DataFrame) -> None:
    lines = [
        "# Figure title / 图题",
        "",
        "**Actionability-aware emergency assessment case panels.**",
        "",
        "每个 case 展示当前 BEV interaction、候选 ego action rollout，以及 compact score/label summary。"
        "These panels compare proximity-defined risk labels with the no-map actionability-label interpretation.",
        "",
        "**Important note.** Action rollout feasibility is evaluated in **no-map** mode: collision and future occupancy are used, "
        "but map/drivable constraints are not used for action feasibility. Lane/map geometry is shown only as visual background.",
        "",
        "## Panel notes",
        "",
    ]
    for _, row in review_df.iterrows():
        sample_id = str(row["sample_id"])
        title = CONTACT_TITLES.get(sample_id, sample_id)
        lines.append(f"- **{title}** (`{sample_id}`): {row['reason_for_selection']}")
    lines.extend(
        [
            "",
            "## Draft caption",
            "",
            "Figure X. Representative no-map actionability case panels. Blue boxes denote the ego vehicle, orange boxes denote "
            "surrounding agents, and grey curves show lane/map context and future motion cues. Green action rollouts are feasible "
            "under the no-map collision-only feasibility audit, while red rollouts become infeasible; the annotated time marks "
            "the first unsafe time. Map geometry is rendered as background only and is not used in the feasibility decision. "
            "Panels include recovered critical states, proximity-warning/high-actionability mismatches, a baseline false positive "
            "corrected by actionability features, and one concept/supplement example where a proximity caution state is critical "
            "under no-map actionability.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_contact_sheet(review_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(18, 22))
    axes_flat = axes.ravel()

    for ax in axes_flat:
        ax.axis("off")

    for idx, sample_id in enumerate(SAMPLE_ORDER):
        ax = axes_flat[idx]
        row = review_df[review_df["sample_id"].astype(str) == sample_id]
        if row.empty:
            ax.set_title(CONTACT_TITLES.get(sample_id, sample_id), fontsize=14, fontweight="bold")
            ax.text(0.5, 0.5, "Missing metadata", ha="center", va="center", fontsize=12)
            continue

        rec = row.iloc[0]
        png_path = Path(str(rec["output_png"]))
        ax.set_title(CONTACT_TITLES.get(sample_id, sample_id), fontsize=14, fontweight="bold", pad=8)
        if not png_path.exists():
            ax.text(0.5, 0.5, "Missing PNG", ha="center", va="center", fontsize=12)
            continue
        img = mpimg.imread(png_path)
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout(pad=1.5)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _copy_artifacts(panel_dir: Path, bundle_dir: Path, review_df: pd.DataFrame) -> List[Path]:
    copied: List[Path] = []
    for _, row in review_df.iterrows():
        for col in ("output_png", "output_pdf"):
            src = Path(str(row[col]))
            if src.exists():
                dst = bundle_dir / src.name
                shutil.copy2(src, dst)
                copied.append(dst)
    for name in ("combined_publication_cases.png", "combined_publication_cases.pdf", "publication_case_panel_metadata.csv"):
        src = panel_dir / name
        if src.exists():
            dst = bundle_dir / name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _metadata_completeness(meta: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    missing_columns = [c for c in REQUIRED_METADATA_FIELDS if c not in meta.columns]
    missing_by_sample: Dict[str, List[str]] = {}
    if "sample_id" not in meta.columns:
        return missing_columns, missing_by_sample
    for _, row in meta.iterrows():
        sid = str(row["sample_id"])
        missing = []
        for col in REQUIRED_METADATA_FIELDS:
            if col not in meta.columns or pd.isna(row.get(col)) or str(row.get(col)).strip() == "":
                missing.append(col)
        if missing:
            missing_by_sample[sid] = missing
    return missing_columns, missing_by_sample


def _write_qa_report(
    out_path: Path,
    panel_dir: Path,
    review_df: pd.DataFrame,
    meta: pd.DataFrame,
    contact_png: Path,
    contact_pdf: Path,
    zip_path: Path,
) -> None:
    missing_columns, missing_by_sample = _metadata_completeness(meta)
    duplicate_samples = []
    if "sample_id" in meta.columns:
        counts = meta["sample_id"].astype(str).value_counts()
        duplicate_samples = counts[counts > 1].index.tolist()

    combined_png = panel_dir / "combined_publication_cases.png"
    combined_pdf = panel_dir / "combined_publication_cases.pdf"
    target = review_df[review_df["sample_id"].astype(str) == "3b007871ad02fbca"]
    if target.empty:
        sample_3b_status = "missing from metadata"
    else:
        row = target.iloc[0]
        score_cols = ["baseline_score", "enhanced_score", "score_delta"]
        missing_scores = [c for c in score_cols if pd.isna(row.get(c))]
        if missing_scores:
            sample_3b_status = "missing " + ", ".join(missing_scores)
        else:
            sample_3b_status = (
                f"present: baseline_score={row['baseline_score']}, "
                f"enhanced_score={row['enhanced_score']}, score_delta={row['score_delta']}"
            )

    lines = [
        "# Publication Case Review Bundle QA Report",
        "",
        "## Artifact Checks",
        "",
        "| sample_id | PNG | PDF | recommended_use | caveat |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in review_df.iterrows():
        lines.append(
            f"| {row['sample_id']} | {bool(row['png_exists'])} | {bool(row['pdf_exists'])} | "
            f"{row['recommended_use']} | {row['caveat']} |"
        )

    lines.extend(
        [
            "",
            "## Metadata QA",
            "",
            f"- Metadata file present: {(panel_dir / 'publication_case_panel_metadata.csv').exists()}",
            f"- Required columns missing: {missing_columns if missing_columns else 'none'}",
            f"- Samples with missing metadata fields: {missing_by_sample if missing_by_sample else 'none'}",
            f"- Duplicate sample_id values: {duplicate_samples if duplicate_samples else 'none'}",
            f"- 3b007871ad02fbca score fields: {sample_3b_status}",
            "",
            "## Combined and Contact Sheet QA",
            "",
            f"- combined_publication_cases.png exists: {combined_png.exists()}",
            f"- combined_publication_cases.pdf exists: {combined_pdf.exists()}",
            f"- publication_case_contact_sheet.png exists: {contact_png.exists()}",
            f"- publication_case_contact_sheet.pdf exists: {contact_pdf.exists()}",
            "",
            "## No-map Feasibility Recording",
            "",
            "- no-rollout-use-map is recorded in figure_candidate_review_table.csv as "
            "`rollout_feasibility_mode=no_map_collision_only; map drawn as background only`.",
            "- paper_figure_caption_draft.md also states that map/drivable constraints are not used for action feasibility.",
            "- The original publication_case_panel_metadata.csv did not include an explicit no-map flag; the review bundle records it.",
            "",
            "## Bundle QA",
            "",
            f"- Zip path: {zip_path}",
            f"- Zip exists: {zip_path.exists()}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _zip_bundle(bundle_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_dir))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    work_dir = _work_dir(cfg)
    panel_dir = Path(args.panel_dir)
    if not panel_dir.exists():
        raise FileNotFoundError(f"panel dir not found: {panel_dir}")

    bundle_dir = ensure_dir(work_dir / "results" / "nc_actionability_cases" / args.out_name)
    meta_path = panel_dir / "publication_case_panel_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata CSV not found: {meta_path}")
    meta = pd.read_csv(meta_path)

    review_df = _build_review_table(meta, panel_dir)
    review_path = bundle_dir / "figure_candidate_review_table.csv"
    review_df.to_csv(review_path, index=False)

    _copy_artifacts(panel_dir, bundle_dir, review_df)

    caption_path = bundle_dir / "paper_figure_caption_draft.md"
    _write_caption(caption_path, review_df)

    contact_png = bundle_dir / "publication_case_contact_sheet.png"
    contact_pdf = bundle_dir / "publication_case_contact_sheet.pdf"
    _make_contact_sheet(review_df, contact_png, contact_pdf)

    zip_path = bundle_dir.parent / f"{args.out_name}.zip"
    qa_path = bundle_dir / "qa_report.md"
    _write_qa_report(qa_path, panel_dir, review_df, meta, contact_png, contact_pdf, zip_path)

    # Re-zip after QA is written so all generated files are included.
    _zip_bundle(bundle_dir, zip_path)
    _write_qa_report(qa_path, panel_dir, review_df, meta, contact_png, contact_pdf, zip_path)
    _zip_bundle(bundle_dir, zip_path)

    print(f"[ok] review table: {review_path}")
    print(f"[ok] contact sheet png: {contact_png}")
    print(f"[ok] contact sheet pdf: {contact_pdf}")
    print(f"[ok] qa report: {qa_path}")
    print(f"[ok] zip: {zip_path}")


if __name__ == "__main__":
    main()
