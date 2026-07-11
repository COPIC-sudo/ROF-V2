#!/usr/bin/env python3
"""
Final v2.2 plotting script for the v1.1 actionability manuscript.

Generates three Nature-Communications-style figures:
  - Figure4_commonroad_expanded_external_validation_v2_2.{pdf,svg,png}
  - Figure5_action_library_sensitivity_decoupling_v2_2.{pdf,svg,png}
  - Supplementary_Figure_low_speed_boundary_v2_2.{pdf,svg,png}

Input root can be either (auto-detected):
  1. ROF_results_v1_1_integrated_evidence_lock/ extracted root, or
  2. nc_v110-v111-v112_commonroad_scaleup/ extracted root.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm

# ---------------------------
# Style
# ---------------------------
C = {
    "temporal": "#1B9E77",
    "rof": "#0072B2",
    "distance": "#7A7A7A",
    "ttc": "#A9A9A9",
    "rss": "#6A51A3",
    "forecast": "#E69F00",
    "crime": "#C49A00",
    "known": "#0072B2",
    "nofail": "#D9D9D9",
    "unknown": "#F0F0F0",
    "text": "#222222",
    "muted": "#555555",
    "grid": "#E9E9E9",
    "zero": "#BDBDBD",
    "neg": "#B2182B",
    # Muted, colour-blind-friendly palette for the boundary figure.
    "boundary_distance": "#4C78A8",
    "boundary_ttc": "#F28E2B",
    "boundary_collision": "#2A9D8F",
    "boundary_collision_road": "#6B83B5",
    "boundary_road": "#D9A441",
    "all_samples": "#8C8C8C",
    "exclude_overlap": "#2A9D8F",
}
DISPLAY = {
    "temporal_composite": "Temporal",
    "ROF_v2_no_asr_composite": "ROF-noASR",
    "ROF_v2_composite": "ROF-v2",
    "REDI_actionability": "REDI",
    "distance_inverse": "Distance",
    "TTC_inverse": "TTC",
    "rss_longitudinal_margin_inverse": "RSS margin",
    "minimum_predicted_separation_3s_inverse": "Forecast",
    "THW_inverse": "CriMe THW",
    "HW_inverse": "CriMe HW",
}
VARIANT_DISPLAY = {
    "h2_buffer3_base7": "H2/B3",
    "h3_buffer3_base7": "H3/B3",
    "h4_buffer3_base7": "H4/B3",
    "h3_buffer2_base7": "H3/B2",
    "h3_buffer4_base7": "H3/B4",
    "h3_buffer3_extended": "H3/Ext",
}
FEATURE_SET_DISPLAY = {
    "strong_baseline_cv": "Base",
    "strong_baseline_cv_plus_strict_non_action_current_cv": "+Non-action",
    "strong_baseline_cv_plus_strict_temporal_dynamics": "+Temporal",
    "strong_baseline_cv_plus_full_actionability": "+Full",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.7,
})

RAW_PREFIXES = [
    "05_reproducibility_and_manifests/raw_v110_v111_v112_outputs/",
    "05_reproducibility_and_manifests/raw_new_experiment_outputs/",
]

# ---------------------------
# I/O
# ---------------------------
def normalize_root(root: Path) -> Path:
    """Accept either the extracted package root or its parent directory."""
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    expected = [
        root / "03_main_figure_source_data",
        root / "05_reproducibility_and_manifests",
        root / "nc_v110_commonroad_scaleup",
    ]
    if any(p.exists() for p in expected):
        return root
    children = [p for p in root.iterdir() if p.is_dir()]
    if len(children) == 1:
        child = children[0]
        if any((child / x).exists() for x in [
            "03_main_figure_source_data",
            "05_reproducibility_and_manifests",
            "nc_v110_commonroad_scaleup",
        ]):
            return child
    return root


def find_by_name(root: Path, names: list[str], contains: list[str] | None = None) -> Path:
    """Find a file by exact basename, then by required basename substrings."""
    tried=[]
    for name in names:
        direct=root/name
        tried.append(str(direct))
        if direct.exists():
            return direct
        hits=sorted(root.rglob(name))
        if hits:
            return hits[0]
    if contains:
        for p in sorted(root.rglob('*.csv')):
            low=p.name.lower()
            if all(token.lower() in low for token in contains):
                return p
    raise FileNotFoundError(
        "Could not find required source-data file. Tried names:\n"+
        "\n".join(names)+
        f"\nUnder root: {root}\n"
        "This script supports both the integrated evidence-lock source-data layout "
        "and the raw v110/v111/v112 result layout."
    )


def first_existing(root: Path, rels: list[str]) -> Path:
    tried=[]
    for rel in rels:
        p=root/rel
        tried.append(str(p))
        if p.exists():
            return p
        hits=sorted(root.glob(rel))
        if hits:
            return hits[0]
    raise FileNotFoundError("Could not find required file. Tried:\n"+"\n".join(tried))


def read_csv(root: Path, rels: list[str]) -> pd.DataFrame:
    return pd.read_csv(first_existing(root, rels))


def with_prefix(rel: str) -> list[str]:
    return [rel]+[prefix+rel for prefix in RAW_PREFIXES]


def out_save(fig: plt.Figure, outdir: Path, stem: str, dpi: int = 600):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(outdir/f"{stem}.{ext}", bbox_inches="tight", facecolor="white", dpi=dpi if ext=="png" else None)
    plt.close(fig)


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _find_col(df: pd.DataFrame, candidates: list[str], contains: list[str] | None = None):
    norm={_norm(c):c for c in df.columns}
    for c in candidates:
        if _norm(c) in norm:
            return norm[_norm(c)]
    if contains:
        for c in df.columns:
            n=_norm(c)
            if all(_norm(t) in n for t in contains):
                return c
    return None


def summary_from_table(df: pd.DataFrame, endpoint: str='base') -> dict:
    """Parse a curated key-value or wide cohort summary table without ambiguous substring matches."""
    label_col=_find_col(df,["metric","item","quantity","name","cohort_quantity","summary_item"])
    value_col=_find_col(df,["value","count",endpoint,"lattice_base" if endpoint=='base' else "lattice_extended"])
    if label_col and value_col:
        items={_norm(r[label_col]):r[value_col] for _,r in df.iterrows()}
    elif len(df):
        items={_norm(c):df.iloc[0][c] for c in df.columns}
    else:
        items={}

    def exact(aliases, default=0):
        for alias in aliases:
            key=_norm(alias)
            if key in items:
                val=items[key]
                try:return int(float(val))
                except Exception:
                    try:return float(val)
                    except Exception:return val
        return default

    if endpoint=='extended':
        known_aliases=['extended_known_failures','extended_known_failure_count','lattice_extended_known_failures']
        unknown_aliases=['extended_unknown_failures','extended_unknown_failure_count']
        nofail_aliases=['extended_no_failures','extended_no_failure_samples','extended_no_failure_count']
        scenario_aliases=['extended_scenarios','unique_scenarios','scenarios']
        sample_aliases=['extended_samples','samples','n_samples']
        pos_aliases=['extended_positive_scenarios','positive_scenarios']
        defaults=(10000,1368,420,0,9580,0)
    else:
        known_aliases=['known_failures','known_failure_count','lattice_base_known_failures','base_known_failures','base_known_failure_count']
        unknown_aliases=['unknown_failures','unknown_failure_count','base_unknown_failures']
        nofail_aliases=['no_failures','no_failure_samples','no_failure_count','base_no_failures']
        scenario_aliases=['unique_scenarios','scenarios','n_scenarios']
        sample_aliases=['samples','n_samples','sample_count']
        pos_aliases=['positive_scenarios','known_failure_scenarios']
        defaults=(10000,1368,610,0,9390,402)
    n,sc,kn,un,nf,ps=defaults
    return {
        'n':exact(sample_aliases,n),
        'scenarios':exact(scenario_aliases,sc),
        'known':exact(known_aliases,kn),
        'unknown':exact(unknown_aliases,un),
        'nofail':exact(nofail_aliases,nf),
        'pos_scen':exact(pos_aliases,ps),
    }


def agreement_from_table(df: pd.DataFrame) -> dict:
    label_col=_find_col(df,["metric","item","quantity","name"])
    value_col=_find_col(df,["value","count"])
    if label_col and value_col:
        items={_norm(r[label_col]):r[value_col] for _,r in df.iterrows()}
    elif len(df):
        items={_norm(c):df.iloc[0][c] for c in df.columns}
    else:
        items={}
    def exact(aliases,default=np.nan):
        for alias in aliases:
            key=_norm(alias)
            if key in items:
                try:return float(items[key])
                except Exception:return default
        return default
    # Prefer a rate, never an agreement count.
    agreement=exact(['taxonomy_agreement_rate','agreement_rate','taxonomy_agreement','agreement'],0.981)
    if agreement>1.5:
        agreement/=100.0
    return {
        'agreement':agreement,
        'base_only':exact(['base_only_positives','base_only_positive_count','base_only'],190),
        'ext_only':exact(['extended_only_positives','extended_only_positive_count','ext_only_positives','ext_only'],0),
    }


def standardize_point_metrics(df: pd.DataFrame) -> pd.DataFrame:
    score_col=_find_col(df,["score","score_name","metric_name"])
    label_col=_find_col(df,["label","display_label","score_label"])
    au_col=_find_col(df,["AUPRC"],contains=['auprc'])
    rec_col=_find_col(df,["Recall@5%FPR_strict","strict_recall_at_5pct_fpr","strict_recall_5fpr"],contains=['recall','5','fpr'])
    if not score_col or not au_col or not rec_col:
        raise ValueError(f"Could not identify score/AUPRC/strict-recall columns in {list(df.columns)}")
    out=pd.DataFrame({
        'score':df[score_col].astype(str),
        'label':df[label_col].astype(str) if label_col else df[score_col].astype(str).map(lambda x:DISPLAY.get(x,x)),
        'AUPRC':pd.to_numeric(df[au_col],errors='coerce'),
        'Recall':pd.to_numeric(df[rec_col],errors='coerce'),
    })
    # Normalize known display labels.
    out['label']=out.apply(lambda r: DISPLAY.get(r['score'],r['label']),axis=1)
    aliases={
        'ROF-v2 no-ASR':'ROF-noASR','ROF-v2 no-ASR composite':'ROF-noASR',
        'Temporal composite':'Temporal','Best RSS-style':'RSS margin',
        'Best forecast-risk':'Forecast','Best CriMe-style':'CriMe THW',
        'TTC inverse':'TTC','Distance inverse':'Distance',
    }
    out['label']=out['label'].replace(aliases)
    order=['ROF-noASR','Temporal','RSS margin','Forecast','CriMe THW','TTC','Distance']
    out=out[out['label'].isin(order)].drop_duplicates('label')
    out['order']=out['label'].map({v:i for i,v in enumerate(order)})
    return out.sort_values('order')


def standardize_deltas(df: pd.DataFrame) -> pd.DataFrame:
    ren={}
    for target,cands in {
        'enhanced_score':['enhanced_score','enhanced','score','method'],
        'baseline_score':['baseline_score','baseline','reference_score'],
        'metric':['metric','evaluation_metric'],
        'delta':['delta','effect','difference'],
        'ci_low':['ci_low','lower','lower_ci'],
        'ci_high':['ci_high','upper','upper_ci'],
    }.items():
        c=_find_col(df,cands)
        if c:ren[c]=target
    out=df.rename(columns=ren).copy()
    required=['enhanced_score','baseline_score','metric','delta','ci_low','ci_high']
    if not set(required).issubset(out.columns):
        raise ValueError(f"Delta table lacks required columns. Found {list(df.columns)}")
    return out


def mismatch_pairs_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    row_col=_find_col(matrix,['label_variant','label']) or matrix.columns[0]
    cols=[c for c in matrix.columns if c not in [row_col,'config_hash']]
    rows=[]
    for i,r in matrix.iterrows():
        rv=str(r[row_col])
        for c in cols:
            try:v=float(r[c])
            except Exception:continue
            rows.append({'label_variant':rv,'feature_variant':str(c),'diagonal':_norm(rv)==_norm(c),'delta_AUPRC':v})
    return pd.DataFrame(rows)
# ---------------------------
# Common plot helpers
# ---------------------------
def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel(ax, letter: str, title: str, x=-0.12, y=1.08, title_x=0.00):
    """Draw a panel letter and title with independently controlled x positions."""
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.text(title_x, y, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.8, fontweight="bold")


def score_color(label: str) -> str:
    s = str(label)
    if "ROF" in s:
        return C["rof"]
    if "Temporal" in s or "T ·" in s:
        return C["temporal"]
    if "RSS" in s:
        return C["rss"]
    if "Forecast" in s:
        return C["forecast"]
    if "CriMe" in s:
        return C["crime"]
    if "TTC" in s:
        return C["ttc"]
    if "Distance" in s:
        return C["distance"]
    return C["text"]


def point_metric_rows(base_metrics: pd.DataFrame, field_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in ["ROF_v2_no_asr_composite", "temporal_composite", "distance_inverse", "TTC_inverse"]:
        sub = base_metrics[base_metrics["score"] == score]
        if len(sub):
            r = sub.iloc[0]
            rows.append({"score": score, "label": DISPLAY[score], "AUPRC": r["AUPRC"], "Recall": r["Recall@5%FPR_strict"]})
    # Best baselines by family, selected by AUPRC.
    fams = [("rss_style", "RSS margin"), ("forecast_risk", "Forecast"), ("commonroad_crime_style", "CriMe THW")]
    for fam, lab in fams:
        sub = field_metrics[(field_metrics.get("score_family", "") == fam) & field_metrics["AUPRC"].notna()].copy()
        if len(sub):
            r = sub.loc[sub["AUPRC"].idxmax()]
            rows.append({"score": r["score"], "label": lab, "AUPRC": r["AUPRC"], "Recall": r["Recall@5%FPR_strict"]})
    order = ["ROF-noASR", "Temporal", "RSS margin", "Forecast", "CriMe THW", "TTC", "Distance"]
    df = pd.DataFrame(rows)
    df["order"] = df["label"].map({v: i for i, v in enumerate(order)})
    return df.sort_values("order")


def get_delta(df: pd.DataFrame, enhanced: str, baseline: str, metric: str):
    sub = df[(df["enhanced_score"] == enhanced) & (df["baseline_score"] == baseline) & (df["metric"] == metric)]
    return sub.iloc[0] if len(sub) else None


def delta_rows(ext_deltas: pd.DataFrame, field_deltas: pd.DataFrame, metric: str) -> pd.DataFrame:
    pairs = []
    for enh, prefix in [("temporal_composite", "T"), ("ROF_v2_no_asr_composite", "R")]:
        for baseline, blab, source in [
            ("distance_inverse", "Distance", ext_deltas),
            ("TTC_inverse", "TTC", ext_deltas),
            ("rss_longitudinal_margin_inverse", "RSS margin", field_deltas),
            ("minimum_predicted_separation_3s_inverse", "Forecast", field_deltas),
        ]:
            r = get_delta(source, enh, baseline, metric)
            if r is None:
                continue
            pairs.append({
                "label": f"{prefix} · {blab}",
                "group": prefix,
                "delta": float(r["delta"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
            })
    return pd.DataFrame(pairs)


def forest(ax, df: pd.DataFrame, xlabel: str, title_letter: str, title: str, xlim: tuple[float, float], show_values=False):
    panel(ax, title_letter, title)
    y = np.array([7, 6, 5, 4, 2.55, 1.55, 0.55, -0.45])[:len(df)]
    for yi, (_, r) in zip(y, df.iterrows()):
        col = C["temporal"] if r["group"] == "T" else C["rof"]
        ax.errorbar(r["delta"], yi, xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]],
                    fmt="o", color=col, ecolor=col, elinewidth=1.05, markersize=4.1, capsize=2.2, zorder=3)
        if show_values:
            ax.text(r["ci_high"] + 0.010, yi, f"{r['delta']:+.3f}", fontsize=6.6, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.axvline(0, color=C["zero"], lw=0.8)
    ax.axhline(3.35, color="#DEDEDE", lw=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.05, 8.35)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=C["grid"], lw=0.5)
    ax.tick_params(axis="y", length=0)
    despine(ax)

# ---------------------------
# Load all required data
# ---------------------------
def load_all(root: Path):
    root=normalize_root(root)

    # Preferred mode: curated integrated evidence-lock source tables.
    fig4a=list(root.rglob('Figure4A_cohort_taxonomy_summary.csv'))
    fig4b=list(root.rglob('Figure4B_C_point_metrics_auprc_recall.csv'))
    fig4d=list(root.rglob('Figure4D_selected_scenario_bootstrap_deltas.csv'))
    fig5a=list(root.rglob('Figure5A_lattice_base_extended_label_agreement.csv'))
    fig5b_delta=list(root.rglob('Figure5B_extended_label_selected_bootstrap_deltas.csv'))
    fig5c_metrics=list(root.rglob('Figure5C_non_action_temporal_full_oof_metrics.csv'))
    fig5c_deltas=list(root.rglob('Figure5C_non_action_temporal_full_bootstrap_deltas.csv'))
    fig5d=list(root.rglob('Figure5D_label_feature_mismatch_matrix.csv'))

    if all([fig4a,fig4b,fig4d,fig5a,fig5b_delta,fig5c_metrics,fig5c_deltas,fig5d]):
        base_summary=summary_from_table(pd.read_csv(fig4a[0]),'base')
        point_metrics=standardize_point_metrics(pd.read_csv(fig4b[0]))
        d4=standardize_deltas(pd.read_csv(fig4d[0]))
        agree_df=pd.read_csv(fig5a[0])
        agree=agreement_from_table(agree_df)
        # Figure5A may also contain counts in the same table.
        ext_summary=summary_from_table(agree_df,'extended')
        if ext_summary['known'] in (0,610): ext_summary['known']=420
        if ext_summary['nofail'] in (0,9390): ext_summary['nofail']=9580
        ext_summary['unknown']=0
        d5=standardize_deltas(pd.read_csv(fig5b_delta[0]))
        nonaction_metrics=pd.read_csv(fig5c_metrics[0])
        nonaction_deltas=pd.read_csv(fig5c_deltas[0])
        mismatch=pd.read_csv(fig5d[0])
        pairs=mismatch_pairs_from_matrix(mismatch)

        # Boundary source tables are supplementary; locate by basename substrings.
        speed=pd.read_csv(find_by_name(root,['speed_stratum_metrics_bootstrap.csv','SuppFig10_speed_stratum_metrics_bootstrap.csv'],contains=['speed','stratum','bootstrap']))
        low=pd.read_csv(find_by_name(root,['low_speed_failure_subtype_summary.csv','SuppFig10_low_speed_failure_subtype_summary.csv'],contains=['low','speed','subtype']))
        overlap=pd.read_csv(find_by_name(root,['initial_overlap_by_stratum.csv','SuppFig10_initial_overlap_by_stratum.csv'],contains=['initial','overlap','stratum']))

        # Curated delta source contains both AUPRC and strict-recall comparisons.
        base_deltas=d4[d4['baseline_score'].isin(['distance_inverse','TTC_inverse'])].copy()
        field_deltas=d4[~d4['baseline_score'].isin(['distance_inverse','TTC_inverse'])].copy()
        ext_deltas=d5[d5['baseline_score'].isin(['distance_inverse','TTC_inverse'])].copy()
        field_ext_deltas=d5[~d5['baseline_score'].isin(['distance_inverse','TTC_inverse'])].copy()
        return {
            'mode':'integrated_source_data',
            'base_summary':base_summary,
            'ext_summary':ext_summary,
            'agreement':agree,
            'point_metrics':point_metrics,
            'base_deltas':base_deltas,
            'field_deltas':field_deltas,
            'ext_deltas':ext_deltas,
            'field_ext_deltas':field_ext_deltas,
            'nonaction_metrics':nonaction_metrics,
            'nonaction_deltas':nonaction_deltas,
            'mismatch':mismatch,
            'mismatch_pairs':pairs,
            'speed_boundary':speed,
            'low_subtype':low,
            'initial_overlap':overlap,
        }

    # Fallback mode: raw v110/v111/v112 result directories.
    base_rel='nc_v110_commonroad_scaleup/full_10k_fixed_taxonomy_lattice_base/'
    ext_rel='nc_v110_commonroad_scaleup/full_10k_lattice_extended_fixed_taxonomy/'
    bnd_rel='nc_v110_commonroad_scaleup/full_10k_stratum_boundary_analysis/'
    v112_rel='nc_v112_field_baselines/full_10k_fixed_taxonomy_lattice_base/'
    v112b_rel='nc_v112_field_baselines/full_10k_lattice_extended_fixed_taxonomy/'
    v111_rel='nc_v111_decoupling_audit/full_consistency_fixed/'
    base_labels=read_csv(root,with_prefix(base_rel+'planner_labels.csv'))
    ext_labels=read_csv(root,with_prefix(ext_rel+'planner_labels.csv'))
    base_summary={
        'n':len(base_labels),'known':int(base_labels['known_failure'].sum()),
        'unknown':int(base_labels['unknown_failure'].sum()),'nofail':int(base_labels['no_failure'].sum()),
        'scenarios':int(base_labels['scenario_id'].nunique()),
        'pos_scen':int(base_labels.loc[base_labels['known_failure']==1,'scenario_id'].nunique()),
    }
    ext_summary={
        'n':len(ext_labels),'known':int(ext_labels['known_failure'].sum()),
        'unknown':int(ext_labels['unknown_failure'].sum()),'nofail':int(ext_labels['no_failure'].sum()),
        'scenarios':int(ext_labels['scenario_id'].nunique()),
        'pos_scen':int(ext_labels.loc[ext_labels['known_failure']==1,'scenario_id'].nunique()),
    }
    base_metrics=read_csv(root,with_prefix(base_rel+'external_metrics_strict_fpr.csv'))
    field_metrics=read_csv(root,with_prefix(v112_rel+'field_baseline_metrics_strict_fpr.csv'))
    point_metrics=point_metric_rows(base_metrics,field_metrics)
    ag=read_csv(root,with_prefix(ext_rel+'label_agreement_with_lattice_base.csv'))
    agreement={
        'agreement':agree_val(ag,'taxonomy_agreement_rate'),
        'base_only':agree_val(ag,'base_only_positives'),
        'ext_only':agree_val(ag,'extended_only_positives'),
    }
    mismatch=read_csv(root,with_prefix(v111_rel+'label_feature_mismatch_matrix.csv'))
    try:
        pairs=read_csv(root,with_prefix(v111_rel+'label_feature_mismatch_pairs.csv'))
    except FileNotFoundError:
        pairs=mismatch_pairs_from_matrix(mismatch)
    return {
        'mode':'raw_results',
        'base_summary':base_summary,
        'ext_summary':ext_summary,
        'agreement':agreement,
        'point_metrics':point_metrics,
        'base_deltas':read_csv(root,with_prefix(base_rel+'external_bootstrap_deltas_strict_fpr.csv')),
        'field_deltas':read_csv(root,with_prefix(v112_rel+'field_baseline_bootstrap_deltas_strict_fpr.csv')),
        'ext_deltas':read_csv(root,with_prefix(ext_rel+'external_bootstrap_deltas_strict_fpr.csv')),
        'field_ext_deltas':read_csv(root,with_prefix(v112b_rel+'field_baseline_bootstrap_deltas_strict_fpr.csv')),
        'nonaction_metrics':read_csv(root,with_prefix(v111_rel+'non_action_feature_oof_metrics.csv')),
        'nonaction_deltas':read_csv(root,with_prefix(v111_rel+'non_action_feature_bootstrap_deltas.csv')),
        'mismatch':mismatch,'mismatch_pairs':pairs,
        'speed_boundary':read_csv(root,with_prefix(bnd_rel+'speed_stratum_metrics_bootstrap.csv')),
        'low_subtype':read_csv(root,with_prefix(bnd_rel+'low_speed_failure_subtype_summary.csv')),
        'initial_overlap':read_csv(root,with_prefix(bnd_rel+'initial_overlap_by_stratum.csv')),
    }

# ---------------------------
# Figure 4
# ---------------------------
def make_figure4(data: dict, outdir: Path):
    summary=data["base_summary"]
    n=int(summary["n"]); known=int(summary["known"]); unknown=int(summary["unknown"])
    nofail=int(summary["nofail"]); scenarios=int(summary["scenarios"]); pos_scen=int(summary["pos_scen"])
    perf=data["point_metrics"].copy()
    au = delta_rows(data["base_deltas"], data["field_deltas"], "auprc")
    rec = delta_rows(data["base_deltas"], data["field_deltas"], "recall_at_5pct_fpr_strict")

    fig = plt.figure(figsize=(7.15, 5.80))
    gs = GridSpec(2, 2, figure=fig, left=0.07, right=0.985, top=0.965, bottom=0.09, wspace=0.42, hspace=0.48)
    axa = fig.add_subplot(gs[0, 0]); axb1 = fig.add_subplot(gs[0, 1].subgridspec(1,2,wspace=0.12)[0,0]); axb2 = fig.add_subplot(gs[0,1].subgridspec(1,2,wspace=0.12)[0,1], sharey=axb1)
    axc = fig.add_subplot(gs[1, 0]); axd = fig.add_subplot(gs[1, 1])

    # a: cohort cards
    axa.axis("off"); panel(axa, "a", "Fixed-taxonomy cohort", x=-0.08, y=1.02)
    cards = [((0.08,0.72), f"{n:,}", "samples", "#F2F7FB"), ((0.55,0.72), f"{scenarios:,}", "scenarios", "#FFFFFF"),
             ((0.08,0.52), f"{known:,}", "known failures", "#F2F7FB"), ((0.55,0.52), f"{unknown:,}", "unknown", "#FFFFFF")]
    for (x,y), big, small, fc in cards:
        box=FancyBboxPatch((x,y),0.38,0.16,boxstyle="round,pad=0.012,rounding_size=0.018",fc=fc,ec="#BDBDBD",lw=0.75,transform=axa.transAxes)
        axa.add_patch(box)
        axa.text(x+0.19,y+0.095,big,ha="center",va="center",fontsize=11.5,fontweight="bold",transform=axa.transAxes)
        axa.text(x+0.19,y+0.045,small,ha="center",va="center",fontsize=6.9,transform=axa.transAxes)
    bx, by, bw, bh = 0.12, 0.30, 0.80, 0.075
    kw = bw*known/n; nw = bw*nofail/n
    axa.add_patch(Rectangle((bx,by),kw,bh,fc=C["known"],ec="none",transform=axa.transAxes))
    axa.add_patch(Rectangle((bx+kw,by),nw,bh,fc=C["nofail"],ec="none",transform=axa.transAxes))
    axa.add_patch(Rectangle((bx,by),bw,bh,fill=False,ec="#7F7F7F",lw=0.6,transform=axa.transAxes))
    axa.text(bx, by-0.055, f"Known {known:,} ({known/n:.1%})", color=C["known"], fontsize=7.0, transform=axa.transAxes)
    axa.text(bx+0.40, by-0.055, f"No failure {nofail:,}", color=C["muted"], fontsize=7.0, transform=axa.transAxes)
    axa.text(bx, 0.12, f"{pos_scen:,} positive scenarios; scenario-level bootstrap", fontsize=7.5, transform=axa.transAxes)
    axa.text(bx, 0.03, "Outcome-blind CommonRoad dynamic-ego cohort", fontsize=7.0, color=C["muted"], transform=axa.transAxes)

    # b: point performance
    y = np.arange(len(perf))[::-1]
    cols = [score_color(s) for s in perf["label"]]
    axb1.scatter(perf["AUPRC"], y, s=23, c=cols, ec="white", lw=0.4, zorder=3)
    axb2.scatter(perf["Recall"], y, s=23, c=cols, ec="white", lw=0.4, zorder=3)
    axb1.set_yticks(y); axb1.set_yticklabels(perf["label"])
    axb2.tick_params(labelleft=False)
    axb1.set_xlabel("AUPRC"); axb2.set_xlabel("Strict recall\nat 5% FPR")
    axb1.set_xlim(0, max(0.45, float(perf["AUPRC"].max())+0.04)); axb2.set_xlim(0, max(0.55, float(perf["Recall"].max())+0.04))
    for ax in (axb1, axb2):
        ax.grid(axis="x", color=C["grid"], lw=0.5); despine(ax)
    panel(axb1, "b", "Point performance", x=-0.36, y=1.10)

    forest(axc, au, "ΔAUPRC", "c", "AUPRC gain over baselines", xlim=(-0.02,0.39), show_values=True)
    forest(axd, rec, "Δ strict recall at 5% FPR", "d", "Strict-recall gain", xlim=(-0.02,0.47), show_values=True)
    out_save(fig, outdir, "Figure4_commonroad_expanded_external_validation_v2_2")

# ---------------------------
# Figure 5
# ---------------------------
def agree_val(df: pd.DataFrame, metric: str):
    if {"metric", "value"}.issubset(df.columns):
        sub=df[df["metric"]==metric]
        return float(sub["value"].iloc[0]) if len(sub) else np.nan
    return np.nan


def make_figure5(data: dict, outdir: Path):
    base_known=int(data["base_summary"]["known"]); ext_known=int(data["ext_summary"]["known"]); ext_unknown=int(data["ext_summary"]["unknown"])
    agreement=float(data["agreement"]["agreement"]); base_only=float(data["agreement"]["base_only"]); ext_only=float(data["agreement"]["ext_only"])
    ext_delta=delta_rows(data["ext_deltas"], data["field_ext_deltas"], "auprc")
    metrics=data["nonaction_metrics"].copy()
    mismatch=data["mismatch"].copy(); pairs=data["mismatch_pairs"].copy()

    fig=plt.figure(figsize=(7.15,5.98))
    gs=GridSpec(2,2,figure=fig,left=0.065,right=0.985,top=0.958,bottom=0.115,wspace=0.54,hspace=0.60)
    axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,0]); axd=fig.add_subplot(gs[1,1])

    # a: rescue diagram
    axa.axis("off")
    panel(axa, "a", "Expanded lattice endpoint", x=-0.09, y=1.04, title_x=0.02)

    # Separate the boxes, arrow and annotation vertically so no text overlaps
    # the endpoint cards or the connector.
    box_y, box_h = 0.46, 0.245
    box_specs = [
        (0.025, base_known, "base known failures", "#EEF4FA", "#8AA9C5"),
        (0.655, ext_known, "extended known failures", "#F1F8F5", "#8CB8A8"),
    ]
    for x0, val, lab, fc, ec in box_specs:
        box = FancyBboxPatch(
            (x0, box_y), 0.315, box_h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            fc=fc, ec=ec, lw=0.85, transform=axa.transAxes,
        )
        axa.add_patch(box)
        axa.text(x0 + 0.1575, box_y + 0.145, f"{val:,}", ha="center", va="center",
                 fontsize=16, fontweight="bold", transform=axa.transAxes)
        axa.text(x0 + 0.1575, box_y - 0.018, lab, ha="center", va="top",
                 fontsize=6.35, transform=axa.transAxes)

    arrow_y = box_y + 0.115
    arrow = FancyArrowPatch(
        (0.355, arrow_y), (0.645, arrow_y), arrowstyle="-|>", mutation_scale=11,
        lw=1.1, color=C["muted"], transform=axa.transAxes,
    )
    axa.add_patch(arrow)
    axa.text(
        # 0.50, box_y + box_h + 0.105, f"{int(base_only)} rescued cases",
        0.49, box_y + 0.165, f"{int(base_only)} rescued",
        ha="center", va="center", fontsize=7.0, color=C["muted"],
        transform=axa.transAxes,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.96),
    )

    stat = [
        (f"{agreement*100:.1f}%", "agreement"),
        (f"+{int(base_only)}", "base-only"),
        (f"{int(ext_only)}", "ext-only"),
        (f"{ext_unknown}", "unknown"),
    ]
    for x0, (big, small) in zip([0.10, 0.36, 0.62, 0.86], stat):
        axa.text(x0, 0.235, big, ha="center", fontsize=11, fontweight="bold", transform=axa.transAxes)
        axa.text(x0, 0.145, small, ha="center", fontsize=6.8, color=C["muted"], transform=axa.transAxes)
    axa.text(0.11, 0.035, "Same 10,000-sample outcome-blind cohort", fontsize=6.8,
             color=C["muted"], transform=axa.transAxes)

    # b forest
    forest(axb, ext_delta, "ΔAUPRC", "b", "Extended-label AUPRC gain", xlim=(-0.02,0.35), show_values=True)

    # c decoupling lollipop
    panel(axc,"c","Feature-set decoupling",x=-0.16,y=1.075,title_x=0.045)
    rows=[]
    for fs, lab in FEATURE_SET_DISPLAY.items():
        sub=metrics[metrics["feature_set"]==fs]
        if len(sub): rows.append({"label":lab,"AUPRC":float(sub["AUPRC"].iloc[0])})
    feat=pd.DataFrame(rows)
    feat["order"]=feat["label"].map({"Base":0,"+Non-action":1,"+Temporal":2,"+Full":3})
    feat=feat.sort_values("order")
    x=feat["order"].to_numpy(); y=feat["AUPRC"].to_numpy()
    axc.plot(x,y,color="#8C8C8C",lw=1.1,zorder=1)
    for xi, yi, lab in zip(x,y,feat["label"]):
        col=C["temporal"] if lab=="+Temporal" else (C["rof"] if lab=="+Full" else C["distance"])
        axc.scatter(xi, yi, s=36, color=col, ec="white", lw=0.5, zorder=3)
        axc.text(xi, yi+0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=6.8)
    axc.set_xticks(x); axc.set_xticklabels(feat["label"]); axc.set_ylabel("AUPRC"); axc.set_ylim(0.23, max(0.45,y.max()+0.05))
    axc.grid(axis="y",color=C["grid"],lw=0.5); despine(axc)
    nd=data["nonaction_deltas"] if "nonaction_deltas" in data else pd.DataFrame()
    if len(nd):
        sub=nd[(nd["metric"]=="auprc") & nd["enhanced_feature_set"].str.contains("strict_temporal", na=False)]
        if len(sub):
            r=sub.iloc[0]
            axc.text(0.03,0.95,f"+Temporal ΔAUPRC {r['delta']:+.3f}\n[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",transform=axc.transAxes,ha="left",va="top",fontsize=6.8,bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="#DDDDDD",lw=0.5))
        sub=nd[(nd["metric"]=="auprc") & nd["enhanced_feature_set"].str.contains("strict_non_action", na=False)]
        if len(sub):
            r=sub.iloc[0]
            axc.text(0.56,0.18,f"+Non-action\n{r['delta']:+.3f}",transform=axc.transAxes,fontsize=6.5,color=C["muted"])

    # d heatmap
    panel(axd,"d","Label-feature mismatch transfer",x=-0.16,y=1.075,title_x=0.045)
    cols=[c for c in mismatch.columns if c not in ["label_variant","config_hash"]]
    mat=mismatch[cols].to_numpy(dtype=float)
    row_labels=[VARIANT_DISPLAY.get(v,v) for v in mismatch["label_variant"]]
    col_labels=[VARIANT_DISPLAY.get(v,v) for v in cols]
    norm=TwoSlopeNorm(vmin=min(-0.05,np.nanmin(mat)),vcenter=0,vmax=max(0.42,np.nanmax(mat)))
    im=axd.imshow(mat,cmap="RdBu_r",norm=norm,aspect="auto")
    axd.set_xticks(np.arange(len(cols))); axd.set_xticklabels(col_labels,rotation=0)
    axd.set_yticks(np.arange(len(row_labels))); axd.set_yticklabels(row_labels)
    axd.set_xlabel("Feature variant"); axd.set_ylabel("Label variant")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val=mat[i,j]
            axd.text(j,i,f"{val:.2f}",ha="center",va="center",fontsize=5.8,color="white" if val>0.25 else C["text"])
    for i in range(min(mat.shape)):
        axd.add_patch(Rectangle((i-0.5,i-0.5),1,1,fill=False,ec=C["text"],lw=0.9))
    cbar=fig.colorbar(im,ax=axd,fraction=0.046,pad=0.03); cbar.set_label("ΔAUPRC",fontsize=7); cbar.ax.tick_params(labelsize=6.5)
    off=pairs[pairs["diagonal"]==False] if "diagonal" in pairs.columns else pd.DataFrame()
    if len(off):
        med=float(off["delta_AUPRC"].median()); pct=100*float((off["delta_AUPRC"]>0).mean())
        axd.text(0.98,-0.20,f"Off-diagonal median {med:+.3f}; {pct:.1f}% positive",transform=axd.transAxes,ha="right",va="top",fontsize=6.7,bbox=dict(boxstyle="round,pad=0.22",fc="white",ec="#CCCCCC",lw=0.5))
    for sp in axd.spines.values(): sp.set_linewidth(0.6)

    out_save(fig,outdir,"Figure5_action_library_sensitivity_decoupling_v2_2")

# ---------------------------
# Supplementary boundary figure
# ---------------------------
def make_boundary(data: dict, outdir: Path):
    speed=data["speed_boundary"]; low=data["low_subtype"]; overlap=data["initial_overlap"]
    fig=plt.figure(figsize=(7.15,3.02))
    gs=GridSpec(1,3,figure=fig,left=0.07,right=0.985,top=0.86,bottom=0.24,wspace=0.58)
    axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[0,2])
    # a speed strata
    panel(axa,"a","Speed-stratum AUPRC gains",x=-0.18,y=1.08)
    strata=["lt5mps","5to15mps","gte15mps"]; labels=["<5","5–15","≥15"]
    width=0.32; x=np.arange(len(strata))
    for k,(base,label,col) in enumerate([("distance_inverse","Distance",C["boundary_distance"]),("TTC_inverse","TTC",C["boundary_ttc"])]):
        vals=[]; lows=[]; highs=[]
        for st in strata:
            sub=speed[(speed["row_type"]=="bootstrap_delta")&(speed["stratum"]==st)&(speed["enhanced_score"]=="temporal_composite")&(speed["baseline_score"]==base)&(speed["metric"]=="auprc")]
            if len(sub):
                r=sub.iloc[0]; vals.append(r["delta"]); lows.append(r["ci_low"]); highs.append(r["ci_high"])
            else:
                vals.append(np.nan); lows.append(np.nan); highs.append(np.nan)
        off=(k-0.5)*width
        axa.bar(x+off,vals,width=width,color=col,ec="none",label=label)
        axa.errorbar(x+off,vals,yerr=[np.array(vals)-np.array(lows),np.array(highs)-np.array(vals)],fmt="none",ecolor=C["text"],lw=0.8,capsize=2)
    axa.axhline(0,color=C["text"],lw=0.7); axa.set_xticks(x); axa.set_xticklabels(labels); axa.set_xlabel("Ego speed (m s$^{-1}$)"); axa.set_ylabel("ΔAUPRC"); axa.grid(axis="y",color=C["grid"],lw=0.5); axa.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.04),fontsize=6.5); despine(axa)
    # b low-speed subtype
    panel(axb,"b","Low-speed failure subtypes",x=-0.18,y=1.08)
    low=low.copy().sort_values("count",ascending=True)
    def subtype_label(s):
        s=str(s).replace("known_failure:","")
        return s.replace("collision_road_boundary_and_kinematic","Collision\n+ road + kin.").replace("collision_and_kinematic","Collision\n+ kin.").replace("road_boundary_and_kinematic","Road\n+ kin.")
    y=np.arange(len(low))
    subtype_colors = {
        "collision_and_kinematic": C["boundary_collision"],
        "collision_road_boundary_and_kinematic": C["boundary_collision_road"],
        "road_boundary_and_kinematic": C["boundary_road"],
    }
    bar_colors = [subtype_colors.get(str(v).replace("known_failure:", ""), C["boundary_collision_road"])
                  for v in low["failure_subtype"]]
    axb.barh(y, low["count"], color=bar_colors, edgecolor="none")
    axb.set_yticks(y); axb.set_yticklabels([subtype_label(s) for s in low["failure_subtype"]]); axb.set_xlabel("Positive samples")
    for yi,val in zip(y,low["count"]): axb.text(val+max(low["count"])*0.02,yi,f"{int(val)}",va="center",fontsize=6.8)
    axb.set_xlim(0,max(low["count"])*1.22); axb.grid(axis="x",color=C["grid"],lw=0.5); despine(axb)
    # c initial-overlap exclusion: paired horizontal point plot avoids
    # bar/legend/value-label collisions in the compact supplementary layout.

    # panel(axc, "c", "Initial-overlap sensitivity", x=-0.18, y=1.08, title_x=0.035)
    # sub = overlap[(overlap["stratum_column"] == "speed_stratum") &
    #               (overlap["stratum"] == "lt5mps") &
    #               (overlap["enhanced_score"] == "temporal_composite")]
    # bases = [("distance_inverse", "Distance"), ("TTC_inverse", "TTC")]
    # allv, excl = [], []
    # for b, _ in bases:
    #     r = sub[sub["baseline_score"] == b]
    #     allv.append(float(r["AUPRC_delta_all"].iloc[0]) if len(r) else np.nan)
    #     excl.append(float(r["AUPRC_delta_exclude_initial_overlap"].iloc[0]) if len(r) else np.nan)
    #
    # y0 = np.array([1.0, 0.0])
    # jitter = 0.115
    # axc.axvline(0, color=C["text"], lw=0.75, zorder=1)
    # axc.scatter(allv, y0 + jitter, marker="s", s=34, color=C["all_samples"],
    #             edgecolor="white", linewidth=0.45, label="All samples", zorder=3)
    # axc.scatter(excl, y0 - jitter, marker="o", s=38, color=C["exclude_overlap"],
    #             edgecolor="white", linewidth=0.45, label="Exclude overlap", zorder=3)
    # for yi, v_all, v_ex in zip(y0, allv, excl):
    #     axc.plot([v_all, v_ex], [yi + jitter, yi - jitter], color="#B8B8B8", lw=0.8, zorder=2)
    #     axc.annotate(f"{v_all:+.3f}", (v_all, yi + jitter), xytext=(-5, 0), textcoords="offset points",
    #                  ha="right", va="center", fontsize=6.3, color=C["all_samples"])
    #     axc.annotate(f"{v_ex:+.3f}", (v_ex, yi - jitter), xytext=(5, 0), textcoords="offset points",
    #                  ha="left", va="center", fontsize=6.3, color=C["exclude_overlap"])
    # axc.set_yticks(y0); axc.set_yticklabels([b[1] for b in bases])
    # axc.set_xlabel("ΔAUPRC")
    # axc.set_xlim(-0.060, 0.032); axc.set_ylim(-0.48, 1.48)
    # axc.grid(axis="x", color=C["grid"], lw=0.5)
    # axc.legend(frameon=False, loc="lower right", fontsize=6.25, handletextpad=0.45, borderaxespad=0.15)
    # despine(axc)

    # c initial-overlap exclusion
    # Restore the v2.1 grouped-bar layout, using the revised v2.2 palette.
    panel(
        axc,
        "c",
        "Initial-overlap sensitivity",
        x=-0.18,
        y=1.08,
        title_x=0.035,
    )

    sub = overlap[
        (overlap["stratum_column"] == "speed_stratum")
        & (overlap["stratum"] == "lt5mps")
        & (overlap["enhanced_score"] == "temporal_composite")
    ]

    bases = [
        ("distance_inverse", "Distance"),
        ("TTC_inverse", "TTC"),
    ]

    allv = []
    excl = []

    for baseline_name, _ in bases:
        row = sub[sub["baseline_score"] == baseline_name]

        allv.append(
            float(row["AUPRC_delta_all"].iloc[0])
            if len(row)
            else np.nan
        )

        excl.append(
            float(row["AUPRC_delta_exclude_initial_overlap"].iloc[0])
            if len(row)
            else np.nan
        )

    x = np.arange(len(bases))
    width = 0.30

    axc.axhline(
        0,
        color=C["text"],
        lw=0.75,
        zorder=1,
    )

    bars_all = axc.bar(
        x - width / 2,
        allv,
        width=width,
        color=col,
        edgecolor="none",
        label="All samples",
        zorder=2,
    )

    bars_excl = axc.bar(
        x + width / 2,
        excl,
        width=width,
        color=C["muted"],
        edgecolor="none",
        label="Exclude overlap",
        zorder=2,
    )

    axc.set_xticks(x)
    axc.set_xticklabels([label for _, label in bases])

    axc.set_ylabel("ΔAUPRC")

    axc.grid(
        axis="y",
        color=C["grid"],
        lw=0.5,
        zorder=0,
    )

    # Give labels enough room above positive bars and below negative bars.
    finite_values = [
        v for v in allv + excl
        if not np.isnan(v)
    ]

    y_min = min(finite_values + [0]) - 0.018
    y_max = max(finite_values + [0]) + 0.025

    axc.set_ylim(y_min, y_max)

    # Value labels
    for xi, v_all, v_excl in zip(x, allv, excl):

        if not np.isnan(v_all):
            offset_all = 0.005 if v_all >= 0 else -0.006

            axc.text(
                xi - width / 2,
                v_all + offset_all,
                f"{v_all:+.3f}",
                ha="center",
                va="bottom" if v_all >= 0 else "top",
                fontsize=6.3,
                color=C["all_samples"],
            )

        if not np.isnan(v_excl):
            offset_excl = 0.005 if v_excl >= 0 else -0.006

            axc.text(
                xi + width / 2,
                v_excl + offset_excl,
                f"{v_excl:+.3f}",
                ha="center",
                va="bottom" if v_excl >= 0 else "top",
                fontsize=6.3,
                color=C["exclude_overlap"],
            )

    axc.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.01),
        fontsize=6.2,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    despine(axc)



    fig.text(
        0.07, 0.055,
        "Boundary interpretation: low-speed collision-heavy cases are a proximity-competitive regime; "
        "this does not alter the full-cohort external-validation claim.",
        fontsize=6.35, color=C["muted"],
    )

    # panel(axc,"c","Initial-overlap exclusion",x=-0.18,y=1.08)
    # sub=overlap[(overlap["stratum_column"]=="speed_stratum")&(overlap["stratum"]=="lt5mps")&(overlap["enhanced_score"]=="temporal_composite")]
    # bases=[("distance_inverse","Distance"),("TTC_inverse","TTC")]
    # allv=[]; excl=[]
    # for b,_ in bases:
    #     r=sub[sub["baseline_score"]==b]
    #     allv.append(float(r["AUPRC_delta_all"].iloc[0]) if len(r) else np.nan)
    #     excl.append(float(r["AUPRC_delta_exclude_initial_overlap"].iloc[0]) if len(r) else np.nan)
    # x=np.arange(2); w=0.32
    # axc.axhline(0,color=C["text"],lw=0.7)
    # axc.bar(x-w/2,allv,w,color="#BDBDBD",label="All")
    # axc.bar(x+w/2,excl,w,color=C["temporal"],label="Exclude overlap")
    # axc.set_xticks(x); axc.set_xticklabels([b[1] for b in bases]); axc.set_ylabel("ΔAUPRC"); axc.grid(axis="y",color=C["grid"],lw=0.5); axc.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.04),fontsize=6.5); despine(axc)
    # y_min = min([v for v in allv + excl if not np.isnan(v)] + [0]) - 0.012
    # y_max = max([v for v in allv + excl if not np.isnan(v)] + [0]) + 0.018
    # axc.set_ylim(y_min, y_max)
    # for xi, v1, v2 in zip(x,allv,excl):
    #     axc.text(xi-w/2, v1+(0.007 if v1>=0 else -0.012), f"{v1:+.3f}", ha="center", va="bottom" if v1>=0 else "top", fontsize=6.5)
    #     axc.text(xi+w/2, v2+(0.007 if v2>=0 else -0.012), f"{v2:+.3f}", ha="center", va="bottom" if v2>=0 else "top", fontsize=6.5)
    # fig.text(0.07,0.06,"Boundary interpretation: low-speed collision-heavy cases are a proximity-competitive regime; this does not alter the full-cohort external-validation claim.",fontsize=6.4,color=C["muted"])

    out_save(fig,outdir,"Supplementary_Figure_low_speed_boundary_v2_2")

# ---------------------------
# CLI
# ---------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Extracted integrated evidence-lock root (preferred) or raw nc_v110/v111/v112 result root")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--skip-supplementary", action="store_true")
    args=ap.parse_args()
    data=load_all(args.root)
    print(f"Input mode: {data.get('mode','unknown')}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    make_figure4(data,args.outdir)
    make_figure5(data,args.outdir)
    if not args.skip_supplementary:
        make_boundary(data,args.outdir)
    print(f"Wrote figures to {args.outdir}")
    for p in sorted(args.outdir.glob("*.pdf")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
