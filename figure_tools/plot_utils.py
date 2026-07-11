from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PALETTE = {
    'baseline': '#6F7A82',
    'enhanced': '#1F78B4',
    'high_actionability': '#D9D9D9',
    'reduced_actionability': '#A6BDD7',
    'critical_actionability': '#F4A582',
    'candidate_set_infeasible': '#CA0020',
    'map_constrained': '#1F78B4',
    'no_map': '#7B8D8E',
    'seed_41': '#1F78B4',
    'seed_42': '#33A02C',
    'seed_43': '#E31A1C',
    'direct_action_ratios_only': '#A6CEE3',
    'explicit_ratio_field_excluded_current': '#1F78B4',
    'strict_spatial_no_action': '#BDBDBD',
    'strict_temporal_dynamics': '#E6550D',
    'reference': '#4D4D4D',
    'horizon': '#1B9E77',
    'lane_buffer': '#7570B3',
    'action_library': '#D95F02',
    'future_handling': '#E7298A',
}

DISPLAY_MAP = {
    'strong_baseline_cv': 'Baseline + CV',
    'strong_baseline_cv_plus_strict_temporal_dynamics': 'Baseline + CV\n+ strict temporal',
    'strong_baseline_cv_plus_direct_action_ratios_only': 'Baseline + CV\n+ direct ratios only',
    'strong_baseline_cv_plus_explicit_ratio_field_excluded_current': 'Baseline + CV\n+ explicit-ratio-field-excluded',
    'strong_baseline_cv_plus_strict_spatial_no_action': 'Baseline + CV\n+ spatial no-action',
    'strong_baseline_cv_plus_strict_temporal_dynamics': 'Baseline + CV\n+ strict temporal',
    'map_critical_or_worse': 'Map critical-or-worse',
    'map_candidate_set_infeasible': 'Map candidate-set infeasible',
    'nomap_critical_or_worse': 'No-map critical-or-worse',
    'nomap_candidate_set_infeasible': 'No-map candidate-set infeasible',
    'actionability_critical_or_worse': 'Critical-or-worse',
    'auprc': 'ΔAUPRC',
    'auroc': 'ΔAUROC',
    'recall_at_5pct_fpr': 'ΔRecall @ 5% FPR',
    'recall_at_1pct_fpr': 'ΔRecall @ 1% FPR',
    'current_min_distance_m': 'Current min distance',
    'valid_ttc_only': 'Valid TTC only',
    'no_ttc_as_category': 'Missing TTC as category',
    'capped_ttc_prespecified_10s': 'Capped TTC (10 s)',
    'inverse_ttc_prespecified': 'Inverse TTC',
    'legacy_sentinel': 'Legacy sentinel TTC',
    'map_actionability_label_id': 'Map-constrained',
    'nomap_actionability_label_id': 'No-map',
    'reference_h3_b3_base7_skip': 'Reference\nh=3, b=3, base7',
    'horizon_h2_b3_base7_skip': 'Horizon\n2 s',
    'horizon_h4_b3_base7_skip': 'Horizon\n4 s',
    'buffer_h3_b2_base7_skip': 'Buffer\n2 m',
    'buffer_h3_b4_base7_skip': 'Buffer\n4 m',
    'action_h3_b3_extended_skip': 'Extended\naction set',
    'future_h3_b3_base7_cvfallback': 'CV\nfallback',
}

FAMILY_COLOR = {
    'reference': PALETTE['reference'],
    'horizon': PALETTE['horizon'],
    'lane_buffer': PALETTE['lane_buffer'],
    'action_library': PALETTE['action_library'],
    'future_handling': PALETTE['future_handling'],
}


class SourceLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.is_zip = self.root.suffix.lower() == '.zip'
        self.zf = zipfile.ZipFile(self.root) if self.is_zip else None
        self.base_prefix = 'ROF_results_v100_evidence_lock/'

    def read_csv(self, rel_path: str) -> pd.DataFrame:
        if self.is_zip:
            full = self.base_prefix + rel_path.replace('\\', '/').lstrip('/')
            data = self.zf.read(full)
            return pd.read_csv(io.BytesIO(data))
        return pd.read_csv(self.root / rel_path)

    def close(self):
        if self.zf is not None:
            self.zf.close()


def apply_nc_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.35,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def add_panel_label(ax, label: str):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')


def nice_axes(ax, grid_axis='y'):
    ax.grid(True, axis=grid_axis, color='#CFCFCF', alpha=0.5)
    ax.set_axisbelow(True)


def save_figure(fig, outdir: Path, stem: str, formats=('png', 'pdf'), dpi=300):
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f'{stem}.{fmt}', dpi=dpi)


def annotate_heatmap(ax, data: np.ndarray, fmt='{:.0f}', textcolor_threshold=None, fontsize=7):
    if textcolor_threshold is None:
        textcolor_threshold = np.nanmax(data) * 0.55 if np.isfinite(data).any() else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                text = '—'
                color = 'black'
            else:
                text = fmt.format(value)
                color = 'white' if value >= textcolor_threshold else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=fontsize, color=color)


def metric_label(metric: str) -> str:
    return DISPLAY_MAP.get(metric, metric)


def display_name(value: str) -> str:
    return DISPLAY_MAP.get(value, value)


def bool_to_symbol(value: bool) -> str:
    return '●' if bool(value) else '○'
