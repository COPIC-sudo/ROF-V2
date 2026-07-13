# ROF Actionability v1.1.0

Research code for quantifying **feasible-action collapse** in autonomous-driving emergencies. The repository implements the paper's actionability endpoint, Waymo out-of-fold analyses, expanded CommonRoad planner-facing validation, strict fixed-FPR evaluation, field baselines, lattice action-library sensitivity, label-feature decoupling audits, and manuscript figures.

## Scientific scope

Proximity metrics answer how close an interaction is. Actionability asks whether feasible ego responses remain. This repository operationalizes actionability as a short-horizon candidate-action feasibility endpoint and evaluates temporal/actionability-derived signals against distance, TTC, CommonRoad-CriMe-style, RSS-style, drivability, and deterministic forecast-risk baselines.

The released code supports the paper's bounded claims:

- actionability measures a feasible-response dimension not fully captured by proximity alone;
- the signal is validated internally on Waymo and externally against CommonRoad lattice-planner outcomes;
- the result is stable to an expanded lattice action library;
- decoupling audits reduce, but do not eliminate, label-feature coupling concerns;
- low-speed collision-heavy cases remain a regime where distance/TTC can be competitive.

The code does **not** claim a closed-loop safety guarantee, native-planner robustness, crash prediction, or uniform superiority in every stratum.

## Repository layout

```text
configs/          Frozen public configurations used by the paper analyses
src/rtbev/        Core ROF, actionability, external-validation and baseline code
scripts/          Waymo/CommonRoad pipelines and versioned paper analyses
figure_tools/     Final manuscript and supplementary plotting scripts
tests/            Synthetic regression and smoke tests
docs/             Reproduction guide, evidence lock and claim boundaries
source_data/      Instructions for the separately deposited evidence-lock archive
```

The versioned paper-analysis modules are:

```text
nc_v090–nc_v097   Waymo confirmatory, robustness and aligned-feature analyses
nc_v110           CommonRoad 10k cohort, fixed taxonomy, strict-FPR and boundary analyses
nc_v111           Non-action, temporal and label-feature mismatch decoupling audits
nc_v112           Field baselines and extended-label baseline evaluation
```

## Installation

Python 3.10 is the locked interpreter family.

### Conda: Waymo/core analyses

```bash
conda env create -f environment.yml
conda activate rof-actionability
```

### Conda: CommonRoad analyses

Use the separate environment to avoid dependency conflicts:

```bash
conda env create -f environment-commonroad.yml
conda activate rof-actionability-commonroad
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

For CommonRoad support:

```bash
pip install -e .[dev,commonroad]
```

## Data requirements

Raw Waymo and CommonRoad data are not distributed here. Obtain them from the official providers and comply with their terms.

Set these environment variables before raw-data reproduction:

```bash
export ROF_WORK_DIR=/path/to/rof_work
export WAYMO_SCENARIO_ROOT=/path/to/waymo/scenario
export COMMONROAD_SCENARIO_ROOT=/path/to/commonroad/scenarios
```

PowerShell:

```powershell
$env:ROF_WORK_DIR="<ROF_WORK_DIR>"
$env:WAYMO_SCENARIO_ROOT="<WAYMO_SCENARIO_ROOT>"
$env:COMMONROAD_SCENARIO_ROOT="<COMMONROAD_SCENARIO_ROOT>"
```

The separately deposited derived-data archive is expected to be named:

```text
ROF_results_v1_1_integrated_evidence_lock.zip
```

See `source_data/README.md` and `docs/REPRODUCE_PAPER.md`.

## Quick validation

```bash
python -m compileall -q src scripts tests figure_tools
pytest -q
python scripts/99_smoke_test.py
python scripts/99_check_github_readiness.py --root .
```

The smoke test uses synthetic data and does not require Waymo or CommonRoad downloads.

## Reproduce manuscript figures from the deposited evidence lock

Extract the evidence-lock archive first. Then run the final plotting scripts:

```bash
# Main Figures 2–3 and legacy robustness panels retained from the v1.0 evidence chain
python figure_tools/make_rof_figures_2_to_6.py \
  --input <EVIDENCE_ROOT>/99_cleanup_QA/legacy_v100_reference \
  --out figures/generated/v100 \
  --figures 2 3

# Main Figures 4–5
python figure_tools/plot_nc_v11_figures_4_5_final_v2_2.py \
  --root <EVIDENCE_ROOT> \
  --outdir figures/generated/v11 \
  --skip-supplementary

# Supplementary Figures S1–S5
python figure_tools/make_supplementary_figures_v100.py \
  --data <EVIDENCE_ROOT>/99_cleanup_QA/legacy_v100_reference \
  --out figures/generated/supp_v100 \
  --formats pdf png svg

# Supplementary Figures S6–S9; this script also accepts the evidence-lock ZIP directly
python figure_tools/plot_supplementary_figures.py \
  --evidence-lock ROF_results_v1_1_integrated_evidence_lock.zip \
  --output-dir figures/generated/supp_v11 \
  --dpi 600
```

Figure 1 is a conceptual information-access schematic; its panel source tables are included in the evidence lock and the final vector artwork is maintained with the manuscript source.

## Raw-data reproduction

Full raw-data reproduction is computationally expensive and requires licensed third-party datasets. The canonical order is documented in `docs/REPRODUCE_PAPER.md`. In brief:

1. scan and extract Waymo samples;
2. generate proximity and actionability labels;
3. generate ROF/current-state/CV features;
4. run `nc_v090–nc_v097` Waymo analyses;
5. build the outcome-blind CommonRoad cohort and lattice-base labels with `nc_v110`;
6. run lattice-extended sensitivity, strict-FPR and boundary analyses;
7. run `nc_v112` field baselines;
8. run `nc_v111` decoupling audits.

The exact final configurations are under `configs/nc_v110`, `configs/nc_v111`, and `configs/nc_v112`.

## Reproducibility and provenance

- `docs/V1_1_EVIDENCE_LOCK.md` identifies the locked analysis chain.
- `docs/V1_1_MANUSCRIPT_NUMBERS.md` records the manuscript-facing results.
- `docs/V1_1_LIMITATIONS_AND_CLAIM_BOUNDARIES.md` defines supported and unsupported claims.
- `docs/RELEASE_VALIDATION.md` records public-release checks.
- `docs/AVAILABILITY_STATEMENTS_TEMPLATE.md` provides manuscript-ready templates.
- Generated results should be written under `ROF_WORK_DIR`; they must not be committed to this repository.

## License and third-party material

Unless otherwise noted, this repository is licensed under the Apache License 2.0. Generated Waymo protocol-buffer modules under `src/waymo_open_dataset/` are attributed in `THIRD_PARTY_NOTICES.md`. Waymo and CommonRoad datasets retain their own terms and are not redistributed.
