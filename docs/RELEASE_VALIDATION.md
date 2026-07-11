# Public-release validation — v1.1.0

This document records the checks performed while assembling the public code release.

## Locked runtime target

- Python: 3.10
- Core environment: `environment.yml`
- CommonRoad environment: `environment-commonroad.yml`
- Continuous integration: `.github/workflows/ci.yml` (Python 3.10)

The release-assembly container used Python 3.13, which is intentionally outside the
paper lock (`>=3.10,<3.11`). Therefore the behavioral checks below were run with
`PYTHONPATH=src`; editable-install validation is delegated to the Python 3.10 CI job.

## Checks completed during release assembly

```text
python -m compileall -q src scripts tests figure_tools      PASS
PYTHONPATH=src pytest -q                                   20 passed, 2 skipped
PYTHONPATH=src python scripts/99_smoke_test.py              PASS
PYTHONPATH=src python scripts/99_check_github_readiness.py  PASS (0 errors, 0 warnings)
PYTHONPATH=src python scripts/99_verify_paper_release.py    PASS
```

The two skipped tests require optional generated Waymo audit artifacts and are not
failures of the code-only release.

## Figure-regeneration checks

The following scripts were executed successfully against
`ROF_results_v1_1_integrated_evidence_lock.zip` or its extracted root:

- `figure_tools/make_rof_figures_2_to_6.py` — manuscript Figures 2–3;
- `figure_tools/plot_nc_v11_figures_4_5_final_v2_2.py` — manuscript Figures 4–5;
- `figure_tools/make_supplementary_figures_v100.py` — Supplementary Figures S1–S5;
- `figure_tools/plot_supplementary_figures.py` — Supplementary Figures S6–S9.

Generated figures were not retained in this source-code archive.

## Public-release hygiene

The release was scanned for:

- machine-specific absolute paths;
- credential-like strings;
- IDE and agent metadata;
- Python caches and test work directories;
- backup files and superseded manuscript-editing utilities;
- embedded raw Waymo/CommonRoad data;
- embedded paper-result archives.

The final archive contains code, frozen public configurations, tests, documentation,
and plotting utilities only. Derived paper results are distributed separately.
