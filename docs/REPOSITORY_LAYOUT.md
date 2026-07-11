# Repository layout

## Core package

- `src/rtbev/pipeline.py`: ROF feature pipeline.
- `src/rtbev/labels.py`: proximity label utilities.
- `src/rtbev/external/taxonomy.py`: fixed CommonRoad planner-outcome taxonomy.
- `src/rtbev/external/metrics.py`: external metrics, strict FPR and scenario bootstrap.
- `src/rtbev/baselines/`: field-baseline and decoupling feature-set code.
- `src/waymo_open_dataset/`: generated Waymo protocol-buffer modules.

## Versioned analyses

- `scripts/nc_v090`: confirmatory Waymo OOF and information-access audit.
- `scripts/nc_v095`: secondary context and future-validity analyses.
- `scripts/nc_v096`: CV fallback and endpoint-design variants.
- `scripts/nc_v097`: aligned label-feature robustness.
- `scripts/nc_v100`: PR-curve derivation used by the retained Waymo figure.
- `scripts/nc_v110`: CommonRoad scale-up, fixed taxonomy, strict-FPR and boundary analyses.
- `scripts/nc_v111`: decoupling audit and consistency fix.
- `scripts/nc_v112`: field baselines and extended-label evaluation.

## Plotting

Only final paper plotting scripts are retained. Superseded drafts and embedded results archives are intentionally excluded.
