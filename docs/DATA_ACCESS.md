# Data access and repository separation

The public release is split into two artifacts:

1. **Code archive / GitHub repository** — this repository.
2. **Derived-data evidence lock** — `ROF_results_v1_1_integrated_evidence_lock.zip`, deposited separately with a persistent identifier.

Raw Waymo and CommonRoad data are not redistributed.

## Expected evidence-lock contents

The derived-data archive contains:

- source data for main and supplementary figures and tables;
- Waymo OOF summaries and paired bootstrap deltas;
- CommonRoad 10k cohort and fixed-taxonomy outputs;
- strict fixed-FPR metrics;
- field-baseline results;
- lattice-extended sensitivity;
- label-feature decoupling outputs;
- low-speed boundary analyses;
- file and SHA256 manifests.

## Local layout

The archive may be stored anywhere. Plotting scripts accept an explicit input path. Raw-data pipelines write all generated artifacts below `${ROF_WORK_DIR}`.

Do not commit raw data, work directories, generated results or the evidence-lock ZIP to GitHub.

## Third-party data

Users are responsible for obtaining the Waymo Open Motion Dataset and CommonRoad scenario collections from their official providers and for complying with the applicable terms.
