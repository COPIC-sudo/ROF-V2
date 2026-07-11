# V1.1 Limitations And Claim Boundaries

## Claims Not Supported

- Uniform superiority across all strata.
- Native-planner robustness.
- Closed-loop safety proof.
- Complete absence of label-feature coupling.
- Segment/family-level grouped Waymo generalization when the metadata are unavailable.

## Claims Supported Within Boundary

- Expanded CommonRoad lattice-base external validation on the locked neutral, outcome-blind full cohort.
- Stronger-baseline superiority under strict-FPR evaluation for the locked endpoint comparisons.
- Action-library extension sensitivity using the lattice_extended endpoint; this is not native-planner robustness.
- Decoupling concern is reduced by strict non-action, label-feature mismatch, and external endpoint sensitivity audits.

## Required Wording

- Refer to current CommonRoad external validation as lattice_base expanded validation unless discussing the separate lattice_extended sensitivity.
- Use strict-FPR metric files for manuscript primary Recall@FPR statements.
- Treat low-speed / near-overlap / collision-heavy strata as boundary analyses, not as claim-gate revisions.
- Do not use v1.1 to claim a closed-loop safety guarantee.
