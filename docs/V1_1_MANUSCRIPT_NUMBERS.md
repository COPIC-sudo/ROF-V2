# V1.1 Manuscript Numbers

This file is generated from the v1.1 locked strict-FPR artifacts. Do not cite non-strict Recall@FPR rows as primary results.

## CommonRoad Lattice Base

- n_samples: 10000
- n_scenarios: 1368
- known_failure: 610
- positive_scenarios: 402
- unknown_failure: 0
- no_failure: 9390
- temporal_composite delta AUPRC:
  - temporal_composite vs distance_inverse AUPRC: delta=0.2913, CI=(0.24, 0.3414)
  - temporal_composite vs TTC_inverse AUPRC: delta=0.2797, CI=(0.2325, 0.3283)
- temporal_composite strict delta Recall@5%FPR:
  - temporal_composite vs distance_inverse strict Recall@5%FPR: delta=0.3459, CI=(0.2837, 0.4035)
  - temporal_composite vs TTC_inverse strict Recall@5%FPR: delta=0.3656, CI=(0.305, 0.4161)
- ROF_v2_no_asr_composite delta AUPRC:
  - ROF_v2_no_asr_composite vs distance_inverse AUPRC: delta=0.3038, CI=(0.2547, 0.354)
  - ROF_v2_no_asr_composite vs TTC_inverse AUPRC: delta=0.2922, CI=(0.2458, 0.344)
- ROF_v2_no_asr_composite strict delta Recall@5%FPR:
  - ROF_v2_no_asr_composite vs distance_inverse strict Recall@5%FPR: delta=0.3475, CI=(0.2859, 0.4048)
  - ROF_v2_no_asr_composite vs TTC_inverse strict Recall@5%FPR: delta=0.3672, CI=(0.3047, 0.4191)

## V112 Strong Baselines

- best CriMe-style baseline: THW_inverse AUPRC=0.1239
- best RSS-style baseline: rss_longitudinal_margin_inverse AUPRC=0.2131
- best forecast-risk baseline: minimum_predicted_separation_3s_inverse AUPRC=0.156
- temporal_composite vs best RSS and forecast-risk:
  - temporal_composite vs rss_longitudinal_margin_inverse AUPRC: delta=0.187, CI=(0.1431, 0.2277)
  - temporal_composite vs rss_longitudinal_margin_inverse strict Recall@5%FPR: delta=0.1738, CI=(0.1313, 0.2248)
  - temporal_composite vs minimum_predicted_separation_3s_inverse AUPRC: delta=0.2441, CI=(0.1905, 0.2953)
  - temporal_composite vs minimum_predicted_separation_3s_inverse strict Recall@5%FPR: delta=0.2738, CI=(0.2054, 0.3356)
- ROF_v2_no_asr_composite vs best RSS and forecast-risk:
  - ROF_v2_no_asr_composite vs rss_longitudinal_margin_inverse AUPRC: delta=0.1994, CI=(0.1577, 0.2409)
  - ROF_v2_no_asr_composite vs rss_longitudinal_margin_inverse strict Recall@5%FPR: delta=0.1754, CI=(0.1353, 0.2258)
  - ROF_v2_no_asr_composite vs minimum_predicted_separation_3s_inverse AUPRC: delta=0.2565, CI=(0.2028, 0.3104)
  - ROF_v2_no_asr_composite vs minimum_predicted_separation_3s_inverse strict Recall@5%FPR: delta=0.2754, CI=(0.2071, 0.337)

## Lattice Extended Endpoint Sensitivity

- known_failure: 420
- unknown_failure: 0
- no_failure: 9580
- base vs extended taxonomy agreement: 0.981
- base-only positives: 190
- extended-only positives: 0
- temporal_composite / ROF_v2_no_asr_composite vs distance/TTC on extended labels:
  - temporal_composite vs distance_inverse AUPRC: delta=0.2431, CI=(0.1841, 0.2989)
  - temporal_composite vs distance_inverse strict Recall@5%FPR: delta=0.2976, CI=(0.2244, 0.3716)
  - temporal_composite vs TTC_inverse AUPRC: delta=0.2358, CI=(0.1836, 0.2931)
  - temporal_composite vs TTC_inverse strict Recall@5%FPR: delta=0.3143, CI=(0.2403, 0.3848)
  - ROF_v2_no_asr_composite vs distance_inverse AUPRC: delta=0.2563, CI=(0.1988, 0.3166)
  - ROF_v2_no_asr_composite vs distance_inverse strict Recall@5%FPR: delta=0.2952, CI=(0.2217, 0.3696)
  - ROF_v2_no_asr_composite vs TTC_inverse AUPRC: delta=0.249, CI=(0.1964, 0.3106)
  - ROF_v2_no_asr_composite vs TTC_inverse strict Recall@5%FPR: delta=0.3119, CI=(0.2382, 0.3811)
- v112b deltas vs best RSS / forecast-risk baselines:
  - temporal_composite vs rss_longitudinal_margin_inverse AUPRC: delta=0.181, CI=(0.1357, 0.2251)
  - temporal_composite vs rss_longitudinal_margin_inverse strict Recall@5%FPR: delta=0.1571, CI=(0.1025, 0.2124)
  - temporal_composite vs minimum_predicted_separation_3s_inverse AUPRC: delta=0.2093, CI=(0.1505, 0.2669)
  - temporal_composite vs minimum_predicted_separation_3s_inverse strict Recall@5%FPR: delta=0.2405, CI=(0.1616, 0.3166)
  - ROF_v2_no_asr_composite vs rss_longitudinal_margin_inverse AUPRC: delta=0.1942, CI=(0.1481, 0.2402)
  - ROF_v2_no_asr_composite vs rss_longitudinal_margin_inverse strict Recall@5%FPR: delta=0.1548, CI=(0.1047, 0.2094)
  - ROF_v2_no_asr_composite vs minimum_predicted_separation_3s_inverse AUPRC: delta=0.2225, CI=(0.1647, 0.2849)
  - ROF_v2_no_asr_composite vs minimum_predicted_separation_3s_inverse strict Recall@5%FPR: delta=0.2381, CI=(0.1605, 0.3158)

## V111 Decoupling

- strong_baseline_cv_plus_strict_non_action_current_cv delta AUPRC over strong_baseline_cv: 0.0715, CI=(0.0569, 0.0847)
- strong_baseline_cv_plus_strict_temporal_dynamics delta AUPRC over strong_baseline_cv: 0.1396, CI=(0.1224, 0.1561)
- strong_baseline_cv_plus_full_actionability delta AUPRC over strong_baseline_cv: 0.1451, CI=(0.1278, 0.1622)
- median off-diagonal delta AUPRC: 0.0827
- percent off-diagonal positive: 96.6667%
- off-diagonal retention vs diagonal: 0.3016
- grouped metadata limitation: segment_id / scenario_family leave-out was unavailable in supplied Waymo v1.0.1 tables; CommonRoad family/topology leave-out was unavailable in the v110 manifest.

## Boundary

- adequate-positive negative delta rows: 14; clearly negative rows: 13.
- low-speed concentration: 14 negative-audit rows had low_speed_fraction >= 0.5.
- low-speed subtype: known_failure:collision_and_kinematic, count=170, scenarios=139
- low-speed subtype: known_failure:collision_road_boundary_and_kinematic, count=10, scenarios=9
- low-speed subtype: known_failure:road_boundary_and_kinematic, count=3, scenarios=3
- initial-overlap-excluded temporal vs distance AUPRC delta: 0.3304
- initial-overlap-excluded temporal vs TTC AUPRC delta: 0.302
- initial-overlap-excluded n=9893, positives=573, excluded_samples=107
- no every-stratum superiority claim is supported.
