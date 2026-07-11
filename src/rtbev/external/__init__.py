from __future__ import annotations

__all__ = [
    "artifact_manifest_rows",
    "classify_failure_taxonomy",
    "config_hash",
    "evaluate_external_scores",
    "evaluate_external_scores_strict",
    "experiment_out_dir",
    "load_yaml_config",
    "recall_at_fpr_strict",
    "scenario_bootstrap_deltas",
    "scenario_bootstrap_deltas_strict",
    "select_neutral_stratified_cohort",
    "stratum_metrics",
    "unknown_failure_sensitivity",
]

_EXPORT_MODULES = {
    "artifact_manifest_rows": ".common",
    "config_hash": ".common",
    "experiment_out_dir": ".common",
    "load_yaml_config": ".common",
    "select_neutral_stratified_cohort": ".cohort",
    "classify_failure_taxonomy": ".metrics",
    "evaluate_external_scores": ".metrics",
    "evaluate_external_scores_strict": ".metrics",
    "recall_at_fpr_strict": ".metrics",
    "scenario_bootstrap_deltas": ".metrics",
    "scenario_bootstrap_deltas_strict": ".metrics",
    "stratum_metrics": ".metrics",
    "unknown_failure_sensitivity": ".metrics",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
