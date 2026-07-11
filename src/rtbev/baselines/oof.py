from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from rtbev.external.metrics import binary_metric_values


def stable_unit_float(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)


def assign_group_folds(groups: pd.Series, n_folds: int, seed: int) -> pd.Series:
    keys = groups.fillna("").astype(str)
    mapping = {k: min(int(stable_unit_float(f"{seed}|{k}") * n_folds), n_folds - 1) for k in keys.unique()}
    return keys.map(mapping).astype(int)


def _numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        if "ttc" in col.lower():
            s = s.mask((s < 0.0) | (~np.isfinite(s)), np.nan)
        out[col] = s.replace([np.inf, -np.inf], np.nan)
    return out


def _make_model(model: str, seed: int) -> Any:
    if model == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=1,
        )
    raise ValueError(f"unknown model: {model}")


def _positive_score(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(x.shape[0], dtype=float)


def grouped_oof_predictions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    group_col: str = "scenario_id",
    n_folds: int = 5,
    seed: int = 42,
    model: str = "rf",
) -> pd.DataFrame:
    if group_col not in df.columns:
        group_col = "sample_id"
    work = df.copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce").fillna(0).astype(int)
    work["outer_fold"] = assign_group_folds(work[group_col].astype(str), int(n_folds), int(seed))
    pred_rows: list[pd.DataFrame] = []
    cols = list(feature_cols)
    for fold in sorted(work["outer_fold"].unique()):
        test_mask = work["outer_fold"].to_numpy(int) == int(fold)
        train = work.loc[~test_mask].copy()
        test = work.loc[test_mask].copy()
        y_train = train[label_col].to_numpy(int)
        y_test = test[label_col].to_numpy(int)
        if len(train) == 0 or len(test) == 0 or len(np.unique(y_train)) < 2:
            continue
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(_numeric_frame(train, cols))
        x_test = imputer.transform(_numeric_frame(test, cols))
        clf = _make_model(model, int(seed) + int(fold))
        clf.fit(x_train, y_train)
        score = _positive_score(clf, x_test)
        pred_rows.append(
            pd.DataFrame(
                {
                    "sample_id": test["sample_id"].astype(str).to_numpy(),
                    "scenario_id": test.get("scenario_id", test["sample_id"]).astype(str).to_numpy(),
                    "group_col": group_col,
                    "group_id": test[group_col].astype(str).to_numpy(),
                    "outer_fold": int(fold),
                    "model": model,
                    "y_true": y_test,
                    "score": score,
                    "n_features": len(cols),
                }
            )
        )
        _ = y_test
    if not pred_rows:
        return pd.DataFrame(columns=["sample_id", "scenario_id", "group_col", "group_id", "outer_fold", "model", "y_true", "score", "n_features"])
    return pd.concat(pred_rows, ignore_index=True)


def oof_metrics(pred: pd.DataFrame) -> dict[str, Any]:
    if pred.empty:
        return {"n": 0, "positive_count": 0, "AUPRC": np.nan, "AUROC": np.nan, "Recall@1%FPR": np.nan, "Recall@5%FPR": np.nan}
    return binary_metric_values(pred["y_true"].to_numpy(int), pd.to_numeric(pred["score"], errors="coerce").to_numpy(float))
