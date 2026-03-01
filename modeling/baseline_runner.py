"""Lightweight baseline model runner."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_squared_error,
    r2_score,
)

from config import (
    MAX_FEATURES_FIRST_PASS,
    MAX_ROWS_FIRST_PASS,
    MODEL_TIME_BUDGET_SECONDS,
)


def _select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    max_features: int,
) -> tuple[pd.DataFrame, bool]:
    """Select top ``max_features`` columns for ``X``."""
    if X.shape[1] <= max_features:
        return X, False

    if task == "regression":
        variances = X.var().sort_values(ascending=False)
        cols = variances.head(max_features).index
    else:
        y_enc, _ = pd.factorize(y)
        mi = mutual_info_classif(X.fillna(0), y_enc, discrete_features=False)
        cols = (
            pd.Series(mi, index=X.columns)
            .sort_values(ascending=False)
            .head(max_features)
            .index
        )
    return X.loc[:, cols], True


def run_baseline(
    df: pd.DataFrame,
    target: str,
    task: str,
    time_budget_s: int = MODEL_TIME_BUDGET_SECONDS,
    sample_rows: int = MAX_ROWS_FIRST_PASS,
    max_features: int = MAX_FEATURES_FIRST_PASS,
) -> Dict[str, Any]:
    """Run a simple baseline model under resource constraints."""
    start = time.time()
    timed_out = False

    # Row sampling
    sampled = False
    if len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=0)
        sampled = True

    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)

    # Feature pruning
    X, pruned = _select_top_features(X, y, task, max_features)

    if time.time() - start > time_budget_s:
        return {
            "metrics": {
                "sampled": sampled,
                "features_pruned": pruned,
                "n_rows": len(X),
                "n_features": X.shape[1],
            },
            "model_artifacts": None,
            "timed_out": True,
        }

    if task == "regression":
        model = Ridge()
        model.fit(X, y)
        preds = model.predict(X)
        metrics = {
            "r2": float(r2_score(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        }
    else:
        y_enc, uniques = pd.factorize(y)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y_enc)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        metrics = {
            "accuracy": float(accuracy_score(y_enc, preds)),
            "log_loss": float(log_loss(y_enc, probs)),
        }

    metrics.update(
        {
            "sampled": sampled,
            "features_pruned": pruned,
            "n_rows": len(X),
            "n_features": X.shape[1],
        }
    )

    if time.time() - start > time_budget_s:
        timed_out = True

    return {
        "metrics": metrics,
        "model_artifacts": {"model": model},
        "timed_out": timed_out,
    }
