"""Models and cross-validation (Person 3).

    python -m src.models                      # 5-fold CV on a 200,000-row training sample
    python -m src.models --cv-sample 0        # 5-fold CV on the full training set (slow)

Uses Person 2's split and preprocessor from src.features (load_split,
make_preprocessor). Every model is wrapped in Pipeline(prep -> model), so the
imputer, scaler and encoders are fitted only on the training folds inside each
CV split: no information from the validation fold leaks into preprocessing.

The test set is NOT used here. It is scored once, at the very end, in P3-c.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import SEED, load_split, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
CV_SAMPLE = 200_000  # rows of the training set used for CV (laptop-friendly)

# sklearn reports errors as negative numbers ("higher is better"); we flip the
# sign back when building the results table.
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def get_models(random_state: int = SEED) -> dict[str, Pipeline]:
    """Return the four candidate models, each as Pipeline(prep -> model).

    - Baseline (DummyRegressor, mean): predicts the average training rating for
      every product. Any useful model must beat it.
    - Ridge: linear and regularized. Shows how far straight-line relationships go.
    - RandomForest: captures non-linear effects and interactions. Kept small
      (100 trees, 30% of rows per tree) so it trains in reasonable time.
    - HistGradientBoosting: fast on hundreds of thousands of rows and usually
      the strongest model on tabular data.
    """
    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_samples=0.3, min_samples_leaf=5,
            n_jobs=-1, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=random_state),
    }
    return {name: Pipeline([("prep", make_preprocessor()), ("model", model)])
            for name, model in models.items()}


def sample_training_rows(X: pd.DataFrame, y: pd.Series, n: int,
                         random_state: int = SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified sample of n training rows, keeping each category's share.

    n <= 0 or n >= len(X) returns the full training set unchanged.
    """
    if n <= 0 or n >= len(X):
        return X, y
    frac = n / len(X)
    idx = (X.groupby("category", group_keys=False, observed=True)
             .sample(frac=frac, random_state=random_state).index)
    return X.loc[idx], y.loc[idx]


def cross_validate_models(models: dict[str, Pipeline], X_train: pd.DataFrame,
                          y_train: pd.Series, cv: int = 5,
                          random_state: int = SEED) -> pd.DataFrame:
    """K-fold CV on the TRAINING set only.

    Returns one row per model with mean and std across folds for RMSE, MAE and
    R2 (RMSE and MAE in stars), plus the mean fit time in seconds. Sorted by
    mean RMSE, best first.
    """
    folds = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, pipe in models.items():
        start = time.time()
        scores = cross_validate(pipe, X_train, y_train, cv=folds, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            if metric in ("rmse", "mae"):
                values = -values
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        row["fit_time_s"] = scores["fit_time"].mean()
        rows.append(row)
        print(f"  {name:<22} RMSE {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}   "
              f"R2 {row['r2_mean']:.4f} ± {row['r2_std']:.4f}   "
              f"({time.time() - start:.0f}s)", flush=True)
    return (pd.DataFrame(rows).sort_values("rmse_mean")
              .reset_index(drop=True).round(4))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="5-fold CV for the P3 models.")
    parser.add_argument("--cv-sample", type=int, default=CV_SAMPLE,
                        help="training rows used for CV; 0 = full training set")
    args = parser.parse_args(argv)

    X_train, _, y_train, _ = load_split()  # test set deliberately ignored
    X_cv, y_cv = sample_training_rows(X_train, y_train, args.cv_sample)
    print(f"models: 5-fold CV on {len(X_cv):,} of {len(X_train):,} training rows",
          flush=True)

    results = cross_validate_models(get_models(), X_cv, y_cv)
    results.insert(1, "cv_rows", len(X_cv))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "cv_results.csv"
    results.to_csv(out, index=False)
    print(f"models: wrote {out}", flush=True)
    print(results.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
