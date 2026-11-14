"""Models, cross-validation, tuning and final test evaluation (Person 3).

    python -m src.models                      # P3-b: 5-fold CV on a 200,000-row training sample
    python -m src.models --cv-sample 0        # P3-b: 5-fold CV on the full training set (slow)
    python -m src.models --final              # P3-c: tune, evaluate ONCE on test, save tables

Uses Person 2's split and preprocessor from src.features (load_split,
make_preprocessor). Every model is wrapped in Pipeline(prep -> model), so the
imputer, scaler and encoders are fitted only on the training folds inside each
CV split: no information from the validation fold leaks into preprocessing.

The test set is used in exactly one place, evaluate_on_test(), after tuning.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from src.features import SEED, load_split, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
CV_SAMPLE = 200_000  # rows of the training set used for CV and tuning (laptop-friendly)
TUNE_ITER = 12       # random settings tried by RandomizedSearchCV
IMPORTANCE_SAMPLE = 20_000  # test rows used for permutation importance

# Settings searched for HistGradientBoosting (the best model in 5-fold CV)
HGB_PARAM_GRID = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_iter": [200, 400, 600],
    "model__max_leaf_nodes": [15, 31, 63, 127],
    "model__min_samples_leaf": [20, 50, 100, 200],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}

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


def _metrics(y_true, y_pred) -> dict:
    """RMSE and MAE (in stars) and R2 for one set of predictions."""
    return {"rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)}


def tune_best_model(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = TUNE_ITER,
                    cv: int = 5, random_state: int = SEED):
    """RandomizedSearchCV over HistGradientBoosting settings, 5-fold, RMSE scoring.

    Runs on the training data only. Returns (search, results_table): the fitted
    search object and one row per setting tried, best first, with the mean and
    std of CV RMSE.
    """
    pipe = get_models(random_state)["HistGradientBoosting"]
    search = RandomizedSearchCV(
        pipe, HGB_PARAM_GRID, n_iter=n_iter, scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
        random_state=random_state, refit=False)
    search.fit(X_train, y_train)
    res = pd.DataFrame(search.cv_results_)
    params = res["params"].apply(pd.Series)
    params.columns = [c.replace("model__", "") for c in params.columns]
    for col in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
        params[col] = params[col].astype(int)
    table = pd.concat([params, pd.DataFrame({
        "cv_rmse_mean": -res["mean_test_score"],
        "cv_rmse_std": res["std_test_score"],
        "fit_time_s": res["mean_fit_time"]})], axis=1)
    table = table.sort_values("cv_rmse_mean").reset_index(drop=True)
    return search, table.round(4)


def evaluate_on_test(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series,
                     name: str = "HistGradientBoosting (tuned)") -> tuple[Pipeline, pd.DataFrame]:
    """Fit on ALL training rows, then score ONCE on the held-out test set.

    The mean baseline is fitted and scored the same way for comparison. Train
    and test metrics are both reported, so overfitting shows up as a gap.
    Returns (fitted_model, results_table).
    """
    rows = []
    baseline = get_models()["Baseline (mean)"]
    fitted = {}
    for label, pipe in [("Baseline (mean)", baseline), (name, clone(model))]:
        start = time.time()
        pipe.fit(X_train, y_train)
        fitted[label] = pipe
        for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            m = _metrics(y, pipe.predict(X))
            rows.append({"model": label, "split": split, "rows": len(X), **m})
        print(f"  {label:<30} fitted on {len(X_train):,} rows ({time.time() - start:.0f}s)",
              flush=True)
    return fitted[name], pd.DataFrame(rows).round(4)


def importance_table(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                     n: int = IMPORTANCE_SAMPLE, n_repeats: int = 5,
                     random_state: int = SEED) -> pd.DataFrame:
    """Permutation importance on a stratified sample of the test set.

    For each original feature, shuffle its values and measure how much test
    RMSE gets worse. A bigger increase means the model relies on that feature
    more. Mean and std over n_repeats shuffles, sorted, biggest first.
    """
    X_s, y_s = sample_training_rows(X_test, y_test, n, random_state)
    result = permutation_importance(model, X_s, y_s, n_repeats=n_repeats,
                                    scoring="neg_root_mean_squared_error",
                                    random_state=random_state)
    table = pd.DataFrame({"feature": X_s.columns,
                          "rmse_increase_mean": result.importances_mean,
                          "rmse_increase_std": result.importances_std,
                          "rows": len(X_s)})
    return table.sort_values("rmse_increase_mean", ascending=False).reset_index(drop=True).round(5)


def save_predictions(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                     path: Path) -> pd.DataFrame:
    """Write actual, predicted, residual (actual - predicted) and category for
    every test product, for Person 4's charts. Returns the table."""
    pred = model.predict(X_test)
    out = pd.DataFrame({"actual": y_test.to_numpy(),
                        "predicted": np.round(pred, 4),
                        "residual": np.round(y_test.to_numpy() - pred, 4),
                        "category": X_test["category"].to_numpy()})
    out.to_csv(path, index=False)
    return out


def errors_by_category(predictions: pd.DataFrame) -> pd.DataFrame:
    """Test RMSE, MAE and mean residual per source category (where the model is weakest)."""
    df = predictions.assign(sq=predictions["residual"] ** 2,
                            abs_res=predictions["residual"].abs())
    table = df.groupby("category", observed=True).agg(
        rows=("residual", "size"), mse=("sq", "mean"), mae=("abs_res", "mean"),
        mean_residual=("residual", "mean"), mean_actual=("actual", "mean"))
    table.insert(1, "rmse", np.sqrt(table.pop("mse")))
    return table.sort_values("rmse", ascending=False).reset_index().round(4)


def run_final(random_state: int = SEED) -> dict[str, Path]:
    """P3-c: tune on the training sample, evaluate once on test, save all tables."""
    X_train, X_test, y_train, y_test = load_split()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    X_tune, y_tune = sample_training_rows(X_train, y_train, CV_SAMPLE, random_state)
    print(f"tuning: {TUNE_ITER} settings x 5 folds on {len(X_tune):,} training rows", flush=True)
    start = time.time()
    search, tuning = tune_best_model(X_tune, y_tune, random_state=random_state)
    tuning.insert(0, "cv_rows", len(X_tune))
    paths["tuning"] = TABLES_DIR / "tuning_results.csv"
    tuning.to_csv(paths["tuning"], index=False)
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"tuning: done in {time.time() - start:.0f}s; best CV RMSE "
          f"{tuning.loc[0, 'cv_rmse_mean']:.4f} ± {tuning.loc[0, 'cv_rmse_std']:.4f}", flush=True)
    print(f"tuning: best settings {best}", flush=True)

    tuned = get_models(random_state)["HistGradientBoosting"].set_params(**search.best_params_)
    print("test: fitting on the full training set, scoring ONCE on test", flush=True)
    model, test_results = evaluate_on_test(tuned, X_train, y_train, X_test, y_test)
    paths["test"] = TABLES_DIR / "test_results.csv"
    test_results.to_csv(paths["test"], index=False)

    print("importance: permutation importance on a test sample", flush=True)
    importance = importance_table(model, X_test, y_test, random_state=random_state)
    paths["importance"] = TABLES_DIR / "feature_importance.csv"
    importance.to_csv(paths["importance"], index=False)

    paths["predictions"] = TABLES_DIR / "test_predictions.csv"
    predictions = save_predictions(model, X_test, y_test, paths["predictions"])
    by_cat = errors_by_category(predictions)
    paths["by_category"] = TABLES_DIR / "test_errors_by_category.csv"
    by_cat.to_csv(paths["by_category"], index=False)

    print("\n=== TEST RESULTS ===")
    print(test_results.to_string(index=False))
    print("\n=== TOP FEATURES (RMSE increase when shuffled) ===")
    print(importance.to_string(index=False))
    print("\n=== TEST ERRORS BY CATEGORY ===")
    print(by_cat.to_string(index=False))
    print("\n=== TOP 5 TUNING SETTINGS ===")
    print(tuning.head(5).to_string(index=False))
    for p in paths.values():
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
    return paths


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="P3 models: CV (default) or --final.")
    parser.add_argument("--cv-sample", type=int, default=CV_SAMPLE,
                        help="training rows used for CV; 0 = full training set")
    parser.add_argument("--final", action="store_true",
                        help="tune the best model, evaluate once on test, save tables")
    args = parser.parse_args(argv)
    if args.final:
        return run_final()

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
