"""Model-ready features, train/test split and preprocessor (Person 2).

    python -m src.features                       # full cleaned table
    python -m src.features --clean PATH --out DIR

Builds on src.clean outputs. Target is average_rating (1-5 stars, regression).
The split is made once: 80/20, stratified by source category, seed 42.
The test set is for the single final evaluation only. All fitted steps
(imputer, scaler, encoders) live in make_preprocessor() so they are fitted
on training rows inside each CV fold, never on test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "products_clean.parquet"
OUT_DIR = ROOT / "data" / "processed" / "features"

TARGET = "average_rating"
ID = "parent_asin"
MIN_RATINGS = 5  # 1-rating products: rating std 1.48 vs 0.69 at 5-9 ratings
TEST_SIZE = 0.2
SEED = 42

NUMERIC = ["log_price", "log_rating_number", "n_features", "n_description",
           "title_len", "n_categories", "n_details"]
BINARY = ["has_price", "price_is_from"]
CATEGORICAL = ["category", "main_category"]
HIGH_CARDINALITY = ["store"]  # ~200k labels; target-encoded with internal cross-fitting
FEATURES = NUMERIC + BINARY + CATEGORICAL + HIGH_CARDINALITY

CLEAN_COLUMNS = [ID, TARGET, "rating_number", "log_price", "log_rating_number",
                 "n_features", "n_description", "has_price", "price_is_from",
                 "category", "main_category", "store", "title", "categories",
                 "details"]


def _n_details(text) -> int:
    """Count keys in the raw details JSON string; unparseable -> 0."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return 0
    return len(parsed) if isinstance(parsed, dict) else 0


def _list_len(x) -> int:
    try:
        return len(x)
    except TypeError:
        return 0


def build_features(clean: pd.DataFrame, min_ratings: int = MIN_RATINGS) -> pd.DataFrame:
    """Return the model table: ID, FEATURES and TARGET, one row per product.

    Drops products with fewer than min_ratings ratings, because their average
    is too noisy to be a reliable target. No imputation here: missing prices
    stay NaN (flagged by has_price) and are imputed inside the preprocessor.
    """
    df = clean.loc[clean["rating_number"] >= min_ratings].copy()
    df["title_len"] = df["title"].astype("string").str.len().fillna(0).astype("int64")
    df["n_categories"] = df["categories"].map(_list_len)
    df["n_details"] = df["details"].map(_n_details)
    for col in BINARY:
        df[col] = df[col].astype("int8")
    for col in CATEGORICAL + HIGH_CARDINALITY:
        df[col] = df[col].astype("string").fillna("Unknown")
    return df[[ID] + FEATURES + [TARGET]].reset_index(drop=True)


def split_data(model_df: pd.DataFrame, test_size: float = TEST_SIZE,
               seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split stratified by source category, so each category keeps its share."""
    return train_test_split(model_df, test_size=test_size, random_state=seed,
                            stratify=model_df["category"])


def make_preprocessor() -> ColumnTransformer:
    """Unfitted preprocessor; put it first in each model pipeline.

    Numeric: median imputation (only log_price has gaps) + scaling for Ridge.
    Categorical: one-hot; main_category labels under 100 training rows are
    grouped as infrequent. Store: TargetEncoder, which cross-fits on the
    training data to avoid leaking each row's own rating.
    """
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("binary", "passthrough", BINARY),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=100, sparse_output=False), CATEGORICAL),
        ("store", TargetEncoder(target_type="continuous", random_state=SEED), HIGH_CARDINALITY),
    ])


def build_model_data(clean_path: Path = CLEAN_PATH, out_dir: Path = OUT_DIR) -> dict[str, Path]:
    """Read the cleaned table, build features, split, and save train/test parquet."""
    clean = pd.read_parquet(clean_path, columns=CLEAN_COLUMNS)
    model_df = build_features(clean)
    train, test = split_data(model_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": out_dir / "train.parquet", "test": out_dir / "test.parquet"}
    train.to_parquet(paths["train"], index=False)
    test.to_parquet(paths["test"], index=False)
    print(f"features: {len(clean):,} cleaned -> {len(model_df):,} with >= {MIN_RATINGS} "
          f"ratings; train {len(train):,}, test {len(test):,}; {out_dir}", flush=True)
    return paths


def load_split(out_dir: Path = OUT_DIR):
    """Return X_train, X_test, y_train, y_test from the saved split."""
    train = pd.read_parquet(out_dir / "train.parquet")
    test = pd.read_parquet(out_dir / "test.parquet")
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    build_model_data(args.clean, args.out)


if __name__ == "__main__":
    main()
