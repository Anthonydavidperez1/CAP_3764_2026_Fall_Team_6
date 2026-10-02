"""Summary statistics for the cleaned Amazon product table (Person 3).

Every function takes the cleaned DataFrame produced by
``src.clean.clean_products`` and returns a pandas DataFrame, so each result can
be displayed in the notebook and cited in the written insights.

Columns used (all created by Person 1's collection or Person 2's cleaning):
    average_rating     target, 1.0-5.0
    rating_number      how many ratings a product has
    price_num          cleaned numeric price (NaN when no price was listed)
    category           our source category tag (4 values)
    main_category      Amazon's own category label ("Unknown" when missing)
    store              store / seller label ("Unknown" when missing)
    has_price          True when price_num is not missing

Note: about half of all products have no price, so anything involving price
describes priced products only.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

NUMERIC_COLS = ["average_rating", "rating_number", "price_num"]

# Review-volume bands for review_volume_summary(). Right edges are exclusive,
# e.g. "10-99" means 10 <= rating_number < 100.
VOLUME_BINS = [0, 10, 100, 1_000, 10_000, float("inf")]
VOLUME_LABELS = ["1-9", "10-99", "100-999", "1k-9.9k", "10k+"]


def numeric_summary(df: pd.DataFrame,
                    cols: Sequence[str] = NUMERIC_COLS) -> pd.DataFrame:
    """Describe the numerical variables (spec item 4a).

    One row per variable: count, n_missing, mean, median, std, min,
    quartiles, max and skew. Skew tells us whether the mean is misleading:
    a large positive skew means a few huge values pull the mean up, so the
    median is the better "typical" value.
    """
    rows = {}
    for col in cols:
        s = df[col].astype("float64")
        rows[col] = {
            "count": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "25%": s.quantile(0.25),
            "75%": s.quantile(0.75),
            "max": s.max(),
            "skew": s.skew(),
        }
    return pd.DataFrame(rows).T.round(3)


def categorical_overview(df: pd.DataFrame,
                         cols: Sequence[str] = ("category", "main_category", "store"),
                         unknown: str = "Unknown") -> pd.DataFrame:
    """One row per categorical variable (spec item 4b).

    n_unique, the most common value and its share, and how many rows are
    the "Unknown" placeholder Person 2 used for missing values.
    """
    rows = {}
    for col in cols:
        counts = df[col].value_counts()
        rows[col] = {
            "n_unique": int(df[col].nunique()),
            "most_common": counts.index[0],
            "most_common_pct": round(100 * counts.iloc[0] / len(df), 2),
            "n_unknown": int((df[col] == unknown).sum()),
            "pct_unknown": round(100 * (df[col] == unknown).mean(), 2),
        }
    return pd.DataFrame(rows).T


def categorical_summary(df: pd.DataFrame, col: str,
                        top: int | None = None) -> pd.DataFrame:
    """Frequency table for one categorical column (spec item 4b).

    Columns: count, pct (share of all rows) and cum_pct (running total).
    The number of distinct values is stored in ``result.attrs["n_unique"]``
    and printed, because it describes the column, not any one row.
    Use ``top`` for high-cardinality columns such as ``store``.
    """
    counts = df[col].value_counts()
    out = pd.DataFrame({
        "count": counts,
        "pct": (100 * counts / len(df)).round(2),
    })
    out["cum_pct"] = out["pct"].cumsum().round(2)
    out.index.name = col
    out.attrs["n_unique"] = int(df[col].nunique())
    print(f"{col}: {out.attrs['n_unique']:,} unique values")
    return out.head(top) if top else out


def rating_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rating, price and review-volume profile for each value of ``col``.

    Used by rating_by_category(), and handy for any other grouping
    (for example ``has_price``). Price columns use priced products only.
    """
    g = df.groupby(col, observed=True)
    out = pd.DataFrame({
        "n_products": g.size(),
        "mean_rating": g["average_rating"].mean(),
        "median_rating": g["average_rating"].median(),
        "std_rating": g["average_rating"].std(),
        "pct_has_price": 100 * g["has_price"].mean(),
        "mean_price": g["price_num"].mean(),
        "median_price": g["price_num"].median(),
        "median_rating_number": g["rating_number"].median(),
    })
    return out.sort_values("mean_rating", ascending=False).round(3)


def rating_by_category(df: pd.DataFrame, col: str = "category") -> pd.DataFrame:
    """Per category: n products, mean/median/std rating, price, review volume
    (spec item 4c). Sorted from highest to lowest mean rating.

    Adds ``diff_from_overall``: how far each category's mean rating sits
    above (+) or below (-) the mean over all products.
    """
    out = rating_by_group(df, col)
    out.insert(2, "diff_from_overall",
               (out["mean_rating"] - df["average_rating"].mean()).round(3))
    return out


def price_band_summary(df: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Mean rating per price quartile, per category (spec item 4c).

    Quartile cut points come from all priced products together, so the bands
    mean the same dollar range in every category. Products without a price
    are left out. Returns one row per price band, one column per category,
    plus an "All" column; ``result.attrs["cut_points"]`` holds the edges and
    ``result.attrs["counts"]`` the number of products in each cell.
    """
    priced = df[df["price_num"].notna()].copy()
    priced["price_band"], edges = pd.qcut(priced["price_num"], 4, retbins=True,
                                          duplicates="drop")
    labels = [f"Q{i + 1}: ${edges[i]:,.2f}-${edges[i + 1]:,.2f}"
              for i in range(len(edges) - 1)]
    priced["price_band"] = priced["price_band"].cat.rename_categories(labels)

    table = priced.pivot_table(index="price_band", columns=by,
                               values="average_rating", aggfunc="mean",
                               observed=True)
    table["All"] = priced.groupby("price_band", observed=True)["average_rating"].mean()
    counts = priced.pivot_table(index="price_band", columns=by,
                                values="average_rating", aggfunc="size",
                                observed=True)
    table = table.round(3)
    table.attrs["cut_points"] = [round(float(e), 2) for e in edges]
    table.attrs["counts"] = counts
    return table


def review_volume_summary(df: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Mean rating by review-volume band, per category.

    Speaks to the project goal of finding heavily reviewed products that
    still rate low. Bands are defined in VOLUME_BINS / VOLUME_LABELS.
    ``result.attrs["counts"]`` holds the number of products in each cell;
    check it before trusting a small cell.
    """
    d = df[[by, "rating_number", "average_rating"]].copy()
    d["volume_band"] = pd.cut(d["rating_number"], VOLUME_BINS,
                              labels=VOLUME_LABELS, right=False)
    table = d.pivot_table(index="volume_band", columns=by,
                          values="average_rating", aggfunc="mean", observed=True)
    table["All"] = d.groupby("volume_band", observed=True)["average_rating"].mean()
    counts = d.pivot_table(index="volume_band", columns=by,
                           values="average_rating", aggfunc="size", observed=True)
    table = table.round(3)
    table.attrs["counts"] = counts
    return table


def top_stores(df: pd.DataFrame, n: int = 20,
               exclude: Sequence[str] = ("Unknown",)) -> pd.DataFrame:
    """Top ``n`` stores by product count, with their rating profile.

    "Unknown" (missing store) is excluded by default because it is not a
    real store. ``diff_from_overall`` compares each store's mean rating to
    the mean over all products.
    """
    d = df[~df["store"].isin(exclude)]
    g = d.groupby("store", observed=True)
    out = pd.DataFrame({
        "n_products": g.size(),
        "main_source_category": g["category"].agg(lambda s: s.value_counts().index[0]),
        "mean_rating": g["average_rating"].mean(),
        "median_rating_number": g["rating_number"].median(),
    })
    out = out.sort_values("n_products", ascending=False).head(n)
    out["diff_from_overall"] = out["mean_rating"] - df["average_rating"].mean()
    return out.round(3)


def correlation_table(df: pd.DataFrame,
                      cols: Sequence[str] = ("average_rating", "rating_number",
                                             "log_rating_number", "price_num",
                                             "log_price", "n_features",
                                             "n_description"),
                      method: str = "spearman") -> pd.DataFrame:
    """Correlation matrix for the numeric columns.

    Default is Spearman (rank-based), because rating_number and price are
    heavily skewed and Pearson would be dominated by a few extreme products.
    Pass ``method="pearson"`` to compare. Each pair uses the rows where both
    values exist, so price correlations cover priced products only.
    """
    cols = [c for c in cols if c in df.columns]
    return df[list(cols)].astype("float64").corr(method=method).round(3)
