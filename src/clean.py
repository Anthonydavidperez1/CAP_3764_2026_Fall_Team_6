import numpy as np
import pandas as pd

PRICE_MISSING_TEXT = {"None", "—", "-", ""}


def clean_price(s: pd.Series) -> pd.DataFrame:
 """Convert raw price strings to numbers.
 'None' and dashes become NaN, 'from X' keeps X and is flagged,
 and prices <= 0 become NaN."""
 raw = s.astype("string").str.strip()
 is_from = raw.str.startswith("from ", na=False)
 text = raw.str.replace(r"^from\s+", "", regex=True)
 text = text.str.replace(r"[$,]", "", regex=True)
 text = text.mask(text.isin(PRICE_MISSING_TEXT))
 num = pd.to_numeric(text, errors="coerce")
 num = num.mask(num <= 0)
 return pd.DataFrame({
     "price_num": num.astype("float64"),
     "price_is_from": is_from.astype(bool),
     "log_price": np.log1p(num).astype("float64"),
 })


def _list_len(x):
    try:
        return len(x)
    except TypeError:
        return 0


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw product table. Pass a freshly loaded raw frame;
    the raw parquet file on disk is never modified."""
    df = df.drop(columns=["subtitle", "author"])
    for col in ["store", "main_category"]:
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s == "").fillna("Unknown")
    df = df.join(clean_price(df["price"]))
    df["has_price"] = df["price_num"].notna()
    df["log_rating_number"] = np.log1p(df["rating_number"])
    df["n_features"] = df["features"].map(_list_len)
    df["n_description"] = df["description"].map(_list_len)
    return df
