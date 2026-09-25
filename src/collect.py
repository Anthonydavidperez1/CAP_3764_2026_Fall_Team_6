"""Collect product metadata; preserve raw values for Person 2's cleaning.

Run ``python -m src.collect`` from the repository root to reproduce all data.
Downloads are cached under the gitignored data directory at a fixed revision.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CATEGORIES = ["All_Beauty", "Musical_Instruments", "Toys_and_Games",
              "Industrial_and_Scientific"]
DROP_COLUMNS = {"images", "videos", "bought_together"}


@lru_cache(maxsize=1)
def _repo_files() -> tuple[str, ...]:
    """List files once per process, using the reproducible source revision."""
    return tuple(list_repo_files(REPO_ID, repo_type="dataset", revision=REVISION))


def _shards(category: str) -> list[str]:
    """Find only product metadata parquet shards, never review/benchmark data."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; choose from {CATEGORIES}")
    files = sorted(p for p in _repo_files()
                   if p.startswith(f"raw_meta_{category}/") and p.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"No metadata parquet shards for {category}")
    return files


def _download(filename: str) -> Path:
    """Retrieve one pinned shard, reusing the local cache when present."""
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset",
                               revision=REVISION, local_dir=DATA_DIR / "source"))


def download_category(category: str) -> list[Path]:
    """Download every metadata parquet shard for one category."""
    return [_download(filename) for filename in _shards(category)]


def load_category(category: str, columns: Sequence[str] | None = None,
                  nrows: int | None = None) -> pd.DataFrame:
    """Read metadata, omit media, and tag source category without cleaning.

    nrows returns the first rows (not a random sample), reading batches and
    downloading only required shards. The first required shard is still downloaded
    in full. A source ``category`` tag is always included, even with projection.
    """
    if nrows is not None and (isinstance(nrows, bool) or not isinstance(nrows, int) or nrows < 0):
        raise ValueError("nrows must be a nonnegative integer or None")
    if columns is not None and isinstance(columns, str):
        raise TypeError("columns must be a sequence of column names, not a string")
    frames: list[pd.DataFrame] = []
    remaining = nrows
    for filename in _shards(category):
        pf = pq.ParquetFile(_download(filename))
        names = pf.schema_arrow.names
        selected = ([c for c in names if c not in DROP_COLUMNS] if columns is None
                    else [c for c in columns if c not in DROP_COLUMNS and c != "category"])
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError(f"Unknown columns: {sorted(unknown)}")
        if remaining == 0:
            empty_schema = pa.schema([pf.schema_arrow.field(c) for c in selected])
            frames.append(pa.Table.from_batches([], schema=empty_schema).to_pandas())
            break
        for batch in pf.iter_batches(batch_size=16384, columns=selected):
            if remaining is not None:
                batch = batch.slice(0, min(remaining, batch.num_rows))
                remaining -= batch.num_rows
            frames.append(batch.to_pandas())
            if remaining == 0:
                break
        if remaining == 0:
            break
    result = pd.concat(frames, ignore_index=True)
    result["category"] = category
    print(f"{category}: {len(result):,} rows", flush=True)
    return result


def load_all(categories: Sequence[str] = CATEGORIES) -> pd.DataFrame:
    """Load the requested categories and concatenate in the supplied order."""
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be nonempty and contain no duplicates")
    return pd.concat([load_category(c) for c in categories], ignore_index=True)


def _raw_path(name: str) -> Path:
    """Keep output names inside data/raw and avoid accidental path traversal."""
    if not name or Path(name).name != name or any(c in name for c in '/\\:'):
        raise ValueError("name must be a simple file stem")
    return DATA_DIR / "raw" / f"{name}.parquet"


def save_raw(df: pd.DataFrame, name: str = "products_raw") -> Path:
    """Atomically save the raw product table as parquet, without its index."""
    path = _raw_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    df.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def load_raw(name: str = "products_raw") -> pd.DataFrame:
    """Read the locally saved raw table; run the CLI first if it is absent."""
    return pd.read_parquet(_raw_path(name))


def row_counts() -> dict[str, int]:
    """Count records from parquet footers; downloads shards but not full tables."""
    return {c: sum(pq.ParquetFile(p).metadata.num_rows for p in download_category(c))
            for c in CATEGORIES}


def collect_to_disk() -> Path:
    """Stream all categories to one parquet file, keeping memory use bounded.

    Read the original Arrow schema directly to preserve nested/raw field types.
    Fail if source schemas differ rather than silently coercing raw values.
    """
    path = _raw_path("products_raw")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    writer = None
    try:
        for category in CATEGORIES:
            count = 0
            for shard in download_category(category):
                pf = pq.ParquetFile(shard)
                columns = [c for c in pf.schema_arrow.names if c not in DROP_COLUMNS]
                for batch in pf.iter_batches(batch_size=16384, columns=columns):
                    table = pa.Table.from_batches([batch])
                    table = table.append_column("category", pa.array([category] * len(table)))
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema)
                    writer.write_table(table)
                    count += len(table)
            print(f"{category}: {count:,} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)
    print(f"Saved {path}", flush=True)
    return path


def main() -> None:
    """Collect all metadata or print footer counts with --counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", action="store_true", help="Print source row counts only")
    args = parser.parse_args()
    if args.counts:
        for category, count in row_counts().items():
            print(f"{category}: {count:,}")
    else:
        collect_to_disk()


if __name__ == "__main__":
    main()
