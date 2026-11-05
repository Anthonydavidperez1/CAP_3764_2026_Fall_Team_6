"""Run the team's available lifecycle stages using the shared modules.

    python -m src.pipeline          # four categories; reuses collected raw data
    python -m src.pipeline --quick  # first 5,000 All_Beauty rows, NOT a random sample

P1-b connects collection, cleaning and EDA. Feature engineering, modeling and
model charts are explicitly pending P2/P3/P4 delivery and P1-c integration.
Verification outputs stay in gitignored data/processed/pipeline/{quick,full}/;
the team's existing cleaned data, result tables and report figures are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import collect, stats
from src.clean import clean_products

ROOT = Path(__file__).resolve().parents[1]
QUICK_ROWS = 5_000
BATCH_ROWS = 16_384


def _rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _output_dir(raw_path: Path) -> Path:
    mode = "quick" if raw_path.name == "pipeline_quick.parquet" else "full"
    return ROOT / "data" / "processed" / "pipeline" / mode


def run_collect(quick: bool = False) -> Path:
    """Use P1's collector; keep the quick sample separate from the full raw file."""
    if quick:
        frame = collect.load_category("All_Beauty", nrows=QUICK_ROWS)
        path = collect.save_raw(frame, name="pipeline_quick")
        print(f"collect: {len(frame):,} rows -> {path.relative_to(ROOT)} "
              "(first rows, not representative)", flush=True)
    else:
        path = collect.DATA_DIR / "raw" / "products_raw.parquet"
        reused = path.is_file()
        if not reused:
            path = collect.collect_to_disk()
        print(f"collect: {_rows(path):,} rows -> {path.relative_to(ROOT)} "
              f"({'reused existing raw table' if reused else 'collected'})", flush=True)
    return path


def run_clean(raw_path: Path) -> Path:
    """Call P2's cleaning function in batches, preserving all its output columns.

    An explicit Arrow schema preserves nested raw types even when one batch
    contains only empty lists. Write atomically, never modifying the input.
    """
    raw = pq.ParquetFile(raw_path)
    path = _output_dir(raw_path) / "products_clean.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    writer = None
    schema = None
    count = 0
    try:
        for batch in raw.iter_batches(batch_size=BATCH_ROWS):
            cleaned = clean_products(batch.to_pandas())
            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if schema is None:
                schema = pa.schema([
                    raw.schema_arrow.field(field.name)
                    if field.name in raw.schema_arrow.names else field
                    for field in table.schema
                ])
                writer = pq.ParquetWriter(temporary, schema)
            writer.write_table(table.cast(schema))
            count += len(cleaned)
        if writer is None:
            raise ValueError(f"No rows to clean in {raw_path}")
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)
    print(f"clean: {raw.metadata.num_rows:,} rows -> {count:,} rows; "
          f"{path.relative_to(ROOT)}", flush=True)
    return path


def run_eda(clean_path: Path) -> dict[str, Path]:
    """Use P3's summary functions; write small tables beside pipeline data."""
    columns = ["category", "average_rating", "rating_number", "price_num",
               "has_price", "log_rating_number", "log_price", "n_features",
               "n_description"]
    frame = pd.read_parquet(clean_path, columns=columns)
    tables = {
        "numeric_summary": stats.numeric_summary(frame),
        "rating_by_category": stats.rating_by_category(frame),
        "correlation": stats.correlation_table(frame),
    }
    directory = clean_path.parent / "tables"
    directory.mkdir(exist_ok=True)
    paths = {}
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path)
        paths[name] = path
    print(f"eda: {len(frame):,} rows -> {len(paths)} tables; "
          f"{directory.relative_to(ROOT)}", flush=True)
    return paths


def run_features(clean_path: Path) -> Path | None:
    """P1-c will connect P2's model-ready features once their PR is delivered."""
    print("features: SKIPPED; awaiting P2 features/split and P1-c integration", flush=True)
    return None


def run_models(features_path: Path | None) -> Path | None:
    """P1-c will connect P3's training/evaluation API; do not invent its results."""
    print("models: SKIPPED; awaiting P3 evaluation and P1-c integration", flush=True)
    return None


def run_figures(clean_path: Path, model_results_path: Path | None = None) -> list[Path]:
    """Call P4's existing EDA plots, redirecting PNGs away from reports/figures.

    FIG_DIR is restored even on failure. This CLI/notebook workflow is sequential;
    do not call it concurrently in threads because P4's module uses a global path.
    P4's fixed random_state=42 controls its scatter-plot samples.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from src import viz

    frame = pd.read_parquet(clean_path, columns=viz.VIZ_COLS)
    directory = clean_path.parent / "figures"
    previous = viz.FIG_DIR
    try:
        viz.FIG_DIR = directory
        paths = viz.save_all_figures(frame)
    finally:
        viz.FIG_DIR = previous
        plt.close("all")
    print(f"figures: {len(frame):,} rows -> {len(paths)} EDA PNGs; "
          f"{directory.relative_to(ROOT)}", flush=True)
    print("model figures: SKIPPED; awaiting P4 model charts and P1-c integration", flush=True)
    return paths


def main(argv: list[str] | None = None) -> dict:
    """Run the available phases and return artifact paths for the main notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="First 5,000 All_Beauty rows; smoke check only")
    args = parser.parse_args(argv)
    started = perf_counter()
    raw = run_collect(quick=args.quick)
    clean = run_clean(raw)
    tables = run_eda(clean)
    features = run_features(clean)
    models = run_models(features)
    figures = run_figures(clean, models)
    print(f"Available collection/cleaning/EDA stages finished in "
          f"{perf_counter() - started:.1f}s. Modeling is NOT complete.", flush=True)
    return {"raw": raw, "clean": clean, "tables": tables, "figures": figures,
            "features": features, "models": models}


if __name__ == "__main__":
    main()
