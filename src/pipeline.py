"""Run the team's available lifecycle stages using the shared modules.

    python -m src.pipeline          # four categories; reuses collected raw data
    python -m src.pipeline --quick  # first 5,000 All_Beauty rows, NOT a random sample

Full mode connects collection, cleaning, features, CV, tuning, final evaluation
and EDA/model charts. Quick mode verifies a small sample and reads the committed
model tables without fitting or evaluating models.
Verification outputs stay in gitignored data/processed/pipeline/{quick,full}/;
the team's existing cleaned data, result tables and report figures are untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
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


RESULT_FILES = {
    "cv": "cv_results.csv", "tuning": "tuning_results.csv",
    "test": "test_results.csv", "importance": "feature_importance.csv",
    "predictions": "test_predictions.csv", "by_category": "test_errors_by_category.csv",
}


def run_features(clean_path: Path) -> dict[str, Path]:
    """Call P2's feature builder and deterministic category-stratified split."""
    from src import features
    return features.build_model_data(clean_path=clean_path,
                                     out_dir=clean_path.parent / "features")


@contextmanager
def _model_paths(split_dir: Path, table_dir: Path):
    """Adapt P3's fixed-path API without editing it or overwriting report tables.

    Restore module globals even on failure. The CLI/notebook calls are sequential;
    this adapter must not be used concurrently by threads in the same process.
    """
    from src import features, models
    old_tables, old_loader = models.TABLES_DIR, models.load_split
    try:
        models.TABLES_DIR = table_dir
        models.load_split = partial(features.load_split, out_dir=split_dir)
        yield models
    finally:
        models.TABLES_DIR, models.load_split = old_tables, old_loader


def run_models(model_data: dict[str, Path], *, quick: bool = False) -> dict[str, Path]:
    """Run P3's unchanged CV/final routines, or read reference tables in quick mode.

    Quick mode NEVER calls CV, tuning or test evaluation. Full mode uses the
    newly generated split and writes only beneath data/processed/pipeline/full/.
    The full reproduction evaluates the frozen test split after training-only
    model selection; it does not tune against test outcomes.
    """
    if quick:
        directory = ROOT / "reports" / "tables"
        paths = {key: directory / name for key, name in RESULT_FILES.items()}
        missing = [p.name for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Committed model tables missing: {missing}; pull main first")
        print("models: skipped in --quick; using committed tables unchanged", flush=True)
        return paths
    split_dir = model_data["train"].parent
    if model_data["test"].parent != split_dir:
        raise ValueError("Train/test files must come from the same feature-builder run")
    directory = split_dir.parent / "model_tables"
    with _model_paths(split_dir, directory) as models:
        cv_path = models.main([])
        paths = models.run_final(random_state=42)
    paths["cv"] = cv_path
    print(f"models: CV, tuning and final evaluation -> {directory.relative_to(ROOT)}", flush=True)
    return paths


def run_figures(clean_path: Path, model_results: dict[str, Path]) -> list[Path]:
    """Call P4's EDA and model plots, keeping PNGs away from reports/figures.

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
        # Explicit table input: reference results in quick mode, reproduced results in full.
        paths = [p for p in paths if p.name[:2] in {f"{n:02d}" for n in range(1, 8)}]
        paths += viz.save_model_figures(table_dir=model_results["test"].parent)
    finally:
        viz.FIG_DIR = previous
        plt.close("all")
    print(f"figures: {len(frame):,} rows -> {len(paths)} EDA/model PNGs; "
          f"{directory.relative_to(ROOT)}", flush=True)
    return paths


def main(argv: list[str] | None = None) -> dict:
    """Run the lifecycle and return artifact paths for the integrated notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="First 5,000 All_Beauty rows; smoke check only")
    args = parser.parse_args(argv)
    started = perf_counter()
    raw = run_collect(quick=args.quick)
    clean = run_clean(raw)
    tables = run_eda(clean)
    model_data = run_features(clean)
    model_results = run_models(model_data, quick=args.quick)
    figures = run_figures(clean, model_results)
    status = ("Quick verification complete; model charts use committed results; no model fitting."
              if args.quick else "Full lifecycle complete; reproduced model results saved separately.")
    print(f"{status} Elapsed: {perf_counter() - started:.1f}s.", flush=True)
    return {"raw": raw, "clean": clean, "tables": tables, "figures": figures,
            "features": model_data, "models": model_results}


if __name__ == "__main__":
    main()
