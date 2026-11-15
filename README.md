# Amazon Product Ratings — What Drives Customer Satisfaction

CAP 3764, Fall 2026, Team 6. Predict product average rating (1–5 stars) from
listing characteristics and identify associations useful to sellers and category
managers. These observational relationships do not establish causation.

## Team and contributions

| Person | Name | GitHub | Built |
|---|---|---|---|
| 1 | Parker Fabian | pfab001 | Collection, environment, integration pipeline, main/collection notebooks, README and reproducibility checks |
| 2 | Michael Font | mfont051 | Cleaning, features, category-stratified split, fold-fitted preprocessing and limitations |
| 3 | Rana Waleed Aftab | wa1eed2 | Statistics, model comparison, CV, tuning, final evaluation and result tables |
| 4 | Anthony Perez | Anthonydavidperez1 | EDA/model charts, visualization notebook and executive-summary assembly |

## Setup

Install Git and Conda, then clone the repository:

```bash
git clone https://github.com/Anthonydavidperez1/CAP_3764_2026_Fall_Team_6.git
cd CAP_3764_2026_Fall_Team_6
```

Select the repository's conda-forge configuration for this shell before creating
or updating the environment. This avoids checking unused default channels.

- Anaconda Prompt (Windows): `set "CONDARC=%CD%\.condarc"`
- PowerShell: `$env:CONDARC = (Resolve-Path .condarc).Path`
- macOS/Linux: `export CONDARC="$PWD/.condarc"`

```bash
conda env create -f amazon-reviews-env.yml
conda activate amazon-reviews
python -c "import pandas, numpy, pyarrow, sklearn, matplotlib, seaborn, huggingface_hub; print('imports OK')"
```

For an existing environment, use this after pulling dependency changes:

```bash
conda env update -n amazon-reviews -f amazon-reviews-env.yml --prune
```

Keep `amazon-reviews` active when launching Python or Jupyter, including on Windows
where compiled plotting dependencies need the activated library paths.

## Reproduce

Run from the repository root:

```bash
python -m src.pipeline --quick
python -m src.pipeline
```

**Quick** collects the first 5,000 All_Beauty entries, cleans them, builds sample
features/splits, produces three EDA tables and generates all eleven charts.
It **does not fit, tune or evaluate models**: its four model charts read Rana's
committed full-data tables. This sample is not random or representative; its EDA
must not be presented as a whole-project finding. A fresh copy downloads the
first source shard even though only 5,000 rows are read.

**Full** uses all four categories, reusing `data/raw/products_raw.parquet` when
present or collecting it when absent. It calls Michael's feature builder and
seed-42, 80/20 category-stratified split, Rana's training-only five-fold CV on
200,000 training rows, 12-setting randomized gradient-boosting tuning, final
training/test evaluation, permutation importance, and Anthony's eleven charts.
This takes longer than quick mode; downloads and model fitting determine runtime.
The test split is evaluated after training-only selection, not used to tune it.
Running full mode is a reproduction of the frozen design, not a new model search
against previously observed test performance.

Generated clean data, splits, tables and figures live under ignored
`data/processed/pipeline/quick/` or `full/`. Full model outputs are in
`full/model_tables/`; model figures are in the corresponding `figures/` folder.
Both modes preserve the published CSVs under `reports/tables/`, the committed
`reports/figures/` images and teammates' original processed data. Full mode adapts
P3's fixed input/output paths temporarily and restores them, without editing his
module. Pipeline calls are sequential; do not call those adapters concurrently
from multiple threads in one Python process.

To deliberately rebuild the complete raw table from its pinned source first:

```bash
python -m src.collect
python -m src.collect --counts
```

Downloads are cached. Data and trained models are never committed. The raw
streaming collector uses bounded memory; cleaning is also batched, while feature
engineering/model fitting need RAM for the modeling table. Allow several GB of
disk space: the source shards total about 1.69 GB and the raw table about 1.15 GB.

Regression checks for quick-mode result protection and model path routing:

```bash
python -m unittest discover -s tests -v
```

## Integrated notebook

```bash
python -m ipykernel install --user --name amazon-reviews --display-name "Python (amazon-reviews)"
jupyter lab
```

Open [00_main_project.ipynb](notebooks/00_main_project.ipynb), select that kernel,
then Restart Kernel and Run All Cells. It runs from the root or `notebooks/`.
Default `QUICK = True` verifies a sample and displays committed full-data model
results. Set `QUICK = False` for a full reproduction, including model fitting.
Each numbered section names the module owner and links to the detailed notebook.

## Results and report

The committed [test results](reports/tables/test_results.csv) show tuned histogram
gradient boosting test RMSE **0.4368 stars**, MAE **0.3207 stars**, and R² **0.2642**
on **235,082** held-out products. Baseline RMSE is **0.5092 stars**, so error is
about **14.2% lower**. Training RMSE is 0.3956; the gap is a limitation, not a claim
that overfitting is absent. [CV results](reports/tables/cv_results.csv) include the
mean and spread across five training folds.

The [importance table](reports/tables/feature_importance.csv) shows store labels
followed by log rating volume as the strongest predictors; log price contributes
little. Store labels are not verified brands. [Category errors](reports/tables/test_errors_by_category.csv)
show All_Beauty has the largest test RMSE (0.5098). These are associations and
model reliance, not causal effects of changing a listing.

- [All committed result tables](reports/tables/)
- [All committed report figures](reports/figures/)
- [Risks and limitations — Michael](reports/risks_limitations_p2.md)
- [Executive summary — Anthony's final report](reports/executive_summary.md)

Anthony assembles and adds the executive summary separately; until that file is
merged, the link above is the agreed destination. The report must be at most
600 words. Parker's Analytical Overview is included in section 8 of the main
notebook for report assembly. Every member submits their own peer-review form.

## Repository structure

```text
amazon-reviews-env.yml             # reproducible Conda specification
src/
  collect.py                      # P1: pinned raw metadata collection
  clean.py                        # P2: cleaning
  features.py                     # P2: features, split and preprocessing
  stats.py                        # P3: summary statistics
  models.py                       # P3: CV, tuning and evaluation
  viz.py                          # P4: EDA and model charts
  pipeline.py                     # P1: integrated lifecycle, safe quick mode
notebooks/
  00_main_project.ipynb            # P1: integrated story
  01_collection_p1.ipynb           # P1: collection checks
  02_cleaning_p2.ipynb             # P2: cleaning decisions
  03_stats_p3.ipynb                # P3: EDA/statistics
  04_viz_p4.ipynb                  # P4: chart interpretation
  05_features_p2.ipynb             # P2: modeling dataset
  06_modeling_p3.ipynb             # P3: validation, tuning and findings
reports/
  tables/                         # committed evaluation tables
  figures/                        # committed report charts
  risks_limitations_p2.md          # P2: limitations draft
  executive_summary.md            # P4: final report, added separately
.github/pull_request_template.md  # meaningful review descriptions
data/                             # ignored raw data and generated outputs
models/                           # ignored trained-model artifacts
```

## Source and scope

[Amazon Reviews 2023, McAuley Lab](https://amazon-reviews-2023.github.io/), via
[Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
pinned revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`. Consult the source's
usage/license terms before redistributing data; this repository does not include
source data. No API key is required for the public parquet source.

| Source category | Raw entries |
|---|---:|
| All_Beauty | 112,590 |
| Musical_Instruments | 213,593 |
| Toys_and_Games | 890,874 |
| Industrial_and_Scientific | 427,564 |
| **Total** | **1,644,621** |

The raw table has 14 columns. One row is a product entry within a source category;
`parent_asin` is the product ID and cross-category overlap is possible. The wider
review corpus spans May 1996–September 2023, but this metadata has no review dates
establishing an observed date range. Only media columns (`images`, `videos`,
`bought_together`) are omitted during collection; raw nested fields are preserved.
Michael's five-rating threshold retains 1,175,407 entries (940,325 train/235,082
test); it excludes new/low-volume products. About 49.1% of modeled entries have
missing prices. Ratings are self-selected, the source is a 2023 snapshot, and the
current row split is not an unseen-product-group or future-time validation.
