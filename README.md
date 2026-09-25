# Amazon Product Ratings — What Drives Customer Satisfaction

CAP 3764, Fall 2026, Team 6. We study associations between product price, category,
store, rating volume, and average customer ratings. Associations are not causation.

## Team

| Person | GitHub | Responsibility |
|---|---|---|
| 1 | pfab001 | Collection, notebook 01, environment, README |
| 2 | mfont051 | Cleaning and features |
| 3 | wa1eed2 | Statistics and hypothesis testing |
| 4 | Anthonydavidperez1 | Visualizations, decks, repository owner |

## Dataset

[Amazon Reviews 2023, McAuley Lab](https://amazon-reviews-2023.github.io/),
via [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
We use product metadata, not individual reviews. The overall source review span
is May 1996–September 2023; this is not a measured date range for our metadata
subset, which contains no review timestamp column.

Pinned source revision: `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`.
Four complete categories: All_Beauty, Musical_Instruments, Toys_and_Games,
Industrial_and_Scientific. Ten source shards total approximately 1.69 GB.
Downloads and generated parquet files stay under gitignored `data/`.

One row is a product entry in a source category. `parent_asin` is the product ID;
cross-category duplicates are possible and belong in Person 2's audit.
`category` is our source tag; `main_category` is the original metadata value.
`store` is a store label, not a verified brand. Numeric variables include
`average_rating` and `rating_number`; `price` remains raw pending cleaning.
We preserve nested categories, details, descriptions, and features. Only
`images`, `videos`, and `bought_together` are omitted. No imputation, duplicate
removal, or value conversion is performed during collection.

## Reproduce

Install Git and conda, then use Anaconda Prompt or a conda-enabled shell:

```bash
git clone https://github.com/Anthonydavidperez1/CAP_3764_2026_Fall_Team_6.git
cd CAP_3764_2026_Fall_Team_6
```

Select the project's conda-forge configuration for this shell. This prevents
Miniconda from checking unused default channels during setup.

Windows Anaconda Prompt (cmd):

```bat
set "CONDARC=%CD%\.condarc"
```

PowerShell: `$env:CONDARC = (Resolve-Path .condarc).Path`

macOS/Linux: `export CONDARC="$PWD/.condarc"`

Then run:

```bash
conda env create -f environment.yml
conda activate amazon-reviews
python -c "import pandas, numpy, pyarrow, seaborn, sklearn, huggingface_hub; print('imports OK')"
python -m src.collect
python -m src.collect --counts
jupyter lab
```

Run `notebooks/01_collection_p1.ipynb` with Restart
Kernel and Run All Cells. It locates the repo from the root or notebooks directory.

The CLI streams to `data/raw/products_raw.parquet`, using bounded memory. Reruns
reuse downloads and rebuild the output. `load_all()` offers an in-memory API.
`--counts` uses parquet footers rather than loading records, but downloads any
missing shards. No API key is needed. Published parquet avoids dependence on
executing the source repository's dataset loading script.

```python
from src.collect import load_category, load_raw
sample = load_category('All_Beauty', nrows=100)
df = load_raw()  # after running the full collection CLI
```

The sample is the first 100 records, not a random or representative sample.

## Workflow and handoff

Person 1 supplies raw data; Person 2 cleans it; Persons 3 and 4 use the clean data.
Use a branch and PR for each task. Pull main before starting and integrate it
before pushing. Never commit data, caches, credentials, or local environments.

P1 collection is due September 27, 2026; P2 cleaning is due September 29.
Submission 1 is October 4; recording is October 3. Every person presents and
submits their own peer review form. Later submission plans are provisional
until the professor supplies the actual requirements.

## Person 2: start cleaning

After cloning and building the environment above, create your task branch:

```bash
git switch main
git pull origin main
git switch -c p2-data-cleaning
python -m src.collect
```

In `notebooks/02_cleaning_p2.ipynb`, import cleaning functions from
`src/clean.py`. Read `data/raw/products_raw.parquet` using `load_raw()`.
Keep that raw file unchanged and save cleaned output under `data/processed/`.
Audit missing values, duplicate rows and `parent_asin` overlap across categories,
and inspect raw prices before conversion. `price` and `details` are strings;
`categories`, `features`, and `description` contain lists. Document every
cleaning decision and before/after row counts. Do not assume `details` is a dict.

Downloads total about 1.69 GB and the combined parquet is about 1.15 GB; allow
several GB of disk space and additional RAM for the full DataFrame. The streaming
collection command uses less memory than the notebook. Data is not included in Git.

Restart and run your notebook, commit only your code and notebook, push your
branch, and open a PR to `main`. Confirm collection reproduces within 24 hours
of the P1 handoff so any setup problems can be resolved before September 29.
For an existing environment, use `conda env update -f environment.yml --prune`
instead of `conda env create` (with the same CONDARC setting).

## Verified raw row counts

| Category | Rows |
|---|---:|
| All_Beauty | 112,590 |
| Musical_Instruments | 213,593 |
| Toys_and_Games | 890,874 |
| Industrial_and_Scientific | 427,564 |
| **Total** | **1,644,621** |

The raw table has 14 columns. Counts were verified against source parquet footers
and the executed collection notebook; these are raw entries, not deduplicated products.
