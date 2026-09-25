# Person 1 verification — September 25, 2026

## Measured full collection

| Source category | Rows |
|---|---:|
| All_Beauty | 112,590 |
| Musical_Instruments | 213,593 |
| Toys_and_Games | 890,874 |
| Industrial_and_Scientific | 427,564 |
| Total | 1,644,621 |

Output: `data/raw/products_raw.parquet`, 14 columns, 1,151,590,690 bytes.
All source rows are retained. Media columns are omitted; source category is added.
Pinned dataset revision: `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`.

## Checks actually completed

- Required libraries import successfully in an isolated Python 3.10 venv.
- All ten source metadata shards downloaded successfully.
- Full streaming collection completed; notebook reran collection successfully.
- Notebook executed all cells in order without errors.
- Source parquet footer counts match the combined table's category counts.
- Column projection, seven-row preview, zero-row preview, and save/load round trip checked.
- pip dependency consistency check passed.

## Raw schema observations for Person 2

`average_rating` is Arrow double; `rating_number` is int64. `price` is string,
not numeric. `details` is string, not an Arrow struct/dictionary. `categories`,
`features`, and `description` are lists of strings. All other retained fields
are strings: main_category, title, store, parent_asin, subtitle, author, category.
These are storage types; inspect missing values and representations before cleaning.
No missing-value treatment or duplicate removal has been performed.

## Remaining setup and publication checks

- Conda is not available on PATH or in the common install locations checked.
  Automated Miniconda installation was rejected by automatic approval review.
  `environment.yml` is prepared but a real conda build is NOT verified.
- Git uses Parker Fabian / parkerfabian0@gmail.com for this repository only.
- GitHub credential access was absent; push permission is NOT verified.
- Branch publication, PR creation, merge, and remote fresh-clone verification remain pending.
- Team chat/deck links, other members' setup, and public/private decision are unconfirmed.

The local Python test environment does not satisfy the professor's conda requirement.
Do not post "env working" or tell Person 2 that main is ready until the remaining
setup/publication checks actually pass.

JupyterLab server check: started on 127.0.0.1, returned HTTP 200, then stopped.
