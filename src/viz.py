"""Visualizations for the cleaned Amazon product table (Person 4).

Every ``plot_*`` function takes the cleaned DataFrame, saves a PNG into
``reports/figures/`` and returns the matplotlib Figure, so the same chart can
be shown in the notebook and dropped onto a slide.

Run ``python -m src.viz`` from the repository root to rebuild every figure
without opening Jupyter.

Columns used (created by Person 1's collection and Person 2's cleaning):
    average_rating   target, 1.0-5.0 in steps of 0.1
    rating_number    how many ratings a product has
    price_num        cleaned numeric price (NaN when no price was listed)
    category         our source category tag (4 values)
    n_features       number of feature bullets on the listing
    n_description    number of description paragraphs on the listing

About half of all products have no listed price, so every chart that uses
price describes priced products only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from src.clean import clean_products

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "products_raw.parquet"
CLEAN_PATH = ROOT / "data" / "processed" / "products_clean.parquet"
FIG_DIR = ROOT / "reports" / "figures"

# Raw columns clean_products() needs. The long text columns (title, details,
# categories) are left out because no chart uses them and they are the bulk
# of the file.
RAW_COLS = ["category", "main_category", "store", "average_rating",
            "rating_number", "price", "features", "description",
            "subtitle", "author"]
# Cleaned columns the charts use.
VIZ_COLS = ["category", "main_category", "store", "average_rating",
            "rating_number", "price_num", "has_price", "n_features",
            "n_description"]

# ---------------------------------------------------------------------------
# One look for every chart. Change a value here and all charts follow.
# ---------------------------------------------------------------------------
# Fixed display order, label, colour and marker for the four categories.
# A category keeps the same colour in every chart.
CATEGORY_ORDER = ["All_Beauty", "Musical_Instruments", "Toys_and_Games",
                  "Industrial_and_Scientific"]
CATEGORY_LABELS = {
    "All_Beauty": "All Beauty",
    "Musical_Instruments": "Musical Instruments",
    "Toys_and_Games": "Toys & Games",
    "Industrial_and_Scientific": "Industrial & Scientific",
}
CATEGORY_COLORS = {
    "All_Beauty": "#2a78d6",                 # blue
    "Musical_Instruments": "#eb6834",        # orange
    "Toys_and_Games": "#1baf7a",             # green
    "Industrial_and_Scientific": "#eda100",  # yellow
}
CATEGORY_MARKERS = {
    "All_Beauty": "o",
    "Musical_Instruments": "s",
    "Toys_and_Games": "^",
    "Industrial_and_Scientific": "D",
}

MAIN_COLOR = "#2a78d6"     # single-colour charts
INK = "#0b0b0b"            # titles and trend lines
INK_SOFT = "#52514e"       # axis labels and notes
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DPI = 150
SAMPLE_PER_CATEGORY = 5_000   # points drawn per panel in the scatter charts
RANDOM_STATE = 42             # fixed so the sampled points never change

# Review-volume bands: the same edges Person 3 uses in src/stats.py.
# Right edges are exclusive, e.g. "10-99" means 10 <= rating_number < 100.
VOLUME_BINS = [0, 10, 100, 1_000, 10_000, float("inf")]
VOLUME_LABELS = ["1-9", "10-99", "100-999", "1k-9.9k", "10k+"]

# Red (negative) -> grey (zero) -> blue (positive) for the correlation heatmap.
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#d03b3b", "#f0efec", "#2a78d6"])
CORR_COLS = {
    "average_rating": "Average rating",
    "rating_number": "Number of ratings",
    "price_num": "Price",
    "n_features": "Feature bullets",
    "n_description": "Description paragraphs",
}


def _apply_style() -> None:
    """Set font sizes and colours once so every chart matches."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 14,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "text.color": INK,
    })


def _save(fig: plt.Figure, name: str) -> Path:
    """Write the figure to reports/figures/<name>.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def _thousands(value: float, _pos=None) -> str:
    """Axis tick text with thousands separators: 250000 -> 250,000."""
    return f"{value:,.0f}"


def _categories_in(df: pd.DataFrame) -> list[str]:
    """The categories present in df, in our fixed display order."""
    present = set(df["category"].unique())
    return [c for c in CATEGORY_ORDER if c in present]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_clean_data() -> pd.DataFrame:
    """Return the cleaned product table with just the columns the charts use.

    If Person 2's processed file exists it is read directly. Otherwise the
    raw file is read in chunks of 200,000 rows and each chunk goes through
    Person 2's ``clean_products``. That function works row by row, so
    cleaning in chunks gives exactly the same result as cleaning the whole
    table at once, while keeping memory use low.
    """
    if CLEAN_PATH.exists():
        return pd.read_parquet(CLEAN_PATH, columns=VIZ_COLS)
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run `python -m src.collect` first.")
    parts = []
    raw_file = pq.ParquetFile(RAW_PATH)
    for batch in raw_file.iter_batches(batch_size=200_000, columns=RAW_COLS):
        parts.append(clean_products(batch.to_pandas())[VIZ_COLS])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_rating_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar for every rating value (1.0 to 5.0 in steps of 0.1).

    Ratings are stored to one decimal, so each bar is one exact value and no
    binning choice is involved. Mean and median are marked.
    """
    _apply_style()
    counts = df["average_rating"].round(1).value_counts().sort_index()
    mean = df["average_rating"].mean()
    median = df["average_rating"].median()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(counts.index, counts.values, width=0.075, color=MAIN_COLOR)
    ax.axvline(mean, color=INK, linewidth=1.5, linestyle="--",
               label=f"Mean {mean:.2f}")
    ax.axvline(median, color=INK, linewidth=1.5, linestyle=":",
               label=f"Median {median:.2f}")
    ax.set_title("Distribution of average product rating")
    ax.set_xlabel("Average rating (stars)")
    ax.set_ylabel("Number of products")
    ax.set_xticks(np.arange(1, 5.01, 0.5))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    _save(fig, "01_rating_distribution")
    return fig


def plot_rating_by_category(df: pd.DataFrame) -> plt.Figure:
    """Box plot of average rating for each category.

    The box spans the middle 50% of products, the line inside is the median
    and the diamond is the mean. Whiskers reach 1.5 box-lengths; individual
    outlier points are hidden because there are hundreds of thousands.
    """
    _apply_style()
    cats = _categories_in(df)
    # Lowest mean at the top so the weakest category is read first.
    means = df.groupby("category")["average_rating"].mean()
    cats = sorted(cats, key=lambda c: means[c], reverse=True)
    data = [df.loc[df["category"] == c, "average_rating"].to_numpy()
            for c in cats]
    positions = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    box = ax.boxplot(data, positions=positions, vert=False, widths=0.5,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "linewidth": 2},
                     whiskerprops={"color": INK_SOFT},
                     capprops={"color": INK_SOFT})
    for patch, cat in zip(box["boxes"], cats):
        patch.set_facecolor(CATEGORY_COLORS[cat])
        patch.set_edgecolor("white")
    ax.scatter([means[c] for c in cats], positions, marker="D", s=70,
               color="white", edgecolor=INK, linewidth=1.5, zorder=3,
               label="Mean")
    for pos, cat in zip(positions, cats):
        ax.text(5.08, pos, f"mean {means[cat]:.2f}", va="center",
                color=INK_SOFT)
    ax.set_yticks(positions)
    ax.set_yticklabels([CATEGORY_LABELS[c] for c in cats])
    ax.set_xlim(0.9, 5.05)
    ax.set_title("Average rating by category")
    ax.set_xlabel("Average rating (stars)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")
    _save(fig, "02_rating_by_category")
    return fig


def plot_category_counts(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bars: number of products in each category."""
    _apply_style()
    counts = df["category"].value_counts()
    cats = sorted(_categories_in(df), key=lambda c: counts[c])
    values = [counts[c] for c in cats]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh([CATEGORY_LABELS[c] for c in cats], values, height=0.55,
            color=[CATEGORY_COLORS[c] for c in cats])
    for i, value in enumerate(values):
        share = 100 * value / len(df)
        ax.text(value, i, f"  {value:,}  ({share:.0f}%)", va="center")
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_title("Number of products per category")
    ax.set_xlabel("Number of products")
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(axis="y", visible=False)
    _save(fig, "03_category_counts")
    return fig


def _binned_mean(x: pd.Series, y: pd.Series, edges: np.ndarray,
                 min_count: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Mean of y inside each x-bin, skipping bins with too few products.

    Returns the bin centres (geometric, because the x-axis is logarithmic)
    and the mean of y in each bin.
    """
    bins = pd.cut(x, edges, include_lowest=True)
    grouped = y.groupby(bins, observed=True).agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= min_count]
    centres = np.array([np.sqrt(iv.left * iv.right) for iv in grouped.index])
    return centres, grouped["mean"].to_numpy()


def _scatter_panels(df: pd.DataFrame, x_col: str, x_min: float, x_max: float,
                    title: str, x_label: str, x_formatter, name: str
                    ) -> plt.Figure:
    """One panel per category: sampled points plus a binned-mean line.

    Shared by the price and review-volume charts. A random sample of points
    is drawn (1.6M points would be a solid blob), but the trend line is
    computed from every product in the panel, not just the sample.
    """
    _apply_style()
    data = df[df[x_col].between(x_min, x_max)]
    cats = _categories_in(data)
    edges = np.logspace(np.log10(x_min), np.log10(x_max), 13)  # 12 bins

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True,
                             layout="constrained")
    for ax, cat in zip(axes.flat, cats):
        panel = data[data["category"] == cat]
        sample = panel.sample(min(SAMPLE_PER_CATEGORY, len(panel)),
                              random_state=RANDOM_STATE)
        ax.scatter(sample[x_col], sample["average_rating"], s=7, alpha=0.25,
                   color=CATEGORY_COLORS[cat], linewidth=0)
        centres, bin_means = _binned_mean(panel[x_col],
                                          panel["average_rating"], edges)
        ax.plot(centres, bin_means, color=INK, linewidth=2, marker="o",
                markersize=5, label="Mean rating in each bin")
        ax.set_xscale("log")
        ax.set_xlim(x_min * 0.8, x_max * 1.25)   # small margin at each end
        ax.set_ylim(0.9, 5.1)
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
        ax.set_title(f"{CATEGORY_LABELS[cat]}  (n = {len(panel):,})",
                     fontsize=13, pad=6)
    for ax in axes.flat[len(cats):]:      # hide unused panels, if any
        ax.set_visible(False)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(title, x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.supxlabel(x_label, color=INK_SOFT, fontsize=14)
    fig.supylabel("Average rating (stars)", color=INK_SOFT, fontsize=14)
    _save(fig, name)
    return fig


def plot_price_vs_rating(df: pd.DataFrame) -> plt.Figure:
    """Price against rating, one panel per category (priced products only).

    Prices between $1 and $1,000 are shown, which covers about 98% of priced
    products; the x-axis is logarithmic because price is heavily skewed.
    """
    return _scatter_panels(
        df[df["price_num"].notna()], "price_num", 1, 1_000,
        title=r"Price vs average rating (priced products, \$1 to \$1,000)",
        x_label="Price (US dollars, log scale)",
        x_formatter=lambda v, _pos: f"${v:,.0f}",
        name="04_price_vs_rating")


def plot_review_volume_vs_rating(df: pd.DataFrame) -> plt.Figure:
    """Number of ratings against average rating, one panel per category.

    The x-axis is logarithmic: most products have a handful of ratings and a
    few have tens of thousands. The range 1 to 300,000 covers every product.
    """
    return _scatter_panels(
        df, "rating_number", 1, 300_000,
        title="Number of ratings vs average rating",
        x_label="Number of ratings per product (log scale)",
        x_formatter=_thousands,
        name="05_review_volume_vs_rating")


def plot_rating_by_volume_band(df: pd.DataFrame, min_count: int = 100
                               ) -> plt.Figure:
    """Mean rating in each review-volume band, one line per category.

    Uses the same bands as Person 3's review_volume_summary table. A point
    is drawn only when at least ``min_count`` products sit behind it, so a
    handful of products cannot draw a misleading spike.
    """
    _apply_style()
    data = df[["category", "rating_number", "average_rating"]].copy()
    data["volume_band"] = pd.cut(data["rating_number"], VOLUME_BINS,
                                 labels=VOLUME_LABELS, right=False)
    grouped = data.groupby(["volume_band", "category"], observed=True)
    means = grouped["average_rating"].mean().unstack("category")
    sizes = grouped["average_rating"].size().unstack("category")
    means = means.where(sizes >= min_count)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(VOLUME_LABELS))
    for cat in _categories_in(df):
        ax.plot(x, means[cat].reindex(VOLUME_LABELS).to_numpy(),
                color=CATEGORY_COLORS[cat], marker=CATEGORY_MARKERS[cat],
                markersize=9, linewidth=2.5, markeredgecolor="white",
                markeredgewidth=1.5, label=CATEGORY_LABELS[cat])
    ax.set_xticks(x)
    ax.set_xticklabels(VOLUME_LABELS)
    ax.set_title("Mean rating by number of ratings")
    ax.set_xlabel("Number of ratings per product")
    ax.set_ylabel("Mean average rating (stars)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    ax.text(0, -0.2, f"Points with fewer than {min_count} products are not "
            "drawn.", transform=ax.transAxes, color=INK_SOFT, fontsize=11)
    _save(fig, "06_rating_by_volume_band")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Spearman correlation between the numeric variables.

    Spearman (rank-based) matches Person 3's choice: price and number of
    ratings are heavily skewed, so Pearson would be driven by a few extreme
    products. Price pairs use priced products only.
    """
    _apply_style()
    cols = [c for c in CORR_COLS if c in df.columns]
    corr = df[cols].astype("float64").corr(method="spearman")
    labels = [CORR_COLS[c] for c in cols]

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr.to_numpy(), cmap=CORR_CMAP, vmin=-1, vmax=1)
    for row in range(len(cols)):
        for col in range(len(cols)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center",
                    va="center", color=INK, fontsize=13)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Spearman correlation between numeric variables")
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("Correlation (-1 to +1)", color=INK_SOFT)
    bar.outline.set_visible(False)
    _save(fig, "07_correlation_heatmap")
    return fig


ALL_PLOTS = [
    plot_rating_distribution,
    plot_rating_by_category,
    plot_category_counts,
    plot_price_vs_rating,
    plot_review_volume_vs_rating,
    plot_rating_by_volume_band,
    plot_correlation_heatmap,
]


def save_all_figures(df: pd.DataFrame) -> list[Path]:
    """Build every chart and return the list of PNG files written."""
    for plot in ALL_PLOTS:
        plt.close(plot(df))
    return sorted(FIG_DIR.glob("*.png"))


def main() -> None:
    """Rebuild every figure from the command line: python -m src.viz"""
    plt.switch_backend("Agg")     # draw to files only, no window needed
    df = load_clean_data()
    print(f"Loaded {len(df):,} products")
    for path in save_all_figures(df):
        print("Saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
