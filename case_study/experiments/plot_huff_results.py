"""Plot Huff-model PPC coverage, posterior distributions, and SBC diagnostics.

Uses saved results in case_study/outputs/huff_sbi; does not run inference.

Run from the repository root:
    python case_study/experiments/plot_huff_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


# =============================================================================
# SETTINGS
# =============================================================================

# Output files: PNG and PDF versions of each figure
SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 250

# Heatmap form selection (based on observations, not PPC results)
FORMS_PER_GOOD = {"olive oil": 3, "wine": 4, "fish sauce": 5}
PREFERRED_REGION = "iberianpeninsula"
PREFERRED_PER_GOOD = {"olive oil": 2, "wine": 2, "fish sauce": 3}
EXCLUDED_REGIONS = ("em",)
MIN_FORM_TOTAL = 30
MIN_FORM_SITES = 8
CUSTOM_AMPHORA_IDS = []       # Empty = automatic selection

# Heatmap site selection (number of selected forms present, then total frequency)
N_HEATMAP_SITES = 15
CUSTOM_SITE_NAMES = []        # Empty = automatic selection

# Heatmap colors: overpredicted, within 90% interval, underpredicted
HEATMAP_COLORS = ["#6B9EC1", "#A4D4BC", "#D98269"]
SHOW_HEATMAP_LEGEND = True
HEATMAP_LEGEND_COLUMNS = 3
SHOW_CELL_GRID = True
CELL_GRID_COLOR = "white"
CELL_GRID_WIDTH = 0.8
SHOW_CONTENT_IN_FORM_LABEL = True
SHOW_SITE_NAMES_ON_HEATMAP = True
HEATMAP_X_ROTATION = 55

# Heatmap layout and text
HEATMAP_SIZE = (11.5, 6.4)
HEATMAP_MARGINS = dict(left=0.27, right=0.97, top=0.92, bottom=0.31)
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 11
TICK_FONT_SIZE = 9
LABEL_FONT_SIZE = 11
TITLE_FONT_SIZE = 12
LEGEND_FONT_SIZE = 9
SHOW_TITLES = False
SHOW_AXIS_LABELS = True
HEATMAP_TITLE = "PPC coverage across amphora forms and sites"
HEATMAP_LEGEND_ANCHOR = (0.5, 1.0)  # Horizontal legend above the heatmap

# Amphora-form spelling for figure labels
FORM_NAME_OVERRIDES = {
    "beltran iia/ pelichet 46": "Beltrán IIa / Pelichet 46",
    "pompeii vii/ vindonissa 586": "Pompeii VII / Vindonissa 586",
    "gauloise 4/ pelichet 47": "Gauloise 4 / Pelichet 47",
}

# Posterior plot (five ASC parameters and beta)
POSTERIOR_SIZE = (10.5, 5.4)
POSTERIOR_BINS = 35
POSTERIOR_TITLE_FONT_SIZE = 9
POSTERIOR_TICK_FONT_SIZE = 8
POSTERIOR_LABEL_FONT_SIZE = 9

# SBC plot (seven ASC parameters and beta)
SBC_SIZE = (8, 5)
SBC_LABEL_FONT_SIZE = 15
SBC_TICK_FONT_SIZE = 13
SBC_LEGEND_FONT_SIZE = 9
SHOW_SBC_LEGEND = True

# =============================================================================

CASE = Path(__file__).resolve().parents[1]
DATA = CASE / "outputs/preprocessed_data/bipartite_obs.csv"
RESULTS = CASE / "outputs/huff_sbi"
PLOTS = RESULTS / "plots"

plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE})


def save(fig, name):
    if SAVE_PNG:
        fig.savefig(PLOTS / f"{name}.png", dpi=SAVE_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(PLOTS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def form_name(raw):
    name = str(raw).strip()
    return FORM_NAME_OVERRIDES.get(name.lower(), name.title())


def select_forms(data):
    forms = (data.groupby("amphora_id", as_index=False)
             .agg(form=("producer_display", "first"), good=("contents", "first"),
                  region=("origin_h1", "first"), total=("frequency_raw", "sum"),
                  n_sites=("frequency_raw", lambda v: int((v > 0).sum()))))

    if CUSTOM_AMPHORA_IDS:
        missing = set(CUSTOM_AMPHORA_IDS) - set(forms["amphora_id"])
        if missing:
            raise ValueError(f"Unknown amphora IDs: {sorted(missing)}")
        return forms.set_index("amphora_id").loc[CUSTOM_AMPHORA_IDS].reset_index()

    candidates = forms[
        (forms["total"] >= MIN_FORM_TOTAL)
        & (forms["n_sites"] >= MIN_FORM_SITES)
        & (~forms["region"].isin(EXCLUDED_REGIONS))
    ].copy()

    chosen = []
    for good, n in FORMS_PER_GOOD.items():
        group = candidates[candidates["good"] == good].sort_values(
            ["total", "n_sites", "amphora_id"], ascending=[False, False, True]
        )
        preferred = group[group["region"] == PREFERRED_REGION].head(
            min(n, PREFERRED_PER_GOOD.get(good, 0))
        )
        other = group[~group["amphora_id"].isin(preferred["amphora_id"])].head(
            n - len(preferred)
        )
        chosen.append(pd.concat([preferred, other]))

    selected = pd.concat(chosen, ignore_index=True)
    if len(selected) < sum(FORMS_PER_GOOD.values()):
        print("Note: fewer forms meet the selection thresholds.")
    return selected


def select_sites(data, forms):
    sites = (data[data["amphora_id"].isin(forms["amphora_id"])]
             .groupby(["site_number", "site_display"], as_index=False)
             .agg(positive_forms=("frequency_raw", lambda v: int((v > 0).sum())),
                  total=("frequency_raw", "sum")))

    if CUSTOM_SITE_NAMES:
        missing = set(CUSTOM_SITE_NAMES) - set(sites["site_display"])
        if missing:
            raise ValueError(f"Unknown site names: {sorted(missing)}")
        return sites.set_index("site_display").loc[CUSTOM_SITE_NAMES].reset_index()

    return sites.sort_values(
        ["positive_forms", "total", "site_display"], ascending=[False, False, True]
    ).head(N_HEATMAP_SITES)


def select_cells(summary, forms, sites):
    """Return the complete selected form-by-site rectangle."""
    keys = pd.MultiIndex.from_product(
        [forms["amphora_id"], sites["site_number"]],
        names=["amphora_id", "site_number"],
    )
    selected = summary.set_index(["amphora_id", "site_number"]).reindex(keys)
    if selected["cell_id"].isna().any():
        raise ValueError("Missing amphora–site cells in the PPC summary.")
    return selected.reset_index()


def plot_coverage_heatmap(summary, data):
    forms = select_forms(data)
    sites = select_sites(data, forms)
    selected = select_cells(summary, forms, sites)

    observed = selected["x_obs"].to_numpy()
    lower = selected["ppc_lower"].to_numpy()
    upper = selected["ppc_upper"].to_numpy()
    states = np.where(observed < lower, -1,
                      np.where(observed > upper, 1, 0))
    states = states.reshape(len(forms), len(sites))

    fig, ax = plt.subplots(figsize=HEATMAP_SIZE)
    cmap = ListedColormap(HEATMAP_COLORS)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(states, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    if SHOW_HEATMAP_LEGEND:
        labels = ["Overpredicted", "Within 90% PPC interval", "Underpredicted"]
        handles = [Patch(facecolor=color, label=label)
                   for color, label in zip(HEATMAP_COLORS, labels)]
        ax.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=HEATMAP_LEGEND_ANCHOR, frameon=False,
                  ncol=HEATMAP_LEGEND_COLUMNS, fontsize=LEGEND_FONT_SIZE)

    form_labels = []
    for row in forms.itertuples(index=False):
        label = form_name(row.form)
        if SHOW_CONTENT_IN_FORM_LABEL:
            label += f" ({row.good})"
        form_labels.append(label)
    ax.set_yticks(range(len(forms)), form_labels, fontsize=TICK_FONT_SIZE)

    if SHOW_SITE_NAMES_ON_HEATMAP:
        ax.set_xticks(range(len(sites)), sites["site_display"],
                      rotation=HEATMAP_X_ROTATION, ha="right", fontsize=TICK_FONT_SIZE)
    else:
        ax.set_xticks([])
    ax.tick_params(length=0)

    if SHOW_CELL_GRID:
        ax.set_xticks(np.arange(-0.5, len(sites), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(forms), 1), minor=True)
        ax.grid(which="minor", color=CELL_GRID_COLOR, linewidth=CELL_GRID_WIDTH)
        ax.tick_params(which="minor", bottom=False, left=False)

    if SHOW_AXIS_LABELS:
        ax.set_xlabel("Consumption site", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Amphora form", fontsize=LABEL_FONT_SIZE)
    if SHOW_TITLES:
        ax.set_title(HEATMAP_TITLE, fontsize=TITLE_FONT_SIZE)

    fig.subplots_adjust(**HEATMAP_MARGINS)
    save(fig, "huff_ppc_coverage_heatmap")

    print("Selected amphora forms (form, region, good, frequency, positive sites):")
    for form in forms.itertuples(index=False):
        print(f"  {form_name(form.form)} | {form.region} | {form.good} | "
              f"{form.total} | {form.n_sites}")
    print("Selected sites:", ", ".join(sites["site_display"]))
    print(f"PPC heatmap: {len(forms)} forms x {len(sites)} sites = {states.size} cells")
    print(f"Selected-cell 90% PPC coverage: {(states == 0).mean():.1%}")


def posteriors(samples):
    names = [name for name in samples.columns if name != "beta"][:5] + ["beta"]
    fig, axes = plt.subplots(2, 3, figsize=POSTERIOR_SIZE)
    for j, (ax, name) in enumerate(zip(axes.flat, names)):
        ax.hist(samples[name].dropna(), bins=POSTERIOR_BINS, density=True)
        ax.set_title(name, fontsize=POSTERIOR_TITLE_FONT_SIZE)
        ax.tick_params(labelsize=POSTERIOR_TICK_FONT_SIZE)
        if j % 3 == 0:
            ax.set_ylabel("Posterior density", fontsize=POSTERIOR_LABEL_FONT_SIZE)
    fig.tight_layout()
    save(fig, "huff_posterior")


def calibration(ranks):
    names = ranks["parameter"].drop_duplicates().tolist()
    names = [name for name in names if name != "beta"][:7] + ["beta"]
    fig, ax = plt.subplots(figsize=SBC_SIZE)
    for name, group in ranks[ranks["parameter"].isin(names)].groupby("parameter"):
        x = np.sort(group["rank_normalized"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Uniform reference")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Normalized SBC rank", fontsize=SBC_LABEL_FONT_SIZE)
    ax.set_ylabel("Empirical cumulative probability", fontsize=SBC_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=SBC_TICK_FONT_SIZE)
    if SHOW_SBC_LEGEND:
        ax.legend(fontsize=SBC_LEGEND_FONT_SIZE)
    fig.tight_layout()
    save(fig, "huff_sbc")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA).sort_values("cell_id").reset_index(drop=True)
    summary = pd.read_csv(RESULTS / "ppc_summary.csv").sort_values("cell_id").reset_index(drop=True)

    if not np.array_equal(summary["cell_id"].to_numpy(), data["cell_id"].to_numpy()):
        raise ValueError("PPC cell IDs do not match the observations.")
    if not np.allclose(summary["x_obs"], data["x_obs_raw"]):
        raise ValueError("PPC observations do not match the input data.")

    plot_coverage_heatmap(summary, data)
    posteriors(pd.read_csv(RESULTS / "posterior_samples.csv"))
    calibration(pd.read_csv(RESULTS / "sbc_ranks.csv"))
    print("Figures saved to:", PLOTS)


if __name__ == "__main__":
    main()
