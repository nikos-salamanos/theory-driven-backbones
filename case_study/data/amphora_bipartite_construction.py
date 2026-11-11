#!/usr/bin/env python3
"""Validate and prepare the complete amphora-type × consumption-site table.

The empirical data contain an observed frequency for each excavated amphora
(type, site) pair. The complete rectangular table additionally contains zeros
for unreported pairs. A zero indicates no recorded observation in this dataset;
it is not proof that the amphora type was historically absent at that site.

Observed counts are normalized separately for each amphora type:

    x_obs_raw[type, site] = frequency[type, site] / sum_sites frequency[type, site]

The result is p(site | amphora type), not p(amphora type | site). All 24
archaeological origins and 79 consumption sites are retained. Transport times
refer to representative ORBIS locations, not verified production workshops.

Input:  case_study/data/processed/bipartite_full_131x79.csv
        case_study/data/raw/ORBIS_transport_matrix_24x79.csv
        case_study/data/raw/RAAD_population_transport_enriched.csv
Output: case_study/outputs/preprocessed_data/bipartite_obs.csv
        case_study/outputs/preprocessed_data/bipartite_obs_full.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

CASE_STUDY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CASE_STUDY_ROOT / "data"
INPUT_FILE = DATA_DIR / "processed" / "bipartite_full_131x79.csv"
TRANSPORT_FILE = DATA_DIR / "raw" / "ORBIS_transport_matrix_24x79.csv"
RAAD_FILE = DATA_DIR / "raw" / "RAAD_population_transport_enriched.csv"
OUTPUT_DIR = CASE_STUDY_ROOT / "outputs" / "preprocessed_data"


def require_columns(table, columns, name):
    """Reject missing input fields before modifying the data."""
    missing = set(columns).difference(table.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def origin_keys(table):
    """Canonicalize categorical keys for joins, not archaeological labels."""
    result = table.copy()
    for field in ("origin_h1", "origin_h2"):
        result[field] = result[field].fillna("").astype(str).str.strip().str.lower()
    return result


def build_observations():
    """Return an independently checked complete rectangular observation table."""
    cells = origin_keys(pd.read_csv(INPUT_FILE, low_memory=False))
    transport = origin_keys(pd.read_csv(TRANSPORT_FILE, low_memory=False))
    raad = origin_keys(pd.read_csv(RAAD_FILE, encoding="cp1252", low_memory=False))

    require_columns(cells, [
        "cell_id", "amphora_id", "RAAD_type_number", "site_number", "origin_h1",
        "origin_h2", "origin_label", "orbis_origin", "orbis_destination",
        "travel_cost", "frequency_raw", "population", "site_display",
    ], "Complete observation matrix")
    require_columns(transport, [
        "origin_h1", "origin_h2", "site_number", "transport_time",
        "orbis_origin", "orbis_destination",
    ], "Transport matrix")
    require_columns(raad, ["RAAD_type_number", "site_number", "frequency"], "RAAD")

    # A complete bipartite matrix must contain exactly one row per type/site
    # combination. This also fixes the ordering required by vectorized models.
    keys = ["amphora_id", "site_number"]
    if cells.duplicated(keys).any() or cells["cell_id"].duplicated().any():
        raise ValueError("Duplicate amphora-site cells or cell identifiers.")
    n_types = cells["amphora_id"].nunique()
    n_sites = cells["site_number"].nunique()
    if len(cells) != n_types * n_sites:
        raise ValueError("Amphora-site table is not rectangular.")
    if cells.groupby("amphora_id")["site_number"].nunique().ne(n_sites).any():
        raise ValueError("At least one amphora type is missing a consumption site.")

    # Archaeological types retain a single recorded provenance category.
    for field in ("RAAD_type_number", "origin_h1", "origin_h2", "origin_label"):
        if cells.groupby("amphora_id")[field].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting {field} within an amphora type.")
    if cells.groupby("site_number")["population"].nunique(dropna=False).gt(1).any():
        raise ValueError("Conflicting populations for a consumption site.")

    # Confirm that the recorded positive counts are unchanged from the source.
    source_counts = raad.groupby(["RAAD_type_number", "site_number"], as_index=False)[
        "frequency"
    ].sum()
    check = cells.merge(source_counts, on=["RAAD_type_number", "site_number"],
                        how="left", validate="one_to_one")
    if not np.allclose(check["frequency_raw"].to_numpy(dtype=float),
                       check["frequency"].fillna(0).to_numpy(dtype=float)):
        raise ValueError("Amphora frequencies do not reproduce the source observations.")

    # Origin/site distances are shared by all amphora types belonging to that
    # origin. Check both route values and the canonical ORBIS endpoint names.
    route_keys = ["origin_h1", "origin_h2", "site_number"]
    if transport.duplicated(route_keys).any():
        raise ValueError("Duplicate origin-site routes in the transport matrix.")
    # Several archaeological origins and consumption sites share ORBIS nodes.
    # Their reused route durations must agree exactly to the recorded precision.
    reused = transport.groupby(["orbis_origin", "orbis_destination"])["transport_time"]
    if reused.nunique().gt(1).any():
        raise ValueError("Conflicting travel times for a shared ORBIS route.")
    expected_routes = transport[route_keys + ["transport_time", "orbis_origin",
                                               "orbis_destination"]]
    check = cells.merge(expected_routes, on=route_keys, how="left",
                        validate="many_to_one", suffixes=("", "_reference"))
    if check["transport_time_reference"].isna().any():
        raise ValueError("Some origin-site transport times are unavailable.")
    if not np.allclose(check["travel_cost"], check["transport_time_reference"]):
        raise ValueError("Travel costs differ from the complete transport matrix.")
    for field in ("orbis_origin", "orbis_destination"):
        if not check[field].eq(check[field + "_reference"]).all():
            raise ValueError(f"Inconsistent ORBIS name: {field}.")

    # Normalization removes differences in total recorded counts between types;
    # it does not model the excavation or recovery process explicitly.
    for field in ("frequency_raw", "travel_cost", "population"):
        cells[field] = pd.to_numeric(cells[field], errors="raise")
    if (cells["frequency_raw"] < 0).any() or not np.isfinite(cells["frequency_raw"]).all():
        raise ValueError("Observed frequencies must be finite and nonnegative.")
    if (cells["travel_cost"] <= 0).any() or not np.isfinite(cells["travel_cost"]).all():
        raise ValueError("Travel times must be finite and positive.")
    if (cells["population"] <= 0).any() or not np.isfinite(cells["population"]).all():
        raise ValueError("Populations must be finite and positive.")

    totals = cells.groupby("amphora_id")["frequency_raw"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Each amphora type must have a positive recorded total.")
    cells["x_obs_raw"] = cells["frequency_raw"] / totals
    cells = cells.sort_values("cell_id").reset_index(drop=True)
    if not np.array_equal(cells["cell_id"], np.arange(len(cells))):
        raise ValueError("cell_id must provide contiguous zero-based ordering.")
    if not np.allclose(cells.groupby("amphora_id")["x_obs_raw"].sum(), 1.0):
        raise ValueError("Within-type probabilities are not normalized.")

    columns = [
        "cell_id", "site_number", "site_display", "site_name_modern",
        "site_name_ancient", "orbis_destination", "lat", "long",
        "amphora_id", "producer_id", "producer_display", "RAAD_type_number",
        "contents", "origin_h1", "origin_h2", "production_origin",
        "origin_label", "orbis_origin", "travel_cost", "population",
        "frequency_raw", "x_obs_raw",
    ]
    require_columns(cells, columns, "Complete observation matrix")
    return cells, columns


def main():
    cells, columns = build_observations()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells[columns].to_csv(OUTPUT_DIR / "bipartite_obs.csv", index=False)
    cells.to_csv(OUTPUT_DIR / "bipartite_obs_full.csv", index=False)
    print(f"Amphora types: {cells['amphora_id'].nunique()}")
    print(f"Consumption sites: {cells['site_number'].nunique()}")
    print(f"Production origins: {cells[['origin_h1', 'origin_h2']].drop_duplicates().shape[0]}")
    print(f"Amphora-site cells: {len(cells)}")
    print(f"Positive observation cells: {int((cells['frequency_raw'] > 0).sum())}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
