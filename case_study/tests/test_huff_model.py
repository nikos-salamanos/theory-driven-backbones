"""Basic reproducibility checks for the Huff amphora-choice model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from abms.huff_amphora_choice import prepare_context, huff_probabilities, sample_amphora_counts


def load():
    table = pd.read_csv(CASE / "outputs/preprocessed_data/bipartite_obs.csv")
    return table, prepare_context(table)


def test_observations():
    table, context = load()
    assert len(table) == 79 * 131
    assert (table["frequency_raw"] > 0).sum() == 891
    assert table["frequency_raw"].sum() == 27531
    assert len(context["origins"]) == 12
    assert len(context["parameter_names"]) == 12
    assert np.allclose(table.groupby("amphora_id")["x_obs_raw"].sum(), 1)


def test_huff_probabilities():
    _, context = load()
    theta = torch.zeros(2, 12)
    theta[:, -1] = 1.0
    shares = huff_probabilities(theta, context).numpy()
    assert shares.shape == (2, 10349)
    matrix = np.empty((2, 79, 131))
    matrix[:, context["cell_site_index"], context["cell_amphora_index"]] = shares
    assert np.allclose(matrix.sum(axis=1), 1)
    assert np.all(shares > 0)


def test_multinomial_type_totals():
    _, context = load()
    theta = torch.tensor([0.] * 11 + [1.])
    shares = huff_probabilities(theta, context).numpy()
    result = sample_amphora_counts(shares, context, np.random.default_rng(0))
    matrix = np.empty((1, 79, 131))
    matrix[:, context["cell_site_index"], context["cell_amphora_index"]] = result
    assert np.allclose(matrix.sum(axis=1), 1)
    counts = matrix[0] * context["amphora_total_counts"][None, :]
    assert np.allclose(counts, np.round(counts))


def test_saved_reference_ppc():
    # The reference results from the manuscript must be present and unchanged.
    results = next((CASE / "results_reference/huff_sbi").glob("*/ppc_metrics_h1.json"))
    import json
    metrics = json.loads(results.read_text())
    assert np.isclose(metrics["positive_cell_coverage_90pct"], 0.4399551066217733)
    assert np.isclose(metrics["cell_coverage_90pct"], 0.913325)
