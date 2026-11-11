"""Make the empirical site-site backbone using positive Pearson-like residuals.

Only the 131-amphora, population-weighted Huff model is implemented.
The residuals operate on p(site | type), not raw counts. The edge score
compares observed positive-residual cosine similarity with its posterior-
predictive null. An edge is a model-relative similarity, not a trade route.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from abms.huff_amphora_choice import prepare_context, huff_probabilities, sample_amphora_counts

INPUT = CASE / "outputs/preprocessed_data/bipartite_obs.csv"
INFERENCE = CASE / "outputs/huff_sbi"
OUTPUT = CASE / "outputs/backbones/positive_cosine"


def matrix_from_cells(values, context):
    """Return arrays of shape (draws, sites, amphora types)."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[None, :]
    matrix = np.empty((len(values), len(context["site_ids"]),
                       len(context["amphora_ids"])), dtype=np.float64)
    matrix[:, context["cell_site_index"], context["cell_amphora_index"]] = values
    return matrix


def positive_cosine(values, expectation, context):
    """Cosine of positive standardized residuals for all site pairs."""
    matrix = matrix_from_cells(values, context)
    residual = (matrix - expectation[None, :, :]) / np.sqrt(np.maximum(expectation, 1e-8))
    positive = np.maximum(residual, 0)
    norm = np.linalg.norm(positive, axis=2)
    numerator = positive @ positive.transpose(0, 2, 1)
    denominator = norm[:, :, None] * norm[:, None, :]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 0)


def bh_adjust(p):
    """Exploratory correction; posterior predictive tails are not frequentist p-values."""
    order = np.argsort(p)
    m = len(p)
    corrected = p[order] * m / np.arange(1, m + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(corrected, 1)
    return result


def calculate_edges(data, posterior, context, draws=1000, seed=0, chunk=32):
    # Expected cell distribution averaged across 512 posterior parameter draws.
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(posterior), min(512, len(posterior)), replace=False)
    with torch.no_grad():
        expected_probabilities = huff_probabilities(torch.tensor(posterior[chosen]), context).numpy()
    expectation = matrix_from_cells(expected_probabilities.mean(axis=0), context)[0]

    observed = positive_cosine(data["x_obs_raw"].to_numpy(), expectation, context)[0]
    src_index, trg_index = np.triu_indices(len(context["site_ids"]), k=1)
    observed_scores = observed[src_index, trg_index]
    total = np.zeros(len(src_index), dtype=float)
    squares = np.zeros(len(src_index), dtype=float)
    exceedances = np.zeros(len(src_index), dtype=np.int64)

    # Each null draw resamples a posterior parameter and archaeological counts.
    # A separate RNG with the same seed reproduces the original null convention.
    rng = np.random.default_rng(seed)
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        chosen = rng.integers(0, len(posterior), size=size)
        with torch.no_grad():
            probabilities = huff_probabilities(torch.tensor(posterior[chosen]), context).numpy()
        simulated = sample_amphora_counts(probabilities, context, rng)
        projected = positive_cosine(simulated, expectation, context)
        scores = projected[:, src_index, trg_index]
        total += scores.sum(axis=0)
        squares += np.square(scores).sum(axis=0)
        exceedances += (scores >= observed_scores[None, :]).sum(axis=0)
        print("Posterior predictive backbone draws:", start + size, "/", draws)

    mean = total / draws
    sd = np.sqrt(np.maximum(squares / draws - mean * mean, 0))
    z = np.divide(observed_scores - mean, sd, out=np.full_like(sd, np.nan), where=sd > 0)
    p_upper = (1 + exceedances) / (draws + 1)

    raw = matrix_from_cells(data["frequency_raw"].to_numpy(), context)[0]
    support = (raw > 0).astype(np.int32)
    shared = (support @ support.T)[src_index, trg_index]
    edges = pd.DataFrame({
        "src": np.asarray(context["site_ids"])[src_index],
        "trg": np.asarray(context["site_ids"])[trg_index],
        "observed": observed_scores, "expected": mean, "null_sd": sd,
        "score": z, "p_upper": p_upper,
        "q_bh_exploratory": bh_adjust(p_upper),
        "shared_recorded_types": shared,
    })
    edges["eligible"] = (shared > 0) & (observed_scores > 0)
    edges["rank"] = edges["p_upper"].rank(method="min", ascending=True).astype(int)
    return edges.sort_values(["eligible", "p_upper", "score"],
                             ascending=[False, True, False]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = pd.read_csv(INPUT).sort_values("cell_id").reset_index(drop=True)
    context = prepare_context(data)
    config = json.loads((INFERENCE / "run_config.json").read_text())
    if config["input_sha256"] != hashlib.sha256(INPUT.read_bytes()).hexdigest():
        raise ValueError("Saved posterior and input data do not match.")
    posterior_csv = pd.read_csv(INFERENCE / "posterior_samples.csv")
    posterior = posterior_csv[context["parameter_names"]].to_numpy(dtype=np.float32)
    edges = calculate_edges(data, posterior, context,
                            draws=args.draws, seed=args.seed, chunk=args.chunk)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    edges.to_csv(OUTPUT / "site_site_edges.csv", index=False)
    edges.loc[edges["eligible"]].head(50).to_csv(OUTPUT / "site_site_top50.csv", index=False)
    edges.loc[edges["eligible"] & (edges["q_bh_exploratory"] <= 0.05)].to_csv(
        OUTPUT / "site_site_q05_exploratory.csv", index=False)
    (OUTPUT / "projection_config.json").write_text(json.dumps({
        "draws": args.draws, "chunk": args.chunk, "seed": args.seed,
        "n_sites": len(context["site_ids"]),
        "n_amphora_types": len(context["amphora_ids"]),
        "similarity": "positive standardized residual cosine",
        "null": "posterior predictive",
        "bh_caveat": "Descriptive only; FDR control is not established.",
    }, indent=2) + "\n")
    print("Saved all", len(edges), "site pairs to", OUTPUT)


if __name__ == "__main__":
    main()
