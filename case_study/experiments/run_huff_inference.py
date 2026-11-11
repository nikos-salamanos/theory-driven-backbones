"""Fit the population-weighted amphora-choice Huff model with SBI.

The paper experiment uses 5,000 training simulations, 2,000 posterior samples,
1,000 PPC draws and 100 SBC trials of 200 posterior samples each.
Run from RUN_CASE_STUDY.py, or run this script with --mode check.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from abms.huff_amphora_choice import prepare_context, simulate

INPUT = CASE / "outputs/preprocessed_data/bipartite_obs.csv"
RESULTS = CASE / "outputs/huff_sbi"


def make_prior(n_groups):
    """Eleven independent ASC uniforms [-4, 4] and beta uniform [0.01, 6]."""
    from sbi.utils import BoxUniform
    low = [-4.0] * (n_groups - 1) + [0.01]
    high = [4.0] * (n_groups - 1) + [6.0]
    return BoxUniform(low=torch.tensor(low), high=torch.tensor(high))


def train(prior, simulator, observed, context, folder, args):
    from sbi.inference import NPE

    theta = prior.sample((args.simulations,))
    x_train = simulator(theta)
    torch.save(theta, folder / "theta_train.pt")
    torch.save(x_train, folder / "x_train.pt")

    inference = NPE(prior=prior).append_simulations(theta, x_train)
    estimator = inference.train(
        training_batch_size=200, validation_fraction=0.1,
        stop_after_epochs=20, max_num_epochs=300,
    )
    posterior = inference.build_posterior(estimator)
    with (folder / "posterior.pkl").open("wb") as file:
        pickle.dump(posterior, file)

    samples = posterior.sample((args.posterior_samples,), x=observed)
    samples_df = pd.DataFrame(samples.detach().cpu().numpy(),
                              columns=context["parameter_names"])
    samples_df.to_csv(folder / "posterior_samples.csv", index=False)
    summary = samples_df.quantile([0.05, 0.5, 0.95]).T
    summary.columns = ["q05", "median", "q95"]
    summary.insert(0, "mean", samples_df.mean())
    summary.to_csv(folder / "posterior_summary.csv", index_label="parameter")
    return posterior, samples


def posterior_predictive(simulator, observed, data, samples, folder, n_draws):
    """Check cellwise p(site | type), preserving the original 90% PPC rule."""
    predictions = simulator(samples[:n_draws]).cpu().numpy()
    actual = observed.cpu().numpy()
    summary = data[[
        "cell_id", "site_number", "site_display", "amphora_id",
        "producer_display", "production_origin", "frequency_raw",
    ]].copy()
    summary["x_obs"] = actual
    summary["ppc_mean"] = predictions.mean(axis=0)
    summary["ppc_median"] = np.median(predictions, axis=0)
    summary["ppc_lower"] = np.quantile(predictions, 0.05, axis=0)
    summary["ppc_upper"] = np.quantile(predictions, 0.95, axis=0)
    summary["inside_interval"] = (actual >= summary["ppc_lower"]) & (actual <= summary["ppc_upper"])
    summary.to_csv(folder / "ppc_summary.csv", index=False)
    np.save(folder / "ppc_draws.npy", predictions)

    positive = summary["frequency_raw"] > 0
    metrics = {
        "mean_absolute_error": float(np.abs(summary["ppc_mean"] - actual).mean()),
        "cell_coverage_90pct": float(summary["inside_interval"].mean()),
        "posterior_samples_used": int(min(len(samples), n_draws)),
        "observation_model": "multinomial",
        "sampling_model": "per-type",
        "demand_model": "population",
        "positive_cell_coverage_90pct": float(summary.loc[positive, "inside_interval"].mean()),
        "positive_cell_mean_absolute_error": float(np.abs(
            summary.loc[positive, "ppc_mean"] - summary.loc[positive, "x_obs"]).mean()),
    }
    (folder / "ppc_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("PPC metrics:", metrics)


def calibrate(prior, simulator, posterior, context, folder, n_runs, n_samples):
    from sbi.diagnostics import run_sbc

    true_parameters = prior.sample((n_runs,))
    simulated_data = simulator(true_parameters)
    ranks, _ = run_sbc(
        true_parameters, simulated_data, posterior,
        num_posterior_samples=n_samples,
        reduce_fns="marginals", use_batched_sampling=False,
        num_workers=1, show_progress_bar=True,
    )
    ranks = ranks.cpu().numpy()
    rows = [
        {"run_id": run, "parameter": name,
         "rank": int(ranks[run, index]),
         "rank_normalized": int(ranks[run, index]) / n_samples}
        for run in range(n_runs)
        for index, name in enumerate(context["parameter_names"])
    ]
    pd.DataFrame(rows).to_csv(folder / "sbc_ranks.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "sbi", "ppc", "sbc", "all"], default="check")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--posterior-samples", type=int, default=2000)
    parser.add_argument("--ppc-samples", type=int, default=1000)
    parser.add_argument("--sbc-runs", type=int, default=100)
    parser.add_argument("--sbc-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = pd.read_csv(INPUT).sort_values("cell_id").reset_index(drop=True)
    context = prepare_context(data)
    observed = torch.tensor(data["x_obs_raw"].to_numpy(dtype=np.float32))
    if not np.allclose(data.groupby("amphora_id")["x_obs_raw"].sum(), 1):
        raise ValueError("Observed fractions must sum to one within each type.")

    # Use one RNG for all calls to the simulator in this invocation.
    # The preview call is retained because it was part of the original RNG order.
    rng = np.random.default_rng(args.seed)
    simulator = lambda theta: simulate(theta, context, rng)
    preview_parameters = torch.zeros(len(context["parameter_names"]))
    preview_parameters[-1] = 1
    preview = simulator(preview_parameters)
    print("Sites:", len(context["site_ids"]), "Amphora types:", len(context["amphora_ids"]))
    print("Major-region–good groups:", len(context["origins"]))
    print("Observation / simulation cells:", len(observed), preview.shape[-1])
    if args.mode == "check":
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args), "parameter_names": context["parameter_names"],
        "groups": context["origins"], "observed_column": "x_obs_raw",
        "n_amphora_types": len(context["amphora_ids"]),
        "n_sites": len(context["site_ids"]), "n_cells": len(data),
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "demand_model": "population", "observation_model": "multinomial",
        "sampling_model": "per-type",
    }
    config_file = RESULTS / "run_config.json"
    if args.mode in ("ppc", "sbc"):
        saved = json.loads(config_file.read_text())
        for field in ("parameter_names", "n_cells", "input_sha256", "demand_model",
                      "observation_model", "sampling_model"):
            if saved[field] != config[field]:
                raise ValueError(f"The saved inference differs in {field}.")
    else:
        config_file.write_text(json.dumps(config, indent=2) + "\n")

    prior = make_prior(len(context["origins"]))
    posterior = None
    samples = None
    if args.mode in ("sbi", "all"):
        posterior, samples = train(prior, simulator, observed, context, RESULTS, args)
    if args.mode in ("ppc", "all"):
        if samples is None:
            samples_file = pd.read_csv(RESULTS / "posterior_samples.csv")
            samples = torch.tensor(samples_file[context["parameter_names"]].to_numpy(dtype=np.float32))
        posterior_predictive(simulator, observed, data, samples, RESULTS, args.ppc_samples)
    if args.mode in ("sbc", "all"):
        if posterior is None:
            with (RESULTS / "posterior.pkl").open("rb") as file:
                posterior = pickle.load(file)
        calibrate(prior, simulator, posterior, context, RESULTS,
                  args.sbc_runs, args.sbc_samples)


if __name__ == "__main__":
    main()
