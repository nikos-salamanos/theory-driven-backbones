"""Huff choices among amphora types, with population-weighted site demand.

For site i and amphora type a:
    u(i,a) = ASC(region(a), good(a)) - beta * log(transport_time(i,a))
    p(a|i) = softmax over all amphora types at site i
    p(i|a) = population(i) * p(a|i), normalized across sites

Only the population-weighted, per-type multinomial experiment is supported.
The archaeologically observed total for each type is held fixed.
"""

import numpy as np
import pandas as pd
import torch

REFERENCE_GROUP = ("gaul", "wine")  # One ASC is fixed to zero.


def prepare_context(data):
    """Build fixed arrays in the order of the observed bipartite cells."""
    df = data.sort_values("cell_id").reset_index(drop=True).copy()
    for column in ("origin_h1", "origin_h2", "contents"):
        df[column] = df[column].fillna("").str.strip().str.lower()

    sites = df["site_number"].drop_duplicates().tolist()
    types = df["amphora_id"].drop_duplicates().tolist()
    if len(df) != len(sites) * len(types) or df.duplicated(["site_number", "amphora_id"]).any():
        raise ValueError("Expected a complete site-by-amphora matrix.")

    metadata = df.drop_duplicates("amphora_id").set_index("amphora_id").loc[types]
    groups_by_type = list(zip(metadata["origin_h1"], metadata["contents"]))
    groups = [REFERENCE_GROUP] + sorted(set(groups_by_type) - {REFERENCE_GROUP})
    group_index = np.array([groups.index(group) for group in groups_by_type])
    names = ["ASC_" + region + "__" + good.replace(" ", "_")
             for region, good in groups[1:]] + ["beta"]

    costs = (df.pivot(index="site_number", columns="amphora_id", values="travel_cost")
             .loc[sites, types].to_numpy(dtype=np.float32))
    population = (df.drop_duplicates("site_number").set_index("site_number")
                  .loc[sites, "population"].to_numpy(dtype=np.float32))
    totals = (df.groupby("amphora_id")["frequency_raw"].sum()
              .reindex(types).to_numpy(dtype=np.int64))
    if np.any(costs <= 0) or np.any(population <= 0) or np.any(totals <= 0):
        raise ValueError("Costs, populations and type counts must be positive.")

    return {
        "log_travel_cost": np.log(costs),
        "population_weights": population / population.sum(),
        "amphora_origin_index": group_index,
        "cell_site_index": pd.Index(sites).get_indexer(df["site_number"]),
        "cell_amphora_index": pd.Index(types).get_indexer(df["amphora_id"]),
        "site_ids": sites,
        "amphora_ids": types,
        "origins": groups,
        "parameter_names": names,
        "amphora_total_counts": totals,
    }


def huff_probabilities(theta, context):
    """Return p(site|type) in the original cell order (batch, cells)."""
    theta = torch.as_tensor(theta, dtype=torch.float32)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    costs = torch.as_tensor(context["log_travel_cost"], device=theta.device)
    population = torch.as_tensor(context["population_weights"], device=theta.device)
    group_index = torch.as_tensor(context["amphora_origin_index"], device=theta.device)

    # The reference group is ASC = 0, so the parameter vector has 11 ASCs + beta.
    asc = torch.cat((torch.zeros_like(theta[:, :1]), theta[:, :-1]), dim=1)
    utilities = (asc[:, group_index][:, None, :]
                 - theta[:, -1, None, None] * costs[None, :, :])
    choices = torch.softmax(utilities, dim=2)  # p(type | site), 131 alternatives
    demand = population[None, :, None] * choices
    site_shares = demand / demand.sum(dim=1, keepdim=True)  # p(site | type)
    return site_shares[:, context["cell_site_index"], context["cell_amphora_index"]]


def sample_amphora_counts(probabilities, context, rng):
    """Draw observed per-type counts and normalize each type over sites."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[None, :]
    n_sites = len(context["site_ids"])
    n_types = len(context["amphora_ids"])
    site_idx = context["cell_site_index"]
    type_idx = context["cell_amphora_index"]

    matrix = np.empty((len(probs), n_sites, n_types), dtype=np.float64)
    matrix[:, site_idx, type_idx] = probs
    simulated = np.zeros(matrix.shape, dtype=np.float32)
    for a, total in enumerate(context["amphora_total_counts"]):
        distribution = matrix[:, :, a]
        distribution = distribution / distribution.sum(axis=1, keepdims=True)
        counts = rng.multinomial(int(total), distribution)
        simulated[:, :, a] = counts / float(total)
    return simulated[:, site_idx, type_idx]


def simulate(theta, context, rng):
    """SBI simulator: normalized recorded fractions from multinomial draws."""
    with torch.no_grad():
        probabilities = huff_probabilities(theta, context).cpu().numpy()
    return torch.tensor(sample_amphora_counts(probabilities, context, rng),
                        dtype=torch.float32)
