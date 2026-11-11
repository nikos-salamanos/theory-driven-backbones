"""Compare the empirical Huff backbone with Michele's NC, DF, and naive methods.

The classical methods use the raw-frequency site-site projection. Their
overlap with TDBB is descriptive; historical ground-truth edges are unknown.
Michele's backboning.py is imported without editing his implementation.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "case_study"
sys.path.insert(0, str(ROOT / "backboning_and_synthetic"))
import backboning as classical

INPUT = CASE / "outputs/preprocessed_data/bipartite_obs.csv"
OUTPUT = CASE / "outputs/backbones/positive_cosine"


def cooccurrence(data):
    counts = data.pivot(index="site_number", columns="amphora_id",
                        values="frequency_raw").sort_index()
    sites = counts.index.to_numpy()
    matrix = counts.to_numpy(dtype=np.float64)
    projected = matrix @ matrix.T
    np.fill_diagonal(projected, 0)
    i, j = np.nonzero(projected > 0)
    return pd.DataFrame({"src": [f"site_{int(sites[k]):03d}" for k in i],
                         "trg": [f"site_{int(sites[k]):03d}" for k in j],
                         "nij": projected[i, j]})


def classical_scores(directed):
    nc = classical.noise_corrected(directed, undirected=True)
    df = classical.disparity_filter(directed, undirected=True)
    nc["score"] = nc["score"] / nc["sdev_cij"].replace(0, np.nan)
    naive = directed.loc[directed["src"] < directed["trg"]].copy()
    naive["score"] = naive["nij"] / naive["nij"].max()
    return {"noise_corrected": nc, "disparity_filter": df, "naive": naive}


def pair_set(frame):
    return {tuple(sorted((str(row.src), str(row.trg)))) for row in frame.itertuples(index=False)}


def undirected(frame):
    frame = frame.copy()
    pairs = np.sort(frame[["src", "trg"]].astype(str).to_numpy(), axis=1)
    frame["src"], frame["trg"] = pairs[:, 0], pairs[:, 1]
    return frame.sort_values("score", ascending=False).drop_duplicates(["src", "trg"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    data = pd.read_csv(INPUT)
    theory = pd.read_csv(OUTPUT / "site_site_edges.csv")
    theory = theory.loc[theory["eligible"].astype(str).str.lower() == "true"].copy()
    theory["src"] = theory["src"].map(lambda x: f"site_{int(x):03d}")
    theory["trg"] = theory["trg"].map(lambda x: f"site_{int(x):03d}")
    theory = theory.sort_values(["p_upper", "score"], ascending=[True, False])
    target = pair_set(theory.head(args.top_k))

    report = []
    for name, frame in classical_scores(cooccurrence(data)).items():
        frame = undirected(frame).replace([np.inf, -np.inf], np.nan)
        frame.to_csv(OUTPUT / f"baseline_{name}.csv", index=False)
        top = frame.dropna(subset=["score"]).sort_values("score", ascending=False)
        top = pair_set(top.head(args.top_k))
        report.append({
            "method": name, "top_k": args.top_k,
            "available_scored_edges": int(len(frame.dropna(subset=["score"]))),
            "intersection_with_theory": len(top & target),
            "jaccard_at_k": len(top & target) / len(top | target) if top | target else np.nan,
        })
    summary = pd.DataFrame(report)
    summary.to_csv(OUTPUT / "baseline_overlap.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
