"""Run the amphora-choice Huff experiments from an IDE.

Edit the settings below. All outputs go to case_study/outputs/.
"""

from pathlib import Path
import json
import hashlib
import subprocess
import sys

# ---------------------------- SETTINGS -----------------------------------
CHECK_ONLY = False        # True = verify model dimensions without fitting SBI.
RUN_PREPROCESS = False      # Verify RAAD/ORBIS inputs and build observed matrix.
RUN_SBI = False             # Train the posterior neural estimator.
RUN_PPC = False            # Posterior predictive checks.
RUN_SBC = False             # Simulation-based calibration.
RUN_PLOTS = False       # Figure 8, Figure 9 and heatmaps.
RUN_BACKBONES = False       # Positive-residual site-site cosine similarities.
RUN_BASELINES = False      # NC, DF and naive projection comparison.
REUSE_RESULTS = False       # Skip completed stages if their files already exist.

SIMULATIONS = 5000
POSTERIOR_SAMPLES = 2000
PPC_SAMPLES = 1000
SBC_RUNS = 100
SBC_SAMPLES = 200
BACKBONE_DRAWS = 1000
TOP_K = 50
SEED = 0
# -------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case_study"
EXPERIMENTS = CASE / "experiments"
RESULTS = CASE / "outputs/huff_sbi"
BACKBONES = CASE / "outputs/backbones/positive_cosine"
INPUT = CASE / "outputs/preprocessed_data/bipartite_obs.csv"


def run(script, *args):
    cmd = [sys.executable, "-u", str(script), *map(str, args)]
    print("\nRUN:", script.name, " ".join(map(str, args)), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def existing(file):
    return REUSE_RESULTS and file.is_file() and file.stat().st_size > 0


def reference_matches():
    config_file = RESULTS / "run_config.json"
    if not config_file.is_file():
        return False
    config = json.loads(config_file.read_text())
    return (config.get("input_sha256") == hashlib.sha256(INPUT.read_bytes()).hexdigest()
            and config.get("demand_model") == "population"
            and config.get("sampling_model") == "per-type"
            and config.get("observation_model") == "multinomial")


def main():
    preprocess = CASE / "data/amphora_bipartite_construction.py"
    inference = EXPERIMENTS / "run_huff_inference.py"
    if RUN_PREPROCESS:
        run(preprocess)
    run(inference, "--mode", "check", "--seed", SEED)
    if CHECK_ONLY:
        return

    if RUN_SBI and (not existing(RESULTS / "posterior.pkl")
                    or not existing(RESULTS / "posterior_samples.csv")
                    or not reference_matches()):
        run(inference, "--mode", "sbi", "--simulations", SIMULATIONS,
            "--posterior-samples", POSTERIOR_SAMPLES, "--seed", SEED)

    if RUN_PPC and not existing(RESULTS / "ppc_metrics.json"):
        run(inference, "--mode", "ppc", "--ppc-samples", PPC_SAMPLES, "--seed", SEED)
    if RUN_SBC and not existing(RESULTS / "sbc_ranks.csv"):
        run(inference, "--mode", "sbc", "--sbc-runs", SBC_RUNS,
            "--sbc-samples", SBC_SAMPLES, "--seed", SEED)
    if RUN_PLOTS and not existing(RESULTS / "plots/huff_sbc.pdf"):
        run(EXPERIMENTS / "plot_huff_results.py")
    if RUN_BACKBONES and not existing(BACKBONES / "site_site_edges.csv"):
        run(EXPERIMENTS / "run_huff_backbones.py", "--draws", BACKBONE_DRAWS, "--seed", SEED)
    if RUN_BASELINES and not existing(BACKBONES / "baseline_overlap.csv"):
        run(EXPERIMENTS / "compare_backbones.py", "--top-k", TOP_K)
    print("\nCompleted. Newly generated outputs:", CASE / "outputs")


if __name__ == "__main__":
    main()
