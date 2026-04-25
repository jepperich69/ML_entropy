#!/usr/bin/env python3
"""Paper reproduction suite for:
  Rich, J. (2026). Mapping the Near-Optimal Space of Transparent Classifiers.
  Computers & Operations Research.
  https://github.com/jepperich69/ML_entropy

Runs all four experiments in order and generates the landscape figure.
All results are written to code/rulelist/results/.

Usage:
    python reproduce.py                  # full suite
    python reproduce.py --experiment 1   # single experiment (1-4)
    python reproduce.py --skip-java      # skip Java bridge (slower for Expts 1-2)
    python reproduce.py --skip-download  # skip data download if already present
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code" / "rulelist"
RESULTS = CODE / "results"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Exact parameters used in the paper (Table 5 of the manuscript)
# ---------------------------------------------------------------------------

SEED = 20_260_423
REGULARIZATION = 0.015
WARM_BETA = 35.0
MH_BETA = 120.0


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print("=" * 72)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR: {label} failed (exit {result.returncode}).")
        sys.exit(result.returncode)
    print(f"\n  Done in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step0_download() -> None:
    run(
        [PYTHON, str(CODE / "download_benchmark_data.py")],
        "Step 0: Download and binarize benchmark datasets",
    )


def experiment1(use_java: bool) -> None:
    """Public benchmark — Tables 2-3 of the paper."""
    cmd = [
        PYTHON, str(CODE / "run_full_benchmark.py"),
        "--mh-steps", "20000",
        "--max-cardinality", "2",
        "--max-depth", "3",
        "--regularization", str(REGULARIZATION),
        "--warm-beta", str(WARM_BETA),
        "--mh-beta", str(MH_BETA),
        "--seed", str(SEED),
        "--out", str(RESULTS / "exp1_public_benchmark.csv"),
    ]
    if use_java:
        cmd.append("--use-java")
    run(cmd, "Experiment 1: Public rule-list benchmarks (Tables 2-3)")


def experiment2() -> None:
    """Warm-start stress test — Table 4 of the paper.

    run_warmstart_stress_grid.py is a thin driver that spawns run_full_benchmark.py
    as child processes. It hardcodes --use-java in those child calls and does not
    forward --seed (child processes use the run_full_benchmark.py default seed).
    """
    cmd = [
        PYTHON, str(CODE / "run_warmstart_stress_grid.py"),
        "--mh-steps", "100", "250", "500",
        "--max-cardinality", "3",
        "--max-depth", "3",
    ]
    run(cmd, "Experiment 2: Warm-start stress test (Table 4)")


def experiment3() -> None:
    """Synthetic large-scale — Table 5 of the paper."""
    cmd = [
        PYTHON, str(CODE / "run_synthetic_scale.py"),
        "--mh-steps", "1000",
        "--max-depth", "5",
        "--mh-beta", str(MH_BETA),
        "--seed", str(SEED),
        "--out", str(RESULTS / "exp3_synthetic_scale.csv"),
    ]
    run(cmd, "Experiment 3: Synthetic large-scale (Table 5)")


def experiment4() -> None:
    """Probability-enabled analysis — Table 6 and Figure 1.

    Note: run_probability_analysis.py uses --samples (not --mh-steps),
    --burn (not --burn-in), and --out-prefix (not --out-dir).
    """
    cmd = [
        PYTHON, str(CODE / "run_probability_analysis.py"),
        "--samples", "1000",
        "--burn", "2000",
        "--thin", "10",
        "--max-cardinality", "2",
        "--max-depth", "3",
        "--mh-beta", str(MH_BETA),
        "--warm-beta", str(WARM_BETA),
        "--regularization", str(REGULARIZATION),
        "--seed", str(SEED),
        "--out-prefix", str(RESULTS / "probability_analysis_card2"),
    ]
    run(cmd, "Experiment 4: Probability-enabled analysis (Table 6)")


def make_figures() -> None:
    """Landscape figure — Figure 1 of the paper.

    make_probability_figures.py writes output to --results-dir; it does not
    accept an --out path. Copy the generated PNG to Overleaf_source/ manually
    or adjust the --results-dir to point there.
    """
    cmd = [
        PYTHON, str(CODE / "make_probability_figures.py"),
        "--results-dir", str(RESULTS),
        "--prefix", "probability_analysis_card2",
    ]
    run(cmd, "Figures: Near-optimal landscape (Figure 1)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2, 3, 4],
        help="Run a single experiment (default: run all four)",
    )
    parser.add_argument(
        "--skip-java", action="store_true",
        help="Skip the Java bridge for Experiments 1-2 (slower but dependency-free)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download (use if code/rulelist/data/ is already populated)",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation after Experiment 4",
    )
    args = parser.parse_args()

    use_java = not args.skip_java
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("\nPaper reproduction suite")
    print("Rich (2026). Mapping the Near-Optimal Space of Transparent Classifiers.")
    print(f"Results: {RESULTS}\n")

    if args.experiment is not None:
        exp = args.experiment
        if not args.skip_download and exp in (1, 2, 4):
            step0_download()
        if exp == 1:
            experiment1(use_java)
        elif exp == 2:
            experiment2()
        elif exp == 3:
            experiment3()
        elif exp == 4:
            experiment4()
            if not args.skip_figures:
                make_figures()
        return

    # Full suite
    if not args.skip_download:
        step0_download()
    experiment1(use_java)
    experiment2()
    experiment3()
    experiment4()
    if not args.skip_figures:
        make_figures()

    print("\n" + "=" * 72)
    print("  All experiments complete.")
    print(f"  Results: {RESULTS}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
