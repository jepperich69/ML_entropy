"""Run a small stress grid for the warm-start question.

This is a thin driver around:
  - run_full_benchmark.py
  - summarize_full_benchmark.py

Purpose:
  Make the same 4 public datasets harder by increasing rule cardinality and
  tightening the MH budget, then check whether warm-started MH beats
  random-started MH under those tighter budgets.

Default grid:
  max_cardinality = 3
  max_depth = 3
  exact_time_limit = 1 second
  mh_steps in {100, 250, 500, 1000}
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RUNNER = SCRIPT_DIR / "run_full_benchmark.py"
SUMMARIZER = SCRIPT_DIR / "summarize_full_benchmark.py"


def run_one(config: dict, python_exe: str) -> None:
    mh_steps = config["mh_steps"]
    max_card = config["max_cardinality"]
    max_depth = config["max_depth"]
    exact_limit = config["exact_time_limit"]
    max_ants = config["max_antecedents"]

    tag = f"card{max_card}_mh{mh_steps}"
    csv_out = RESULTS_DIR / f"warmstart_stress_{tag}.csv"
    md_out = RESULTS_DIR / f"warmstart_stress_{tag}.md"

    cmd_run = [
        python_exe,
        str(RUNNER),
        "--use-java",
        "--mh-steps", str(mh_steps),
        "--max-cardinality", str(max_card),
        "--max-depth", str(max_depth),
        "--max-antecedents", str(max_ants),
        "--exact-time-limit", str(exact_limit),
        "--out", str(csv_out),
    ]

    cmd_summary = [
        python_exe,
        str(SUMMARIZER),
        "--csv", str(csv_out),
        "--out", str(md_out),
    ]

    print("=" * 72)
    print(f"Warm-start stress run: {tag}")
    print("Benchmark command:")
    print(" ".join(cmd_run))
    subprocess.run(cmd_run, check=True, cwd=str(PROJECT_ROOT))

    print(f"\nSummarizing: {md_out.name}")
    subprocess.run(cmd_summary, check=True, cwd=str(PROJECT_ROOT))
    print(f"Done: {csv_out.name}, {md_out.name}\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", default=sys.executable,
                        help="Python executable to use for child runs")
    parser.add_argument("--mh-steps", nargs="+", type=int, default=[100, 250, 500, 1000],
                        help="MH step counts to run")
    parser.add_argument("--max-cardinality", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-antecedents", type=int, default=250000)
    parser.add_argument("--exact-time-limit", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for mh_steps in args.mh_steps:
        run_one(
            {
                "mh_steps": mh_steps,
                "max_cardinality": args.max_cardinality,
                "max_depth": args.max_depth,
                "max_antecedents": args.max_antecedents,
                "exact_time_limit": args.exact_time_limit,
            },
            args.python_exe,
        )


if __name__ == "__main__":
    main()
