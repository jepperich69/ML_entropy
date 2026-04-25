#!/usr/bin/env python3
"""Run the entropy/MH benchmark on downloaded CORELS COMPAS binary CSVs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.path.join("code", "rulelist", "data", "corels_compas"),
        help="directory containing compas_train-binary.csv and compas_test-binary.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("code", "rulelist", "results", "corels_compas_results.csv"),
    )
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-cardinality", type=int, default=2)
    parser.add_argument("--regularization", type=float, default=0.015)
    parser.add_argument("--mh-steps", type=int, default=10000)
    parser.add_argument(
        "--exact-max-antecedents",
        type=int,
        default=120,
        help="skip exact enumeration above this antecedent count",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="run download_corels_compas.py first if binary CSVs are missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    data_dir = args.data_dir
    out_path = args.out
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    if not os.path.isabs(out_path):
        out_path = os.path.join(project_root, out_path)

    train_csv = os.path.join(data_dir, "compas_train-binary.csv")
    test_csv = os.path.join(data_dir, "compas_test-binary.csv")

    if args.download_if_missing and (not os.path.exists(train_csv) or not os.path.exists(test_csv)):
        downloader = os.path.join(here, "download_corels_compas.py")
        cmd = [sys.executable, downloader, "--binary-only", "--out-dir", data_dir]
        subprocess.check_call(cmd, cwd=project_root)

    missing = [path for path in [train_csv, test_csv] if not os.path.exists(path)]
    if missing:
        print("Missing COMPAS binary CSV files:", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        print(
            "Run: python code/rulelist/download_corels_compas.py --binary-only",
            file=sys.stderr,
        )
        return 1

    benchmark = os.path.join(here, "benchmark_rulelist_entropy_mh.py")
    cmd = [
        sys.executable,
        benchmark,
        "--train-csv",
        train_csv,
        "--test-csv",
        test_csv,
        "--max-depth",
        str(args.max_depth),
        "--max-cardinality",
        str(args.max_cardinality),
        "--regularization",
        str(args.regularization),
        "--mh-steps",
        str(args.mh_steps),
        "--exact-max-antecedents",
        str(args.exact_max_antecedents),
        "--out",
        out_path,
    ]
    subprocess.check_call(cmd, cwd=project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
