#!/usr/bin/env python3
"""Download the public CORELS COMPAS sample data.

The CORELS repository ships COMPAS train/test splits generated from the
ProPublica two-year recidivism dataset. For this project we mainly need the
`*-binary.csv` files, because they are directly compatible with the lightweight
rule-list benchmark harness.

The script also attempts to download the CORELS-native `.out`, `.label`, and
`.minor` files so a later baseline wrapper can call CORELS itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from typing import Iterable, List


FILES = [
    "compas_train.csv",
    "compas_train-binary.csv",
    "compas_train.out",
    "compas_train.label",
    "compas_train.minor",
    "compas_test.csv",
    "compas_test-binary.csv",
    "compas_test.out",
    "compas_test.label",
]

BASE_URLS = [
    "https://raw.githubusercontent.com/corels/corels/master/data/{name}",
    "https://raw.githubusercontent.com/corels/corels/main/data/{name}",
]


def download_one(name: str, out_dir: str, bases: Iterable[str], overwrite: bool) -> str:
    out_path = os.path.join(out_dir, name)
    if os.path.exists(out_path) and not overwrite:
        return f"exists  {out_path}"

    last_error = None
    for template in bases:
        url = template.format(name=name)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty response")
            with open(out_path, "wb") as handle:
                handle.write(data)
            return f"fetched {out_path} ({len(data)} bytes)"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download {name}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=os.path.join("code", "rulelist", "data", "corels_compas"),
        help="where to place downloaded files",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="download only compas_train-binary.csv and compas_test-binary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    names: List[str]
    if args.binary_only:
        names = ["compas_train-binary.csv", "compas_test-binary.csv"]
    else:
        names = FILES

    failures = []
    for name in names:
        try:
            print(download_one(name, args.out_dir, BASE_URLS, args.overwrite))
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"failed  {name}: {exc}", file=sys.stderr)

    if failures:
        print("\nSome files were not downloaded.", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nCORELS COMPAS data ready.")
    print(f"Directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
