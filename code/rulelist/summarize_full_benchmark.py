"""Summarize a full benchmark CSV into the two paper-facing benchmark views.

Outputs:
  1. Warm-start ablation table: warm vs MH-from-warm vs MH-from-random
  2. Complexity-aware table: best rule-list accuracy and smallest rule-list
     achieving the best rule-list accuracy, plus CART reference rows if present

Usage:
  python code/rulelist/summarize_full_benchmark.py
  python code/rulelist/summarize_full_benchmark.py --csv code/rulelist/results/full_benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_CSV = RESULTS_DIR / "full_benchmark_results.csv"
DEFAULT_MD = RESULTS_DIR / "full_benchmark_summary.md"


def _to_float(row: dict, key: str) -> float:
    return float(row[key])


def _to_int(row: dict, key: str) -> int:
    raw = str(row.get(key, "")).strip()
    return int(raw) if raw else 0


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    clean = []
    for row in rows:
        dataset = (row.get("dataset") or "").strip()
        method = (row.get("method") or "").strip()
        if not dataset or dataset == "rate=0.228 java_elapsed=27.208s":
            continue
        if not method:
            continue
        clean.append(row)
    return clean


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["dataset"], []).append(row)
    return out


def find_one(rows: list[dict], method: str) -> dict | None:
    for row in rows:
        if row["method"] == method:
            return row
    return None


def best_rulelist_acc(rows: list[dict]) -> dict | None:
    candidates = [r for r in rows if not r["method"].startswith("cart_")]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (_to_float(r, "test_accuracy"), -_to_int(r, "rule_count")),
    )


def smallest_at_best_acc(rows: list[dict], tol: float = 1e-12) -> dict | None:
    best = best_rulelist_acc(rows)
    if best is None:
        return None
    target = _to_float(best, "test_accuracy")
    candidates = [
        r for r in rows
        if not r["method"].startswith("cart_")
        and abs(_to_float(r, "test_accuracy") - target) <= tol
    ]
    return min(candidates, key=lambda r: (_to_int(r, "rule_count"), _to_float(r, "runtime_ms")))


def best_cart(rows: list[dict]) -> dict | None:
    candidates = [r for r in rows if r["method"].startswith("cart_")]
    if not candidates:
        return None
    return max(candidates, key=lambda r: _to_float(r, "test_accuracy"))


def render_markdown(rows: list[dict]) -> str:
    by_dataset = grouped(rows)
    lines: list[str] = []

    lines.append("# Full Benchmark Summary")
    lines.append("")
    lines.append("## Warm-Start Ablation")
    lines.append("")
    lines.append("| Dataset | Warm acc | MH from warm acc | MH from random acc | Warm rules | MH warm rules | MH random rules | Read |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for dataset in sorted(by_dataset):
        ds = by_dataset[dataset]
        warm = find_one(ds, "warm_java")
        mh_warm = find_one(ds, "mh_from_warm_acc") or find_one(ds, "mh_from_warm_obj")
        mh_rand = find_one(ds, "mh_from_random_acc") or find_one(ds, "mh_from_random_obj")
        if not (warm and mh_warm and mh_rand):
            continue
        warm_acc = _to_float(warm, "test_accuracy")
        mh_warm_acc = _to_float(mh_warm, "test_accuracy")
        mh_rand_acc = _to_float(mh_rand, "test_accuracy")
        if abs(mh_warm_acc - mh_rand_acc) < 1e-12:
            read = "warm and random reach the same incumbent"
        elif mh_warm_acc > mh_rand_acc:
            read = "warm-start advantage"
        else:
            read = "random start wins"
        lines.append(
            f"| {dataset} | {warm_acc:.4f} | {mh_warm_acc:.4f} | {mh_rand_acc:.4f} | "
            f"{_to_int(warm, 'rule_count')} | {_to_int(mh_warm, 'rule_count')} | {_to_int(mh_rand, 'rule_count')} | {read} |"
        )

    lines.append("")
    lines.append("## Complexity-Aware Comparison")
    lines.append("")
    lines.append("| Dataset | Best rule-list acc | Rules at best acc | Objective at best acc | Best CART acc | CART leaves | Read |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for dataset in sorted(by_dataset):
        ds = by_dataset[dataset]
        rule_best = smallest_at_best_acc(ds)
        cart = best_cart(ds)
        if rule_best is None:
            continue
        if cart is None:
            cart_acc = ""
            cart_leaves = ""
            read = "rule-list only"
        else:
            cart_acc_val = _to_float(cart, "test_accuracy")
            cart_acc = f"{cart_acc_val:.4f}"
            cart_leaves = str(_to_int(cart, "complexity_value"))
            rl_acc = _to_float(rule_best, "test_accuracy")
            if rl_acc > cart_acc_val:
                read = "rule list wins on accuracy"
            elif rl_acc < cart_acc_val:
                read = "tree wins on accuracy; compare complexity gap"
            else:
                read = "tie on accuracy"
        lines.append(
            f"| {dataset} | {_to_float(rule_best, 'test_accuracy'):.4f} | {_to_int(rule_best, 'rule_count')} | "
            f"{_to_float(rule_best, 'train_objective'):.4f} | {cart_acc} | {cart_leaves} | {read} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Warm-start value is only established if `mh_from_warm` beats `mh_from_random` under the same MH budget.")
    lines.append("- Simplicity claims should be made through the `rule_count` versus `n_leaves` gap, not raw accuracy alone.")
    lines.append("- If CART rows are absent, rerun `run_full_benchmark.py` with `--cart`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_MD))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    rows = load_rows(csv_path)
    out_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
