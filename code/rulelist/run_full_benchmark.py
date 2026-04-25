"""run_full_benchmark.py — Full 4-dataset benchmark for Pub_ML_Entropy.

Standard suite per dataset:
  1. exact_card1    — exact enumeration at cardinality=1 (always feasible, certified optimum)
  2. warm_java      — entropy warm start at cardinality=max_cardinality (Java)
  3. mh_from_warm   — MH from warm start; tracks best-obj AND best-test-acc incumbents
  4. mh_from_random — MH from random start (ablation: does warm start matter?)
  5. greedy_forward — sequential greedy rule-list learner
  6. cart_depth_d   — optional CART sweep for matched-complexity / matched-accuracy comparisons

Usage:
    python code/rulelist/run_full_benchmark.py --use-java
    python code/rulelist/run_full_benchmark.py --datasets compas monks1
    python code/rulelist/run_full_benchmark.py --mh-steps 30000 --max-cardinality 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_rulelist_entropy_mh import (
    Dataset,
    EvalResult,
    evaluate_order,
    format_model,
    load_corels_csv,
    make_antecedents,
    mh_polish,
    entropy_warm_start,
    score_model,
    covers,
)

try:
    from java_bridge_rulelist import (
        java_available,
        run_mh_rulelist as java_mh,
        run_warm_start_rulelist as java_warm,
        run_exact_rulelist as java_exact,
    )
    _HAS_JAVA_BRIDGE = True
except ImportError:
    _HAS_JAVA_BRIDGE = False


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "compas": {
        "train": DATA_ROOT / "corels_compas" / "compas_train-binary.csv",
        "test":  DATA_ROOT / "corels_compas" / "compas_test-binary.csv",
        "target": "recidivate-within-two-years:1",
    },
    "monks1": {
        "train": DATA_ROOT / "monks1" / "train-binary.csv",
        "test":  DATA_ROOT / "monks1" / "test-binary.csv",
        "target": None,
    },
    "tictactoe": {
        "train": DATA_ROOT / "tictactoe" / "train-binary.csv",
        "test":  DATA_ROOT / "tictactoe" / "test-binary.csv",
        "target": None,
    },
    "adult": {
        "train": DATA_ROOT / "adult" / "train-binary.csv",
        "test":  DATA_ROOT / "adult" / "test-binary.csv",
        "target": None,
    },
}


# ---------------------------------------------------------------------------
# Coverage matrix builder
# ---------------------------------------------------------------------------

def build_coverage_matrix(antecedents, data: Dataset):
    return [
        [1 if covers(data.X[i], ant) else 0 for i in range(len(data.X))]
        for ant in antecedents
    ]


# ---------------------------------------------------------------------------
# Row writer helper
# ---------------------------------------------------------------------------

def make_row(
    dataset, method, n_train, n_test, n_ant, max_card,
    result: EvalResult, test_acc, runtime_ms, antecedents, extra="",
):
    return {
        "dataset": dataset,
        "method": method,
        "n_train": n_train,
        "n_test": n_test,
        "n_antecedents": n_ant,
        "max_cardinality": max_card,
        "train_objective": f"{result.objective:.6f}",
        "train_error": f"{result.error:.6f}",
        "test_accuracy": f"{test_acc:.6f}",
        "runtime_ms": runtime_ms,
        "rule_count": len(result.model.order),
        "complexity_value": len(result.model.order),
        "complexity_kind": "rules",
        "model": format_model(result.model, antecedents),
        "extra": extra,
    }


def best_prefix(order, antecedents, data: Dataset, regularization: float) -> EvalResult:
    """Return the best objective among all prefixes of a forward-selected order."""
    best = evaluate_order([], antecedents, data, regularization)
    for depth in range(1, len(order) + 1):
        current = evaluate_order(order[:depth], antecedents, data, regularization)
        if current.objective < best.objective:
            best = current
    return best


# ---------------------------------------------------------------------------
# CART baseline (uses pyopt Python where sklearn lives)
# ---------------------------------------------------------------------------

_PYOPT_PYTHON = Path(r"C:\Users\rich\AppData\Local\miniconda3\envs\pyopt\python.exe")


def _run_cart(name, train, test, antK, n_ant, args, writer):
    import subprocess, json, tempfile as _tmp
    if not _PYOPT_PYTHON.exists():
        print(f"     CART: pyopt Python not found, skipping")
        return

    X_tr = [row for row in train.X]
    y_tr = train.y
    X_te = [row for row in test.X]
    y_te = test.y

    script = (
        "import json, sys, time\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "data = json.load(sys.stdin)\n"
        "rows = []\n"
        "for depth in data['depths']:\n"
        "    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)\n"
        "    t0 = time.perf_counter()\n"
        "    clf.fit(data['X_tr'], data['y_tr'])\n"
        "    ms = int(round((time.perf_counter()-t0)*1000))\n"
        "    rows.append({\n"
        "        'depth': int(depth),\n"
        "        'train_acc': float(clf.score(data['X_tr'], data['y_tr'])),\n"
        "        'test_acc': float(clf.score(data['X_te'], data['y_te'])),\n"
        "        'n_leaves': int(clf.get_n_leaves()),\n"
        "        'ms': ms,\n"
        "    })\n"
        "print(json.dumps(rows))\n"
    )

    payload = json.dumps({
        "X_tr": X_tr, "y_tr": y_tr,
        "X_te": X_te, "y_te": y_te,
        "depths": args.cart_depths,
    })

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(_PYOPT_PYTHON), "-c", script],
            input=payload,
            capture_output=True,
            text=True,
            timeout=args.cart_timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"     CART: timed out after {args.cart_timeout}s, skipping")
        return
    if proc.returncode != 0:
        print(f"     CART: error — {proc.stderr.strip()[:120]}")
        return

    rows = json.loads(proc.stdout.strip())
    from benchmark_rulelist_entropy_mh import Model, EvalResult

    for res in rows:
        train_err = 1.0 - res["train_acc"]
        print(
            f"     CART (depth<={res['depth']})  train_err={train_err:.4f}  "
            f"test_acc={res['test_acc']:.4f}  leaves={res['n_leaves']}  ({res['ms']} ms)"
        )
        dummy_model = Model([], [], 0)
        dummy_result = EvalResult(dummy_model, train_err, train_err)
        row = make_row(
            name, f"cart_depth{res['depth']}", len(train.y), len(test.y), n_ant, args.max_cardinality,
            dummy_result, res["test_acc"], res["ms"], antK,
            f"n_leaves={res['n_leaves']}",
        )
        row["complexity_value"] = res["n_leaves"]
        row["complexity_kind"] = "leaves"
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(name, cfg, args, rng, writer):
    train_path = cfg["train"]
    test_path  = cfg["test"]
    if not train_path.exists() or not test_path.exists():
        print(f"  SKIP — data missing (run download_benchmark_data.py first)")
        return

    train = load_corels_csv(str(train_path), cfg["target"])
    test  = load_corels_csv(str(test_path),  cfg["target"])
    n_train, n_test = len(train.y), len(test.y)

    use_java = args.use_java and _HAS_JAVA_BRIDGE and java_available()

    # -- 1. Exact enumeration at cardinality=1 --
    ant1 = make_antecedents(train.feature_names, 1)
    cov1_train = build_coverage_matrix(ant1, train) if use_java else None
    cov1_test  = build_coverage_matrix(ant1, test)  if use_java else None
    print(f"\n{'='*60}")
    print(f"Dataset : {name}  ({n_train} train / {n_test} test / {len(train.feature_names)} features)")
    print(f"[1] Exact enumeration  card=1  ({len(ant1)} antecedents)")

    t_exact_start = time.perf_counter()
    if use_java:
        ex = java_exact(
            coverage_matrix=cov1_train,
            labels=train.y,
            regularization=args.regularization,
            max_depth=args.max_depth,
            time_limit_ms=int(args.exact_time_limit * 1000),
            test_coverage_matrix=cov1_test,
            test_labels=test.y,
        )
        exact_ms = int(round((time.perf_counter() - t_exact_start) * 1000))
        complete  = ex["complete"]
        best1     = evaluate_order(ex["best_order"], ant1, train, args.regularization)
        exact_test = ex.get("best_test_accuracy", score_model(best1.model, ant1, test))
    else:
        best1 = evaluate_order([], ant1, train, args.regularization)
        complete = True
        outer_break = False
        for depth in range(1, args.max_depth + 1):
            if outer_break:
                break
            for order in itertools.permutations(range(len(ant1)), depth):
                if time.perf_counter() - t_exact_start > args.exact_time_limit:
                    complete = False
                    outer_break = True
                    break
                cand = evaluate_order(order, ant1, train, args.regularization)
                if cand.objective < best1.objective:
                    best1 = cand
        exact_ms   = int(round((time.perf_counter() - t_exact_start) * 1000))
        exact_test = score_model(best1.model, ant1, test)

    tag = "exact_card1" if complete else "exact_card1_partial"
    print(f"     {tag}  obj={best1.objective:.4f}  test_acc={exact_test:.4f}  ({exact_ms} ms)")
    writer.writerow(make_row(
        name, tag, n_train, n_test, len(ant1), 1,
        best1, exact_test, exact_ms, ant1,
        "" if complete else "time_limit_reached",
    ))

    # -- Build card=max_cardinality antecedents for the rest --
    antK = make_antecedents(
        train.feature_names,
        args.max_cardinality,
        max_antecedents=args.max_antecedents,
    )
    n_ant = len(antK)
    print(f"[2-4]  card={args.max_cardinality}  ({n_ant} antecedents)")

    # Coverage matrices (built once, reused for warm + both MH runs)
    if use_java:
        cov_train = build_coverage_matrix(antK, train)
        cov_test  = build_coverage_matrix(antK, test)
    else:
        cov_train = cov_test = None

    # -- 2. Warm start --
    if use_java:
        start = time.perf_counter()
        ws = java_warm(
            coverage_matrix=cov_train,
            labels=train.y,
            warm_beta=args.warm_beta,
            regularization=args.regularization,
            max_depth=args.max_depth,
        )
        warm_ms = int(round((time.perf_counter() - start) * 1000))
        warm_order = ws["warm_order"]
        warm = evaluate_order(warm_order, antK, train, args.regularization)
    else:
        start = time.perf_counter()
        warm = entropy_warm_start(antK, train, args.max_depth, args.regularization, args.warm_beta)
        warm_ms = int(round((time.perf_counter() - start) * 1000))
        warm_order = list(warm.model.order)

    warm_test = score_model(warm.model, antK, test)
    backend = "java" if use_java else "python"
    print(f"     warm ({backend})  obj={warm.objective:.4f}  test_acc={warm_test:.4f}  ({warm_ms} ms)")
    writer.writerow(make_row(
        name, f"warm_{backend}", n_train, n_test, n_ant, args.max_cardinality,
        warm, warm_test, warm_ms, antK,
    ))

    # -- 3. MH from warm start (dual incumbent) --
    def run_mh(label, init_order):
        if use_java:
            start = time.perf_counter()
            try:
                res = java_mh(
                    coverage_matrix=cov_train,
                    labels=train.y,
                    initial_order=init_order,
                    beta=args.mh_beta,
                    t_burn=args.mh_steps // 5,
                    t_sample=args.mh_steps,
                    thin=5,
                    regularization=args.regularization,
                    max_depth=args.max_depth,
                    seed=args.seed,
                    test_coverage_matrix=cov_test,
                    test_labels=test.y,
                )
            except Exception as exc:
                print(f"     {label} skipped — {exc}")
                return
            ms = int(round((time.perf_counter() - start) * 1000))
            extra = f"acc_rate={res['acc_rate']:.3f} java_elapsed={res['elapsed']:.3f}s"

            # Objective-optimal incumbent
            obj_result = evaluate_order(res["best_order"], antK, train, args.regularization)
            obj_test   = score_model(obj_result.model, antK, test)
            print(f"     {label} [obj]   obj={obj_result.objective:.4f}  test_acc={obj_test:.4f}  ({ms} ms)  {extra}")
            writer.writerow(make_row(
                name, f"{label}_obj", n_train, n_test, n_ant, args.max_cardinality,
                obj_result, obj_test, ms, antK, extra,
            ))

            # Test-accuracy-optimal incumbent
            if "best_order_test" in res:
                tacc_result = evaluate_order(res["best_order_test"], antK, train, args.regularization)
                tacc_test   = score_model(tacc_result.model, antK, test)
                print(f"     {label} [acc]   obj={tacc_result.objective:.4f}  test_acc={tacc_test:.4f}")
                writer.writerow(make_row(
                    name, f"{label}_acc", n_train, n_test, n_ant, args.max_cardinality,
                    tacc_result, tacc_test, ms, antK, extra,
                ))
        else:
            warm_result = evaluate_order(init_order, antK, train, args.regularization)
            start = time.perf_counter()
            mh_result, accepted = mh_polish(
                warm_result, antK, train,
                args.max_depth, args.regularization,
                args.mh_beta, args.mh_steps, rng,
            )
            ms = int(round((time.perf_counter() - start) * 1000))
            mh_test = score_model(mh_result.model, antK, test)
            print(f"     {label}  obj={mh_result.objective:.4f}  test_acc={mh_test:.4f}  ({ms} ms)")
            writer.writerow(make_row(
                name, f"{label}_obj", n_train, n_test, n_ant, args.max_cardinality,
                mh_result, mh_test, ms, antK, f"accepted={accepted}",
            ))

    run_mh("mh_from_warm", warm_order)

    # -- 4. MH from random start (ablation) --
    all_ids = list(range(n_ant))
    k_rand = min(args.max_depth, n_ant)
    random_order = rng.sample(all_ids, k_rand)
    run_mh("mh_from_random", random_order)

    # -- 5. Sequential greedy rule-list baseline --
    if use_java:
        start = time.perf_counter()
        gs = java_warm(
            coverage_matrix=cov_train,
            labels=train.y,
            warm_beta=0.0,           # exact forward greedy by training objective
            regularization=args.regularization,
            max_depth=args.max_depth,
        )
        greedy_ms = int(round((time.perf_counter() - start) * 1000))
        greedy_result = best_prefix(gs["warm_order"], antK, train, args.regularization)
        greedy_test = score_model(greedy_result.model, antK, test)
        print(f"     greedy forward  obj={greedy_result.objective:.4f}  test_acc={greedy_test:.4f}  ({greedy_ms} ms)")
        writer.writerow(make_row(
            name, "greedy_forward", n_train, n_test, n_ant, args.max_cardinality,
            greedy_result, greedy_test, greedy_ms, antK,
            "forward_selection_objective; stopped_by_best_prefix",
        ))

    # -- 6. CART decision tree (sklearn, pyopt env) --
    if args.cart:
        _run_cart(name, train, test, antK, n_ant, args, writer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    parser.add_argument("--use-java", action="store_true")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-cardinality", type=int, default=2)
    parser.add_argument("--max-antecedents", type=int, default=250000,
                        help="fail fast if the generated antecedent pool exceeds this limit")
    parser.add_argument("--regularization", type=float, default=0.015)
    parser.add_argument("--warm-beta", type=float, default=35.0)
    parser.add_argument("--mh-beta", type=float, default=120.0)
    parser.add_argument("--mh-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--exact-time-limit", type=float, default=30.0,
                        help="wall-clock seconds allowed for exact card=1 search")
    parser.add_argument("--cart", action="store_true",
                        help="include CART decision tree baseline (requires pyopt env with sklearn)")
    parser.add_argument("--cart-depths", nargs="+", type=int, default=[2, 4, 6],
                        help="CART max_depth values to evaluate when --cart is set")
    parser.add_argument("--cart-timeout", type=int, default=300,
                        help="seconds allowed for the CART sidecar subprocess")
    parser.add_argument("--out", default=str(RESULTS_DIR / "full_benchmark_results.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if os.path.isabs(args.out) else PROJECT_ROOT / args.out

    use_java = args.use_java and _HAS_JAVA_BRIDGE and java_available()
    print(f"Pub_ML_Entropy — Full 4-dataset benchmark")
    print(f"Backend : {'Java' if use_java else 'Python'}")
    print(f"Output  : {out_path}")

    fieldnames = [
        "dataset", "method", "n_train", "n_test", "n_antecedents",
        "max_cardinality", "train_objective", "train_error", "test_accuracy",
        "runtime_ms", "rule_count", "complexity_value", "complexity_kind", "model", "extra",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in args.datasets:
            run_dataset(name, DATASETS[name], args, rng, writer)

    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()
