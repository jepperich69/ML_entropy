#!/usr/bin/env python3
"""Synthetic large-scale rule-list speed experiments.

These runs are not meant to replace public benchmarks. They create controlled
large candidate libraries to show what happens when the discrete classifier
space is much larger than the four public datasets.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


@dataclass(frozen=True)
class Rule:
    features: tuple[int, ...]
    values: tuple[int, ...]
    name: str


@dataclass
class EvalResult:
    order: tuple[int, ...]
    predictions: tuple[int, ...]
    default_pred: int
    error: float
    objective: float


def make_rule(features: tuple[int, ...], values: tuple[int, ...]) -> Rule:
    name = " & ".join(f"x{f}={v}" for f, v in zip(features, values))
    return Rule(features, values, name)


def generate_candidate_rules(
    n_features: int,
    n_candidates: int,
    max_cardinality: int,
    true_rules: list[Rule],
    rng: random.Random,
) -> list[Rule]:
    rules = list(true_rules)
    seen = {(r.features, r.values) for r in rules}
    while len(rules) < n_candidates:
        card = rng.randint(1, max_cardinality)
        feats = tuple(sorted(rng.sample(range(n_features), card)))
        vals = tuple(rng.randint(0, 1) for _ in feats)
        key = (feats, vals)
        if key in seen:
            continue
        seen.add(key)
        rules.append(make_rule(feats, vals))
    return rules


def coverage(rule: Rule, X: np.ndarray) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    for f, v in zip(rule.features, rule.values):
        mask &= X[:, f] == v
    return mask


def apply_true_rule_list(X: np.ndarray, true_rules: list[Rule], true_preds: list[int], default: int) -> np.ndarray:
    y = np.full(X.shape[0], default, dtype=np.uint8)
    remaining = np.ones(X.shape[0], dtype=bool)
    for rule, pred in zip(true_rules, true_preds):
        captured = remaining & coverage(rule, X)
        y[captured] = pred
        remaining[captured] = False
    return y


def generate_problem(
    n_train: int,
    n_test: int,
    n_features: int,
    n_candidates: int,
    max_cardinality: int,
    noise: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    X_train = (np_rng.random((n_train, n_features)) < 0.5).astype(np.uint8)
    X_test = (np_rng.random((n_test, n_features)) < 0.5).astype(np.uint8)

    true_specs = [
        ((0, 1, 2), (1, 1, 0), 1),
        ((3, 4, 5), (0, 1, 1), 1),
        ((6, 7), (1, 0), 0),
        ((8, 9, 10), (1, 0, 1), 1),
        ((11, 12), (0, 0), 0),
    ]
    true_rules = [make_rule(tuple(f), tuple(v)) for f, v, _ in true_specs]
    true_preds = [p for _, _, p in true_specs]
    y_train = apply_true_rule_list(X_train, true_rules, true_preds, default=0)
    y_test = apply_true_rule_list(X_test, true_rules, true_preds, default=0)

    if noise > 0:
        flip_train = np_rng.random(n_train) < noise
        flip_test = np_rng.random(n_test) < noise
        y_train[flip_train] = 1 - y_train[flip_train]
        y_test[flip_test] = 1 - y_test[flip_test]

    rules = generate_candidate_rules(n_features, n_candidates, max_cardinality, true_rules, rng)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "rules": rules,
        "true_rules": true_rules,
    }


def majority(pos: int, total: int) -> int:
    return 1 if total > 0 and 2 * pos >= total else 0


def eval_order(
    order: tuple[int, ...],
    rules: list[Rule],
    X: np.ndarray,
    y: np.ndarray,
    regularization: float,
    cache: dict[int, np.ndarray],
) -> EvalResult:
    remaining = np.ones(X.shape[0], dtype=bool)
    pred_bits = np.zeros(X.shape[0], dtype=np.uint8)
    predictions = []
    for rid in order:
        if rid not in cache:
            cache[rid] = coverage(rules[rid], X)
        captured = remaining & cache[rid]
        total = int(captured.sum())
        pos = int(y[captured].sum())
        pred = majority(pos, total)
        predictions.append(pred)
        pred_bits[captured] = pred
        remaining[captured] = False
    total_rem = int(remaining.sum())
    pos_rem = int(y[remaining].sum())
    default = majority(pos_rem, total_rem)
    pred_bits[remaining] = default
    error = float(np.mean(pred_bits != y))
    return EvalResult(tuple(order), tuple(predictions), default, error, error + regularization * len(order))


def score_order(order: tuple[int, ...], train_result: EvalResult, rules: list[Rule], X: np.ndarray, y: np.ndarray) -> float:
    remaining = np.ones(X.shape[0], dtype=bool)
    pred_bits = np.zeros(X.shape[0], dtype=np.uint8)
    for rid, pred in zip(order, train_result.predictions):
        captured = remaining & coverage(rules[rid], X)
        pred_bits[captured] = pred
        remaining[captured] = False
    pred_bits[remaining] = train_result.default_pred
    return float(np.mean(pred_bits == y))


def one_rule_scores(
    rules: list[Rule],
    X: np.ndarray,
    y: np.ndarray,
    regularization: float,
    chunk_size: int,
) -> np.ndarray:
    n = X.shape[0]
    y_bool = y.astype(bool)
    scores = np.empty(len(rules), dtype=np.float64)
    for start in range(0, len(rules), chunk_size):
        chunk = rules[start : start + chunk_size]
        cover = np.ones((len(chunk), n), dtype=bool)
        for row, rule in enumerate(chunk):
            for f, v in zip(rule.features, rule.values):
                cover[row] &= X[:, f] == v
        total = cover.sum(axis=1)
        pos = cover @ y.astype(np.int32)
        pred_one = 2 * pos >= total
        mistakes_captured = np.where(pred_one, total - pos, pos)
        rem_total = n - total
        rem_pos = int(y.sum()) - pos
        pred_default = 2 * rem_pos >= rem_total
        mistakes_default = np.where(pred_default, rem_total - rem_pos, rem_pos)
        scores[start : start + len(chunk)] = (mistakes_captured + mistakes_default) / n + regularization
    return scores


def warm_start(
    rules: list[Rule],
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    regularization: float,
    chunk_size: int,
) -> EvalResult:
    scores = one_rule_scores(rules, X, y, regularization, chunk_size)
    order = tuple(int(i) for i in np.argsort(scores)[:max_depth])
    cache: dict[int, np.ndarray] = {}
    return eval_order(order, rules, X, y, regularization, cache)


def greedy_forward(
    rules: list[Rule],
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    regularization: float,
    chunk_size: int,
) -> EvalResult:
    """Sequential covering baseline: add the rule with best objective improvement."""
    n = X.shape[0]
    y_i = y.astype(np.int32)
    y_total = int(y_i.sum())
    order: list[int] = []
    used: set[int] = set()
    cache: dict[int, np.ndarray] = {}
    best = eval_order(tuple(order), rules, X, y, regularization, cache)

    for _ in range(max_depth):
        remaining = np.ones(n, dtype=bool)
        assigned_mistakes = 0
        for rid, pred in zip(best.order, best.predictions):
            if rid not in cache:
                cache[rid] = coverage(rules[rid], X)
            captured = remaining & cache[rid]
            total = int(captured.sum())
            pos = int(y_i[captured].sum())
            assigned_mistakes += (total - pos) if pred == 1 else pos
            remaining[captured] = False

        best_candidate = -1
        best_candidate_obj = best.objective
        for start in range(0, len(rules), chunk_size):
            chunk = rules[start : start + chunk_size]
            ids = np.arange(start, start + len(chunk))
            usable = np.array([int(i) not in used for i in ids], dtype=bool)
            if not usable.any():
                continue

            cover = np.ones((len(chunk), n), dtype=bool)
            for row, rule in enumerate(chunk):
                for f, v in zip(rule.features, rule.values):
                    cover[row] &= X[:, f] == v
            cover &= remaining
            total = cover.sum(axis=1).astype(np.int32)
            pos = cover @ y_i
            pred_one = 2 * pos >= total
            mistakes_captured = np.where(pred_one, total - pos, pos)
            rem_total = int(remaining.sum()) - total
            rem_pos = (y_i & remaining).sum() - pos
            pred_default = 2 * rem_pos >= rem_total
            mistakes_default = np.where(pred_default, rem_total - rem_pos, rem_pos)
            obj = (
                assigned_mistakes
                + mistakes_captured
                + mistakes_default
            ) / n + regularization * (len(order) + 1)
            obj[~usable] = np.inf

            local = int(np.argmin(obj))
            local_obj = float(obj[local])
            if local_obj < best_candidate_obj:
                best_candidate_obj = local_obj
                best_candidate = start + local

        if best_candidate < 0:
            break
        order.append(best_candidate)
        used.add(best_candidate)
        best = eval_order(tuple(order), rules, X, y, regularization, cache)
    return best


def propose(order: tuple[int, ...], n_rules: int, max_depth: int, rng: random.Random) -> tuple[int, ...]:
    nxt = list(order)
    used = set(nxt)
    move = rng.random()
    if move < 0.25 and nxt:
        del nxt[rng.randrange(len(nxt))]
    elif move < 0.50 and len(nxt) < max_depth:
        avail = [j for j in range(n_rules) if j not in used]
        if avail:
            nxt.insert(rng.randrange(len(nxt) + 1), rng.choice(avail))
    elif move < 0.80 and nxt:
        pos = rng.randrange(len(nxt))
        avail = [j for j in range(n_rules) if j not in used or j == nxt[pos]]
        if avail:
            nxt[pos] = rng.choice(avail)
    elif len(nxt) > 1:
        a, b = rng.sample(range(len(nxt)), 2)
        nxt[a], nxt[b] = nxt[b], nxt[a]
    return tuple(nxt)


def mh_polish(
    initial: EvalResult,
    rules: list[Rule],
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    regularization: float,
    beta: float,
    steps: int,
    seed: int,
) -> tuple[EvalResult, int]:
    rng = random.Random(seed)
    cache: dict[int, np.ndarray] = {}
    current = initial
    best = initial
    accepted = 0
    for _ in range(steps):
        cand_order = propose(current.order, len(rules), max_depth, rng)
        cand = eval_order(cand_order, rules, X, y, regularization, cache)
        delta = cand.objective - current.objective
        if delta <= 0 or rng.random() < math.exp(-beta * delta):
            current = cand
            accepted += 1
            if current.objective < best.objective:
                best = current
    return best, accepted


def run_case(name: str, cfg: dict, args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    problem = generate_problem(seed=args.seed, **cfg)
    gen_s = time.perf_counter() - t0

    rules = problem["rules"]
    X_train = problem["X_train"]
    y_train = problem["y_train"]
    X_test = problem["X_test"]
    y_test = problem["y_test"]

    t1 = time.perf_counter()
    warm = warm_start(rules, X_train, y_train, args.max_depth, args.regularization, args.chunk_size)
    warm_s = time.perf_counter() - t1
    warm_acc = score_order(warm.order, warm, rules, X_test, y_test)

    tg = time.perf_counter()
    greedy = greedy_forward(rules, X_train, y_train, args.max_depth, args.regularization, args.chunk_size)
    greedy_s = time.perf_counter() - tg
    greedy_acc = score_order(greedy.order, greedy, rules, X_test, y_test)

    t2 = time.perf_counter()
    mh_warm, accepted_warm = mh_polish(
        warm,
        rules,
        X_train,
        y_train,
        args.max_depth,
        args.regularization,
        args.mh_beta,
        args.mh_steps,
        args.seed + 17,
    )
    mh_warm_s = time.perf_counter() - t2
    mh_warm_acc = score_order(mh_warm.order, mh_warm, rules, X_test, y_test)

    rng = random.Random(args.seed + 31)
    random_order = tuple(rng.sample(range(len(rules)), min(args.max_depth, len(rules))))
    random_initial = eval_order(random_order, rules, X_train, y_train, args.regularization, {})
    random_initial_acc = score_order(random_initial.order, random_initial, rules, X_test, y_test)

    t3 = time.perf_counter()
    mh_random, accepted_random = mh_polish(
        random_initial,
        rules,
        X_train,
        y_train,
        args.max_depth,
        args.regularization,
        args.mh_beta,
        args.mh_steps,
        args.seed + 47,
    )
    mh_random_s = time.perf_counter() - t3
    mh_random_acc = score_order(mh_random.order, mh_random, rules, X_test, y_test)

    total_s = time.perf_counter() - t0
    true_ids = set(range(5))
    recovered_warm = len(true_ids & set(mh_warm.order))
    recovered_random = len(true_ids & set(mh_random.order))
    recovered_greedy = len(true_ids & set(greedy.order))
    return {
        "case": name,
        "n_train": cfg["n_train"],
        "n_test": cfg["n_test"],
        "n_features": cfg["n_features"],
        "n_candidates": cfg["n_candidates"],
        "max_cardinality": cfg["max_cardinality"],
        "noise": cfg["noise"],
        "warm_objective": f"{warm.objective:.6f}",
        "warm_test_accuracy": f"{warm_acc:.6f}",
        "greedy_objective": f"{greedy.objective:.6f}",
        "greedy_test_accuracy": f"{greedy_acc:.6f}",
        "random_initial_objective": f"{random_initial.objective:.6f}",
        "random_initial_test_accuracy": f"{random_initial_acc:.6f}",
        "mh_warm_objective": f"{mh_warm.objective:.6f}",
        "mh_warm_test_accuracy": f"{mh_warm_acc:.6f}",
        "mh_random_objective": f"{mh_random.objective:.6f}",
        "mh_random_test_accuracy": f"{mh_random_acc:.6f}",
        "mh_warm_accepted": accepted_warm,
        "mh_random_accepted": accepted_random,
        "recovered_true_rules_warm": recovered_warm,
        "recovered_true_rules_greedy": recovered_greedy,
        "recovered_true_rules_random": recovered_random,
        "generation_seconds": f"{gen_s:.3f}",
        "warm_seconds": f"{warm_s:.3f}",
        "greedy_seconds": f"{greedy_s:.3f}",
        "mh_warm_seconds": f"{mh_warm_s:.3f}",
        "mh_random_seconds": f"{mh_random_s:.3f}",
        "total_seconds": f"{total_s:.3f}",
        "warm_order": " ".join(str(x) for x in warm.order),
        "greedy_order": " ".join(str(x) for x in greedy.order),
        "random_initial_order": " ".join(str(x) for x in random_initial.order),
        "mh_warm_order": " ".join(str(x) for x in mh_warm.order),
        "mh_random_order": " ".join(str(x) for x in mh_random.order),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=0.005)
    parser.add_argument("--mh-beta", type=float, default=150.0)
    parser.add_argument("--mh-steps", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--out", default=str(RESULTS_DIR / "synthetic_scale_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.quick:
        cases = {
            "synthetic_20k_50k": dict(
                n_train=20_000, n_test=10_000, n_features=60,
                n_candidates=50_000, max_cardinality=3, noise=0.03,
            ),
            "synthetic_50k_100k": dict(
                n_train=50_000, n_test=20_000, n_features=100,
                n_candidates=100_000, max_cardinality=3, noise=0.03,
            ),
        }
    else:
        cases = {
            "synthetic_50k_100k": dict(
                n_train=50_000, n_test=20_000, n_features=100,
                n_candidates=100_000, max_cardinality=3, noise=0.03,
            ),
            "synthetic_100k_200k": dict(
                n_train=100_000, n_test=40_000, n_features=150,
                n_candidates=200_000, max_cardinality=3, noise=0.03,
            ),
        }

    rows = []
    for name, cfg in cases.items():
        print(f"\n{name}")
        print(
            f"  train={cfg['n_train']} test={cfg['n_test']} "
            f"features={cfg['n_features']} candidates={cfg['n_candidates']}"
        )
        row = run_case(name, cfg, args)
        rows.append(row)
        print(
            f"  warm_acc={row['warm_test_accuracy']} "
            f"greedy_acc={row['greedy_test_accuracy']} "
            f"mh_warm_acc={row['mh_warm_test_accuracy']} "
            f"mh_random_acc={row['mh_random_test_accuracy']} "
            f"total={row['total_seconds']}s "
            f"recovered_warm={row['recovered_true_rules_warm']}/5 "
            f"recovered_greedy={row['recovered_true_rules_greedy']}/5 "
            f"recovered_random={row['recovered_true_rules_random']}/5"
        )

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
