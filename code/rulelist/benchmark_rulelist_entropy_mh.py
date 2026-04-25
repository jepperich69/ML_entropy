#!/usr/bin/env python3
"""Rule-list entropy/MH benchmark harness.

This is the Python path for the Pub_ML_Entropy classifier experiments.
It is dependency-free so it can run before we introduce OpenML/PMLB.

Features:
- synthetic smoke-test data,
- CORELS-style binary CSV loading,
- deterministic train/test split if only one CSV is provided,
- exact enumeration for small instances,
- entropy-weighted warm start,
- Metropolis-Hastings polishing,
- CSV result output.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Literal:
    feature: int
    value: int


@dataclass(frozen=True)
class Antecedent:
    id: int
    literals: Tuple[Literal, ...]
    name: str


@dataclass
class Dataset:
    name: str
    feature_names: List[str]
    X: List[List[int]]
    y: List[int]


@dataclass
class Model:
    order: List[int]
    predictions: List[int]
    default_pred: int


@dataclass
class EvalResult:
    model: Model
    error: float
    objective: float


def make_dataset(name: str, n: int, n_features: int, rng: random.Random) -> Dataset:
    X: List[List[int]] = []
    y: List[int] = []
    for _ in range(n):
        row = []
        for j in range(n_features):
            p = 0.35 + 0.08 * (j % 3)
            row.append(1 if rng.random() < p else 0)

        label = 0
        if row[0] == 1 and row[1] == 1:
            label = 1
        elif row[2] == 1 and row[3] == 0:
            label = 1
        elif row[4] == 1 and row[5] == 1:
            label = 0

        if rng.random() < 0.04:
            label = 1 - label

        X.append(row)
        y.append(label)

    return Dataset(name, [f"x{j}" for j in range(n_features)], X, y)


def _binary_value(raw: str, column: str) -> int:
    value = raw.strip()
    if value in {"0", "0.0", "false", "False", "FALSE", "no", "No", "NO"}:
        return 0
    if value in {"1", "1.0", "true", "True", "TRUE", "yes", "Yes", "YES"}:
        return 1
    raise ValueError(f"Column {column!r} must be binary, got {raw!r}")


def load_corels_csv(path: str, target: Optional[str] = None) -> Dataset:
    """Load a CORELS-style CSV: binary feature columns plus one binary target.

    If target is omitted, the last column is used. This accepts CORELS-style
    files where all values are 0/1, and also simple true/false variants.
    """

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        columns = list(reader.fieldnames)
        target_col = target or columns[-1]
        if target_col not in columns:
            raise ValueError(f"Target {target_col!r} not found in {path}")

        feature_names = [col for col in columns if col != target_col]
        X: List[List[int]] = []
        y: List[int] = []
        for row in reader:
            X.append([_binary_value(row[col], col) for col in feature_names])
            y.append(_binary_value(row[target_col], target_col))

    if not X:
        raise ValueError(f"{path} has no data rows")
    return Dataset(os.path.splitext(os.path.basename(path))[0], feature_names, X, y)


def split_dataset(data: Dataset, test_fraction: float, seed: int) -> Tuple[Dataset, Dataset]:
    rng = random.Random(seed)
    indices = list(range(len(data.y)))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(indices) * test_fraction)))
    test_idx = set(indices[:n_test])

    train_X, train_y, test_X, test_y = [], [], [], []
    for i, row in enumerate(data.X):
        if i in test_idx:
            test_X.append(row)
            test_y.append(data.y[i])
        else:
            train_X.append(row)
            train_y.append(data.y[i])

    train = Dataset(data.name + "_train", data.feature_names, train_X, train_y)
    test = Dataset(data.name + "_test", data.feature_names, test_X, test_y)
    return train, test


def make_antecedents(
    feature_names: Sequence[str],
    max_cardinality: int,
    max_antecedents: Optional[int] = None,
) -> List[Antecedent]:
    if max_cardinality < 1:
        raise ValueError("max_cardinality must be at least 1")

    feature_ids = list(range(len(feature_names)))
    raw: List[Tuple[Literal, ...]] = []

    for card in range(1, max_cardinality + 1):
        for feature_combo in itertools.combinations(feature_ids, card):
            for values in itertools.product((0, 1), repeat=card):
                lits = tuple(
                    Literal(feature, value)
                    for feature, value in zip(feature_combo, values)
                )
                raw.append(lits)
                if max_antecedents is not None and len(raw) > max_antecedents:
                    raise ValueError(
                        f"Antecedent pool exceeded limit ({max_antecedents}) "
                        f"while generating up to cardinality {max_cardinality}"
                    )

    antecedents = []
    for idx, lits in enumerate(raw):
        name = " & ".join(f"{feature_names[lit.feature]}={lit.value}" for lit in lits)
        antecedents.append(Antecedent(idx, lits, name))
    return antecedents


def covers(row: Sequence[int], antecedent: Antecedent) -> bool:
    return all(row[lit.feature] == lit.value for lit in antecedent.literals)


def majority_label(labels: Iterable[int]) -> int:
    values = list(labels)
    if not values:
        return 0
    return 1 if 2 * sum(values) >= len(values) else 0


def fit_predictions(order: Sequence[int], antecedents: Sequence[Antecedent], data: Dataset) -> Model:
    remaining = set(range(len(data.y)))
    predictions = []
    for rule_id in order:
        captured = [i for i in remaining if covers(data.X[i], antecedents[rule_id])]
        pred = majority_label(data.y[i] for i in captured)
        predictions.append(pred)
        for i in captured:
            remaining.remove(i)
    default_pred = majority_label(data.y[i] for i in remaining)
    return Model(list(order), predictions, default_pred)


def predict_one(row: Sequence[int], model: Model, antecedents: Sequence[Antecedent]) -> int:
    for pos, rule_id in enumerate(model.order):
        if covers(row, antecedents[rule_id]):
            return model.predictions[pos]
    return model.default_pred


def evaluate_order(
    order: Sequence[int],
    antecedents: Sequence[Antecedent],
    data: Dataset,
    regularization: float,
) -> EvalResult:
    model = fit_predictions(order, antecedents, data)
    mistakes = sum(
        1 for row, label in zip(data.X, data.y) if predict_one(row, model, antecedents) != label
    )
    error = mistakes / len(data.y)
    return EvalResult(model, error, error + regularization * len(order))


def score_model(model: Model, antecedents: Sequence[Antecedent], data: Dataset) -> float:
    correct = sum(
        1 for row, label in zip(data.X, data.y) if predict_one(row, model, antecedents) == label
    )
    return correct / len(data.y)


def exact_search(
    antecedents: Sequence[Antecedent],
    data: Dataset,
    max_depth: int,
    regularization: float,
) -> Tuple[EvalResult, int]:
    best = evaluate_order([], antecedents, data, regularization)
    checked = 1
    for depth in range(1, max_depth + 1):
        for order in itertools.permutations(range(len(antecedents)), depth):
            current = evaluate_order(order, antecedents, data, regularization)
            checked += 1
            if current.objective < best.objective:
                best = current
    return best, checked


def entropy_warm_start(
    antecedents: Sequence[Antecedent],
    data: Dataset,
    max_depth: int,
    regularization: float,
    beta: float,
) -> EvalResult:
    one_rule = [
        (ant.id, evaluate_order([ant.id], antecedents, data, regularization).objective)
        for ant in antecedents
    ]
    max_utility = max(-obj for _, obj in one_rule)
    weights = {rule_id: math.exp(beta * (-obj - max_utility)) for rule_id, obj in one_rule}

    order: List[int] = []
    used = set()
    while len(order) < max_depth:
        best_id = None
        best_score = -math.inf
        best_objective = math.inf
        for ant in antecedents:
            if ant.id in used:
                continue
            current = evaluate_order(order + [ant.id], antecedents, data, regularization)
            entropy_score = math.log(weights[ant.id] + 1e-300)
            score = -current.objective + 0.02 * entropy_score
            if score > best_score or (
                score == best_score and current.objective < best_objective
            ):
                best_id = ant.id
                best_score = score
                best_objective = current.objective
        if best_id is None:
            break
        order.append(best_id)
        used.add(best_id)
    return evaluate_order(order, antecedents, data, regularization)


def propose_move(
    order: Sequence[int],
    n_rules: int,
    max_depth: int,
    rng: random.Random,
) -> List[int]:
    next_order = list(order)
    used = set(next_order)
    move = rng.random()

    if move < 0.25 and next_order:
        del next_order[rng.randrange(len(next_order))]
        return next_order

    if move < 0.50 and len(next_order) < max_depth:
        available = [idx for idx in range(n_rules) if idx not in used]
        if available:
            next_order.insert(rng.randrange(len(next_order) + 1), rng.choice(available))
        return next_order

    if move < 0.80 and next_order:
        pos = rng.randrange(len(next_order))
        available = [idx for idx in range(n_rules) if idx not in used or idx == next_order[pos]]
        if available:
            next_order[pos] = rng.choice(available)
        return next_order

    if len(next_order) > 1:
        a, b = rng.sample(range(len(next_order)), 2)
        next_order[a], next_order[b] = next_order[b], next_order[a]
    return next_order


def mh_polish(
    initial: EvalResult,
    antecedents: Sequence[Antecedent],
    data: Dataset,
    max_depth: int,
    regularization: float,
    beta: float,
    steps: int,
    rng: random.Random,
) -> Tuple[EvalResult, int]:
    current = initial
    best = initial
    accepted = 0
    for _ in range(steps):
        proposal_order = propose_move(current.model.order, len(antecedents), max_depth, rng)
        proposal = evaluate_order(proposal_order, antecedents, data, regularization)
        delta = proposal.objective - current.objective
        if delta <= 0 or rng.random() < math.exp(-beta * delta):
            current = proposal
            accepted += 1
            if current.objective < best.objective:
                best = current
    return best, accepted


def format_model(model: Model, antecedents: Sequence[Antecedent]) -> str:
    parts = [
        f"if {antecedents[rule_id].name} then {model.predictions[pos]}"
        for pos, rule_id in enumerate(model.order)
    ]
    parts.append(f"else {model.default_pred}")
    return "; ".join(parts)


def write_results(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "train_objective",
        "train_error",
        "test_accuracy",
        "runtime_ms",
        "checked_or_steps",
        "accepted_moves",
        "model",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_row(
    dataset: str,
    method: str,
    result: EvalResult,
    test_accuracy: float,
    runtime_ms: int,
    checked_or_steps: str,
    accepted_moves: str,
    antecedents: Sequence[Antecedent],
) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "train_objective": f"{result.objective:.6f}",
        "train_error": f"{result.error:.6f}",
        "test_accuracy": f"{test_accuracy:.6f}",
        "runtime_ms": str(runtime_ms),
        "checked_or_steps": checked_or_steps,
        "accepted_moves": accepted_moves,
        "model": format_model(result.model, antecedents),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", help="CORELS-style binary training CSV")
    parser.add_argument("--test-csv", help="CORELS-style binary test CSV")
    parser.add_argument("--target", help="target column; defaults to last column")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-cardinality", type=int, default=2)
    parser.add_argument("--regularization", type=float, default=0.015)
    parser.add_argument("--warm-beta", type=float, default=35.0)
    parser.add_argument("--mh-beta", type=float, default=120.0)
    parser.add_argument("--mh-steps", type=int, default=5000)
    parser.add_argument("--exact-max-antecedents", type=int, default=90)
    parser.add_argument(
        "--out",
        default=os.path.join("code", "rulelist", "results", "benchmark_results.csv"),
        help="output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.train_csv:
        train_all = load_corels_csv(args.train_csv, args.target)
        if args.test_csv:
            train = train_all
            test = load_corels_csv(args.test_csv, args.target)
        else:
            train, test = split_dataset(train_all, args.test_fraction, args.seed)
    else:
        train = make_dataset("synthetic_rulelist_train", args.n_train, args.n_features, rng)
        test = make_dataset("synthetic_rulelist_test", args.n_test, args.n_features, rng)

    antecedents = make_antecedents(train.feature_names, args.max_cardinality)
    dataset_name = train.name.replace("_train", "")

    rows: List[dict] = []
    print("Pub_ML_Entropy Python rule-list benchmark")
    print("=========================================")
    print(f"dataset: {dataset_name}")
    print(f"train rows: {len(train.y)}, test rows: {len(test.y)}")
    print(f"features: {len(train.feature_names)}, antecedents: {len(antecedents)}")

    if len(antecedents) <= args.exact_max_antecedents:
        start = time.perf_counter()
        exact, checked = exact_search(
            antecedents, train, args.max_depth, args.regularization
        )
        exact_ms = int(round((time.perf_counter() - start) * 1000))
        exact_test = score_model(exact.model, antecedents, test)
        rows.append(
            result_row(
                dataset_name,
                "exact",
                exact,
                exact_test,
                exact_ms,
                str(checked),
                "",
                antecedents,
            )
        )
        print(f"exact objective: {exact.objective:.4f}, test accuracy: {exact_test:.4f}")
    else:
        exact = None
        print(
            f"exact skipped: {len(antecedents)} antecedents exceeds "
            f"--exact-max-antecedents={args.exact_max_antecedents}"
        )

    start = time.perf_counter()
    warm = entropy_warm_start(
        antecedents, train, args.max_depth, args.regularization, args.warm_beta
    )
    warm_ms = int(round((time.perf_counter() - start) * 1000))
    warm_test = score_model(warm.model, antecedents, test)
    rows.append(
        result_row(
            dataset_name,
            "entropy_warm_start",
            warm,
            warm_test,
            warm_ms,
            "",
            "",
            antecedents,
        )
    )

    start = time.perf_counter()
    mh, accepted = mh_polish(
        warm,
        antecedents,
        train,
        args.max_depth,
        args.regularization,
        args.mh_beta,
        args.mh_steps,
        rng,
    )
    mh_ms = int(round((time.perf_counter() - start) * 1000))
    mh_test = score_model(mh.model, antecedents, test)
    rows.append(
        result_row(
            dataset_name,
            "mh_polish",
            mh,
            mh_test,
            mh_ms,
            str(args.mh_steps),
            str(accepted),
            antecedents,
        )
    )

    print(f"warm objective:  {warm.objective:.4f}, test accuracy: {warm_test:.4f}")
    print(f"mh objective:    {mh.objective:.4f}, test accuracy: {mh_test:.4f}")
    if exact is not None:
        print(f"mh-exact objective gap: {mh.objective - exact.objective:.4f}")

    write_results(args.out, rows)
    print(f"results written: {args.out}")


if __name__ == "__main__":
    main()
