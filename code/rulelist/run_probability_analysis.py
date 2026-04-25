#!/usr/bin/env python3
"""PMIP-style probability analysis for sampled rule lists.

This script is intentionally separate from the Java benchmark runner. The Java
backend is optimized for incumbents; Experiment 3 needs thinned MH samples.
The implementation below uses Python integer bitsets for fast rule-list
evaluation while keeping the outputs simple CSV files.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_rulelist_entropy_mh import (  # noqa: E402
    Dataset,
    Model,
    format_model,
    load_corels_csv,
    make_antecedents,
)
from run_full_benchmark import DATASETS  # noqa: E402


@dataclass
class FastProblem:
    dataset: str
    train: Dataset
    test: Dataset
    antecedents: list
    train_masks: list[int]
    test_masks: list[int]
    train_pos: int
    test_pos: int
    train_all: int
    test_all: int
    regularization: float


@dataclass
class EvalFast:
    order: tuple[int, ...]
    predictions: tuple[int, ...]
    default_pred: int
    train_error: float
    objective: float
    test_accuracy: float
    test_pred_bits: int


def _mask_from_indices(indices: Iterable[int]) -> int:
    mask = 0
    for i in indices:
        mask |= 1 << i
    return mask


def _build_masks(antecedents, data: Dataset) -> list[int]:
    masks = []
    for ant in antecedents:
        mask = 0
        for i, row in enumerate(data.X):
            ok = True
            for lit in ant.literals:
                if row[lit.feature] != lit.value:
                    ok = False
                    break
            if ok:
                mask |= 1 << i
        masks.append(mask)
    return masks


def _majority_pred(pos_count: int, total_count: int) -> int:
    if total_count <= 0:
        return 0
    return 1 if 2 * pos_count >= total_count else 0


def evaluate_fast(order: Sequence[int], problem: FastProblem) -> EvalFast:
    remaining = problem.train_all
    mistakes = 0
    predictions: list[int] = []
    for rid in order:
        captured = remaining & problem.train_masks[rid]
        total = captured.bit_count()
        pos = (captured & problem.train_pos).bit_count()
        pred = _majority_pred(pos, total)
        predictions.append(pred)
        mistakes += (total - pos) if pred == 1 else pos
        remaining &= ~captured

    total_rem = remaining.bit_count()
    pos_rem = (remaining & problem.train_pos).bit_count()
    default_pred = _majority_pred(pos_rem, total_rem)
    mistakes += (total_rem - pos_rem) if default_pred == 1 else pos_rem

    train_error = mistakes / len(problem.train.y)
    objective = train_error + problem.regularization * len(order)

    pred_bits = predict_bits(order, predictions, default_pred, problem.test_masks, problem.test_all)
    test_correct = (~(pred_bits ^ problem.test_pos) & problem.test_all).bit_count()
    test_accuracy = test_correct / len(problem.test.y)

    return EvalFast(
        order=tuple(order),
        predictions=tuple(predictions),
        default_pred=default_pred,
        train_error=train_error,
        objective=objective,
        test_accuracy=test_accuracy,
        test_pred_bits=pred_bits,
    )


def predict_bits(
    order: Sequence[int],
    predictions: Sequence[int],
    default_pred: int,
    masks: Sequence[int],
    all_mask: int,
) -> int:
    remaining = all_mask
    pred_bits = all_mask if default_pred == 1 else 0
    for rid, pred in zip(order, predictions):
        captured = remaining & masks[rid]
        if pred == 1:
            pred_bits |= captured
        else:
            pred_bits &= ~captured
        remaining &= ~captured
    return pred_bits


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


def greedy_warm_start(problem: FastProblem, max_depth: int, beta: float) -> EvalFast:
    one_rule = [(j, evaluate_fast((j,), problem).objective) for j in range(len(problem.antecedents))]
    max_utility = max(-obj for _, obj in one_rule)
    weights = {j: math.exp(beta * (-obj - max_utility)) for j, obj in one_rule}

    order: list[int] = []
    used = set()
    while len(order) < max_depth:
        best_id = None
        best_score = -math.inf
        best_objective = math.inf
        for j in range(len(problem.antecedents)):
            if j in used:
                continue
            current = evaluate_fast(tuple(order + [j]), problem)
            entropy_score = math.log(weights[j] + 1e-300)
            score = -current.objective + 0.02 * entropy_score
            if score > best_score or (score == best_score and current.objective < best_objective):
                best_id = j
                best_score = score
                best_objective = current.objective
        if best_id is None:
            break
        order.append(best_id)
        used.add(best_id)
    return evaluate_fast(tuple(order), problem)


def run_chain(
    problem: FastProblem,
    initial: EvalFast,
    max_depth: int,
    beta: float,
    burn: int,
    samples: int,
    thin: int,
    seed: int,
) -> tuple[list[EvalFast], EvalFast, float]:
    rng = random.Random(seed)
    current = initial
    best = initial
    accepted = 0
    proposed = 0
    kept: list[EvalFast] = []

    total_steps = burn + samples * thin
    for step in range(total_steps):
        cand_order = propose(current.order, len(problem.antecedents), max_depth, rng)
        cand = evaluate_fast(cand_order, problem)
        delta = cand.objective - current.objective
        if delta <= 0 or rng.random() < math.exp(-beta * delta):
            current = cand
            accepted += 1
            if current.objective < best.objective:
                best = current
        proposed += 1
        if step >= burn and (step - burn + 1) % thin == 0:
            kept.append(current)
    return kept, best, accepted / max(proposed, 1)


def quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_samples(problem: FastProblem, sampled: list[EvalFast], best: EvalFast, prefix: Path) -> None:
    n_samples = len(sampled)
    n_rules = len(problem.antecedents)
    max_depth = max((len(s.order) for s in sampled), default=0)

    inclusion = [0] * n_rules
    positions = [[0] * max_depth for _ in range(n_rules)]
    pred_counts = [0] * len(problem.test.y)
    obj = []
    test_acc = []
    train_err = []
    rule_count = []

    for sample in sampled:
        obj.append(sample.objective)
        test_acc.append(sample.test_accuracy)
        train_err.append(sample.train_error)
        rule_count.append(len(sample.order))
        for pos, rid in enumerate(sample.order):
            inclusion[rid] += 1
            if pos < max_depth:
                positions[rid][pos] += 1
        bits = sample.test_pred_bits
        for i in range(len(problem.test.y)):
            if (bits >> i) & 1:
                pred_counts[i] += 1

    with (prefix.with_name(prefix.name + "_summary.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "n_samples", "best_objective", "best_test_accuracy",
            "objective_mean", "objective_q05", "objective_q95",
            "test_accuracy_mean", "test_accuracy_q05", "test_accuracy_q95",
            "train_error_mean", "rule_count_mean",
            "prediction_variance_mean", "prediction_variance_q95",
            "ambiguous_share_025_075",
            "positive_rate_y0_q05", "positive_rate_y0_q95",
            "positive_rate_y1_q05", "positive_rate_y1_q95",
        ])
        writer.writeheader()
        probs = [c / n_samples for c in pred_counts]
        variances = [p * (1 - p) for p in probs]
        ambiguous = sum(1 for p in probs if 0.25 <= p <= 0.75) / len(probs)
        group_rates = {0: [], 1: []}
        group_masks = {
            y: _mask_from_indices(i for i, yy in enumerate(problem.test.y) if yy == y)
            for y in (0, 1)
        }
        for sample in sampled:
            for y in (0, 1):
                denom = group_masks[y].bit_count()
                group_rates[y].append((sample.test_pred_bits & group_masks[y]).bit_count() / denom if denom else 0.0)
        writer.writerow({
            "dataset": problem.dataset,
            "n_samples": n_samples,
            "best_objective": f"{best.objective:.6f}",
            "best_test_accuracy": f"{best.test_accuracy:.6f}",
            "objective_mean": f"{sum(obj)/n_samples:.6f}",
            "objective_q05": f"{quantile(obj, 0.05):.6f}",
            "objective_q95": f"{quantile(obj, 0.95):.6f}",
            "test_accuracy_mean": f"{sum(test_acc)/n_samples:.6f}",
            "test_accuracy_q05": f"{quantile(test_acc, 0.05):.6f}",
            "test_accuracy_q95": f"{quantile(test_acc, 0.95):.6f}",
            "train_error_mean": f"{sum(train_err)/n_samples:.6f}",
            "rule_count_mean": f"{sum(rule_count)/n_samples:.6f}",
            "prediction_variance_mean": f"{sum(variances)/len(variances):.6f}",
            "prediction_variance_q95": f"{quantile(variances, 0.95):.6f}",
            "ambiguous_share_025_075": f"{ambiguous:.6f}",
            "positive_rate_y0_q05": f"{quantile(group_rates[0], 0.05):.6f}",
            "positive_rate_y0_q95": f"{quantile(group_rates[0], 0.95):.6f}",
            "positive_rate_y1_q05": f"{quantile(group_rates[1], 0.05):.6f}",
            "positive_rate_y1_q95": f"{quantile(group_rates[1], 0.95):.6f}",
        })

    rows = []
    for rid, count in enumerate(inclusion):
        p = count / n_samples
        if p <= 0:
            continue
        row = {
            "dataset": problem.dataset,
            "rule_id": rid,
            "rule": problem.antecedents[rid].name,
            "inclusion_prob": f"{p:.6f}",
            "inclusion_variance": f"{p * (1 - p):.6f}",
        }
        for pos in range(max_depth):
            row[f"pos{pos+1}_prob"] = f"{positions[rid][pos] / n_samples:.6f}"
        rows.append(row)
    rows.sort(key=lambda r: float(r["inclusion_prob"]), reverse=True)
    fieldnames = ["dataset", "rule_id", "rule", "inclusion_prob", "inclusion_variance"] + [
        f"pos{pos+1}_prob" for pos in range(max_depth)
    ]
    with (prefix.with_name(prefix.name + "_rules.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (prefix.with_name(prefix.name + "_samples.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "sample", "objective", "train_error", "test_accuracy",
            "rule_count", "order", "model",
        ])
        writer.writeheader()
        for idx, sample in enumerate(sampled, 1):
            model = Model(list(sample.order), list(sample.predictions), sample.default_pred)
            writer.writerow({
                "dataset": problem.dataset,
                "sample": idx,
                "objective": f"{sample.objective:.6f}",
                "train_error": f"{sample.train_error:.6f}",
                "test_accuracy": f"{sample.test_accuracy:.6f}",
                "rule_count": len(sample.order),
                "order": " ".join(str(x) for x in sample.order),
                "model": format_model(model, problem.antecedents),
            })


def load_problem(name: str, max_cardinality: int, regularization: float, max_antecedents: int | None) -> FastProblem:
    cfg = DATASETS[name]
    train = load_corels_csv(str(cfg["train"]), cfg["target"])
    test = load_corels_csv(str(cfg["test"]), cfg["target"])
    antecedents = make_antecedents(train.feature_names, max_cardinality, max_antecedents=max_antecedents)
    train_masks = _build_masks(antecedents, train)
    test_masks = _build_masks(antecedents, test)
    train_pos = _mask_from_indices(i for i, y in enumerate(train.y) if y == 1)
    test_pos = _mask_from_indices(i for i, y in enumerate(test.y) if y == 1)
    return FastProblem(
        dataset=name,
        train=train,
        test=test,
        antecedents=antecedents,
        train_masks=train_masks,
        test_masks=test_masks,
        train_pos=train_pos,
        test_pos=test_pos,
        train_all=(1 << len(train.y)) - 1,
        test_all=(1 << len(test.y)) - 1,
        regularization=regularization,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["compas", "monks1", "tictactoe", "adult"], choices=list(DATASETS))
    parser.add_argument("--max-cardinality", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-antecedents", type=int, default=250000)
    parser.add_argument("--regularization", type=float, default=0.015)
    parser.add_argument("--warm-beta", type=float, default=35.0)
    parser.add_argument("--mh-beta", type=float, default=120.0)
    parser.add_argument("--burn", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--out-prefix", default=str(RESULTS_DIR / "probability_analysis_card2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(args.out_prefix)
    combined_rows = []
    for offset, name in enumerate(args.datasets):
        print(f"\n{name}: loading card-{args.max_cardinality} problem")
        t0 = time.perf_counter()
        problem = load_problem(name, args.max_cardinality, args.regularization, args.max_antecedents)
        print(
            f"  {len(problem.train.y)} train / {len(problem.test.y)} test / "
            f"{len(problem.antecedents)} antecedents"
        )
        warm = greedy_warm_start(problem, args.max_depth, args.warm_beta)
        print(f"  warm: obj={warm.objective:.4f} test_acc={warm.test_accuracy:.4f}")
        sampled, best, acc_rate = run_chain(
            problem=problem,
            initial=warm,
            max_depth=args.max_depth,
            beta=args.mh_beta,
            burn=args.burn,
            samples=args.samples,
            thin=args.thin,
            seed=args.seed + offset,
        )
        prefix = base.with_name(f"{base.name}_{name}")
        summarize_samples(problem, sampled, best, prefix)
        elapsed = time.perf_counter() - t0
        print(
            f"  sampled={len(sampled)} acc_rate={acc_rate:.3f} "
            f"best_obj={best.objective:.4f} best_test_acc={best.test_accuracy:.4f} "
            f"elapsed={elapsed:.1f}s"
        )
        summary_path = prefix.with_name(prefix.name + "_summary.csv")
        with summary_path.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
            combined_rows.append(row)

    combined_path = base.with_name(base.name + "_summary_all.csv")
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"\ncombined summary: {combined_path}")


if __name__ == "__main__":
    main()
