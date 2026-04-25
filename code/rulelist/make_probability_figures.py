#!/usr/bin/env python3
"""Create exploratory figures for Experiment 3 probability analysis."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

DATASETS = ["compas", "monks1", "tictactoe", "adult"]
LABELS = {
    "compas": "COMPAS",
    "monks1": "Monks-1",
    "tictactoe": "Tic-Tac-Toe",
    "adult": "Adult",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_order(text: str) -> tuple[int, ...]:
    if not text.strip():
        return tuple()
    return tuple(int(x) for x in text.split())


def jaccard_distance(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def make_landscape(prefix: str, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True)
    axes = axes.ravel()
    cmap = "viridis"
    all_acc = []
    for dataset in DATASETS:
        rows = read_rows(out_dir / f"{prefix}_{dataset}_samples.csv")
        all_acc.extend(float(r["test_accuracy"]) for r in rows)
    norm = Normalize(vmin=min(all_acc), vmax=max(all_acc))

    for ax, dataset in zip(axes, DATASETS):
        rows = read_rows(out_dir / f"{prefix}_{dataset}_samples.csv")
        best = min(rows, key=lambda r: float(r["objective"]))
        best_order = parse_order(best["order"])
        best_obj = float(best["objective"])

        grouped: dict[tuple[float, float, float], int] = Counter()
        for row in rows:
            order = parse_order(row["order"])
            x = jaccard_distance(order, best_order)
            y = float(row["objective"]) - best_obj
            acc = float(row["test_accuracy"])
            grouped[(round(x, 4), round(y, 6), round(acc, 6))] += 1

        xs = [k[0] for k in grouped]
        ys = [k[1] for k in grouped]
        accs = [k[2] for k in grouped]
        sizes = [16 + 10 * math.sqrt(v) for v in grouped.values()]

        sc = ax.scatter(xs, ys, c=accs, s=sizes, cmap=cmap, norm=norm, alpha=0.78, edgecolor="none")
        for gap in (0.01, 0.03, 0.05):
            ax.axhline(gap, color="0.75", lw=0.8, ls="--", zorder=0)
        ax.scatter([0], [0], marker="*", s=120, c="black", label="best")
        ax.set_title(LABELS[dataset])
        ax.set_xlim(-0.03, 1.03)
        ymax = max(0.055, max(ys) * 1.08 if ys else 0.055)
        ax.set_ylim(-0.003, ymax)
        ax.grid(ls=":", color="0.86")
        ax.set_xlabel("Structural distance from best list")
        ax.set_ylabel("Objective gap")

    cbar = fig.colorbar(sc, ax=axes.tolist(), orientation="horizontal", fraction=0.045, pad=0.13)
    cbar.set_label("Test accuracy")
    fig.suptitle("Near-optimal rule-list landscapes", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.91, wspace=0.24, hspace=0.38)
    out = out_dir / f"{prefix}_landscape_2x2.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def mass_thresholds(H: np.ndarray, levels: tuple[float, ...]) -> list[float]:
    vals = np.sort(H.ravel())[::-1]
    total = vals.sum()
    if total <= 0:
        return [0.0 for _ in levels]
    csum = np.cumsum(vals)
    thresholds = []
    for level in levels:
        idx = int(np.searchsorted(csum, level * total, side="left"))
        idx = min(idx, len(vals) - 1)
        thresholds.append(float(vals[idx]))
    return thresholds


def make_heatmap(prefix: str, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True)
    axes = axes.ravel()
    levels = (0.50, 0.80, 0.95)

    for ax, dataset in zip(axes, DATASETS):
        rows = read_rows(out_dir / f"{prefix}_{dataset}_samples.csv")
        best = min(rows, key=lambda r: float(r["objective"]))
        best_order = parse_order(best["order"])
        best_obj = float(best["objective"])

        xs = np.array([jaccard_distance(parse_order(r["order"]), best_order) for r in rows])
        ys = np.array([float(r["objective"]) - best_obj for r in rows])
        ymax = max(0.055, float(np.quantile(ys, 0.995)) * 1.08)
        H, xedges, yedges = np.histogram2d(
            xs,
            ys,
            bins=(36, 36),
            range=[[0.0, 1.0], [0.0, ymax]],
        )
        H = H.T
        H_smooth = smooth2d(H)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        mesh = ax.pcolormesh(xedges, yedges, H_smooth, cmap="magma", shading="auto")
        thresholds = mass_thresholds(H_smooth, levels)
        contour_levels = sorted(set(t for t in thresholds if t > 0))
        if contour_levels:
            cs = ax.contour(
                xcenters,
                ycenters,
                H_smooth,
                levels=contour_levels,
                colors="white",
                linewidths=1.2,
            )
            label_map = {thresholds[i]: f"{int(levels[i] * 100)}%" for i in range(len(levels))}
            fmt = {}
            for lev in cs.levels:
                nearest = min(thresholds, key=lambda t: abs(t - lev))
                fmt[lev] = label_map.get(nearest, "")
            ax.clabel(cs, inline=True, fontsize=8, fmt=fmt)
        ax.scatter([0], [0], marker="*", s=120, c="cyan", edgecolor="black", linewidth=0.8)
        ax.set_title(LABELS[dataset])
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Structural distance from best list")
        ax.set_ylabel("Objective gap")
        ax.grid(ls=":", color="white", alpha=0.18)

    cbar = fig.colorbar(mesh, ax=axes.tolist(), orientation="horizontal", fraction=0.045, pad=0.13)
    cbar.set_label("Sample density")
    fig.suptitle("Near-optimal policy mass contours", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.91, wspace=0.24, hspace=0.38)
    out = out_dir / f"{prefix}_heatmap_contours_2x2.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def smooth2d(H: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],
        dtype=float,
    )
    kernel /= kernel.sum()
    padded = np.pad(H, 1, mode="edge")
    out = np.zeros_like(H, dtype=float)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            out[i, j] = np.sum(padded[i : i + 3, j : j + 3] * kernel)
    return out


def make_rule_stability(prefix: str, out_dir: Path, top_n: int = 8) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes = axes.ravel()

    for ax, dataset in zip(axes, DATASETS):
        rows = read_rows(out_dir / f"{prefix}_{dataset}_rules.csv")[:top_n]
        labels = [short_rule_label(r["rule"]) for r in rows][::-1]
        vals = [float(r["inclusion_prob"]) for r in rows][::-1]
        pos1 = [float(r.get("pos1_prob", 0.0)) for r in rows][::-1]
        pos2 = [float(r.get("pos2_prob", 0.0)) for r in rows][::-1]
        pos3 = [float(r.get("pos3_prob", 0.0)) for r in rows][::-1]
        y = list(range(len(rows)))

        ax.barh(y, pos1, color="#4C78A8", label="pos. 1")
        ax.barh(y, pos2, left=pos1, color="#F58518", label="pos. 2")
        left3 = [a + b for a, b in zip(pos1, pos2)]
        ax.barh(y, pos3, left=left3, color="#54A24B", label="pos. 3")
        ax.plot(vals, y, "ko", ms=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, min(1.0, max(0.18, max(vals) * 1.08 if vals else 0.18)))
        ax.set_title(LABELS[dataset])
        ax.set_xlabel("Rule inclusion probability")
        ax.grid(axis="x", ls=":", color="0.86")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Rule stability and ordering across sampled near-optimal lists", fontsize=14)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.10, top=0.90, wspace=0.42, hspace=0.34)
    out = out_dir / f"{prefix}_rule_stability_2x2.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def short_rule_label(rule: str, max_part: int = 24) -> str:
    parts = rule.split(" & ")
    compact = []
    for part in parts[:3]:
        compact.append(part if len(part) <= max_part else part[: max_part - 1] + ".")
    if len(parts) > 3:
        compact.append("...")
    return "\n& ".join(compact)


def make_summary_bars(prefix: str, out_dir: Path) -> Path:
    rows = read_rows(out_dir / f"{prefix}_summary_all.csv")
    datasets = [LABELS[r["dataset"]] for r in rows]
    ambiguity = [float(r["ambiguous_share_025_075"]) for r in rows]
    pred_var = [float(r["prediction_variance_mean"]) for r in rows]
    obj_width = [float(r["objective_q95"]) - float(r["objective_q05"]) for r in rows]
    acc_width = [float(r["test_accuracy_q95"]) - float(r["test_accuracy_q05"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
    panels = [
        ("Ambiguous test-case share", ambiguity),
        ("Mean prediction variance", pred_var),
        ("Objective interval width", obj_width),
        ("Accuracy interval width", acc_width),
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for ax, (title, vals), color in zip(axes.ravel(), panels, colors):
        ax.bar(datasets, vals, color=color)
        ax.set_title(title)
        ax.grid(axis="y", ls=":", color="0.86")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Uncertainty summaries from sampled rule-list ensembles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / f"{prefix}_uncertainty_summary_2x2.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="probability_analysis_card2")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--top-n", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    outputs = [
        make_landscape(args.prefix, out_dir),
        make_heatmap(args.prefix, out_dir),
        make_rule_stability(args.prefix, out_dir, args.top_n),
        make_summary_bars(args.prefix, out_dir),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
