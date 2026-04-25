# Mapping the Near-Optimal Space of Transparent Classifiers

Code for:

> Rich, J. (2026). Mapping the Near-Optimal Space of Transparent Classifiers.
> *Computers & Operations Research* (under review).

---

## What this paper does

Learning a transparent classifier — a rule list, rule set, sparse tree, or sparse
linear model — is a combinatorial search problem. The standard question is *which
classifier is best*. This paper asks a different question: **what does the space
of good transparent classifiers look like?**

We propose a three-stage pipeline:

1. **Boltzmann soft weighting** — assign Gibbs weights to candidate conditions
   based on their individual training quality.
2. **KL-optimal integerization** — select exactly *K* conditions to minimize
   information loss relative to the soft baseline (warm start).
3. **Metropolis–Hastings polish and sample** — improve the warm start by local
   search and retain the MH chain to characterize the near-optimal neighborhood.

From the retained chain we estimate rule inclusion probabilities, prediction
uncertainty at individual test cases, and coverage stability across demographic
groups. This probability-enabled analysis reveals structure that the single best
classifier does not: near-optimality can be concentrated in one stable policy or
spread across many substitutable alternatives.

---

## Repository layout

```
reproduce.py              Master reproduction script (runs all four experiments)
requirements.txt          Python dependencies
code/rulelist/
  benchmark_rulelist_entropy_mh.py   Core algorithm (warm start + MH sampler)
  run_full_benchmark.py              Experiment 1 — public benchmarks
  run_warmstart_stress_grid.py       Experiment 2 — warm-start stress test
  run_synthetic_scale.py             Experiment 3 — synthetic large-scale
  run_probability_analysis.py        Experiment 4 — probability-enabled analysis
  make_probability_figures.py        Figure 1 — near-optimal landscape
  download_benchmark_data.py         Download and binarize all four datasets
  java_bridge_rulelist.py            Optional Java backend bridge
  summarize_full_benchmark.py        CSV → markdown summary helper
```

Results are written to `code/rulelist/results/`.

---

## Quick start

```bash
pip install -r requirements.txt
python reproduce.py          # downloads data and runs all four experiments
```

Run a single experiment:

```bash
python reproduce.py --experiment 1   # public benchmarks (Tables 2-3)
python reproduce.py --experiment 2   # warm-start stress test (Table 4)
python reproduce.py --experiment 3   # synthetic large-scale (Table 5)
python reproduce.py --experiment 4   # probability-enabled analysis (Table 6 + Figure 1)
```

Skip the Java backend (slower, but no JVM required):

```bash
python reproduce.py --skip-java
```

Skip data download if `code/rulelist/data/` is already populated:

```bash
python reproduce.py --skip-download
```

---

## Datasets

All four benchmark datasets are downloaded automatically by `reproduce.py` (or
`python code/rulelist/download_benchmark_data.py`):

| Dataset | Train | Test | Source |
|---|---|---|---|
| COMPAS | 6,489 | 721 | CORELS repository / ProPublica |
| Monks-1 | 124 | 432 | UCI Machine Learning Repository |
| Tic-Tac-Toe | 766 | 192 | UCI Machine Learning Repository |
| Adult | 30,162 | 15,060 | UCI Machine Learning Repository |

---

## Algorithm parameters

All paper results use the following settings (Table 5 of the manuscript):

| Parameter | Value |
|---|---|
| Regularization weight λ | 0.015 |
| Warm-start inverse temperature β_warm | 35.0 |
| MH polish inverse temperature β_MH | 120.0 |
| Random seed | 20,260,423 |

---

## Requirements

- Python 3.10+
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- Java (optional — accelerates Experiments 1 and 2)

---

## Citation

```bibtex
@article{rich2026transparent,
  author  = {Rich, Jeppe},
  title   = {Mapping the Near-Optimal Space of Transparent Classifiers},
  journal = {Computers \& Operations Research},
  year    = {2026},
  note    = {Under review}
}
```
