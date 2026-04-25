"""download_benchmark_data.py — Download and binarize the 4 benchmark datasets.

Datasets:
  compas   — CORELS COMPAS binary (already in data/corels_compas/, skip if present)
  monks1   — UCI Monks-1 (one-hot encode 6 categorical features)
  tictactoe — UCI Tic-Tac-Toe Endgame (one-hot encode 9 ternary board features)
  adult    — UCI Adult census income (threshold continuous + one-hot categorical)

Output: code/rulelist/data/<name>/train-binary.csv and test-binary.csv
"""

from __future__ import annotations

import csv
import os
import random
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = SCRIPT_DIR / "data"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _write_binary_csv(path: Path, feature_names: list, X: list, y: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(feature_names + ["label"])
        for row, label in zip(X, y):
            w.writerow(list(row) + [label])


def _split(X: list, y: list, test_frac: float = 0.20, seed: int = 20260424):
    rng = random.Random(seed)
    idx = list(range(len(y)))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(y) * test_frac)))
    test_set = set(idx[:n_test])
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for i, (row, label) in enumerate(zip(X, y)):
        if i in test_set:
            X_te.append(row); y_te.append(label)
        else:
            X_tr.append(row); y_tr.append(label)
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# COMPAS — already downloaded; just verify
# ---------------------------------------------------------------------------

def check_compas() -> bool:
    d = DATA_ROOT / "corels_compas"
    ok = (d / "compas_train-binary.csv").exists() and (d / "compas_test-binary.csv").exists()
    if ok:
        print("compas: already present")
    else:
        print("compas: missing — run download_corels_compas.py --binary-only")
    return ok


# ---------------------------------------------------------------------------
# Monks-1
# ---------------------------------------------------------------------------

def _monks_onehot(vals: list, ranges: list) -> list:
    """One-hot encode Monks features given per-feature value ranges."""
    out = []
    for v, rng in zip(vals, ranges):
        for r in rng:
            out.append(1 if int(v) == r else 0)
    return out


_MONKS_RANGES = [[1, 2, 3], [1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2]]
_MONKS_FEAT_NAMES = [
    f"a{fi+1}_eq_{v}"
    for fi, rng in enumerate(_MONKS_RANGES)
    for v in rng
]

_MONKS1_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"
_MONKS1_TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test"


def _parse_monks(path: Path):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            label = int(parts[0])
            feats = _monks_onehot(parts[1:7], _MONKS_RANGES)
            X.append(feats)
            y.append(label)
    return X, y


def download_monks1() -> None:
    out = DATA_ROOT / "monks1"
    out.mkdir(parents=True, exist_ok=True)

    tr_raw = out / "monks-1.train"
    te_raw = out / "monks-1.test"
    if not tr_raw.exists():
        _download(_MONKS1_TRAIN_URL, tr_raw)
    if not te_raw.exists():
        _download(_MONKS1_TEST_URL, te_raw)

    X_tr, y_tr = _parse_monks(tr_raw)
    X_te, y_te = _parse_monks(te_raw)

    _write_binary_csv(out / "train-binary.csv", _MONKS_FEAT_NAMES, X_tr, y_tr)
    _write_binary_csv(out / "test-binary.csv",  _MONKS_FEAT_NAMES, X_te, y_te)
    print(f"monks1: {len(y_tr)} train / {len(y_te)} test, {len(_MONKS_FEAT_NAMES)} features")


# ---------------------------------------------------------------------------
# Tic-Tac-Toe Endgame
# ---------------------------------------------------------------------------

_TTT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
_TTT_POSITIONS = [f"sq{i+1}" for i in range(9)]
_TTT_VALUES = ["x", "o", "b"]
_TTT_FEAT_NAMES = [f"{pos}_is_{v}" for pos in _TTT_POSITIONS for v in _TTT_VALUES]


def _parse_ttt(path: Path):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) < 10:
                continue
            feats = []
            for cell in parts[:9]:
                cell = cell.strip().lower()
                for v in _TTT_VALUES:
                    feats.append(1 if cell == v else 0)
            label = 1 if parts[9].strip().lower() == "positive" else 0
            X.append(feats)
            y.append(label)
    return X, y


def download_tictactoe() -> None:
    out = DATA_ROOT / "tictactoe"
    out.mkdir(parents=True, exist_ok=True)

    raw = out / "tic-tac-toe.data"
    if not raw.exists():
        _download(_TTT_URL, raw)

    X, y = _parse_ttt(raw)
    X_tr, y_tr, X_te, y_te = _split(X, y, test_frac=0.20)

    _write_binary_csv(out / "train-binary.csv", _TTT_FEAT_NAMES, X_tr, y_tr)
    _write_binary_csv(out / "test-binary.csv",  _TTT_FEAT_NAMES, X_te, y_te)
    print(f"tictactoe: {len(y_tr)} train / {len(y_te)} test, {len(_TTT_FEAT_NAMES)} features")


# ---------------------------------------------------------------------------
# Adult (Census Income)
# ---------------------------------------------------------------------------

_ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
_ADULT_TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

_ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "label",
]

# Thresholds for continuous features (inclusive lower bound of second bucket)
_CONT_THRESHOLDS = {
    "age": [25, 60],              # <25, 25-59, >=60
    "education_num": [10],         # <10, >=10
    "capital_gain": [1],           # =0 vs >0
    "capital_loss": [1],           # =0 vs >0
    "hours_per_week": [40],        # <40, >=40
}

# Categorical features: map to binary presence indicators
_CAT_VALUES = {
    "workclass": ["Private"],
    "marital_status": ["Married-civ-spouse", "Married-AF-spouse"],
    "occupation": ["Exec-managerial", "Prof-specialty", "Tech-support"],
    "relationship": ["Husband", "Wife"],
    "race": ["White"],
    "sex": ["Male"],
    "native_country": ["United-States"],
}

# Drop: fnlwgt, education (redundant with education_num)
_DROP_COLS = {"fnlwgt", "education"}


def _binarize_adult_row(row: dict) -> tuple[list, list]:
    """Convert one Adult row to binary features + label. Returns (features, names)."""
    feats = []
    names = []

    for col, thresholds in _CONT_THRESHOLDS.items():
        val = float(row[col])
        prev = 0.0
        for t in thresholds:
            names.append(f"{col}_lt_{t}")
            feats.append(1 if val < t else 0)
        # last bucket: >= last threshold (implicit, derived from others)

    for col, vals in _CAT_VALUES.items():
        raw = row[col].strip()
        for v in vals:
            names.append(f"{col}_is_{v.replace('-', '_').replace(' ', '_')}")
            feats.append(1 if raw == v else 0)

    label_raw = row["label"].strip().rstrip(".")
    label = 1 if ">50K" in label_raw else 0
    return feats, names, label


def _parse_adult(path: Path, skip_first_line: bool = False):
    X, y, feat_names = [], [], None
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if skip_first_line:
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line or line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 15:
            continue
        row = dict(zip(_ADULT_COLS, parts))
        if "?" in row.values():
            continue
        feats, names, label = _binarize_adult_row(row)
        if feat_names is None:
            feat_names = names
        X.append(feats)
        y.append(label)
    return X, y, feat_names or []


def download_adult() -> None:
    out = DATA_ROOT / "adult"
    out.mkdir(parents=True, exist_ok=True)

    tr_raw = out / "adult.data"
    te_raw = out / "adult.test"
    if not tr_raw.exists():
        _download(_ADULT_TRAIN_URL, tr_raw)
    if not te_raw.exists():
        _download(_ADULT_TEST_URL, te_raw)

    X_tr, y_tr, feat_names = _parse_adult(tr_raw, skip_first_line=False)
    X_te, y_te, _ = _parse_adult(te_raw, skip_first_line=True)

    _write_binary_csv(out / "train-binary.csv", feat_names, X_tr, y_tr)
    _write_binary_csv(out / "test-binary.csv",  feat_names, X_te, y_te)
    print(f"adult: {len(y_tr)} train / {len(y_te)} test, {len(feat_names)} features")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+",
        default=["compas", "monks1", "tictactoe", "adult"],
        choices=["compas", "monks1", "tictactoe", "adult"],
        help="which datasets to download/prepare",
    )
    args = parser.parse_args()

    print(f"Data root: {DATA_ROOT}")
    for name in args.datasets:
        print(f"\n--- {name} ---")
        if name == "compas":
            check_compas()
        elif name == "monks1":
            download_monks1()
        elif name == "tictactoe":
            download_tictactoe()
        elif name == "adult":
            download_adult()


if __name__ == "__main__":
    main()
