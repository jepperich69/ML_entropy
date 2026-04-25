"""java_bridge_rulelist.py — Python bridge for the Java rule-list MH backend.

Protocol (file-based, same pattern as Pub_SAA_PMIP_MC):
  request dir  : meta.properties, coverage_matrix.csv, labels.csv, initial_order.csv
  response dir : summary.json, best_order.csv, objective_trace.csv
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence

_JAR_PATH = Path(
    r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet"
    r"\JR\Publikationer\Pub_SAA_PMIP_MC"
    r"\github_repo\java_backend\build\libs\pmip-java-backend.jar"
)


def _find_java() -> str:
    jh = os.environ.get("JAVA_HOME")
    if jh:
        p = Path(jh) / "bin" / "java.exe"
        if p.exists():
            return str(p)
    default = Path(r"C:\Program Files\Eclipse Adoptium\jdk-21.0.10.7-hotspot\bin\java.exe")
    if default.exists():
        return str(default)
    return "java"


def java_available() -> bool:
    return _JAR_PATH.exists()


def _require_jar() -> None:
    if not _JAR_PATH.exists():
        raise FileNotFoundError(
            f"Java backend jar not found: {_JAR_PATH}\n"
            "Rebuild in Pub_SAA_PMIP_MC/github_repo/java_backend/build.ps1"
        )


def _write_int_matrix(path: Path, matrix: List[List[int]]) -> None:
    lines = [",".join(str(v) for v in row) for row in matrix]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_int_vector(path: Path, v: Sequence[int]) -> None:
    path.write_text(",".join(str(x) for x in v) + "\n", encoding="utf-8")


def run_warm_start_rulelist(
    coverage_matrix: List[List[int]],
    labels: List[int],
    warm_beta: float = 35.0,
    regularization: float = 0.015,
    max_depth: int = 3,
    seed: int = 20260424,
) -> dict:
    """Run the greedy entropy warm start via Java.

    Returns dict with keys: warm_order (list[int]), warm_objective (float), elapsed (float).
    """
    _require_jar()

    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "req"
        req.mkdir()
        resp = Path(tmp) / "resp"

        meta = (
            f"warm_beta={warm_beta}\n"
            f"regularization={regularization:.17g}\n"
            f"max_depth={max_depth}\n"
            f"seed={seed}\n"
        )
        (req / "meta.properties").write_text(meta, encoding="utf-8")
        _write_int_matrix(req / "coverage_matrix.csv", coverage_matrix)
        _write_int_vector(req / "labels.csv", labels)

        cmd = [_find_java(), "-jar", str(_JAR_PATH), "run-warm-start-rulelist", str(req), str(resp)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        summary = json.loads((resp / "summary.json").read_text(encoding="utf-8"))
        order_text = (resp / "warm_order.csv").read_text(encoding="utf-8").strip()
        warm_order = [int(x) for x in order_text.split(",") if x.strip()] if order_text else []
        summary["warm_order"] = warm_order
        return summary


def run_exact_rulelist(
    coverage_matrix: List[List[int]],
    labels: List[int],
    regularization: float = 0.015,
    max_depth: int = 3,
    time_limit_ms: int = 300_000,
    test_coverage_matrix: List[List[int]] = None,
    test_labels: List[int] = None,
) -> dict:
    """Exact enumeration of all ordered subsets (size 1..max_depth) via Java.

    Returns dict with keys: best_order, best_objective, elapsed, complete,
    and optionally best_order_test, best_test_accuracy.
    """
    _require_jar()

    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "req"
        req.mkdir()
        resp = Path(tmp) / "resp"

        meta = (
            f"regularization={regularization:.17g}\n"
            f"max_depth={max_depth}\n"
            f"time_limit_ms={time_limit_ms}\n"
        )
        (req / "meta.properties").write_text(meta, encoding="utf-8")
        _write_int_matrix(req / "coverage_matrix.csv", coverage_matrix)
        _write_int_vector(req / "labels.csv", labels)
        if test_coverage_matrix is not None and test_labels is not None:
            _write_int_matrix(req / "test_coverage_matrix.csv", test_coverage_matrix)
            _write_int_vector(req / "test_labels.csv", test_labels)

        cmd = [_find_java(), "-jar", str(_JAR_PATH), "run-exact-rulelist", str(req), str(resp)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        summary = json.loads((resp / "summary.json").read_text(encoding="utf-8"))

        order_text = (resp / "best_order.csv").read_text(encoding="utf-8").strip()
        summary["best_order"] = [int(x) for x in order_text.split(",") if x.strip()] if order_text else []

        test_order_path = resp / "best_order_test.csv"
        if test_order_path.exists():
            t = test_order_path.read_text(encoding="utf-8").strip()
            summary["best_order_test"] = [int(x) for x in t.split(",") if x.strip()] if t else []

        return summary


def run_mh_rulelist(
    coverage_matrix: List[List[int]],           # [nAntecedents][nTrain], 0/1
    labels: List[int],                           # [nTrain], 0/1
    initial_order: List[int],                    # rule IDs in starting order
    beta: float = 120.0,
    t_burn: int = 5_000,
    t_sample: int = 20_000,
    thin: int = 5,
    regularization: float = 0.015,
    max_depth: int = 3,
    seed: int = 20260424,
    test_coverage_matrix: List[List[int]] = None,  # optional, for dual-incumbent
    test_labels: List[int] = None,
) -> dict:
    """Run variable-length sequence MH on a rule-list problem via Java.

    Returns dict with keys:
      best_order (list[int])        — best by training objective
      best_order_test (list[int])   — best by test accuracy (if test data provided)
      best_objective (float)
      best_test_accuracy (float)    — if test data provided
      acc_rate (float), elapsed (float)
    """
    _require_jar()

    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "req"
        req.mkdir()
        resp = Path(tmp) / "resp"

        meta = (
            f"beta={beta}\n"
            f"t_burn={t_burn}\n"
            f"t_sample={t_sample}\n"
            f"thin={thin}\n"
            f"seed={seed}\n"
            f"regularization={regularization:.17g}\n"
            f"max_depth={max_depth}\n"
        )
        (req / "meta.properties").write_text(meta, encoding="utf-8")
        _write_int_matrix(req / "coverage_matrix.csv", coverage_matrix)
        _write_int_vector(req / "labels.csv", labels)
        if initial_order:
            _write_int_vector(req / "initial_order.csv", initial_order)
        if test_coverage_matrix is not None and test_labels is not None:
            _write_int_matrix(req / "test_coverage_matrix.csv", test_coverage_matrix)
            _write_int_vector(req / "test_labels.csv", test_labels)

        cmd = [_find_java(), "-jar", str(_JAR_PATH), "run-mh-rulelist", str(req), str(resp)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr if stderr else stdout
            raise RuntimeError(
                "Java rule-list MH failed"
                + (f": {detail}" if detail else "")
            ) from exc

        summary = json.loads((resp / "summary.json").read_text(encoding="utf-8"))

        order_text = (resp / "best_order.csv").read_text(encoding="utf-8").strip()
        summary["best_order"] = [int(x) for x in order_text.split(",") if x.strip()] if order_text else []

        test_order_path = resp / "best_order_test.csv"
        if test_order_path.exists():
            t = test_order_path.read_text(encoding="utf-8").strip()
            summary["best_order_test"] = [int(x) for x in t.split(",") if x.strip()] if t else []

        return summary
