#!/usr/bin/env python3
"""Reproducible coverage and latency report for the deterministic control path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from hyperjev.control import (
    deterministic_control_action,
    explicit_stop_action,
    explicit_stop_signal,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError(f"{path}:{line_number}: queue row must be an object")
        state = raw.get("state")
        if not isinstance(state, str) or not state.strip():
            raise ValueError(f"{path}:{line_number}: state must be a non-empty string")
        rows.append(raw)
    if not rows:
        raise ValueError(f"queue is empty: {path}")
    return rows


def _human_target(row: dict[str, Any]) -> str | None:
    labels = row.get("labels")
    if not isinstance(labels, dict):
        return None
    human = labels.get("human")
    if isinstance(human, dict):
        selected = human.get("selected")
        return str(selected) if selected is not None and str(selected).strip() else None
    if isinstance(human, str) and human.strip():
        return human.strip()
    return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1)
    return round(ordered[index], 3)


def evaluate_fast_path(queue_path: str | Path) -> dict[str, Any]:
    """Evaluate rule-only decisions without invoking a model or reading targets."""

    path = Path(queue_path)
    rows = _load_rows(path)
    latencies_us: list[float] = []
    source_counts: dict[str, int] = {}
    resolved_count = 0
    synthetic_target_count = 0
    synthetic_correct_count = 0
    human_target_count = 0
    human_correct_count = 0

    for row in rows:
        started = time.perf_counter_ns()
        if explicit_stop_signal(row["state"]):
            action = explicit_stop_action()
        else:
            action = deterministic_control_action(row["state"])
        latencies_us.append((time.perf_counter_ns() - started) / 1000.0)
        source = action.source if action is not None else "model"
        source_counts[source] = source_counts.get(source, 0) + 1
        if action is None:
            continue
        resolved_count += 1
        target = row.get("target")
        if target is not None:
            synthetic_target_count += 1
            synthetic_correct_count += int(action.skill == str(target))
        human_target = _human_target(row)
        if human_target is not None:
            human_target_count += 1
            human_correct_count += int(action.skill == human_target)

    row_count = len(rows)
    human_label_gate = human_target_count == row_count
    report = {
        "record_type": "control_fast_path_evaluation",
        "queue": str(path.resolve()),
        "queue_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": row_count,
        "resolved_count": resolved_count,
        "unresolved_count": row_count - resolved_count,
        "coverage": round(resolved_count / row_count, 6),
        "source_counts": dict(sorted(source_counts.items())),
        "synthetic_target": {
            "labeled_count": synthetic_target_count,
            "correct_count": synthetic_correct_count,
            "accuracy": round(synthetic_correct_count / synthetic_target_count, 6)
            if synthetic_target_count
            else None,
        },
        "human_target": {
            "labeled_count": human_target_count,
            "correct_count": human_correct_count,
            "accuracy": round(human_correct_count / human_target_count, 6)
            if human_target_count
            else None,
        },
        "human_label_gate": human_label_gate,
        "production_ready": False,
        "latency_us": {
            "p50": _percentile(latencies_us, 50),
            "p95": _percentile(latencies_us, 95),
            "p99": _percentile(latencies_us, 99),
            "max": round(max(latencies_us), 3),
            "mean": round(sum(latencies_us) / len(latencies_us), 3),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-full-coverage", action="store_true")
    args = parser.parse_args()
    report = evaluate_fast_path(args.queue)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_full_coverage and report["unresolved_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
