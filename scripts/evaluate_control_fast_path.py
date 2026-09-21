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
    ControlObservation,
    ControlStudentClient,
    deterministic_control_action,
    explicit_stop_action,
    explicit_stop_signal,
)
from hyperjev.registry import TaskRegistry


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


def _latency_report(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": round(max(values), 3) if values else None,
        "mean": round(sum(values) / len(values), 3) if values else None,
    }


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
        "latency_us": _latency_report(latencies_us),
    }
    return report


def evaluate_runtime(
    queue_path: str | Path,
    checkpoint_path: str | Path,
    registry_path: str | Path = "registry/control_tasks",
    *,
    device: str = "cpu",
    enable_fast_path: bool = True,
    warmup_count: int = 0,
    synchronize_cuda: bool = False,
) -> dict[str, Any]:
    """Replay the integrated rule -> Student -> safety runtime on a queue."""

    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative")
    if synchronize_cuda and device != "cuda":
        raise ValueError("synchronize_cuda requires device='cuda'")
    queue = Path(queue_path)
    checkpoint = Path(checkpoint_path)
    rows = _load_rows(queue)
    registry = TaskRegistry.load(registry_path)
    client = ControlStudentClient(
        checkpoint,
        registry,
        enable_fast_path=enable_fast_path,
        device=device,
    )
    torch = client._client._torch if synchronize_cuda else None

    def _synchronize() -> None:
        if torch is not None:
            torch.cuda.synchronize()

    warmup_rows = rows[: min(warmup_count, len(rows))]
    for row in warmup_rows:
        client.decide(
            ControlObservation(
                observation_id=f"warmup-{row.get('sample_id', 'runtime-row')}",
                state=row["state"],
                domain=str(row.get("domain", "control-evaluation")),
                timestamp_ms=0.0,
            ),
            now_ms=1.0,
        )
    _synchronize()
    latencies_us: list[float] = []
    source_counts: dict[str, int] = {}
    source_latencies_us: dict[str, list[float]] = {}
    synthetic_correct_count = 0
    synthetic_target_count = 0
    human_correct_count = 0
    human_target_count = 0
    stop_target_count = 0
    stop_hit_count = 0

    for row in rows:
        observation = ControlObservation(
            observation_id=str(row.get("sample_id", "runtime-row")),
            state=row["state"],
            domain=str(row.get("domain", "control-evaluation")),
            timestamp_ms=0.0,
        )
        _synchronize()
        started = time.perf_counter_ns()
        action = client.decide(observation, now_ms=1.0)
        _synchronize()
        latencies_us.append((time.perf_counter_ns() - started) / 1000.0)
        source_counts[action.source] = source_counts.get(action.source, 0) + 1
        source_latencies_us.setdefault(action.source, []).append(latencies_us[-1])
        target = row.get("target")
        if target is not None:
            synthetic_target_count += 1
            synthetic_correct_count += int(action.skill == str(target))
            if str(target) == "STOP":
                stop_target_count += 1
                stop_hit_count += int(action.skill == "STOP")
        human_target = _human_target(row)
        if human_target is not None:
            human_target_count += 1
            human_correct_count += int(action.skill == human_target)

    row_count = len(rows)
    human_label_gate = human_target_count == row_count
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "runtime_mode": "integrated" if enable_fast_path else "model_only_with_safety_policy",
        "warmup_count": len(warmup_rows),
        "cuda_synchronized": synchronize_cuda,
        "row_count": row_count,
        "source_counts": dict(sorted(source_counts.items())),
        "synthetic_target": {
            "labeled_count": synthetic_target_count,
            "correct_count": synthetic_correct_count,
            "accuracy": round(synthetic_correct_count / synthetic_target_count, 6)
            if synthetic_target_count
            else None,
            "stop_recall": round(stop_hit_count / stop_target_count, 6)
            if stop_target_count
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
        "latency_us": _latency_report(latencies_us),
        "latency_by_source_us": {
            source: _latency_report(values)
            for source, values in sorted(source_latencies_us.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="also replay the integrated runtime")
    parser.add_argument("--registry", type=Path, default=Path("registry/control_tasks"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="run this many queue rows before recording latency",
    )
    parser.add_argument(
        "--cuda-sync",
        action="store_true",
        help="synchronize CUDA before and after each measured decision",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="disable deterministic control rules while keeping safety policy",
    )
    parser.add_argument("--require-full-coverage", action="store_true")
    args = parser.parse_args()
    report = evaluate_fast_path(args.queue)
    if args.checkpoint:
        report["runtime"] = evaluate_runtime(
            args.queue,
            args.checkpoint,
            args.registry,
            device=args.device,
            enable_fast_path=not args.model_only,
            warmup_count=args.warmup,
            synchronize_cuda=args.cuda_sync,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_full_coverage and report["unresolved_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
