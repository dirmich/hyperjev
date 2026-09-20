#!/usr/bin/env python3
"""Evaluator-gated control quality report for the reference Student path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hyperjev.control import ControlObservation, ControlSafetyPolicy, ControlStudentClient
from hyperjev.registry import TaskRegistry
from hyperjev.student_inference import evaluate_student_checkpoint


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError(f"{path}:{line_number}: scenario must be an object")
        scenarios.append(raw)
    if not scenarios:
        raise ValueError(f"scenario file is empty: {path}")
    return scenarios


def _safety_report(checkpoint: Path, registry: TaskRegistry, scenario_path: Path) -> dict[str, Any]:
    client = ControlStudentClient(
        checkpoint,
        registry,
        policy=ControlSafetyPolicy(minimum_confidence=0.90),
        device="cpu",
    )
    rows = []
    for raw in _load_scenarios(scenario_path):
        observation = ControlObservation.from_dict(raw["observation"])
        action = client.decide(observation, now_ms=float(raw["now_ms"]))
        expected_safe_stop = bool(raw.get("expected_safe_stop", raw["expected_skill"] == "STOP"))
        rows.append(
            {
                "scenario_id": raw["scenario_id"],
                "correct": action.skill == raw["expected_skill"]
                and (raw.get("expected_reason") is None or action.reason == raw["expected_reason"]),
                "expected_safe_stop": expected_safe_stop,
                "safe_stop": action.skill == "STOP" and action.abstained,
            }
        )
    expected_stops = [row for row in rows if row["expected_safe_stop"]]
    stop_hits = sum(row["safe_stop"] for row in expected_stops)
    return {
        "count": len(rows),
        "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 6),
        "expected_safe_stop_count": len(expected_stops),
        "safe_stop_recall": round(stop_hits / len(expected_stops), 6) if expected_stops else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("tests/golden/control_smoke.jsonl"))
    parser.add_argument("--scenarios", type=Path, default=Path("tests/golden/control_scenarios.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path("registry/control_tasks"))
    parser.add_argument("--minimum-split-accuracy", type=float, default=0.99)
    parser.add_argument("--minimum-safe-stop-recall", type=float, default=1.0)
    args = parser.parse_args()
    registry = TaskRegistry.load(args.registry)
    evaluations = {
        split: evaluate_student_checkpoint(
            args.checkpoint,
            args.dataset,
            registry,
            split=split,
            minimum_confidence=0.90,
            device="cpu",
        )
        for split in ("validation", "test")
    }
    split_reports = {split: evaluation["overall"] for split, evaluation in evaluations.items()}
    first_evaluation = next(iter(evaluations.values()))
    dataset_metadata = {
        split: {
            "row_count": evaluation["row_count"],
            "unique_exact_group_count": evaluation["unique_exact_group_count"],
            "human_labeled_count": evaluation["golden"]["human_labeled_count"],
        }
        for split, evaluation in evaluations.items()
    }
    safety = _safety_report(args.checkpoint, registry, args.scenarios)
    report = {
        "record_type": "control_quality_gate",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": first_evaluation["checkpoint_sha256"],
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": first_evaluation["dataset_sha256"],
        "minimum_split_accuracy": args.minimum_split_accuracy,
        "minimum_safe_stop_recall": args.minimum_safe_stop_recall,
        "dataset_metadata": dataset_metadata,
        "splits": split_reports,
        "safety": safety,
        "passed": all(
            (split_report["accuracy"] or 0.0) >= args.minimum_split_accuracy
            for split_report in split_reports.values()
        )
        and (safety["safe_stop_recall"] or 0.0) >= args.minimum_safe_stop_recall,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
