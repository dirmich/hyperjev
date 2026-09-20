#!/usr/bin/env python3
"""Evaluator-gated control quality report for the reference Student path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hyperjev.control import ControlObservation, ControlSafetyPolicy, ControlStudentClient
from hyperjev.metrics import threshold_risk_coverage
from hyperjev.registry import TaskRegistry
from hyperjev.student_inference import evaluate_student_checkpoint

_WILSON_Z_95 = 1.959963984540054


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
    scenario_ids = [scenario.get("scenario_id") for scenario in scenarios]
    if any(not isinstance(scenario_id, str) or not scenario_id.strip() for scenario_id in scenario_ids):
        raise ValueError(f"{path}: every scenario must have a non-empty scenario_id")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError(f"{path}: scenario_id values must be unique")
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
        "accuracy_ci95": _binomial_interval(sum(row["correct"] for row in rows), len(rows)),
        "expected_safe_stop_count": len(expected_stops),
        "safe_stop_recall": round(stop_hits / len(expected_stops), 6) if expected_stops else None,
        "safe_stop_recall_ci95": _binomial_interval(stop_hits, len(expected_stops)),
    }


def _binomial_interval(successes: int, total: int) -> dict[str, float | None]:
    """Return a Wilson 95% interval without treating a small 100% sample as proof."""

    if total < 1:
        return {"lower": None, "upper": None}
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    denominator = 1.0 + (_WILSON_Z_95**2 / total)
    center = (proportion + (_WILSON_Z_95**2 / (2.0 * total))) / denominator
    margin = (
        _WILSON_Z_95
        * ((proportion * (1.0 - proportion) / total) + (_WILSON_Z_95**2 / (4.0 * total**2))) ** 0.5
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
    }


def _annotate_split_report(report: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(report)
    annotated["accuracy_ci95"] = _binomial_interval(
        int(report.get("correct", 0)), int(report.get("count", 0))
    )
    annotated["accepted_accuracy_ci95"] = _binomial_interval(
        int(report.get("accepted_correct", 0)), int(report.get("accepted_count", 0))
    )
    return annotated


def _skill_report(evaluation: dict[str, Any], registry: TaskRegistry) -> dict[str, Any]:
    """Expose per-target accuracy and confusion instead of only aggregate accuracy."""

    candidates = [str(candidate) for candidate in registry.get("control.skill", 1).output["candidates"]]
    counts = {
        candidate: {
            "count": 0,
            "correct": 0,
            "accepted_count": 0,
            "accepted_correct": 0,
        }
        for candidate in candidates
    }
    confusion = {candidate: {predicted: 0 for predicted in candidates} for candidate in candidates}
    for row in evaluation["predictions"]:
        target = str(row["target"])
        predicted = str(row.get("prediction", {}).get("selected", ""))
        counts.setdefault(
            target,
            {"count": 0, "correct": 0, "accepted_count": 0, "accepted_correct": 0},
        )
        counts[target]["count"] += 1
        counts[target]["correct"] += int(bool(row["correct"]))
        if row.get("accepted"):
            counts[target]["accepted_count"] += 1
            counts[target]["accepted_correct"] += int(bool(row["correct"]))
        confusion.setdefault(target, {})[predicted] = confusion.setdefault(target, {}).get(predicted, 0) + 1
    metrics = {
        skill: {
            "count": values["count"],
            "correct": values["correct"],
            "accuracy": round(values["correct"] / values["count"], 6) if values["count"] else None,
            "accuracy_ci95": _binomial_interval(values["correct"], values["count"]),
            "accepted_count": values["accepted_count"],
            "accepted_correct": values["accepted_correct"],
            "accepted_accuracy": (
                round(values["accepted_correct"] / values["accepted_count"], 6)
                if values["accepted_count"]
                else None
            ),
            "accepted_coverage": (
                round(values["accepted_count"] / values["count"], 6) if values["count"] else None
            ),
            "accepted_accuracy_ci95": _binomial_interval(
                values["accepted_correct"], values["accepted_count"]
            ),
        }
        for skill, values in sorted(counts.items())
    }
    return {"metrics": metrics, "confusion_matrix": confusion}


def _risk_coverage_report(
    evaluation: dict[str, Any], thresholds: tuple[float, ...]
) -> list[dict[str, Any]]:
    return threshold_risk_coverage(
        [bool(row["correct"]) for row in evaluation["predictions"]],
        [float(row["confidence"]) for row in evaluation["predictions"]],
        thresholds=thresholds,
    )


def _quality_gate_failures(
    split_reports: dict[str, dict[str, Any]],
    skill_reports: dict[str, dict[str, Any]],
    safety: dict[str, Any],
    *,
    minimum_split_accuracy: float,
    minimum_skill_accuracy: float,
    minimum_accepted_accuracy: float,
    minimum_accepted_coverage: float,
    minimum_safety_accuracy: float,
    minimum_safe_stop_recall: float,
    minimum_safe_stop_lower_bound: float,
    minimum_safe_stop_count: int,
) -> dict[str, list[str]]:
    """Return explicit failures for raw, accepted, safety, and confidence gates."""

    if minimum_safe_stop_count < 1:
        raise ValueError("minimum_safe_stop_count must be positive")
    split_failures = [
        split
        for split, report in split_reports.items()
        if (report.get("accuracy") or 0.0) < minimum_split_accuracy
    ]
    accepted_accuracy_failures = [
        split
        for split, report in split_reports.items()
        if report.get("accepted_accuracy") is None
        or report["accepted_accuracy"] < minimum_accepted_accuracy
    ]
    accepted_coverage_failures = [
        split
        for split, report in split_reports.items()
        if (report.get("coverage") or 0.0) < minimum_accepted_coverage
    ]
    skill_failures = {
        split: [
            skill
            for skill, metrics in skill_report["metrics"].items()
            if metrics["count"] == 0 or (metrics["accuracy"] or 0.0) < minimum_skill_accuracy
        ]
        for split, skill_report in skill_reports.items()
    }
    safety_failures: list[str] = []
    if (safety.get("accuracy") or 0.0) < minimum_safety_accuracy:
        safety_failures.append("safety_accuracy")
    if (safety.get("safe_stop_recall") or 0.0) < minimum_safe_stop_recall:
        safety_failures.append("safe_stop_recall")
    lower_bound = safety.get("safe_stop_recall_ci95", {}).get("lower")
    if lower_bound is None or lower_bound < minimum_safe_stop_lower_bound:
        safety_failures.append("safe_stop_recall_ci95_lower_bound")
    if int(safety.get("expected_safe_stop_count", 0)) < minimum_safe_stop_count:
        safety_failures.append("safe_stop_sample_count")
    return {
        "split_accuracy": split_failures,
        "accepted_accuracy": accepted_accuracy_failures,
        "accepted_coverage": accepted_coverage_failures,
        "skill_accuracy": [
            f"{split}:{skill}" for split, skills in skill_failures.items() for skill in skills
        ],
        "safety": safety_failures,
    }


def _human_label_status(dataset_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize full and held-out human-label gates by split."""

    by_split = {
        split: metadata["human_labeled_count"] == metadata["row_count"]
        for split, metadata in dataset_metadata.items()
    }
    return {
        "by_split": by_split,
        "all_splits": all(by_split.values()),
        "test": bool(by_split.get("test", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("tests/golden/control_smoke.jsonl"))
    parser.add_argument("--scenarios", type=Path, default=Path("tests/golden/control_scenarios.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path("registry/control_tasks"))
    parser.add_argument("--minimum-split-accuracy", type=float, default=0.99)
    parser.add_argument("--minimum-skill-accuracy", type=float, default=0.99)
    parser.add_argument("--minimum-accepted-accuracy", type=float, default=0.995)
    parser.add_argument("--minimum-accepted-coverage", type=float, default=0.99)
    parser.add_argument("--minimum-safety-accuracy", type=float, default=1.0)
    parser.add_argument("--minimum-safe-stop-recall", type=float, default=1.0)
    parser.add_argument("--minimum-safe-stop-lower-bound", type=float, default=0.99)
    parser.add_argument(
        "--minimum-safe-stop-count",
        type=int,
        default=500,
        help="minimum independent expected safe-stop scenarios for the safety gate",
    )
    parser.add_argument(
        "--risk-thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.70, 0.80, 0.90, 0.95, 0.99),
        help="confidence thresholds used for accepted-risk/coverage reporting",
    )
    parser.add_argument("--require-human-labels", action="store_true")
    parser.add_argument(
        "--require-human-test",
        action="store_true",
        help="fail unless every held-out test row has a typed human label",
    )
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
    split_reports = {
        split: _annotate_split_report(evaluation["overall"])
        for split, evaluation in evaluations.items()
    }
    skill_reports = {split: _skill_report(evaluation, registry) for split, evaluation in evaluations.items()}
    risk_coverage = {
        split: _risk_coverage_report(evaluation, tuple(args.risk_thresholds))
        for split, evaluation in evaluations.items()
    }
    first_evaluation = next(iter(evaluations.values()))
    dataset_metadata = {
        split: {
            "row_count": evaluation["row_count"],
            "unique_exact_group_count": evaluation["unique_exact_group_count"],
            "human_labeled_count": evaluation["golden"]["human_labeled_count"],
            "human_label_gate": evaluation["golden"]["human_labeled_count"] == evaluation["row_count"],
        }
        for split, evaluation in evaluations.items()
    }
    human_status = _human_label_status(dataset_metadata)
    human_label_gate = human_status["all_splits"]
    human_test_gate = human_status["test"]
    safety = _safety_report(args.checkpoint, registry, args.scenarios)
    gate_failures = _quality_gate_failures(
        split_reports,
        skill_reports,
        safety,
        minimum_split_accuracy=args.minimum_split_accuracy,
        minimum_skill_accuracy=args.minimum_skill_accuracy,
        minimum_accepted_accuracy=args.minimum_accepted_accuracy,
        minimum_accepted_coverage=args.minimum_accepted_coverage,
        minimum_safety_accuracy=args.minimum_safety_accuracy,
        minimum_safe_stop_recall=args.minimum_safe_stop_recall,
        minimum_safe_stop_lower_bound=args.minimum_safe_stop_lower_bound,
        minimum_safe_stop_count=args.minimum_safe_stop_count,
    )
    if args.require_human_labels and not human_label_gate:
        gate_failures["human_labels"] = [
            split for split, metadata in dataset_metadata.items() if not metadata["human_label_gate"]
        ]
    if args.require_human_test and not human_test_gate:
        gate_failures["human_test"] = ["test"]
    report = {
        "record_type": "control_quality_gate",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": first_evaluation["checkpoint_sha256"],
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": first_evaluation["dataset_sha256"],
        "minimum_split_accuracy": args.minimum_split_accuracy,
        "minimum_skill_accuracy": args.minimum_skill_accuracy,
        "minimum_accepted_accuracy": args.minimum_accepted_accuracy,
        "minimum_accepted_coverage": args.minimum_accepted_coverage,
        "minimum_safety_accuracy": args.minimum_safety_accuracy,
        "minimum_safe_stop_recall": args.minimum_safe_stop_recall,
        "minimum_safe_stop_lower_bound": args.minimum_safe_stop_lower_bound,
        "minimum_safe_stop_count": args.minimum_safe_stop_count,
        "require_human_labels": args.require_human_labels,
        "require_human_test": args.require_human_test,
        "dataset_metadata": dataset_metadata,
        "splits": split_reports,
        "skills": skill_reports,
        "risk_coverage": risk_coverage,
        "raw_student_head": {
            "splits": split_reports,
            "skills": skill_reports,
            "risk_coverage": risk_coverage,
        },
        "safety": safety,
        "safety_policy": safety,
        "gate_failures": gate_failures,
        "passed": not any(gate_failures.values()),
        "human_label_gate": human_label_gate,
        "human_test_gate": human_test_gate,
        "human_label_status": human_status,
    }
    report["production_ready"] = report["passed"] and human_label_gate
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
