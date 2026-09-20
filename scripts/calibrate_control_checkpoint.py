#!/usr/bin/env python3
"""Fit task-local temperature manifests from a held-out Student split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from hyperjev.calibration import fit_temperature
from hyperjev.registry import TaskRegistry
from hyperjev.student_inference import evaluate_student_checkpoint


def _target_index(target: Any, task: Any) -> int:
    if task.output_type == "boolean":
        return int(bool(target))
    candidates = [str(candidate) for candidate in task.output.get("candidates", [])]
    try:
        return candidates.index(str(target))
    except ValueError as exc:
        raise ValueError(f"target {target!r} is not a registered candidate") from exc


def build_calibration_manifest(
    checkpoint: Path,
    dataset: Path,
    registry_path: Path,
    *,
    split: str = "validation",
    minimum_samples: int = 10,
) -> dict[str, Any]:
    if minimum_samples < 2:
        raise ValueError("minimum_samples must be at least 2")
    registry = TaskRegistry.load(registry_path)
    evaluation = evaluate_student_checkpoint(
        checkpoint,
        dataset,
        registry,
        split=split,
        minimum_confidence=0.0,
        include_logits=True,
        device="cpu",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in evaluation["predictions"]:
        if "logits" not in row:
            continue
        grouped.setdefault(str(row["task_id"]), []).append(row)

    tasks: dict[str, Any] = {}
    task_failures: list[str] = []
    for task_id, rows in sorted(grouped.items()):
        task = registry.get(task_id)
        if len(rows) < minimum_samples:
            task_failures.append(task_id)
            tasks[task_id] = {"sample_count": len(rows), "status": "insufficient_samples"}
            continue
        labels = [_target_index(row["target"], task) for row in rows]
        result = fit_temperature([row["logits"] for row in rows], labels)
        tasks[task_id] = {
            "status": "fitted",
            **result.to_dict(),
            "human_labeled_count": sum(row["target_source"] == "human" for row in rows),
        }

    human_labeled_count = sum(
        row["target_source"] == "human" for row in evaluation["predictions"]
    )
    manifest = {
        "record_type": "control_calibration_manifest",
        "calibration_version": "cal-control-"
        + hashlib.sha256(
            f"{evaluation['checkpoint_sha256']}:{evaluation['dataset_sha256']}:{split}".encode()
        ).hexdigest()[:12],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "dataset": str(dataset.resolve()),
        "dataset_sha256": evaluation["dataset_sha256"],
        "split": split,
        "sample_count": evaluation["row_count"],
        "human_labeled_count": human_labeled_count,
        "tasks": tasks,
        "task_failures": task_failures,
        "production_eligible": not task_failures
        and human_labeled_count == evaluation["row_count"],
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("registry/control_tasks"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_calibration_manifest(
        args.checkpoint,
        args.dataset,
        args.registry,
        split=args.split,
        minimum_samples=args.minimum_samples,
    )
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if manifest["production_eligible"] else 1


if __name__ == "__main__":
    sys.exit(main())
