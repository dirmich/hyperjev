#!/usr/bin/env python3
"""Measure a control teacher draft without treating it as human gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from hyperjev.registry import TaskRegistry
from hyperjev.samples import load_jsonl


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1)
    return round(ordered[index], 3)


def _load_draft(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not isinstance(rows[0], dict):
        raise ValueError("teacher draft is empty")
    manifest = rows[0]
    if manifest.get("record_type") != "golden_teacher_draft_manifest":
        raise ValueError("first teacher draft record must be a manifest")
    records: dict[str, dict[str, Any]] = {}
    for record in rows[1:]:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in records:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        records[sample_id] = record
    return manifest, records


def evaluate_control_teacher_draft(
    queue_path: str | Path,
    draft_path: str | Path,
    registry_path: str | Path = "registry/control_tasks",
) -> dict[str, Any]:
    queue = Path(queue_path)
    draft = Path(draft_path)
    registry = TaskRegistry.load(registry_path)
    samples = load_jsonl(queue, registry)
    manifest, records = _load_draft(draft)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if manifest.get("queue_sha256") != hashlib.sha256(queue.read_bytes()).hexdigest():
        raise ValueError("teacher draft queue SHA-256 does not match queue")
    unknown = set(records) - set(sample_by_id)
    if unknown:
        raise ValueError(f"teacher draft references unknown samples: {sorted(unknown)[:3]}")

    per_skill: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    valid_count = 0
    correct_count = 0
    repaired_count = 0
    for sample in samples:
        record = records.get(sample.sample_id)
        skill = str(sample.target)
        metrics = per_skill.setdefault(skill, {"count": 0, "valid": 0, "correct": 0, "repaired": 0})
        metrics["count"] += 1
        if record is None:
            continue
        if record.get("latency_ms") is not None:
            latencies.append(float(record["latency_ms"]))
        if record.get("schema_valid") is not True:
            continue
        result = record.get("normalized_result") or {}
        if result.get("type") != "choice" or not result.get("selected"):
            continue
        valid_count += 1
        metrics["valid"] += 1
        correct = str(result["selected"]) == skill
        correct_count += int(correct)
        metrics["correct"] += int(correct)
        repaired = record.get("schema_repaired") is True
        repaired_count += int(repaired)
        metrics["repaired"] += int(repaired)

    for metrics in per_skill.values():
        metrics["coverage"] = round(metrics["valid"] / metrics["count"], 6)
        metrics["accuracy_on_valid"] = (
            round(metrics["correct"] / metrics["valid"], 6) if metrics["valid"] else None
        )
    human_labeled_count = sum(sample.labels.get("human") is not None for sample in samples)
    sample_count = len(samples)
    report = {
        "record_type": "control_teacher_draft_quality",
        "queue": str(queue.resolve()),
        "queue_sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
        "draft": str(draft.resolve()),
        "draft_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "sample_count": sample_count,
        "record_count": len(records),
        "completed_count": sum(record.get("status") == "completed" for record in records.values()),
        "schema_valid_count": valid_count,
        "schema_valid_coverage": round(valid_count / sample_count, 6) if sample_count else 0.0,
        "synthetic_target_correct_count": correct_count,
        "synthetic_target_accuracy_on_valid": round(correct_count / valid_count, 6)
        if valid_count
        else None,
        "synthetic_target_accuracy_overall": round(correct_count / sample_count, 6)
        if sample_count
        else None,
        "schema_repaired_count": repaired_count,
        "human_labeled_count": human_labeled_count,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "per_skill": per_skill,
        "synthetic_reference_only": True,
        "complete_and_valid": valid_count == sample_count and len(records) == sample_count,
        "human_label_gate": human_labeled_count == sample_count,
    }
    report["production_ready"] = report["complete_and_valid"] and report["human_label_gate"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("registry/control_tasks"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_control_teacher_draft(args.queue, args.draft, args.registry)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["complete_and_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
