"""Phase 0 baseline and golden-set evaluation utilities."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .registry import TaskRegistry
from .samples import load_jsonl


class EvaluationError(ValueError):
    """Raised when a baseline run cannot be evaluated safely."""


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(ordered[rank], 3)


def _load_run(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"cannot read run file {source}: {exc}") from exc
    if not lines:
        raise EvaluationError(f"run file is empty: {source}")
    try:
        manifest = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:] if line.strip()]
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSONL in {source}: {exc.msg}") from exc
    if not isinstance(manifest, dict) or manifest.get("record_type") != "manifest":
        raise EvaluationError("first run record must be a manifest")
    if not all(isinstance(record, dict) for record in records):
        raise EvaluationError("sample records must be objects")
    return manifest, records


def evaluate_run(
    run_path: str | Path,
    samples_path: str | Path,
    registry: TaskRegistry,
) -> dict[str, Any]:
    """Summarize latency, schema validity, quality, and teacher agreement."""

    manifest, records = _load_run(run_path)
    samples = {sample.sample_id: sample for sample in load_jsonl(samples_path, registry)}
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id not in samples:
            raise EvaluationError(f"run references unknown sample: {sample_id}")
        by_provider[str(record.get("provider", "unknown"))].append(record)

    provider_reports: dict[str, Any] = {}
    for provider, provider_records in sorted(by_provider.items()):
        latencies = [float(record["latency_ms"]) for record in provider_records if record.get("latency_ms") is not None]
        valid_records = [record for record in provider_records if record.get("schema_valid") is True]
        task_stats: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"count": 0, "correct": 0, "absolute_error": 0.0}
        )
        for record in valid_records:
            sample = samples[record["sample_id"]]
            result = record.get("normalized_result") or {}
            stats = task_stats[sample.task_id]
            stats["count"] += 1
            if result.get("type") == "boolean":
                correct = result.get("value") == sample.target
                stats["correct"] += int(correct)
            elif result.get("type") == "choice":
                correct = result.get("selected") == sample.target
                stats["correct"] += int(correct)
            elif result.get("type") == "score":
                stats["absolute_error"] += abs(float(result.get("value", 0)) - float(sample.target))
        quality: dict[str, Any] = {}
        for task_id, stats in sorted(task_stats.items()):
            count = int(stats["count"])
            report: dict[str, Any] = {"count": count}
            if count and stats["correct"]:
                report["accuracy"] = round(int(stats["correct"]) / count, 4)
            if count and stats["absolute_error"]:
                report["mae"] = round(float(stats["absolute_error"]) / count, 4)
            quality[task_id] = report
        provider_reports[provider] = {
            "records": len(provider_records),
            "completed": sum(record.get("status") == "completed" for record in provider_records),
            "errors": sum(record.get("status") == "error" for record in provider_records),
            "schema_valid": len(valid_records),
            "schema_valid_rate": round(len(valid_records) / len(provider_records), 4)
            if provider_records
            else 0.0,
            "latency_ms": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "p99": _percentile(latencies, 99),
            },
            "total_tokens": sum(int(record.get("total_tokens") or 0) for record in provider_records),
            "quality": quality,
        }

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for provider, provider_records in by_provider.items():
        for record in provider_records:
            if record.get("schema_valid") is True:
                grouped[record["sample_id"]][provider] = record.get("normalized_result")
    comparable = [values for values in grouped.values() if len(values) >= 2]
    agreements = sum(len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1 for values in comparable)
    return {
        "manifest": manifest,
        "run_path": str(Path(run_path).resolve()),
        "sample_count": len(samples),
        "providers": provider_reports,
        "agreement": {
            "comparable_samples": len(comparable),
            "agreed_samples": agreements,
            "rate": round(agreements / len(comparable), 4) if comparable else None,
        },
    }


def validate_golden_set(
    samples_path: str | Path,
    registry: TaskRegistry,
    *,
    minimum_count: int = 1000,
    require_human: bool = True,
) -> dict[str, Any]:
    """Report whether a dataset satisfies the Phase 0 golden-set gate."""

    samples = load_jsonl(samples_path, registry)
    human_count = sum(sample.labels.get("human") is not None for sample in samples)
    count_ok = len(samples) >= minimum_count
    human_ok = not require_human or human_count == len(samples)
    return {
        "path": str(Path(samples_path).resolve()),
        "sample_count": len(samples),
        "human_labeled_count": human_count,
        "minimum_count": minimum_count,
        "require_human": require_human,
        "ready": count_ok and human_ok,
        "missing_count": max(0, minimum_count - len(samples)),
    }
