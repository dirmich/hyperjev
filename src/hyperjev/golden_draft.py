"""Teacher-assisted draft labels for a human-reviewed golden queue."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Phase0Config
from .contracts import ContractError, parse_decision_result, validate_result_for_task
from .prompts import PROMPT_VERSION, messages_for_sample
from .registry import TaskRegistry
from .samples import load_jsonl
from .teachers import TeacherClient, parse_json_object, response_hash

GOLDEN_DRAFT_VERSION = "golden-teacher-draft-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_teacher_draft(
    config: Phase0Config,
    registry: TaskRegistry,
    queue_path: str | Path,
    output_path: str | Path,
    *,
    provider: str = "gemma",
    limit: int | None = None,
    timeout_s: float | None = None,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Generate non-human teacher draft labels bound to one queue hash.

    The output deliberately excludes raw queue state and raw teacher text. A
    reviewer joins records by ``sample_id`` and creates human feedback only
    after checking the original state and question independently.
    """

    queue = Path(queue_path)
    samples = load_jsonl(queue, registry)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    if not samples:
        raise ValueError("golden draft queue has no samples")
    settings = config.teachers.get(provider)
    if settings is None:
        raise ValueError(f"teacher is not configured: {provider}")
    client = TeacherClient(settings, timeout_s=timeout_s)
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    for sample in samples:
        task = registry.get(sample.task_id, sample.task_version)
        record: dict[str, Any] = {
            "record_type": "golden_teacher_draft",
            "draft_version": GOLDEN_DRAFT_VERSION,
            "run_id": run_id,
            "sample_id": sample.sample_id,
            "task": f"{task.id}@{task.version}",
            "queue_sha256": queue_sha256,
            "provider": provider,
            "model": settings.model,
            "prompt_version": PROMPT_VERSION,
            "status": "pending",
            "latency_ms": None,
            "response_sha256": None,
            "normalized_result": None,
            "schema_valid": None,
            "error": None,
        }
        try:
            completion = client.complete(messages_for_sample(provider, sample, task), max_tokens=max_tokens)
            record.update(
                {
                    "status": "completed",
                    "model": completion.model,
                    "latency_ms": round(completion.elapsed_ms, 3),
                    "response_sha256": response_hash(completion.content),
                }
            )
            try:
                parsed = parse_decision_result(parse_json_object(completion.content))
                validate_result_for_task(task, parsed)
                record["normalized_result"] = parsed.to_dict()
                record["schema_valid"] = True
            except (ContractError, TypeError, ValueError) as exc:
                record["schema_valid"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
        except (OSError, TypeError, ValueError) as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    manifest = {
        "record_type": "golden_teacher_draft_manifest",
        "draft_version": GOLDEN_DRAFT_VERSION,
        "created_at": _utc_now(),
        "run_id": run_id,
        "queue_path": str(queue.resolve()),
        "queue_sha256": queue_sha256,
        "provider": provider,
        "model": settings.model,
        "prompt_version": PROMPT_VERSION,
        "max_tokens": max_tokens,
        "sample_count": len(samples),
        "schema_valid_count": sum(record["schema_valid"] is True for record in records),
        "completed_count": sum(record["status"] == "completed" for record in records),
        "error_count": sum(record["status"] == "error" for record in records),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"output": str(output.resolve()), "manifest": manifest, "records": records}


def generate_gemma_draft(
    config: Phase0Config,
    registry: TaskRegistry,
    queue_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    timeout_s: float | None = None,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Backward-compatible Gemma-specific wrapper."""

    return generate_teacher_draft(
        config,
        registry,
        queue_path,
        output_path,
        provider="gemma",
        limit=limit,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )
