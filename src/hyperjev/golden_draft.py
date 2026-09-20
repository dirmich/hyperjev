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
GOLDEN_ADJUDICATION_VERSION = "golden-teacher-adjudication-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_draft_file(output: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _append_draft_record(output: Path, record: dict[str, Any]) -> None:
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


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
    resume: bool = False,
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_manifest: dict[str, Any] | None = None
    existing_records: dict[str, dict[str, Any]] = {}
    if resume and output.exists():
        existing_manifest, existing_records = _draft_records(output)
        if existing_manifest.get("record_type") != "golden_teacher_draft_manifest":
            raise ValueError("resume output must be a teacher draft, not an adjudication")
        if existing_manifest.get("queue_sha256") != queue_sha256:
            raise ValueError("resume output queue SHA-256 does not match queue")
        if existing_manifest.get("provider") != provider:
            raise ValueError("resume output provider does not match provider")
        selected_ids = {sample.sample_id for sample in samples}
        if not set(existing_records).issubset(selected_ids):
            raise ValueError("resume output contains samples outside the selected queue range")
    run_id = str(existing_manifest.get("run_id")) if existing_manifest else datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = [
        existing_records[sample.sample_id] for sample in samples if sample.sample_id in existing_records
    ]
    initial_manifest = {
        "record_type": "golden_teacher_draft_manifest",
        "draft_version": GOLDEN_DRAFT_VERSION,
        "created_at": existing_manifest.get("created_at", _utc_now()) if existing_manifest else _utc_now(),
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
        "resumable": True,
    }
    if not resume or not output.exists():
        _write_draft_file(output, initial_manifest, records)
    for sample in samples:
        if sample.sample_id in existing_records:
            continue
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
        _append_draft_record(output, record)

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
        "resumable": True,
    }
    _write_draft_file(output, manifest, records)
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
    resume: bool = False,
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
        resume=resume,
    )


def _draft_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or not isinstance(records[0], dict):
        raise ValueError(f"teacher draft is empty: {path}")
    manifest = records[0]
    if manifest.get("record_type") not in {
        "golden_teacher_draft_manifest",
        "golden_teacher_adjudication_manifest",
    }:
        raise ValueError(f"unsupported teacher draft manifest: {path}")
    by_sample: dict[str, dict[str, Any]] = {}
    for record in records[1:]:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in by_sample:
            raise ValueError(f"duplicate or empty teacher draft sample_id: {sample_id!r}")
        by_sample[sample_id] = record
    return manifest, by_sample


def _result_signature(raw: Any, task: Any) -> tuple[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    try:
        parsed = parse_decision_result(raw)
        validate_result_for_task(task, parsed)
    except (ContractError, TypeError, ValueError):
        return None
    if parsed.abstained:
        return None
    if parsed.type == "boolean":
        return parsed.type, parsed.value
    if parsed.type == "choice":
        return parsed.type, parsed.selected
    return parsed.type, round(parsed.value, 6)


def adjudicate_teacher_drafts(
    config: Phase0Config,
    registry: TaskRegistry,
    queue_path: str | Path,
    qwen_draft_path: str | Path,
    gemma_draft_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Join Qwen/Gemma drafts; only exact typed agreement becomes silver."""

    queue = Path(queue_path)
    qwen_manifest, qwen_records = _draft_records(Path(qwen_draft_path))
    gemma_manifest, gemma_records = _draft_records(Path(gemma_draft_path))
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    for label, manifest in (("qwen", qwen_manifest), ("gemma", gemma_manifest)):
        if manifest.get("queue_sha256") != queue_sha256:
            raise ValueError(f"{label} draft queue SHA-256 does not match queue")
    samples = load_jsonl(queue, registry)
    sample_ids = {sample.sample_id for sample in samples}
    if set(qwen_records) != sample_ids or set(gemma_records) != sample_ids:
        raise ValueError("Qwen and Gemma drafts must cover every queue sample exactly once")
    records: list[dict[str, Any]] = []
    agreement_count = 0
    disagreement_count = 0
    invalid_count = 0
    for sample in samples:
        task = registry.get(sample.task_id, sample.task_version)
        qwen = qwen_records[sample.sample_id]
        gemma = gemma_records[sample.sample_id]
        qwen_result = qwen.get("normalized_result") if qwen.get("schema_valid") is True else None
        gemma_result = gemma.get("normalized_result") if gemma.get("schema_valid") is True else None
        qwen_signature = _result_signature(qwen_result, task)
        gemma_signature = _result_signature(gemma_result, task)
        if qwen_signature is not None and qwen_signature == gemma_signature:
            status = "agreed"
            normalized_result = qwen_result
            schema_valid = True
            agreement_count += 1
        elif qwen_signature is None or gemma_signature is None:
            status = "invalid_or_abstained"
            normalized_result = None
            schema_valid = False
            invalid_count += 1
        else:
            status = "disagreement"
            normalized_result = None
            schema_valid = False
            disagreement_count += 1
        records.append(
            {
                "record_type": "golden_teacher_adjudication",
                "draft_version": GOLDEN_ADJUDICATION_VERSION,
                "sample_id": sample.sample_id,
                "task": f"{task.id}@{task.version}",
                "queue_sha256": queue_sha256,
                "provider": "qwen+gemma",
                "model": f"{qwen.get('model')}|{gemma.get('model')}",
                "normalized_result": normalized_result,
                "schema_valid": schema_valid,
                "status": status,
                "error": None if status == "agreed" else status,
                "teacher_comparison": {
                    "qwen": {
                        "normalized_result": qwen_result,
                        "schema_valid": qwen.get("schema_valid"),
                        "status": qwen.get("status"),
                        "latency_ms": qwen.get("latency_ms"),
                        "error": qwen.get("error"),
                    },
                    "gemma": {
                        "normalized_result": gemma_result,
                        "schema_valid": gemma.get("schema_valid"),
                        "status": gemma.get("status"),
                        "latency_ms": gemma.get("latency_ms"),
                        "error": gemma.get("error"),
                    },
                },
            }
        )
    manifest = {
        "record_type": "golden_teacher_adjudication_manifest",
        "draft_version": GOLDEN_ADJUDICATION_VERSION,
        "queue_path": str(queue.resolve()),
        "queue_sha256": queue_sha256,
        "qwen_draft_path": str(Path(qwen_draft_path).resolve()),
        "gemma_draft_path": str(Path(gemma_draft_path).resolve()),
        "qwen_model": config.teachers.get("qwen").model if config.teachers.get("qwen") else None,
        "gemma_model": config.teachers.get("gemma").model if config.teachers.get("gemma") else None,
        "sample_count": len(samples),
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "invalid_count": invalid_count,
        "review_required_count": disagreement_count + invalid_count,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"output": str(output.resolve()), "manifest": manifest, "records": records}
