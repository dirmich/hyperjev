"""Reproducible teacher-labelled dataset construction for Phase 2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Phase0Config
from .contracts import (
    BooleanDecision,
    ChoiceDecision,
    ContractError,
    DecisionResult,
    ScoreDecision,
    parse_decision_result,
    parse_task_reference,
    validate_result_for_task,
)
from .prompts import PROMPT_VERSION, messages_for_question
from .registry import TaskDefinition, TaskRegistry
from .teachers import TeacherClient, parse_json_object, response_hash

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|Bearer\s+[A-Za-z0-9._-]+)\b"
)


@dataclass(frozen=True)
class DatasetBuildReport:
    output_path: Path
    review_path: Path
    counts: Mapping[str, int]
    dataset_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "review_path": str(self.review_path),
            "counts": dict(self.counts),
            "dataset_hash": self.dataset_hash,
        }


def redact_text(value: str) -> tuple[str, bool]:
    """Apply deterministic, conservative redaction before dataset output."""

    redacted = _SECRET.sub("[SECRET]", value)
    redacted = _EMAIL.sub("[EMAIL]", redacted)
    redacted = _PHONE.sub("[PHONE]", redacted)
    return redacted, redacted != value


def _redact_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        values: list[Any] = []
        changed = False
        for item in value:
            redacted, item_changed = _redact_value(item)
            values.append(redacted)
            changed = changed or item_changed
        return values, changed
    if isinstance(value, Mapping):
        values: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            redacted, item_changed = _redact_value(item)
            values[str(key)] = redacted
            changed = changed or item_changed
        return values, changed
    return value, False


def _confidence(result: DecisionResult) -> float:
    if isinstance(result, BooleanDecision):
        return result.probability
    if isinstance(result, ChoiceDecision):
        return result.probabilities.get(result.selected, 0.0)
    if isinstance(result, ScoreDecision):
        return max(0.0, 1.0 - (result.interval_90[1] - result.interval_90[0]))
    return 0.0


def _result_target(result: DecisionResult) -> Any:
    if isinstance(result, BooleanDecision):
        return result.value
    if isinstance(result, ChoiceDecision):
        return result.selected
    return result.value


def _soft_target(first: DecisionResult, second: DecisionResult) -> Any:
    if isinstance(first, BooleanDecision) and isinstance(second, BooleanDecision):
        first_probability = first.probability if first.value else 1.0 - first.probability
        second_probability = second.probability if second.value else 1.0 - second.probability
        return round((first_probability + second_probability) / 2.0, 6)
    if isinstance(first, ChoiceDecision) and isinstance(second, ChoiceDecision):
        candidates = sorted(set(first.probabilities) | set(second.probabilities))
        return {
            candidate: round(
                (first.probabilities.get(candidate, 0.0) + second.probabilities.get(candidate, 0.0))
                / 2.0,
                6,
            )
            for candidate in candidates
        }
    if isinstance(first, ScoreDecision) and isinstance(second, ScoreDecision):
        return round((first.value + second.value) / 2.0, 6)
    return None


def _same_label(task: TaskDefinition, first: DecisionResult, second: DecisionResult) -> bool:
    if isinstance(first, BooleanDecision) and isinstance(second, BooleanDecision):
        return first.value == second.value
    if isinstance(first, ChoiceDecision) and isinstance(second, ChoiceDecision):
        return first.selected == second.selected
    if isinstance(first, ScoreDecision) and isinstance(second, ScoreDecision):
        return abs(first.value - second.value) <= 0.20
    return False


def _split_for(seed: Mapping[str, Any], redacted_source: Mapping[str, Any]) -> str:
    """Assign by source/entity group, never by an ungrouped random row."""

    group = (
        redacted_source.get("split_group")
        or redacted_source.get("document_id")
        or redacted_source.get("entity_id")
        or redacted_source.get("source_id")
        or seed.get("sample_id")
    )
    bucket = int(hashlib.sha256(str(group).encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _read_seed(path: str | Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"seed:{line_number}: record must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"seed file is empty: {path}")
    return tuple(records)


def _task_for(seed: Mapping[str, Any], registry: TaskRegistry) -> TaskDefinition:
    if seed.get("task"):
        task_id, version = parse_task_reference(str(seed["task"]))
    else:
        task_id = str(seed.get("task_id", ""))
        version = int(seed.get("task_version", 0))
    return registry.get(task_id, version)


def _human_result(seed: Mapping[str, Any], task: TaskDefinition) -> DecisionResult | None:
    labels = seed.get("labels", {})
    if not isinstance(labels, Mapping) or labels.get("human") is None:
        return None
    human = labels["human"]
    if not isinstance(human, Mapping):
        raise ContractError("labels.human must be a typed result object")
    result = parse_decision_result(human)
    validate_result_for_task(task, result)
    return result


def _teacher_result(
    provider: str,
    client: Any,
    task: TaskDefinition,
    state: str,
    question: str,
) -> tuple[DecisionResult | None, dict[str, Any]]:
    try:
        completion = client.complete(
            messages_for_question(
                provider,
                task,
                state=state,
                question=question,
                candidates=list(task.output.get("candidates", [])),
            )
        )
        result = parse_decision_result(parse_json_object(completion.content))
        validate_result_for_task(task, result)
        return result, {
            "provider": provider,
            "model": completion.model,
            "status": "valid",
            "schema_valid": True,
            "response_sha256": response_hash(completion.content),
            "latency_ms": round(float(completion.elapsed_ms), 3),
        }
    except (ContractError, OSError, TypeError, ValueError) as exc:
        return None, {
            "provider": provider,
            "status": "invalid",
            "schema_valid": False,
            "response_sha256": None,
            "latency_ms": None,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def build_dataset(
    config: Phase0Config,
    registry: TaskRegistry,
    seed_path: str | Path,
    output_path: str | Path,
    *,
    review_path: str | Path | None = None,
    clients: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> DatasetBuildReport:
    """Build a privacy-filtered, agreement-gated JSONL dataset."""

    seeds = _read_seed(seed_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        seeds = seeds[:limit]
    teacher_clients: dict[str, Any] = {
        "qwen": TeacherClient(config.teachers["qwen"]),
        "gemma": TeacherClient(config.teachers["gemma"]),
    }
    if clients:
        teacher_clients.update(clients)
    output = Path(output_path)
    review = Path(review_path) if review_path else output.with_name(f"{output.stem}.review.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    review.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    counts: Counter[str] = Counter(total=len(seeds))
    seen: set[str] = set()

    for seed in seeds:
        sample_id = str(seed.get("sample_id", ""))
        try:
            task = _task_for(seed, registry)
            state = str(seed.get("state", ""))
            question = str(seed.get("question", ""))
            language = str(seed.get("language", ""))
            domain = str(seed.get("domain", ""))
            source = seed.get("source", {})
            if not sample_id or not state or not question or not isinstance(source, Mapping):
                raise ValueError("sample_id, state, question, and source are required")
            if language not in task.languages:
                raise ValueError(f"language is not registered for {task.id}: {language}")
            if (
                str(source.get("sensitivity", "")).lower() == "secret"
                or str(source.get("kind", "")).lower() == "user"
                and not config.privacy_allow_training_from_user_data
            ):
                counts["excluded_privacy"] += 1
                continue
            redacted_state, state_changed = redact_text(state)
            redacted_question, question_changed = redact_text(question)
            redacted_source, source_changed = _redact_value(dict(source))
            if not isinstance(redacted_source, Mapping):
                raise TypeError("redacted source must be an object")
            dedup_key = hashlib.sha256(
                json.dumps(
                    {
                        "task": f"{task.id}@{task.version}",
                        "state": redacted_state,
                        "question": redacted_question,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if dedup_key in seen:
                counts["excluded_duplicate"] += 1
                continue
            seen.add(dedup_key)
            qwen, qwen_meta = _teacher_result(
                "qwen", teacher_clients["qwen"], task, redacted_state, redacted_question
            )
            gemma, gemma_meta = _teacher_result(
                "gemma", teacher_clients["gemma"], task, redacted_state, redacted_question
            )
            human = _human_result(seed, task)
            if human is not None:
                chosen = human
                counts["accepted_human"] += 1
            elif qwen is None or gemma is None:
                reviews.append(
                    _review_record(
                        seed,
                        task,
                        redacted_state,
                        redacted_question,
                        "teacher_schema_invalid",
                        qwen_meta,
                        gemma_meta,
                    )
                )
                counts["review_schema_invalid"] += 1
                continue
            elif _confidence(qwen) < 0.5 and _confidence(gemma) < 0.5:
                reviews.append(
                    _review_record(
                        seed,
                        task,
                        redacted_state,
                        redacted_question,
                        "both_teachers_low_confidence",
                        qwen_meta,
                        gemma_meta,
                    )
                )
                counts["excluded_low_confidence"] += 1
                continue
            elif not _same_label(task, qwen, gemma) or abs(_confidence(qwen) - _confidence(gemma)) > 0.35:
                reviews.append(
                    _review_record(
                        seed,
                        task,
                        redacted_state,
                        redacted_question,
                        "teacher_disagreement",
                        qwen_meta,
                        gemma_meta,
                    )
                )
                counts["review_disagreement"] += 1
                continue
            else:
                chosen = qwen
                counts["accepted_silver"] += 1
            record = {
                "sample_id": sample_id,
                "task_id": task.id,
                "task_version": task.version,
                "state": redacted_state,
                "question": redacted_question,
                "target": _result_target(chosen),
                "soft_target": _soft_target(qwen or chosen, gemma or chosen),
                "language": language,
                "domain": domain,
                "source": dict(redacted_source),
                "labels": {
                    "qwen": qwen.to_dict() if qwen else None,
                    "gemma": gemma.to_dict() if gemma else None,
                    "human": human.to_dict() if human else None,
                },
                "provenance": {
                    "prompt_version": PROMPT_VERSION,
                    "teacher_models": {
                        "qwen": config.teachers["qwen"].model,
                        "gemma": config.teachers["gemma"].model,
                    },
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "split": _split_for(seed, redacted_source),
                    "redaction_applied": state_changed or question_changed or source_changed,
                    "privacy_raw_inputs_stored": config.privacy_store_raw_inputs,
                },
            }
            records.append(record)
        except (ContractError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reviews.append(
                {
                    "record_type": "review",
                    "sample_id": sample_id,
                    "reason": "invalid_seed",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            )
            counts["review_invalid_seed"] += 1

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with review.open("w", encoding="utf-8") as handle:
        for record in reviews:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    dataset_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    counts["accepted_total"] = len(records)
    counts["review_total"] = len(reviews)
    return DatasetBuildReport(output_path=output, review_path=review, counts=dict(counts), dataset_hash=dataset_hash)


def _review_record(
    seed: Mapping[str, Any],
    task: TaskDefinition,
    state: str,
    question: str,
    reason: str,
    qwen_meta: Mapping[str, Any],
    gemma_meta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "review",
        "sample_id": str(seed.get("sample_id", "")),
        "task": f"{task.id}@{task.version}",
        "state": state,
        "question": question,
        "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
        "reason": reason,
        "teacher_metadata": {"qwen": dict(qwen_meta), "gemma": dict(gemma_meta)},
        "reviewed": False,
    }


def validate_dataset(path: str | Path, registry: TaskRegistry) -> dict[str, Any]:
    """Validate generated records and summarize task/split coverage."""

    records = _read_seed(path)
    task_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for record in records:
        task = registry.get(str(record.get("task_id")), int(record.get("task_version", 0)))
        if not record.get("state") or not record.get("question"):
            raise ValueError(f"{record.get('sample_id')}: state and question are required")
        task_counts[task.id] += 1
        split = str(record.get("provenance", {}).get("split", ""))
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{record.get('sample_id')}: invalid split")
        split_counts[split] += 1
    return {
        "path": str(Path(path).resolve()),
        "sample_count": len(records),
        "tasks": dict(sorted(task_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "ready": bool(records),
    }
