"""Deterministic synthetic review-queue generation for Phase 0."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import parse_decision_result, validate_result_for_task
from .prompts import PROMPT_VERSION
from .registry import TaskRegistry
from .samples import load_jsonl

GENERATOR_VERSION = "phase0-synthetic-v1"
GOLDEN_FEEDBACK_VERSION = "golden-feedback-v1"


class GoldenReviewError(ValueError):
    """Raised when a golden queue or feedback record is invalid."""


TASK_ORDER = (
    "memory.remember_worthy",
    "memory.type",
    "memory.importance",
    "query.route",
    "memory.relation",
    "wiki.semantic_change",
)


TEMPLATES: dict[str, dict[str, list[tuple[str, str, Any]]]] = {
    "memory.remember_worthy": {
        "ko": [
            ("다음 분기부터 배포 승인 절차를 바꾸기로 했다.", "장기 기억으로 저장할 가치가 있는가?", True),
            ("오늘 점심으로 비빔밥을 먹었다.", "장기 기억으로 저장할 가치가 있는가?", False),
            ("나는 답변을 받을 때 코드 예제를 먼저 보고 싶다.", "장기 기억으로 저장할 가치가 있는가?", True),
        ],
        "en": [
            ("We decided to change the deployment approval process next quarter.", "Is this worth long-term memory?", True),
            ("I had noodles for lunch today.", "Is this worth long-term memory?", False),
            ("I prefer seeing a code example before a long explanation.", "Is this worth long-term memory?", True),
        ],
    },
    "memory.type": {
        "ko": [
            ("기본 배포 브랜치는 main으로 유지하기로 했다.", "이 기억의 유형은 무엇인가?", "decision"),
            ("나는 매일 아침 차를 마신다.", "이 기억의 유형은 무엇인가?", "preference"),
            ("팀의 API는 PostgreSQL을 사용한다.", "이 기억의 유형은 무엇인가?", "fact"),
        ],
        "en": [
            ("We decided to keep main as the default deployment branch.", "What type of memory is this?", "decision"),
            ("I prefer tea in the morning.", "What type of memory is this?", "preference"),
            ("The team's API uses PostgreSQL.", "What type of memory is this?", "fact"),
        ],
    },
    "memory.importance": {
        "ko": [
            ("서비스의 기본 리전은 서울이다.", "이 기억의 중요도를 0에서 1로 평가하라.", 0.65),
            ("분기별 보안 키 교체 일정이 확정되었다.", "이 기억의 중요도를 0에서 1로 평가하라.", 0.9),
            ("이번 주에 회의실을 예약했다.", "이 기억의 중요도를 0에서 1로 평가하라.", 0.25),
        ],
        "en": [
            ("The service's default region is Seoul.", "Score the importance of this memory from 0 to 1.", 0.65),
            ("The quarterly security-key rotation schedule is finalized.", "Score the importance of this memory from 0 to 1.", 0.9),
            ("I reserved a meeting room for this week.", "Score the importance of this memory from 0 to 1.", 0.25),
        ],
    },
    "query.route": {
        "ko": [
            ("지난달에 결정한 데이터베이스 마이그레이션 계획을 찾아줘.", "이 질의의 검색 경로는 무엇인가?", "TEMPORAL"),
            ("내 프로필에 저장된 선호 언어를 알려줘.", "이 질의의 검색 경로는 무엇인가?", "PROFILE"),
            ("로그에 정확히 적힌 오류 코드를 찾아줘.", "이 질의의 검색 경로는 무엇인가?", "EXACT"),
        ],
        "en": [
            ("Find the database migration plan we decided on last month.", "What retrieval route should be used?", "TEMPORAL"),
            ("Tell me the preferred language stored in my profile.", "What retrieval route should be used?", "PROFILE"),
            ("Find the exact error code written in the logs.", "What retrieval route should be used?", "EXACT"),
        ],
    },
    "memory.relation": {
        "ko": [
            ("기존 기억: 기본 브랜치는 main이다.\n새 기억: 기본 브랜치는 trunk이다.", "두 기억의 관계는 무엇인가?", "CONTRADICT"),
            ("기존 기억: 배포는 금요일이다.\n새 기억: 배포는 금요일 오후다.", "두 기억의 관계는 무엇인가?", "EXTEND"),
            ("기존 기억: API는 PostgreSQL이다.\n새 기억: API는 PostgreSQL이다.", "두 기억의 관계는 무엇인가?", "DUPLICATE"),
        ],
        "en": [
            ("Old memory: the default branch is main.\nNew memory: the default branch is trunk.", "What is the relation between these memories?", "CONTRADICT"),
            ("Old memory: deployment is on Friday.\nNew memory: deployment is Friday afternoon.", "What is the relation between these memories?", "EXTEND"),
            ("Old memory: the API uses PostgreSQL.\nNew memory: the API uses PostgreSQL.", "What is the relation between these memories?", "DUPLICATE"),
        ],
    },
    "wiki.semantic_change": {
        "ko": [
            ("문서의 제목 스타일만 변경되었다.", "사실 의미가 변경되었는가?", False),
            ("문서에서 기본 리전이 서울에서 도쿄로 변경되었다.", "사실 의미가 변경되었는가?", True),
            ("문단의 공백만 정리되었다.", "사실 의미가 변경되었는가?", False),
        ],
        "en": [
            ("Only the document title style changed.", "Did the factual meaning change?", False),
            ("The default region changed from Seoul to Tokyo.", "Did the factual meaning change?", True),
            ("Only paragraph spacing was normalized.", "Did the factual meaning change?", False),
        ],
    },
}


def _sample(index: int, task_id: str, language: str, rng: random.Random) -> dict[str, Any]:
    state, question, target = rng.choice(TEMPLATES[task_id][language])
    return {
        "sample_id": f"phase0-synthetic-{index:04d}",
        "task_id": task_id,
        "task_version": 1,
        "state": state,
        "question": question,
        "target": target,
        "language": language,
        "domain": "phase0-synthetic",
        "source": {"kind": "synthetic", "document_id": None, "span": None},
        "labels": {"qwen": None, "gemma": None, "human": None},
        "review": {"status": "pending", "reviewer": None},
        "provenance": {
            "prompt_version": PROMPT_VERSION,
            "teacher_model": None,
            "generator": GENERATOR_VERSION,
            "seeded_index": index,
            "created_at": "2026-09-20T00:00:00Z",
        },
    }


def generate_review_queue(
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    count: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a deterministic synthetic queue awaiting human review."""

    if count < 1:
        raise ValueError("count must be positive")
    for task_id in TASK_ORDER:
        registry.get(task_id, 1)
    rng = random.Random(seed)
    task_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(1, count + 1):
            task_id = TASK_ORDER[(index - 1) % len(TASK_ORDER)]
            language = "en" if rng.random() < 0.2 else "ko"
            sample = _sample(index, task_id, language, rng)
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            task_counts[task_id] += 1
            language_counts[language] += 1
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "sample_count": count,
        "seed": seed,
        "generator": GENERATOR_VERSION,
        "task_counts": dict(sorted(task_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "sha256": digest,
        "review_status": "pending",
        "human_reviewed": False,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: str | Path, *, label: str) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {label} {source}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise GoldenReviewError(f"{source}:{line_number}: {label} record must be an object")
        records.append(record)
    if not records:
        raise ValueError(f"{label} is empty: {source}")
    return records


def append_golden_feedback(
    queue_path: str | Path,
    feedback_path: str | Path,
    registry: TaskRegistry,
    *,
    sample_id: str,
    correction: dict[str, Any],
    reviewer: str,
    reason: str = "",
) -> dict[str, Any]:
    """Append one validated typed human label without changing the source queue."""

    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    queue_records = _read_jsonl(queue_path, label="golden queue")
    samples = {sample.sample_id: sample for sample in load_jsonl(queue_path, registry)}
    sample = samples.get(sample_id)
    if sample is None:
        raise ValueError(f"unknown golden sample_id: {sample_id}")
    task = registry.get(sample.task_id, sample.task_version)
    parsed = parse_decision_result(correction)
    validate_result_for_task(task, parsed)
    queue_digest = hashlib.sha256(Path(queue_path).read_bytes()).hexdigest()
    record = {
        "record_type": "golden_feedback",
        "feedback_version": GOLDEN_FEEDBACK_VERSION,
        "created_at": _utc_now(),
        "sample_id": sample_id,
        "task": f"{task.id}@{task.version}",
        "queue_sha256": queue_digest,
        "correction": parsed.to_dict(),
        "reviewer": reviewer.strip(),
        "reason": reason,
    }
    path = Path(feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": str(path.resolve()),
        "sample_id": sample_id,
        "task": record["task"],
        "queue_sha256": queue_digest,
        "feedback_version": GOLDEN_FEEDBACK_VERSION,
        "records": 1,
        "queue_records": len(queue_records),
    }


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def apply_golden_feedback(
    queue_path: str | Path,
    feedback_path: str | Path,
    output_path: str | Path,
    registry: TaskRegistry,
) -> dict[str, Any]:
    """Create a reviewed queue copy from append-only feedback records."""

    queue = Path(queue_path)
    output = Path(output_path)
    if queue.resolve() == output.resolve():
        raise ValueError("reviewed queue output must differ from the source queue")
    queue_records = _read_jsonl(queue, label="golden queue")
    load_jsonl(queue, registry)
    queue_digest = hashlib.sha256(queue.read_bytes()).hexdigest()
    samples = {str(record["sample_id"]): record for record in queue_records}
    feedback_records = _read_jsonl(feedback_path, label="golden feedback")
    latest: dict[str, dict[str, Any]] = {}
    for record in feedback_records:
        if record.get("record_type") != "golden_feedback":
            raise ValueError("feedback file contains a non-golden feedback record")
        sample_id = str(record.get("sample_id", ""))
        if sample_id not in samples:
            raise ValueError(f"feedback references unknown golden sample_id: {sample_id}")
        if record.get("queue_sha256") != queue_digest:
            raise ValueError(f"feedback queue digest does not match source queue: {sample_id}")
        raw_task = str(record.get("task", ""))
        task_id, task_version_text = raw_task.rsplit("@", 1)
        try:
            task_version = int(task_version_text)
        except ValueError as exc:
            raise ValueError(f"invalid feedback task reference: {raw_task!r}") from exc
        sample = samples[sample_id]
        if raw_task != f"{sample['task_id']}@{sample['task_version']}":
            raise ValueError(f"feedback task does not match sample: {sample_id}")
        task = registry.get(task_id, task_version)
        correction = record.get("correction")
        if not isinstance(correction, dict):
            raise GoldenReviewError(f"feedback correction must be an object: {sample_id}")
        parsed = parse_decision_result(correction)
        validate_result_for_task(task, parsed)
        latest[sample_id] = record | {"correction": parsed.to_dict()}

    reviewed_records: list[dict[str, Any]] = []
    for original in queue_records:
        record = json.loads(json.dumps(original))
        feedback = latest.get(str(record["sample_id"]))
        if feedback is not None:
            labels = dict(record.get("labels", {}))
            labels["human"] = feedback["correction"]
            record["labels"] = labels
            review = dict(record.get("review", {}))
            review.update(
                {
                    "status": "reviewed",
                    "reviewer": feedback["reviewer"],
                    "reviewed_at": feedback["created_at"],
                    "reason": feedback.get("reason", ""),
                }
            )
            record["review"] = review
        reviewed_records.append(record)
    _atomic_write_jsonl(output, reviewed_records)
    reviewed_count = len(latest)
    return {
        "path": str(output.resolve()),
        "source_path": str(queue.resolve()),
        "feedback_path": str(Path(feedback_path).resolve()),
        "sample_count": len(reviewed_records),
        "human_reviewed_count": reviewed_count,
        "pending_count": len(reviewed_records) - reviewed_count,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "ready": reviewed_count == len(reviewed_records),
    }
