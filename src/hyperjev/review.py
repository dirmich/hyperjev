"""Append-only provenance, review, and human feedback stores."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    DecisionResult,
    parse_decision_result,
    validate_result_for_task,
)
from .registry import TaskDefinition


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ReviewStore:
    """Write review cases without persisting raw request state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        *,
        request_id: str,
        question_id: str,
        task: TaskDefinition,
        state_sha256: str,
        result: DecisionResult,
        route: str,
        trace: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        record = {
            "record_type": "review",
            "created_at": _utc_now(),
            "request_id": request_id,
            "question_id": question_id,
            "task": f"{task.id}@{task.version}",
            "state_sha256": state_sha256,
            "route": route,
            "reason": reason,
            "result": result.to_dict(),
            "trace": trace,
            "reviewed": False,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record


class FeedbackStore:
    """Append validated human corrections for later dataset construction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        *,
        request_id: str,
        question_id: str,
        task: TaskDefinition,
        correction: dict[str, Any],
        reviewer: str,
        reason: str = "",
    ) -> dict[str, Any]:
        if not reviewer.strip():
            raise ContractError("reviewer must not be empty")
        parsed = parse_decision_result(correction)
        validate_result_for_task(task, parsed)
        record = {
            "record_type": "feedback",
            "created_at": _utc_now(),
            "request_id": request_id,
            "question_id": question_id,
            "task": f"{task.id}@{task.version}",
            "correction": parsed.to_dict(),
            "reviewer": reviewer,
            "reason": reason,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record
