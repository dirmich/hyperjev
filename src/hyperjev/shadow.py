"""Privacy-preserving shadow comparison records for model promotion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import DecisionResponse, DecisionResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _result_signature(result: DecisionResult) -> str:
    return hashlib.sha256(
        json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class ShadowLogger:
    """Append Student-vs-baseline comparisons without raw request state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        *,
        request_id: str,
        state_sha256: str,
        primary: DecisionResponse,
        shadow: DecisionResponse,
    ) -> dict[str, Any]:
        question_ids = sorted(set(primary.results) | set(shadow.results))
        comparisons = []
        for question_id in question_ids:
            first = primary.results.get(question_id)
            second = shadow.results.get(question_id)
            comparisons.append(
                {
                    "question_id": question_id,
                    "primary_signature": _result_signature(first) if first else None,
                    "shadow_signature": _result_signature(second) if second else None,
                    "agree": first is not None and second is not None and first.to_dict() == second.to_dict(),
                }
            )
        record = {
            "record_type": "shadow",
            "created_at": _now(),
            "request_id": request_id,
            "state_sha256": state_sha256,
            "primary_model": primary.model,
            "shadow_model": shadow.model,
            "comparisons": comparisons,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record
