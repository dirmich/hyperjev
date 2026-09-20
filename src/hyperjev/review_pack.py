"""Local human-review packs that join queue context with teacher drafts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import TaskRegistry
from .samples import load_jsonl

REVIEW_PACK_VERSION = "golden-review-pack-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"review draft is empty: {path}")
    return records


def export_review_pack(
    queue_path: str | Path,
    draft_path: str | Path,
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Join queue text and a teacher draft for explicit local human review.

    This is intentionally opt-in because it writes raw state/question. The
    synthetic target and queue labels are never copied into the review pack,
    preventing target leakage and accidental human-review anchoring.
    """

    if not include_raw:
        raise ValueError("review pack requires explicit include_raw=True")
    queue = Path(queue_path)
    draft = Path(draft_path)
    samples = load_jsonl(queue, registry)
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    draft_sha256 = hashlib.sha256(draft.read_bytes()).hexdigest()
    draft_records = _read_records(draft)
    manifest = draft_records[0]
    if manifest.get("record_type") != "golden_teacher_draft_manifest":
        raise ValueError("draft must start with a golden teacher draft manifest")
    if manifest.get("queue_sha256") != queue_sha256:
        raise ValueError("queue SHA-256 does not match the draft manifest")
    by_sample: dict[str, dict[str, Any]] = {}
    for record in draft_records[1:]:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in by_sample:
            raise ValueError(f"draft contains duplicate or empty sample_id: {sample_id!r}")
        by_sample[sample_id] = record
    missing = [sample.sample_id for sample in samples if sample.sample_id not in by_sample]
    if missing:
        raise ValueError(f"draft is missing {len(missing)} queue samples")

    pack_manifest = {
        "record_type": "golden_review_pack_manifest",
        "review_pack_version": REVIEW_PACK_VERSION,
        "created_at": _utc_now(),
        "queue_path": str(queue.resolve()),
        "queue_sha256": queue_sha256,
        "draft_path": str(draft.resolve()),
        "draft_sha256": draft_sha256,
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "prompt_version": manifest.get("prompt_version"),
        "sample_count": len(samples),
        "raw_inputs_included": True,
        "target_excluded": True,
    }
    output_records: list[dict[str, Any]] = [pack_manifest]
    for sample in samples:
        task = registry.get(sample.task_id, sample.task_version)
        draft_record = by_sample[sample.sample_id]
        output_records.append(
            {
                "record_type": "golden_review_item",
                "review_pack_version": REVIEW_PACK_VERSION,
                "sample_id": sample.sample_id,
                "task": f"{task.id}@{task.version}",
                "state": sample.state,
                "question": sample.question,
                "language": sample.language,
                "domain": sample.domain,
                "teacher": {
                    "provider": draft_record.get("provider"),
                    "model": draft_record.get("model"),
                    "normalized_result": draft_record.get("normalized_result"),
                    "schema_valid": draft_record.get("schema_valid"),
                    "status": draft_record.get("status"),
                    "error": draft_record.get("error"),
                    "response_sha256": draft_record.get("response_sha256"),
                },
                "human_correction": None,
                "review_status": "pending",
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "output": str(output.resolve()),
        "manifest": pack_manifest,
        "records": len(output_records) - 1,
    }
