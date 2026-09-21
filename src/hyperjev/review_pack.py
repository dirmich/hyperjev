"""Local human-review packs that join queue context with teacher drafts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import parse_decision_result, parse_task_reference, validate_result_for_task
from .golden import GOLDEN_FEEDBACK_VERSION, append_golden_feedback
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


def _teacher_priority(item: dict[str, Any]) -> tuple[int, float, str]:
    """Sort uncertain teacher items before confident items without target leakage."""

    teacher = item.get("teacher", {})
    if teacher.get("schema_valid") is not True:
        return (0, 0.0, str(item.get("sample_id", "")))
    if teacher.get("schema_repaired") is True:
        return (1, 0.0, str(item.get("sample_id", "")))
    result = teacher.get("normalized_result") or {}
    confidence = 0.0
    if result.get("type") == "boolean":
        probability = float(result.get("probability", 0.0))
        confidence = probability if result.get("value") else 1.0 - probability
    elif result.get("type") == "choice":
        probabilities = result.get("probabilities", {})
        if isinstance(probabilities, dict) and probabilities:
            confidence = max(float(value) for value in probabilities.values())
    elif result.get("type") == "score":
        interval = result.get("interval_90", [])
        if isinstance(interval, list) and len(interval) == 2:
            confidence = max(0.0, 1.0 - float(interval[1]) + float(interval[0]))
    return (2, confidence, str(item.get("sample_id", "")))


def _counterfactual_group(item: dict[str, Any]) -> str:
    source = item.get("source")
    if isinstance(source, dict):
        group = source.get("counterfactual_group_id") or source.get("semantic_group_id")
        if group:
            return str(group)
    return str(item.get("sample_id", ""))


def _teacher_signature(item: dict[str, Any]) -> tuple[str, Any] | None:
    result = item.get("teacher", {}).get("normalized_result") or {}
    result_type = result.get("type")
    if result_type == "boolean" and isinstance(result.get("value"), bool):
        return (result_type, result["value"])
    if result_type == "choice" and result.get("selected"):
        return (result_type, str(result["selected"]))
    if result_type == "score" and isinstance(result.get("value"), (int, float)):
        return (result_type, round(float(result["value"]), 6))
    return None


def _teacher_action(item: dict[str, Any]) -> str | None:
    """Return a teacher-selected choice for focus ordering, if available."""

    signature = _teacher_signature(item)
    if signature is None or signature[0] != "choice":
        return None
    return str(signature[1])


def _student_signature(prediction: dict[str, Any]) -> tuple[str, Any] | None:
    rendered = prediction.get("prediction") or {}
    result_type = rendered.get("type")
    if result_type == "boolean" and isinstance(rendered.get("value"), bool):
        return (result_type, rendered["value"])
    if result_type == "choice" and rendered.get("selected"):
        return (result_type, str(rendered["selected"]))
    if result_type == "score" and isinstance(rendered.get("value"), (int, float)):
        return (result_type, round(float(rendered["value"]), 6))
    return None


def _is_teacher_collision(items: list[dict[str, Any]]) -> bool:
    signatures = [_teacher_signature(item) for item in items]
    return len(signatures) > 1 and signatures[0] is not None and len(set(signatures)) == 1


def _prioritize_review_items(
    items: list[dict[str, Any]],
    *,
    student_confidence: dict[str, float] | None = None,
    student_disagreement: set[str] | None = None,
    focus_actions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Order uncertain items while keeping counterfactual siblings adjacent.

    Student confidence is used only for ordering. It is never copied into a
    review item, so a blind reviewer cannot be anchored by the model prediction.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for item in items:
        group_id = _counterfactual_group(item)
        if group_id not in groups:
            groups[group_id] = []
            group_order.append(group_id)
        groups[group_id].append(item)
    ordered_groups = sorted(
        group_order,
        key=lambda group_id: (
            0 if _is_teacher_collision(groups[group_id]) else 1,
            0
            if focus_actions is not None
            and any(_teacher_action(item) in focus_actions for item in groups[group_id])
            else 1,
            0
            if student_disagreement is not None
            and any(str(item.get("sample_id", "")) in student_disagreement for item in groups[group_id])
            else 1,
            min(
                student_confidence.get(str(item.get("sample_id", "")), 1.0)
                for item in groups[group_id]
            )
            if student_confidence is not None
            else 1.0,
            min(_teacher_priority(item) for item in groups[group_id]),
            group_id,
        ),
    )
    ordered: list[dict[str, Any]] = []
    for group_id in ordered_groups:
        ordered.extend(sorted(groups[group_id], key=_teacher_priority))
    return ordered


def _priority_stats(items: list[dict[str, Any]]) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(_counterfactual_group(item), []).append(item)
    collision_groups = [group for group in groups.values() if _is_teacher_collision(group)]
    return {
        "counterfactual_group_count": len(groups),
        "collision_group_count": len(collision_groups),
        "collision_item_count": sum(len(group) for group in collision_groups),
    }


def export_review_pack(
    queue_path: str | Path,
    draft_path: str | Path,
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    include_raw: bool = False,
    prioritize: bool = False,
    student_checkpoint: str | Path | None = None,
    student_device: str = "cpu",
    focus_actions: set[str] | None = None,
    review_split: str | None = None,
) -> dict[str, Any]:
    """Join queue text and a teacher draft for explicit local human review.

    This is intentionally opt-in because it writes raw state/question. The
    synthetic target and queue labels are never copied into the review pack,
    preventing target leakage and accidental human-review anchoring.
    """

    if not include_raw:
        raise ValueError("review pack requires explicit include_raw=True")
    if student_checkpoint is not None and not prioritize:
        raise ValueError("student checkpoint priority requires prioritize=True")
    if focus_actions and not prioritize:
        raise ValueError("action focus priority requires prioritize=True")
    if review_split is not None and review_split not in {"train", "validation", "test"}:
        raise ValueError("review_split must be train, validation, or test")
    queue = Path(queue_path)
    draft = Path(draft_path)
    all_samples = load_jsonl(queue, registry)
    samples = [
        sample
        for sample in all_samples
        if review_split is None or sample.provenance.get("split") == review_split
    ]
    if not samples:
        raise ValueError(f"review pack has no samples for split: {review_split}")
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    draft_sha256 = hashlib.sha256(draft.read_bytes()).hexdigest()
    draft_records = _read_records(draft)
    manifest = draft_records[0]
    if manifest.get("record_type") not in {
        "golden_teacher_draft_manifest",
        "golden_teacher_adjudication_manifest",
    }:
        raise ValueError("draft must start with a supported teacher draft manifest")
    if manifest.get("queue_sha256") != queue_sha256:
        raise ValueError("queue SHA-256 does not match the draft manifest")
    pack_provider = manifest.get("provider")
    pack_model = manifest.get("model")
    pack_prompt_version = manifest.get("prompt_version")
    if manifest.get("record_type") == "golden_teacher_adjudication_manifest":
        pack_provider = manifest.get("provider") or "qwen+gemma"
        pack_model = "|".join(
            str(model)
            for model in (manifest.get("qwen_model"), manifest.get("gemma_model"))
            if model
        ) or None
        prompt_versions = {
            value
            for value in (manifest.get("qwen_prompt_version"), manifest.get("gemma_prompt_version"))
            if value is not None
        }
        pack_prompt_version = next(iter(prompt_versions)) if len(prompt_versions) == 1 else None
    by_sample: dict[str, dict[str, Any]] = {}
    for record in draft_records[1:]:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in by_sample:
            raise ValueError(f"draft contains duplicate or empty sample_id: {sample_id!r}")
        by_sample[sample_id] = record
    missing = [sample.sample_id for sample in samples if sample.sample_id not in by_sample]
    if missing:
        raise ValueError(f"draft is missing {len(missing)} queue samples")

    student_confidence: dict[str, float] | None = None
    student_disagreement: set[str] | None = None
    student_predictions: dict[str, dict[str, Any]] | None = None
    student_checkpoint_sha256: str | None = None
    if student_checkpoint is not None:
        from .student_inference import evaluate_student_checkpoint

        checkpoint = Path(student_checkpoint)
        evaluation = evaluate_student_checkpoint(
            checkpoint,
            queue,
            registry,
            split="all",
            minimum_confidence=0.0,
            device=student_device,
        )
        student_confidence = {
            str(row["sample_id"]): float(row["confidence"])
            for row in evaluation["predictions"]
        }
        student_predictions = {
            str(row["sample_id"]): row for row in evaluation["predictions"]
        }
        if set(student_confidence) != {sample.sample_id for sample in samples}:
            raise ValueError("student evaluation does not cover every review sample")
        student_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    pack_manifest = {
        "record_type": "golden_review_pack_manifest",
        "review_pack_version": REVIEW_PACK_VERSION,
        "created_at": _utc_now(),
        "queue_path": str(queue.resolve()),
        "queue_sha256": queue_sha256,
        "draft_path": str(draft.resolve()),
        "draft_sha256": draft_sha256,
        "provider": pack_provider,
        "model": pack_model,
        "prompt_version": pack_prompt_version,
        "sample_count": len(samples),
        "review_split": review_split,
        "raw_inputs_included": True,
        "target_excluded": True,
        "priority_order": (
            "student_uncertainty_then_teacher"
            if student_confidence is not None
            else "uncertain_first"
            if prioritize
            else "queue_order"
        ),
        "priority_stats": None,
        "student_checkpoint": str(Path(student_checkpoint).resolve())
        if student_checkpoint is not None
        else None,
        "student_checkpoint_sha256": student_checkpoint_sha256,
        "focus_actions": sorted(focus_actions) if focus_actions else None,
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
                "source": {
                    field: sample.source[field]
                    for field in (
                        "kind",
                        "scenario_id",
                        "episode_id",
                        "semantic_group_id",
                        "counterfactual_group_id",
                        "pair_side",
                    )
                    if field in sample.source
                },
                "split": sample.provenance.get("split"),
                "teacher": {
                    "provider": draft_record.get("provider"),
                    "model": draft_record.get("model"),
                    "normalized_result": draft_record.get("normalized_result"),
                    "schema_valid": draft_record.get("schema_valid"),
                    "schema_repaired": draft_record.get("schema_repaired", False),
                    "status": draft_record.get("status"),
                    "error": draft_record.get("error"),
                    "response_sha256": draft_record.get("response_sha256"),
                    "comparison": draft_record.get("teacher_comparison"),
                },
                "human_correction": None,
                "review_status": "pending",
            }
        )
    if prioritize:
        pack_manifest["priority_stats"] = _priority_stats(output_records[1:])
        if focus_actions:
            pack_manifest["priority_stats"]["focus_action_item_count"] = sum(
                _teacher_action(item) in focus_actions for item in output_records[1:]
            )
        if student_confidence is not None:
            student_disagreement = {
                str(item["sample_id"])
                for item in output_records[1:]
                if (
                    _teacher_signature(item) is not None
                    and student_predictions is not None
                    and _student_signature(student_predictions[str(item["sample_id"])])
                    != _teacher_signature(item)
                )
            }
            pack_manifest["priority_stats"]["student_uncertain_item_count"] = sum(
                confidence < 0.90 for confidence in student_confidence.values()
            )
            pack_manifest["priority_stats"]["student_teacher_disagreement_item_count"] = len(
                student_disagreement
            )
            if student_disagreement:
                pack_manifest["priority_order"] = (
                    "student_disagreement_then_uncertainty_then_teacher"
                )
            if focus_actions:
                if student_disagreement:
                    pack_manifest["priority_order"] = (
                        "focus_action_then_student_disagreement_then_uncertainty_then_teacher"
                    )
                elif student_confidence is not None:
                    pack_manifest["priority_order"] = "focus_action_then_uncertainty_then_teacher"
                else:
                    pack_manifest["priority_order"] = "focus_action_then_teacher"
        elif focus_actions:
            pack_manifest["priority_order"] = "focus_action_then_teacher"
        output_records[1:] = _prioritize_review_items(
            output_records[1:],
            student_confidence=student_confidence,
            student_disagreement=student_disagreement,
            focus_actions=focus_actions,
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


def _feedback_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    return {
        str(record["sample_id"]): record
        for record in _read_records(path)
        if record.get("record_type") == "golden_feedback" and record.get("sample_id")
    }


def _validated_feedback_by_sample(
    path: str | Path,
    queue: Path,
    registry: TaskRegistry,
) -> dict[str, dict[str, Any]]:
    """Load one reviewer stream and validate every append-only correction."""

    feedback_path = Path(path)
    if not feedback_path.exists() or not feedback_path.read_text(encoding="utf-8").strip():
        return {}
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    samples = {sample.sample_id: sample for sample in load_jsonl(queue, registry)}
    latest: dict[str, dict[str, Any]] = {}
    for record in _read_records(feedback_path):
        if record.get("record_type") != "golden_feedback":
            raise ValueError(f"feedback contains an unsupported record: {feedback_path}")
        sample_id = str(record.get("sample_id", ""))
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(f"feedback references unknown sample_id: {sample_id}")
        if record.get("queue_sha256") != queue_sha256:
            raise ValueError(f"feedback queue digest does not match: {sample_id}")
        if not str(record.get("reviewer", "")).strip():
            raise ValueError(f"feedback reviewer is empty: {sample_id}")
        expected_task = f"{sample.task_id}@{sample.task_version}"
        if record.get("task") != expected_task:
            raise ValueError(f"feedback task does not match queue sample: {sample_id}")
        task = registry.get(sample.task_id, sample.task_version)
        correction = record.get("correction")
        if not isinstance(correction, dict):
            raise TypeError(f"feedback correction is invalid: {sample_id}")
        parsed = parse_decision_result(correction)
        validate_result_for_task(task, parsed)
        latest[sample_id] = record | {"correction": parsed.to_dict()}
    return latest


def control_review_status(
    queue_path: str | Path,
    feedback_path: str | Path,
    registry: TaskRegistry,
) -> dict[str, Any]:
    """Report human-review progress without reading synthetic targets."""

    queue = Path(queue_path)
    feedback = Path(feedback_path)
    samples = load_jsonl(queue, registry)
    latest = _validated_feedback_by_sample(feedback, queue, registry)
    total_by_split: dict[str, int] = {}
    reviewed_by_split: dict[str, int] = {}
    reviewed_by_action: dict[str, int] = {}
    for sample in samples:
        split = str(sample.provenance.get("split", "unknown"))
        total_by_split[split] = total_by_split.get(split, 0) + 1
        if sample.sample_id not in latest:
            continue
        reviewed_by_split[split] = reviewed_by_split.get(split, 0) + 1
        correction = latest[sample.sample_id]["correction"]
        if correction.get("type") == "choice":
            action = str(correction.get("selected", ""))
        elif correction.get("type") == "boolean":
            action = str(bool(correction.get("value"))).lower()
        else:
            action = str(correction.get("type", "unknown"))
        reviewed_by_action[action] = reviewed_by_action.get(action, 0) + 1
    reviewed_count = len(latest)
    sample_count = len(samples)
    by_split = {
        split: {
            "count": count,
            "reviewed_count": reviewed_by_split.get(split, 0),
            "pending_count": count - reviewed_by_split.get(split, 0),
            "coverage": round(reviewed_by_split.get(split, 0) / count, 6),
        }
        for split, count in sorted(total_by_split.items())
    }
    test = by_split.get("test", {"count": 0, "reviewed_count": 0, "pending_count": 0})
    return {
        "record_type": "control_review_status",
        "queue_path": str(queue.resolve()),
        "queue_sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
        "feedback_path": str(feedback.resolve()),
        "feedback_sha256": hashlib.sha256(feedback.read_bytes()).hexdigest()
        if feedback.exists()
        else None,
        "sample_count": sample_count,
        "reviewed_count": reviewed_count,
        "pending_count": sample_count - reviewed_count,
        "coverage": round(reviewed_count / sample_count, 6) if sample_count else 0.0,
        "by_split": by_split,
        "reviewed_by_action": dict(sorted(reviewed_by_action.items())),
        "test_ready": bool(test["count"] and test["pending_count"] == 0),
        "ready_for_materialize": reviewed_count == sample_count,
    }


def _correction_signature(correction: dict[str, Any]) -> str:
    return json.dumps(correction, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare_control_reviews(
    queue_path: str | Path,
    reviewer_a_feedback_path: str | Path,
    reviewer_b_feedback_path: str | Path,
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    minimum_agreement: float = 0.98,
) -> dict[str, Any]:
    """Compare two teacher-blind reviewer streams without using synthetic targets."""

    if not 0.0 <= minimum_agreement <= 1.0:
        raise ValueError("minimum_agreement must be between 0 and 1")
    queue = Path(queue_path)
    samples = load_jsonl(queue, registry)
    reviewer_a = _validated_feedback_by_sample(reviewer_a_feedback_path, queue, registry)
    reviewer_b = _validated_feedback_by_sample(reviewer_b_feedback_path, queue, registry)
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    items: list[dict[str, Any]] = []
    pair_counts: dict[str, int] = {}
    action_agreement: dict[str, dict[str, int]] = {}
    agreement_count = 0
    disagreement_count = 0
    missing_count = 0
    for sample in samples:
        first = reviewer_a.get(sample.sample_id)
        second = reviewer_b.get(sample.sample_id)
        first_correction = first.get("correction") if first else None
        second_correction = second.get("correction") if second else None
        if first_correction is None or second_correction is None:
            status = "missing_reviewer"
            missing_count += 1
        elif _correction_signature(first_correction) == _correction_signature(second_correction):
            status = "agreement"
            agreement_count += 1
        else:
            status = "disagreement"
            disagreement_count += 1
        if first_correction is not None and second_correction is not None:
            first_value = str(first_correction.get("selected", first_correction.get("value", "")))
            second_value = str(second_correction.get("selected", second_correction.get("value", "")))
            pair = f"{first_value}->{second_value}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            for action, reviewer_key in (
                (first_value, "reviewer_a_count"),
                (second_value, "reviewer_b_count"),
            ):
                stats = action_agreement.setdefault(
                    action,
                    {
                        "reviewer_a_count": 0,
                        "reviewer_b_count": 0,
                        "agreement_count": 0,
                        "disagreement_count": 0,
                    },
                )
                stats[reviewer_key] += 1
            if first_value == second_value:
                action_agreement[first_value]["agreement_count"] += 1
            else:
                action_agreement[first_value]["disagreement_count"] += 1
                action_agreement[second_value]["disagreement_count"] += 1
        items.append(
            {
                "record_type": "control_dual_review_item",
                "sample_id": sample.sample_id,
                "task": f"{sample.task_id}@{sample.task_version}",
                "state": sample.state,
                "question": sample.question,
                "language": sample.language,
                "domain": sample.domain,
                "source": {
                    field: sample.source[field]
                    for field in (
                        "kind",
                        "scenario_id",
                        "episode_id",
                        "semantic_group_id",
                        "counterfactual_group_id",
                        "pair_side",
                    )
                    if field in sample.source
                },
                "reviewer_a": {
                    "reviewer": first.get("reviewer") if first else None,
                    "correction": first_correction,
                },
                "reviewer_b": {
                    "reviewer": second.get("reviewer") if second else None,
                    "correction": second_correction,
                },
                "status": status,
            }
        )
    comparable_count = agreement_count + disagreement_count
    manifest = {
        "record_type": "control_dual_review_manifest",
        "review_version": "control-dual-review-v2",
        "created_at": _utc_now(),
        "queue_path": str(queue.resolve()),
        "queue_sha256": queue_sha256,
        "reviewer_a_feedback_path": str(Path(reviewer_a_feedback_path).resolve()),
        "reviewer_b_feedback_path": str(Path(reviewer_b_feedback_path).resolve()),
        "sample_count": len(samples),
        "reviewer_a_count": len(reviewer_a),
        "reviewer_b_count": len(reviewer_b),
        "both_labeled_count": comparable_count,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "missing_count": missing_count,
        "agreement_rate": round(agreement_count / comparable_count, 6) if comparable_count else None,
        "minimum_agreement": minimum_agreement,
        "agreement_gate": (
            missing_count == 0
            and comparable_count > 0
            and agreement_count / comparable_count >= minimum_agreement
        ),
        "target_excluded": True,
        "label_pair_counts": dict(sorted(pair_counts.items())),
        "label_agreement_by_action": {
            action: {
                **stats,
                "comparable_count": stats["agreement_count"] + stats["disagreement_count"],
                "agreement_rate": round(
                    stats["agreement_count"]
                    / (stats["agreement_count"] + stats["disagreement_count"]),
                    6,
                )
                if stats["agreement_count"] + stats["disagreement_count"]
                else None,
            }
            for action, stats in sorted(action_agreement.items())
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return {"manifest": manifest, "output": str(output.resolve()), "records": len(items)}


def format_control_review_action_summary(manifest: dict[str, Any]) -> str:
    """Format a concise human-facing summary without exposing synthetic targets."""

    action_stats = manifest.get("label_agreement_by_action", {})
    if not isinstance(action_stats, dict) or not action_stats:
        return "dual-review action summary: no comparable action labels"
    ordered = sorted(
        action_stats.items(),
        key=lambda entry: (
            entry[1].get("agreement_rate") is None,
            entry[1].get("agreement_rate", 1.0),
            -entry[1].get("disagreement_count", 0),
            entry[0],
        ),
    )
    lines = [
        "dual-review action summary (diagnostic only; not accuracy):",
        "  action  agreement_rate  disagreement  comparable",
    ]
    for action, stats in ordered:
        rate = stats.get("agreement_rate")
        rate_text = "n/a" if rate is None else f"{float(rate):.3f}"
        lines.append(
            f"  {action:<7} {rate_text:>15} {stats.get('disagreement_count', 0):>13}"
            f" {stats.get('comparable_count', 0):>11}"
        )
    return "\n".join(lines)


def load_control_review_focus_actions(
    manifest_path: str | Path,
    *,
    maximum_agreement: float = 0.98,
) -> set[str]:
    """Load low-agreement actions from a target-free dual-review manifest."""

    if not 0.0 <= maximum_agreement <= 1.0:
        raise ValueError("maximum_agreement must be between 0 and 1")
    records = _read_records(Path(manifest_path))
    manifest = records[0]
    if manifest.get("record_type") != "control_dual_review_manifest":
        raise ValueError("focus manifest must be a control dual-review manifest")
    if manifest.get("target_excluded") is not True:
        raise ValueError("focus manifest must explicitly exclude synthetic targets")
    action_stats = manifest.get("label_agreement_by_action", {})
    if not isinstance(action_stats, dict):
        raise TypeError("focus manifest action agreement is invalid")
    return {
        str(action)
        for action, stats in action_stats.items()
        if isinstance(stats, dict)
        and stats.get("agreement_rate") is not None
        and float(stats["agreement_rate"]) < maximum_agreement
    }


def finalize_control_reviews(
    queue_path: str | Path,
    agreement_path: str | Path,
    adjudication_feedback_path: str | Path,
    output_feedback_path: str | Path,
    registry: TaskRegistry,
) -> dict[str, Any]:
    """Create a normal feedback stream from agreements plus adjudicated conflicts."""

    queue = Path(queue_path)
    agreement_records = _read_records(Path(agreement_path))
    manifest = agreement_records[0]
    if manifest.get("record_type") != "control_dual_review_manifest":
        raise ValueError("agreement report must start with a dual-review manifest")
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    if manifest.get("queue_sha256") != queue_sha256:
        raise ValueError("agreement queue digest does not match the source queue")
    adjudications = _validated_feedback_by_sample(adjudication_feedback_path, queue, registry)
    output_records: list[dict[str, Any]] = []
    agreement_count = 0
    adjudicated_count = 0
    for item in agreement_records[1:]:
        status = item.get("status")
        if status == "agreement":
            correction = item["reviewer_a"]["correction"]
            reviewer_ids = (item["reviewer_a"].get("reviewer"), item["reviewer_b"].get("reviewer"))
            reviewer = "dual-agreement:" + "+".join(str(value) for value in reviewer_ids)
            reason = "control dual-review agreement"
            agreement_count += 1
        elif status == "disagreement":
            adjudication = adjudications.get(str(item["sample_id"]))
            if adjudication is None:
                raise ValueError(f"disagreement is not adjudicated: {item['sample_id']}")
            correction = adjudication["correction"]
            reviewer = str(adjudication["reviewer"])
            reason = "control dual-review adjudication"
            adjudicated_count += 1
        else:
            raise ValueError(f"dual review is incomplete: {item['sample_id']}")
        output_records.append(
            {
                "record_type": "golden_feedback",
                "feedback_version": GOLDEN_FEEDBACK_VERSION,
                "created_at": _utc_now(),
                "sample_id": item["sample_id"],
                "task": item["task"],
                "queue_sha256": queue_sha256,
                "correction": correction,
                "reviewer": reviewer,
                "reason": reason,
            }
        )
    output = Path(output_feedback_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "record_type": "control_dual_review_finalization",
        "output": str(output.resolve()),
        "sample_count": len(output_records),
        "agreement_count": agreement_count,
        "adjudicated_count": adjudicated_count,
        "ready": len(output_records) == int(manifest.get("sample_count", -1)),
    }


def run_adjudication_session(
    agreement_path: str | Path,
    queue_path: str | Path,
    feedback_path: str | Path,
    registry: TaskRegistry,
    *,
    reviewer: str,
    input_fn: Any = input,
    output_fn: Any = print,
) -> dict[str, Any]:
    """Resolve dual-review disagreements with a third typed human decision."""

    agreement_records = _read_records(Path(agreement_path))
    manifest = agreement_records[0]
    if manifest.get("record_type") != "control_dual_review_manifest":
        raise ValueError("agreement report must start with a dual-review manifest")
    queue = Path(queue_path)
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    if manifest.get("queue_sha256") != queue_sha256:
        raise ValueError("agreement queue digest does not match the source queue")
    items = [item for item in agreement_records[1:] if item.get("status") == "disagreement"]
    feedback = Path(feedback_path)
    latest = _validated_feedback_by_sample(feedback, queue, registry)
    index = next((position for position, item in enumerate(items) if item["sample_id"] not in latest), 0)
    saved = 0
    stopped = False
    while 0 <= index < len(items):
        item = items[index]
        sample_id = str(item["sample_id"])
        output_fn("")
        output_fn(f"[adjudication {index + 1}/{len(items)}] {sample_id}  {item.get('task')}")
        output_fn(f"state: {item.get('state')}")
        output_fn(f"question: {item.get('question')}")
        output_fn("Reviewer A: " + json.dumps(item["reviewer_a"], ensure_ascii=False, sort_keys=True))
        output_fn("Reviewer B: " + json.dumps(item["reviewer_b"], ensure_ascii=False, sort_keys=True))
        if sample_id in latest:
            output_fn("Current adjudication: " + json.dumps(latest[sample_id]["correction"], ensure_ascii=False))
        output_fn("Commands: [e]nter final value  [n]ext  [p]revious  [s]kip  [q]uit")
        try:
            command = input_fn("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            stopped = True
            break
        if command in {"q", "quit"}:
            stopped = True
            break
        if command in {"n", "next", "s", "skip"}:
            if index < len(items) - 1:
                index += 1
            continue
        if command in {"p", "previous", "prev"}:
            if index > 0:
                index -= 1
            continue
        if command not in {"e", "edit"}:
            output_fn("Use e, n, p, s, or q.")
            continue
        try:
            raw_value = input_fn("final value: ")
            correction = _correction_from_value(item, raw_value, registry)
            record = append_golden_feedback(
                queue,
                feedback,
                registry,
                sample_id=sample_id,
                correction=correction,
                reviewer=reviewer,
                reason="control dual-review adjudication",
            )
        except (EOFError, KeyboardInterrupt):
            stopped = True
            break
        except (TypeError, ValueError) as exc:
            output_fn(f"Correction rejected: {exc}")
            continue
        latest[sample_id] = {
            "sample_id": sample_id,
            "correction": correction,
            "created_at": record.get("created_at"),
        }
        saved += 1
        if index < len(items) - 1:
            index += 1
        else:
            break
    return {
        "disagreement_count": len(items),
        "adjudicated_count": len(latest),
        "pending_count": len(items) - len(latest),
        "saved_in_session": saved,
        "stopped": stopped,
    }


def _correction_from_value(item: dict[str, Any], raw_value: str, registry: TaskRegistry) -> dict[str, Any]:
    """Build a valid typed correction from one human-entered task value."""

    task_id, task_version = parse_task_reference(str(item.get("task", "")))
    task = registry.get(task_id, task_version)
    value = raw_value.strip()
    if not value:
        raise ValueError("value must not be empty")
    if task.output_type == "boolean":
        normalized = {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False}
        if value.lower() not in normalized:
            raise ValueError("boolean value must be true or false")
        return {
            "type": "boolean",
            "value": normalized[value.lower()],
            "probability": 1.0,
            "abstained": False,
        }
    if task.output_type == "choice":
        candidates = [str(candidate) for candidate in task.output.get("candidates", [])]
        selected = next((candidate for candidate in candidates if candidate.lower() == value.lower()), None)
        if selected is None:
            raise ValueError(f"choice value must be one of: {', '.join(candidates)}")
        return {
            "type": "choice",
            "selected": selected,
            "probabilities": {candidate: float(candidate == selected) for candidate in candidates},
            "abstained": False,
        }
    if task.output_type == "score":
        try:
            score = float(value)
        except ValueError as exc:
            raise ValueError("score value must be a number between 0 and 1") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError("score value must be a number between 0 and 1")
        return {
            "type": "score",
            "value": score,
            "interval_90": [score, score],
            "abstained": False,
        }
    raise ValueError(f"unsupported task output type: {task.output_type}")


def _review_value_options(item: dict[str, Any], registry: TaskRegistry) -> str:
    """Return the flat values shown by the control value-review shell."""

    task_id, task_version = parse_task_reference(str(item.get("task", "")))
    task = registry.get(task_id, task_version)
    if task.output_type == "boolean":
        return "t=true / f=false"
    if task.output_type == "choice":
        return " / ".join(
            f"{alias}={candidate}"
            for candidate in task.output.get("candidates", [])
            for alias in (_control_value_alias(str(candidate)),)
        )
    if task.output_type == "score":
        return "number between 0 and 1"
    return str(task.output_type)


def _control_value_alias(candidate: str) -> str:
    """Return an unambiguous one- or two-letter control action alias."""

    aliases = {
        "APPROACH": "a",
        "HOLD": "h",
        "INTERACT": "i",
        "MOVE": "m",
        "STOP": "s",
        "ROTATE": "ro",
        "RETREAT": "rt",
        "RECOVER": "rc",
    }
    return aliases.get(candidate.upper(), candidate[:2].lower())


def _control_value_from_input(
    item: dict[str, Any], raw_value: str, registry: TaskRegistry
) -> dict[str, Any]:
    """Translate a short control alias, while retaining full-value compatibility."""

    task_id, task_version = parse_task_reference(str(item.get("task", "")))
    task = registry.get(task_id, task_version)
    value = raw_value.strip()
    if task.output_type == "boolean":
        aliases = {"t": "true", "f": "false"}
        value = aliases.get(value.lower(), value)
    elif task.output_type == "choice":
        aliases = {
            _control_value_alias(str(candidate)): str(candidate)
            for candidate in task.output.get("candidates", [])
        }
        value = aliases.get(value.lower(), value)
    return _correction_from_value(item, value, registry)


def _review_value_label(correction: dict[str, Any]) -> str:
    """Render a stored typed correction as one scalar value for local review."""

    result_type = correction.get("type")
    if result_type == "choice":
        return str(correction.get("selected", ""))
    if result_type == "boolean":
        return str(bool(correction.get("value"))).lower()
    if result_type == "score":
        return str(correction.get("value", ""))
    return str(result_type or "")


def _review_configured_value(item: dict[str, Any]) -> str:
    """Render the configured scalar without exposing nested teacher payloads."""

    configured = item.get("configured_value")
    if isinstance(configured, dict):
        return f"{_review_value_label(configured)} (configured)"
    if configured is not None and str(configured).strip():
        return f"{configured} (configured)"

    teacher = item.get("teacher")
    if isinstance(teacher, dict):
        result = teacher.get("normalized_result")
        if isinstance(result, dict) and result.get("type") in {"choice", "boolean", "score"}:
            return f"{_review_value_label(result)} (teacher draft)"
    return "not provided"


def _exact_review_group_key(item: dict[str, Any]) -> tuple[str, ...]:
    """Return the immutable content key used for exact-duplicate review groups."""

    return tuple(
        str(item.get(field, ""))
        for field in ("task", "language", "domain", "state", "question")
    )


def run_control_review_shell(
    review_pack_path: str | Path,
    queue_path: str | Path,
    feedback_path: str | Path,
    registry: TaskRegistry,
    *,
    reviewer: str,
    input_fn: Any = input,
    output_fn: Any = print,
    batch_offset: int = 0,
    batch_limit: int | None = None,
) -> dict[str, Any]:
    """Review control samples with one flat value per prompt.

    Exact duplicates are always collapsed into one group, teacher/nested JSON is
    never displayed, and the reviewer enters only the typed scalar action. The
    append-only feedback stream still records one validated label per sample in
    the collapsed group. ``c``/``confirm`` accepts the hidden teacher draft as
    the reviewer's label after the same feedback validation used for manual values.
    """

    if batch_offset < 0:
        raise ValueError("batch_offset must be non-negative")
    if batch_limit is not None and batch_limit < 1:
        raise ValueError("batch_limit must be positive")
    pack_records = _read_records(Path(review_pack_path))
    manifest = pack_records[0]
    if manifest.get("record_type") != "golden_review_pack_manifest":
        raise ValueError("review pack must start with a golden review pack manifest")
    queue = Path(queue_path)
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    if manifest.get("queue_sha256") != queue_sha256:
        raise ValueError("queue SHA-256 does not match the review pack manifest")
    items = pack_records[1:]
    if not items:
        raise ValueError("review pack has no review items")

    feedback = Path(feedback_path)
    latest = _validated_feedback_by_sample(feedback, queue, registry)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(_exact_review_group_key(item), []).append(item)
    groups = list(grouped.values())
    selected_groups = groups[batch_offset:]
    if batch_limit is not None:
        selected_groups = selected_groups[:batch_limit]
    if not selected_groups:
        raise ValueError("review batch is empty")

    reviewed_count = sum(
        str(item.get("sample_id", "")) in latest
        for item in items
    )
    output_fn(
        f"review progress: {reviewed_count}/{len(items)} reviewed; "
        f"{len(items) - reviewed_count} pending; "
        f"groups={len(groups)}; batch={len(selected_groups)}"
    )

    propagated = 0
    for group in selected_groups:
        reviewed = [
            str(item.get("sample_id"))
            for item in group
            if str(item.get("sample_id")) in latest
        ]
        if not reviewed or len(reviewed) == len(group):
            continue
        source_id = reviewed[-1]
        correction = latest[source_id].get("correction")
        if not isinstance(correction, dict):
            raise TypeError(f"feedback correction is invalid for exact duplicate {source_id}")
        for item in group:
            sample_id = str(item.get("sample_id", ""))
            if sample_id in latest:
                continue
            record = append_golden_feedback(
                queue,
                feedback,
                registry,
                sample_id=sample_id,
                correction=correction,
                reviewer=reviewer,
                reason=f"control review shell: propagated exact duplicate of {source_id}",
            )
            latest[sample_id] = {
                "sample_id": sample_id,
                "correction": correction,
                "created_at": record.get("created_at"),
            }
            propagated += 1

    index = next(
        (
            position
            for position, group in enumerate(selected_groups)
            if not all(str(item.get("sample_id")) in latest for item in group)
        ),
        len(selected_groups),
    )
    saved = 0
    decisions = 0
    stopped = False
    while 0 <= index < len(selected_groups):
        group = selected_groups[index]
        item = group[0]
        sample_id = str(item.get("sample_id", ""))
        output_fn("")
        output_fn(
            f"[{batch_offset + index + 1}/{len(groups)}] {item.get('task')} "
            f"exact_duplicates={len(group)}"
        )
        output_fn(f"state: {item.get('state')}")
        output_fn(f"question: {item.get('question')}")
        output_fn(f"configured value: {_review_configured_value(item)}")
        output_fn(f"allowed values: {_review_value_options(item, registry)}")
        if sample_id in latest:
            output_fn(f"current value: {_review_value_label(latest[sample_id]['correction'])}")
        output_fn(
            "Enter a value directly, or use [c]onfirm teacher "
            "[n]ext [p]revious [sk]ip [q]uit"
        )
        try:
            raw_value = input_fn("value> ").strip()
        except (EOFError, KeyboardInterrupt):
            stopped = True
            break
        command = raw_value.lower()
        if command in {"q", "quit"}:
            stopped = True
            break
        if command in {"n", "next", "sk", "skip"}:
            if index < len(selected_groups) - 1:
                index += 1
            continue
        if command in {"p", "previous", "prev"}:
            if index > 0:
                index -= 1
            continue
        if command in {"c", "confirm"}:
            teacher = item.get("teacher")
            correction = teacher.get("normalized_result") if isinstance(teacher, dict) else None
            if not isinstance(correction, dict):
                output_fn("Teacher draft is unavailable or schema-invalid; enter a value manually.")
                continue
            reason = "control review shell: reviewer confirmed teacher draft"
        else:
            try:
                correction = _control_value_from_input(item, raw_value, registry)
            except (TypeError, ValueError) as exc:
                output_fn(f"Invalid value: {exc}")
                continue
            reason = "control review shell: reviewer entered value"
        affected = group
        if len(affected) > 1:
            reason += f"; applied to {len(affected)} exact duplicate samples"
        try:
            for affected_item in affected:
                affected_id = str(affected_item.get("sample_id", ""))
                record = append_golden_feedback(
                    queue,
                    feedback,
                    registry,
                    sample_id=affected_id,
                    correction=correction,
                    reviewer=reviewer,
                    reason=reason,
                )
                latest[affected_id] = {
                    "sample_id": affected_id,
                    "correction": correction,
                    "created_at": record.get("created_at"),
                }
                saved += 1
        except (TypeError, ValueError) as exc:
            output_fn(f"Correction rejected: {exc}")
            continue
        decisions += 1
        output_fn(f"Saved {len(affected)} label(s).")
        if index < len(selected_groups) - 1:
            index += 1
        else:
            break

    reviewed_count = sum(str(item.get("sample_id")) in latest for item in items)
    return {
        "reviewed_count": reviewed_count,
        "pending_count": len(items) - reviewed_count,
        "saved_in_session": saved,
        "decisions_in_session": decisions,
        "review_group_count": len(groups),
        "batch_offset": batch_offset,
        "batch_limit": batch_limit,
        "batch_count": len(selected_groups),
        "batch_pending_count": sum(
            1
            for group in selected_groups
            for item in group
            if str(item.get("sample_id")) not in latest
        ),
        "deduplicated_exact": True,
        "propagated_count": propagated,
        "stopped": stopped,
        "flat_value_input": True,
        "teacher_hidden": True,
    }


def run_review_session(
    review_pack_path: str | Path,
    queue_path: str | Path,
    feedback_path: str | Path,
    registry: TaskRegistry,
    *,
    reviewer: str,
    input_fn: Any = input,
    output_fn: Any = print,
    deduplicate_exact: bool = False,
    blind_teacher: bool = False,
    batch_offset: int = 0,
    batch_limit: int | None = None,
) -> dict[str, Any]:
    """Review a pack in one resumable session with next/previous navigation.

    Commands are ``a`` (accept the displayed teacher draft), ``e`` (enter only
    the corrected task value), ``n``/``p`` (next/previous), ``s`` (leave pending and
    move next), and ``q`` (save and quit). Every accepted or edited label is
    appended immediately; revising an earlier item appends a newer record that
    wins when feedback is applied. With ``blind_teacher=True``, teacher output
    is hidden and only ``e`` can create a label. ``batch_offset`` and
    ``batch_limit`` bound the deterministic review-group slice for resumable
    batch labeling.
    """

    if batch_offset < 0:
        raise ValueError("batch_offset must be non-negative")
    if batch_limit is not None and batch_limit < 1:
        raise ValueError("batch_limit must be positive")
    pack_records = _read_records(Path(review_pack_path))
    manifest = pack_records[0]
    if manifest.get("record_type") != "golden_review_pack_manifest":
        raise ValueError("review pack must start with a golden review pack manifest")
    queue = Path(queue_path)
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    if manifest.get("queue_sha256") != queue_sha256:
        raise ValueError("queue SHA-256 does not match the review pack manifest")
    items = pack_records[1:]
    if not items:
        raise ValueError("review pack has no review items")
    feedback = Path(feedback_path)
    latest = _feedback_by_sample(feedback)
    if deduplicate_exact:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(_exact_review_group_key(item), []).append(item)
        groups = list(grouped.values())
    else:
        groups = [[item] for item in items]

    selected_groups = groups[batch_offset:]
    if batch_limit is not None:
        selected_groups = selected_groups[:batch_limit]
    if not selected_groups:
        raise ValueError("review batch is empty")

    propagated = 0
    if deduplicate_exact:
        for group in selected_groups:
            reviewed = [
                str(item.get("sample_id"))
                for item in group
                if str(item.get("sample_id")) in latest
            ]
            if not reviewed or len(reviewed) == len(group):
                continue
            source_id = reviewed[-1]
            correction = latest[source_id].get("correction")
            if not isinstance(correction, dict):
                raise TypeError(f"feedback correction is invalid for exact duplicate {source_id}")
            for item in group:
                sample_id = str(item.get("sample_id", ""))
                if sample_id in latest:
                    continue
                record = append_golden_feedback(
                    queue,
                    feedback,
                    registry,
                    sample_id=sample_id,
                    correction=correction,
                    reviewer=reviewer,
                    reason=f"interactive review: propagated exact duplicate of {source_id}",
                )
                latest[sample_id] = {
                    "sample_id": sample_id,
                    "correction": correction,
                    "created_at": record.get("created_at"),
                }
                propagated += 1
            output_fn(
                f"Propagated human label from {source_id} to {len(group) - 1} exact duplicate(s)."
            )

    index = next(
        (
            position
            for position, group in enumerate(selected_groups)
            if not all(str(item.get("sample_id")) in latest for item in group)
        ),
        len(selected_groups),
    )
    saved = 0
    decisions = 0
    stopped = False

    while 0 <= index < len(selected_groups):
        group = selected_groups[index]
        item = group[0]
        sample_id = str(item.get("sample_id", ""))
        teacher = item.get("teacher", {})
        output_fn("")
        if deduplicate_exact:
            group_ids = [str(member.get("sample_id", "")) for member in group]
            output_fn(
                f"[group {batch_offset + index + 1}/{len(groups)}; "
                f"batch {index + 1}/{len(selected_groups)}] {item.get('task')} "
                f"exact_duplicates={len(group)}"
            )
            output_fn("sample_ids: " + ", ".join(group_ids[:5]))
            if len(group_ids) > 5:
                output_fn(f"... and {len(group_ids) - 5} more")
        else:
            output_fn(
                f"[{batch_offset + index + 1}/{len(items)}; "
                f"batch {index + 1}/{len(selected_groups)}] {sample_id}  {item.get('task')}"
            )
        output_fn(f"state: {item.get('state')}")
        output_fn(f"question: {item.get('question')}")
        if not blind_teacher:
            output_fn(
                "Qwen draft: "
                + json.dumps(teacher.get("normalized_result"), ensure_ascii=False, sort_keys=True)
            )
            if teacher.get("comparison"):
                output_fn(
                    "Qwen/Gemma comparison: "
                    + json.dumps(teacher["comparison"], ensure_ascii=False, sort_keys=True)
                )
            if teacher.get("error"):
                output_fn(f"Qwen error: {teacher['error']}")
        if sample_id in latest:
            output_fn(
                "Current human label: "
                + json.dumps(latest[sample_id].get("correction"), ensure_ascii=False, sort_keys=True)
            )
        if blind_teacher:
            output_fn("Commands: [e]nter label  [n]ext  [p]revious  [s]kip  [q]uit")
        else:
            output_fn("Commands: [a]ccept  [e]dit  [n]ext  [p]revious  [s]kip  [q]uit")
        try:
            command = input_fn("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            stopped = True
            break
        if command in {"q", "quit"}:
            stopped = True
            break
        if command in {"n", "next", "s", "skip"}:
            if index < len(selected_groups) - 1:
                index += 1
            else:
                output_fn("Already at the last item.")
            continue
        if command in {"p", "previous", "prev"}:
            if index > 0:
                index -= 1
            else:
                output_fn("Already at the first item.")
            continue
        allowed_decisions = {"e", "edit"} if blind_teacher else {"a", "accept", "e", "edit"}
        if command not in allowed_decisions:
            output_fn("Use a, e, n, p, s, or q.")
            continue

        if command in {"a", "accept"} and not blind_teacher:
            correction = teacher.get("normalized_result")
            if not isinstance(correction, dict):
                output_fn("Qwen draft is not schema-valid; use e to enter a correction.")
                continue
            reason = "interactive review: reviewer accepted teacher draft"
        else:
            try:
                raw_value = input_fn("correct value: ")
            except (EOFError, KeyboardInterrupt):
                stopped = True
                break
            try:
                correction = _correction_from_value(item, raw_value, registry)
            except (TypeError, ValueError) as exc:
                output_fn(f"Invalid value: {exc}")
                continue
            reason = "interactive review: reviewer entered correction"
        affected = group if deduplicate_exact else [item]
        if deduplicate_exact and len(affected) > 1:
            reason += f"; applied to {len(affected)} exact duplicate samples"
        try:
            for affected_item in affected:
                affected_id = str(affected_item.get("sample_id", ""))
                record = append_golden_feedback(
                    queue,
                    feedback,
                    registry,
                    sample_id=affected_id,
                    correction=correction,
                    reviewer=reviewer,
                    reason=reason,
                )
                latest[affected_id] = {
                    "sample_id": affected_id,
                    "correction": correction,
                    "created_at": record.get("created_at"),
                }
                saved += 1
        except (TypeError, ValueError) as exc:
            output_fn(f"Correction rejected: {exc}")
            continue
        decisions += 1
        output_fn(f"Saved {len(affected)} label(s). Moving to the next review group.")
        if index < len(selected_groups) - 1:
            index += 1
        else:
            break

    reviewed_count = sum(str(item.get("sample_id")) in latest for item in items)
    return {
        "reviewed_count": reviewed_count,
        "pending_count": len(items) - reviewed_count,
        "saved_in_session": saved,
        "decisions_in_session": decisions,
        "review_group_count": len(groups),
        "batch_offset": batch_offset,
        "batch_limit": batch_limit,
        "batch_count": len(selected_groups),
        "batch_pending_count": sum(
            1
            for group in selected_groups
            for item in group
            if str(item.get("sample_id")) not in latest
        ),
        "deduplicated_exact": deduplicate_exact,
        "propagated_count": propagated,
        "stopped": stopped,
        "blind_teacher": blind_teacher,
    }
