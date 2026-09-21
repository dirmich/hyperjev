"""Reference Student checkpoint loading and guarded quality evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import (
    BooleanDecision,
    ChoiceDecision,
    ScoreDecision,
    parse_decision_result,
    validate_result_for_task,
)
from .registry import TaskDefinition, TaskRegistry
from .rules import match_rule
from .samples import CanonicalSample
from .student import StudentConfig, StudentDependencyError, build_torch_model
from .training import _encode_reference_sample, load_training_dataset


class StudentInferenceError(ValueError):
    """Raised when a Student checkpoint cannot be evaluated safely."""


def _exact_group_key(sample: CanonicalSample) -> tuple[str, int, str, str, str, str]:
    """Return the semantic identity used by golden exact-duplicate review."""

    return (
        sample.task_id,
        sample.task_version,
        sample.language,
        sample.domain,
        sample.state,
        sample.question,
    )


def _semantic_group_key(sample: CanonicalSample) -> str:
    """Return the provenance group used to measure independent examples."""

    value = sample.provenance.get("semantic_group_id")
    return str(value) if value else f"sample:{sample.sample_id}"


def _deduplicate_exact(samples: list[CanonicalSample]) -> list[CanonicalSample]:
    """Keep one deterministic representative per exact semantic group."""

    selected: list[CanonicalSample] = []
    seen: set[tuple[str, int, str, str, str, str]] = set()
    for sample in samples:
        key = _exact_group_key(sample)
        if key in seen:
            continue
        seen.add(key)
        selected.append(sample)
    return selected


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise StudentDependencyError("PyTorch is required to evaluate the Student checkpoint") from exc
    return torch


def _student_config(raw: dict[str, Any]) -> StudentConfig:
    fields = {
        "model_id": str(raw.get("model_id", "hyperjev-student-dev")),
        "backbone": str(raw.get("backbone", "reference-byte-encoder")),
        "hidden_size": int(raw.get("hidden_size", 256)),
        "vocab_size": int(raw.get("vocab_size", 32768)),
        "max_sequence_length": int(raw.get("max_sequence_length", 1024)),
        "precision": str(raw.get("precision", "fp32")),
    }
    return StudentConfig(**fields)


def _predict_sample(
    model: Any,
    sample: CanonicalSample,
    task: TaskDefinition,
    target: Any,
    *,
    torch: Any,
    device: str,
    score_tolerance: float,
    include_logits: bool = False,
) -> dict[str, Any]:
    config = model._hyperjev_student_config
    token_ids, attention = _encode_reference_sample(
        sample,
        vocab_size=config.vocab_size,
        max_length=config.max_sequence_length,
        pad_to_max=False,
        backbone=config.backbone,
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(sample.task_id, input_ids, attention_mask)

    raw_logits: list[float] | None = None
    if task.output_type == "boolean":
        if include_logits:
            raw_logits = [float(value) for value in output["logits"][0].detach().cpu().tolist()]
        probabilities = torch.softmax(output["logits"], dim=-1)[0].detach().cpu().tolist()
        index = int(torch.argmax(output["logits"], dim=-1)[0].item())
        predicted: Any = bool(index)
        confidence = float(probabilities[index])
        correct = predicted == bool(target)
        rendered = {"type": "boolean", "value": predicted, "probability": confidence}
    elif task.output_type == "choice":
        candidates = [str(candidate) for candidate in task.output.get("candidates", [])]
        if include_logits:
            raw_logits = [float(value) for value in output["logits"][0].detach().cpu().tolist()]
        probabilities = torch.softmax(output["logits"], dim=-1)[0].detach().cpu().tolist()
        index = int(torch.argmax(output["logits"], dim=-1)[0].item())
        predicted = candidates[index]
        confidence = float(probabilities[index])
        correct = predicted == str(target)
        rendered = {
            "type": "choice",
            "selected": predicted,
            "probabilities": {
                candidate: float(probability)
                for candidate, probability in zip(candidates, probabilities)
            },
        }
    else:
        raw_value = float(output["parameters"][0, 0].detach().cpu().item())
        predicted = min(1.0, max(0.0, raw_value))
        width = float(torch.sigmoid(output["parameters"][0, 1]).detach().cpu().item()) * 0.5
        interval = (max(0.0, predicted - width), min(1.0, predicted + width))
        confidence = max(0.0, 1.0 - (interval[1] - interval[0]))
        correct = abs(predicted - float(target)) <= score_tolerance
        rendered = {"type": "score", "value": predicted, "interval_90": interval}

    result = {
        "sample_id": sample.sample_id,
        "task_id": sample.task_id,
        "split": sample.provenance.get("split"),
        "target": target,
        "prediction": rendered,
        "confidence": round(confidence, 6),
        "correct": correct,
    }
    if raw_logits is not None:
        result["logits"] = raw_logits
    return result


def _evaluation_target(
    sample: CanonicalSample,
    task: TaskDefinition,
) -> tuple[Any, str]:
    """Use a reviewed typed label when present; otherwise expose synthetic target."""

    raw_human = sample.labels.get("human")
    if raw_human is None:
        return sample.target, "sample.target"
    if not isinstance(raw_human, dict):
        raise StudentInferenceError(f"{sample.sample_id}: human label must be an object")
    try:
        human_result = parse_decision_result(raw_human)
        validate_result_for_task(task, human_result)
    except (TypeError, ValueError) as exc:
        raise StudentInferenceError(f"{sample.sample_id}: invalid human label: {exc}") from exc
    if human_result.abstained:
        raise StudentInferenceError(f"{sample.sample_id}: human label must not abstain")
    if isinstance(human_result, BooleanDecision):
        return human_result.value, "human"
    if isinstance(human_result, ChoiceDecision):
        return human_result.selected, "human"
    if isinstance(human_result, ScoreDecision):
        return human_result.value, "human"
    raise StudentInferenceError(f"{sample.sample_id}: unsupported human label type")


def _summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(predictions)
    correct = sum(bool(row["correct"]) for row in predictions)
    accepted = [row for row in predictions if row["accepted"]]
    accepted_correct = sum(bool(row["correct"]) for row in accepted)
    return {
        "count": count,
        "correct": correct,
        "accuracy": round(correct / count, 6) if count else None,
        "accepted_count": len(accepted),
        "accepted_correct": accepted_correct,
        "accepted_accuracy": round(accepted_correct / len(accepted), 6) if accepted else None,
        "coverage": round(len(accepted) / count, 6) if count else None,
        "fallback_count": count - len(accepted),
    }


def evaluate_student_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    registry: TaskRegistry,
    *,
    split: str = "all",
    minimum_confidence: float = 0.95,
    allow_score: bool = False,
    with_rules: bool = False,
    score_tolerance: float = 0.10,
    minimum_accuracy: float = 0.99,
    minimum_accepted_accuracy: float = 0.995,
    minimum_task_accuracy: float = 0.98,
    deduplicate_exact: bool = False,
    include_logits: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a checkpoint and report accuracy separately from safe coverage."""

    if split not in {"all", "train", "validation", "test"}:
        raise StudentInferenceError("split must be all, train, validation, or test")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise StudentInferenceError("minimum_confidence must be between 0 and 1")
    if score_tolerance < 0:
        raise StudentInferenceError("score_tolerance must not be negative")
    for name, value in (
        ("minimum_accuracy", minimum_accuracy),
        ("minimum_accepted_accuracy", minimum_accepted_accuracy),
        ("minimum_task_accuracy", minimum_task_accuracy),
    ):
        if not 0.0 <= value <= 1.0:
            raise StudentInferenceError(f"{name} must be between 0 and 1")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise StudentInferenceError(f"checkpoint not found: {checkpoint}")
    dataset = load_training_dataset(dataset_path, registry)
    torch = _load_torch()
    try:
        raw_checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise StudentInferenceError(f"cannot load checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(raw_checkpoint, dict) or not isinstance(raw_checkpoint.get("student"), dict):
        raise StudentInferenceError("checkpoint does not contain a Student manifest")
    config = _student_config(raw_checkpoint["student"])
    model = build_torch_model(registry, config).to(device)
    try:
        model.load_state_dict(raw_checkpoint["model_state_dict"])
    except (KeyError, RuntimeError, TypeError) as exc:
        raise StudentInferenceError(f"checkpoint weights do not match Student manifest: {exc}") from exc
    model._hyperjev_student_config = config
    model.eval()

    selected_samples = [
        sample
        for sample in dataset.samples
        if split == "all" or sample.provenance.get("split") == split
    ]
    if not selected_samples:
        raise StudentInferenceError(f"dataset has no samples for split: {split}")
    unique_exact_group_count = len({_exact_group_key(sample) for sample in selected_samples})
    unique_semantic_group_count = len({_semantic_group_key(sample) for sample in selected_samples})
    samples = _deduplicate_exact(selected_samples) if deduplicate_exact else selected_samples
    targets: dict[str, tuple[Any, str]] = {}
    predictions = []
    for sample in samples:
        task = registry.get(sample.task_id, sample.task_version)
        target, target_source = _evaluation_target(sample, task)
        targets[sample.sample_id] = (target, target_source)
        prediction = _predict_sample(
            model,
            sample,
            task,
            target,
            torch=torch,
            device=device,
            score_tolerance=score_tolerance,
            include_logits=include_logits,
        )
        prediction["target_source"] = target_source
        predictions.append(prediction)
    for row in predictions:
        output_type = registry.get(row["task_id"]).output_type
        row["accepted"] = (
            row["confidence"] >= minimum_confidence
            and (output_type != "score" or allow_score)
        )
        row["route"] = "student"

    rule_covered = 0
    rule_correct = 0
    if with_rules:
        by_sample = {sample.sample_id: sample for sample in samples}
        for row in predictions:
            sample = by_sample[row["sample_id"]]
            task = registry.get(sample.task_id, sample.task_version)
            rule = match_rule(task, state=sample.state, question=sample.question)
            if rule is None or rule.result.abstained:
                continue
            rule_covered += 1
            rule_result = rule.result.to_dict()
            if task.output_type == "boolean":
                expected_target = targets[sample.sample_id][0]
                rule_correct_for_sample = bool(rule_result["value"]) == bool(expected_target)
                confidence = float(rule_result["probability"])
            elif task.output_type == "choice":
                expected_target = targets[sample.sample_id][0]
                rule_correct_for_sample = str(rule_result["selected"]) == str(expected_target)
                confidence = max(float(value) for value in rule_result["probabilities"].values())
            else:
                expected_target = targets[sample.sample_id][0]
                rule_correct_for_sample = (
                    abs(float(rule_result["value"]) - float(expected_target)) <= score_tolerance
                )
                confidence = 1.0
            rule_correct += int(rule_correct_for_sample)
            row["prediction"] = rule_result
            row["confidence"] = round(confidence, 6)
            row["correct"] = rule_correct_for_sample
            row["accepted"] = True
            row["route"] = "rule"

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        by_task.setdefault(row["task_id"], []).append(row)
    report = {
        "record_type": "student_evaluation",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_sha256": dataset.dataset_hash,
        "model_id": config.model_id,
        "backbone": config.backbone,
        "split": split,
        "deduplicate_exact": deduplicate_exact,
        "row_count": len(selected_samples),
        "unique_exact_group_count": unique_exact_group_count,
        "unique_semantic_group_count": unique_semantic_group_count,
        "minimum_confidence": minimum_confidence,
        "score_auto_accept": allow_score,
        "rules_enabled": with_rules,
        "rule_covered": rule_covered,
        "rule_correct": rule_correct,
        "score_tolerance": score_tolerance,
        "overall": _summary(predictions),
        "tasks": {task_id: _summary(rows) for task_id, rows in sorted(by_task.items())},
        "predictions": predictions,
    }
    human_labeled_count = sum(sample.labels.get("human") is not None for sample in samples)
    task_failures = [
        task_id
        for task_id, task_report in report["tasks"].items()
        if (task_report["accuracy"] or 0.0) < minimum_task_accuracy
    ]
    gate_reasons: list[str] = []
    if human_labeled_count != len(samples):
        gate_reasons.append("human_labels_required")
    if (report["overall"]["accuracy"] or 0.0) < minimum_accuracy:
        gate_reasons.append("overall_accuracy_below_threshold")
    accepted_accuracy = report["overall"]["accepted_accuracy"]
    if accepted_accuracy is None or accepted_accuracy < minimum_accepted_accuracy:
        gate_reasons.append("accepted_accuracy_below_threshold")
    if task_failures:
        gate_reasons.append("task_accuracy_below_threshold")
    report["golden"] = {
        "human_labeled_count": human_labeled_count,
        "human_labeled": human_labeled_count == len(samples),
    }
    report["quality_gate"] = {
        "ready": not gate_reasons,
        "minimum_accuracy": minimum_accuracy,
        "minimum_accepted_accuracy": minimum_accepted_accuracy,
        "minimum_task_accuracy": minimum_task_accuracy,
        "task_failures": task_failures,
        "reasons": gate_reasons,
    }
    return report


def write_student_evaluation(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
