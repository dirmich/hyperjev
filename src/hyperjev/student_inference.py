"""Reference Student checkpoint loading and guarded quality evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .registry import TaskRegistry
from .rules import match_rule
from .samples import CanonicalSample
from .student import StudentConfig, StudentDependencyError, build_torch_model
from .training import _encode_reference_sample, load_training_dataset


class StudentInferenceError(ValueError):
    """Raised when a Student checkpoint cannot be evaluated safely."""


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
    registry: TaskRegistry,
    *,
    torch: Any,
    device: str,
    score_tolerance: float,
) -> dict[str, Any]:
    task = registry.get(sample.task_id, sample.task_version)
    config = model._hyperjev_student_config
    token_ids, attention = _encode_reference_sample(
        sample,
        vocab_size=config.vocab_size,
        max_length=config.max_sequence_length,
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(sample.task_id, input_ids, attention_mask)

    if task.output_type == "boolean":
        probabilities = torch.softmax(output["logits"], dim=-1)[0].detach().cpu().tolist()
        index = int(torch.argmax(output["logits"], dim=-1)[0].item())
        predicted: Any = bool(index)
        confidence = float(probabilities[index])
        correct = predicted == bool(sample.target)
        rendered = {"type": "boolean", "value": predicted, "probability": confidence}
    elif task.output_type == "choice":
        candidates = [str(candidate) for candidate in task.output.get("candidates", [])]
        probabilities = torch.softmax(output["logits"], dim=-1)[0].detach().cpu().tolist()
        index = int(torch.argmax(output["logits"], dim=-1)[0].item())
        predicted = candidates[index]
        confidence = float(probabilities[index])
        correct = predicted == str(sample.target)
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
        correct = abs(predicted - float(sample.target)) <= score_tolerance
        rendered = {"type": "score", "value": predicted, "interval_90": interval}

    return {
        "sample_id": sample.sample_id,
        "task_id": sample.task_id,
        "split": sample.provenance.get("split"),
        "target": sample.target,
        "prediction": rendered,
        "confidence": round(confidence, 6),
        "correct": correct,
    }


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
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a checkpoint and report accuracy separately from safe coverage."""

    if split not in {"all", "train", "validation", "test"}:
        raise StudentInferenceError("split must be all, train, validation, or test")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise StudentInferenceError("minimum_confidence must be between 0 and 1")
    if score_tolerance < 0:
        raise StudentInferenceError("score_tolerance must not be negative")
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

    samples = [
        sample
        for sample in dataset.samples
        if split == "all" or sample.provenance.get("split") == split
    ]
    if not samples:
        raise StudentInferenceError(f"dataset has no samples for split: {split}")
    predictions = [
        _predict_sample(
            model,
            sample,
            registry,
            torch=torch,
            device=device,
            score_tolerance=score_tolerance,
        )
        for sample in samples
    ]
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
                rule_correct_for_sample = bool(rule_result["value"]) == bool(sample.target)
                confidence = float(rule_result["probability"])
            elif task.output_type == "choice":
                rule_correct_for_sample = str(rule_result["selected"]) == str(sample.target)
                confidence = max(float(value) for value in rule_result["probabilities"].values())
            else:
                rule_correct_for_sample = (
                    abs(float(rule_result["value"]) - float(sample.target)) <= score_tolerance
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
    return report


def write_student_evaluation(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
