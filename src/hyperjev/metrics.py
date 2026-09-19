"""Dependency-free quality and calibration metrics for typed decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class MetricError(ValueError):
    """Raised when metric inputs are malformed or have incompatible lengths."""


def _validate_lengths(*values: Sequence[Any]) -> int:
    if not values:
        raise MetricError("at least one metric input is required")
    length = len(values[0])
    if length == 0:
        raise MetricError("metric inputs must not be empty")
    if any(len(value) != length for value in values[1:]):
        raise MetricError("metric inputs must have equal lengths")
    return length


def _probability(value: Any) -> float:
    if isinstance(value, bool):
        raise MetricError("probabilities must be numbers between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricError("probabilities must be numbers between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise MetricError("probabilities must be numbers between 0 and 1")
    return result


def _rounded(value: float) -> float:
    return round(value, 6)


def _expected_calibration_error(labels: Sequence[bool], probabilities: Sequence[float], bins: int) -> float:
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, probability in enumerate(probabilities)
            if (lower <= probability < upper) or (index == bins - 1 and probability == upper)
        ]
        if not members:
            continue
        confidence = sum(probabilities[position] for position in members) / len(members)
        accuracy = sum(labels[position] for position in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def binary_metrics(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
    bins: int = 10,
) -> dict[str, float | int]:
    """Return classification, Brier, and ECE metrics for positive-class probabilities."""

    count = _validate_lengths(labels, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise MetricError("threshold must be between 0 and 1")
    if bins < 1:
        raise MetricError("bins must be positive")
    normalized_labels = [bool(label) for label in labels]
    normalized_probabilities = [_probability(probability) for probability in probabilities]
    predictions = [probability >= threshold for probability in normalized_probabilities]
    true_positive = sum(prediction and label for prediction, label in zip(predictions, normalized_labels))
    false_positive = sum(prediction and not label for prediction, label in zip(predictions, normalized_labels))
    false_negative = sum(not prediction and label for prediction, label in zip(predictions, normalized_labels))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(prediction == label for prediction, label in zip(predictions, normalized_labels)) / count
    brier = sum((probability - label) ** 2 for probability, label in zip(normalized_probabilities, normalized_labels)) / count
    return {
        "count": count,
        "accuracy": _rounded(accuracy),
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "brier": _rounded(brier),
        "ece": _rounded(_expected_calibration_error(normalized_labels, normalized_probabilities, bins)),
    }


def choice_metrics(
    labels: Sequence[str],
    selected: Sequence[str],
    probabilities: Sequence[Mapping[str, float]],
) -> dict[str, float | int]:
    """Return accuracy, top-2 accuracy, macro-F1, and multiclass Brier score."""

    count = _validate_lengths(labels, selected, probabilities)
    normalized_probabilities: list[dict[str, float]] = []
    classes = {str(label) for label in labels}
    for raw in probabilities:
        if not isinstance(raw, Mapping) or not raw:
            raise MetricError("choice probabilities must be non-empty mappings")
        parsed = {str(key): _probability(value) for key, value in raw.items()}
        normalized_probabilities.append(parsed)
        classes.update(parsed)
    normalized_labels = [str(label) for label in labels]
    normalized_selected = [str(value) for value in selected]
    correct = [prediction == label for prediction, label in zip(normalized_selected, normalized_labels)]
    per_class_f1: list[float] = []
    for class_name in sorted(classes):
        true_positive = sum(prediction == class_name and label == class_name for prediction, label in zip(normalized_selected, normalized_labels))
        false_positive = sum(prediction == class_name and label != class_name for prediction, label in zip(normalized_selected, normalized_labels))
        false_negative = sum(prediction != class_name and label == class_name for prediction, label in zip(normalized_selected, normalized_labels))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        per_class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    top2 = [
        label in [key for key, _ in sorted(row.items(), key=lambda item: (-item[1], item[0]))[:2]]
        for label, row in zip(normalized_labels, normalized_probabilities)
    ]
    brier = 0.0
    for label, row in zip(normalized_labels, normalized_probabilities):
        brier += sum((row.get(class_name, 0.0) - int(class_name == label)) ** 2 for class_name in classes)
    return {
        "count": count,
        "accuracy": _rounded(sum(correct) / count),
        "top2_accuracy": _rounded(sum(top2) / count),
        "macro_f1": _rounded(sum(per_class_f1) / len(per_class_f1)) if per_class_f1 else 0.0,
        "brier": _rounded(brier / count),
    }


def score_metrics(
    targets: Sequence[float],
    values: Sequence[float],
    intervals: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    """Return regression error and interval coverage metrics."""

    count = _validate_lengths(targets, values, intervals)
    normalized_targets = [float(target) for target in targets]
    normalized_values = [float(value) for value in values]
    absolute_errors = [abs(value - target) for value, target in zip(normalized_values, normalized_targets)]
    squared_errors = [(value - target) ** 2 for value, target in zip(normalized_values, normalized_targets)]
    coverage = 0
    widths: list[float] = []
    for target, interval in zip(normalized_targets, intervals):
        if len(interval) != 2:
            raise MetricError("score intervals must contain lower and upper bounds")
        lower, upper = float(interval[0]), float(interval[1])
        if lower > upper:
            raise MetricError("score interval lower bound must not exceed upper bound")
        coverage += int(lower <= target <= upper)
        widths.append(upper - lower)
    return {
        "count": count,
        "mae": _rounded(sum(absolute_errors) / count),
        "rmse": _rounded(math.sqrt(sum(squared_errors) / count)),
        "interval_90_coverage": _rounded(coverage / count),
        "interval_90_width": _rounded(sum(widths) / count),
    }
