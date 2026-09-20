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


def _adaptive_calibration_error(labels: Sequence[bool], probabilities: Sequence[float], bins: int) -> float:
    """Compute ECE over equal-frequency bins ordered by confidence."""

    ordered = sorted(range(len(labels)), key=lambda position: probabilities[position])
    bin_count = min(bins, len(ordered))
    error = 0.0
    for index in range(bin_count):
        start = index * len(ordered) // bin_count
        end = (index + 1) * len(ordered) // bin_count
        members = ordered[start:end]
        if not members:
            continue
        confidence = sum(probabilities[position] for position in members) / len(members)
        accuracy = sum(labels[position] for position in members) / len(members)
        error += len(members) / len(ordered) * abs(accuracy - confidence)
    return error


def _selective_curve(correct: Sequence[bool], confidence: Sequence[float]) -> list[dict[str, float]]:
    ordered = sorted(range(len(correct)), key=lambda position: (-confidence[position], position))
    correct_so_far = 0
    curve: list[dict[str, float]] = []
    for rank, position in enumerate(ordered, start=1):
        correct_so_far += int(correct[position])
        curve.append(
            {
                "coverage": _rounded(rank / len(ordered)),
                "risk": _rounded(1.0 - correct_so_far / rank),
                "threshold": _rounded(confidence[position]),
            }
        )
    return curve


def threshold_risk_coverage(
    correct: Sequence[bool],
    confidence: Sequence[float],
    *,
    thresholds: Sequence[float] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99),
) -> list[dict[str, Any]]:
    """Summarize accepted accuracy and risk at fixed confidence thresholds.

    This is intentionally separate from a model's raw accuracy: a controller may
    abstain below a threshold, so each row reports the quality and coverage of
    only the decisions that would be accepted at that threshold.
    """

    count = _validate_lengths(correct, confidence)
    if not thresholds:
        raise MetricError("risk-coverage thresholds must not be empty")
    normalized_correct = [bool(value) for value in correct]
    normalized_confidence = [_probability(value) for value in confidence]
    normalized_thresholds = [_probability(value) for value in thresholds]
    report: list[dict[str, Any]] = []
    for threshold in normalized_thresholds:
        accepted = [
            position
            for position, value in enumerate(normalized_confidence)
            if value >= threshold
        ]
        accepted_count = len(accepted)
        accepted_correct = sum(normalized_correct[position] for position in accepted)
        accepted_accuracy = accepted_correct / accepted_count if accepted_count else None
        report.append(
            {
                "threshold": _rounded(threshold),
                "count": count,
                "accepted_count": accepted_count,
                "coverage": _rounded(accepted_count / count),
                "accepted_correct": accepted_correct,
                "accepted_accuracy": _rounded(accepted_accuracy)
                if accepted_accuracy is not None
                else None,
                "accepted_risk": _rounded(1.0 - accepted_accuracy)
                if accepted_accuracy is not None
                else None,
            }
        )
    return report


def _binary_auroc(labels: Sequence[bool], probabilities: Sequence[float]) -> float | None:
    positives = [probability for label, probability in zip(labels, probabilities) if label]
    negatives = [probability for label, probability in zip(labels, probabilities) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _binary_auprc(labels: Sequence[bool], probabilities: Sequence[float]) -> float | None:
    positive_count = sum(labels)
    if not positive_count:
        return None
    ordered = sorted(range(len(labels)), key=lambda position: (-probabilities[position], position))
    true_positives = 0
    precision_sum = 0.0
    for rank, position in enumerate(ordered, start=1):
        if labels[position]:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positive_count


def binary_metrics(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
    bins: int = 10,
) -> dict[str, Any]:
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
    negative_log_likelihood = sum(
        -math.log(max(probability if label else 1.0 - probability, 1e-12))
        for probability, label in zip(normalized_probabilities, normalized_labels)
    ) / count
    return {
        "count": count,
        "accuracy": _rounded(accuracy),
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "brier": _rounded(brier),
        "ece": _rounded(_expected_calibration_error(normalized_labels, normalized_probabilities, bins)),
        "adaptive_ece": _rounded(
            _adaptive_calibration_error(normalized_labels, normalized_probabilities, bins)
        ),
        "nll": _rounded(negative_log_likelihood),
        "auroc": _rounded(value) if (value := _binary_auroc(normalized_labels, normalized_probabilities)) is not None else None,
        "auprc": _rounded(value) if (value := _binary_auprc(normalized_labels, normalized_probabilities)) is not None else None,
        "risk_coverage": _selective_curve(
            [prediction == label for prediction, label in zip(predictions, normalized_labels)],
            [max(probability, 1.0 - probability) for probability in normalized_probabilities],
        ),
    }


def choice_metrics(
    labels: Sequence[str],
    selected: Sequence[str],
    probabilities: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
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
    negative_log_likelihood = 0.0
    confidence: list[float] = []
    for label, row in zip(normalized_labels, normalized_probabilities):
        brier += sum((row.get(class_name, 0.0) - int(class_name == label)) ** 2 for class_name in classes)
        negative_log_likelihood -= math.log(max(row.get(label, 0.0), 1e-12))
        confidence.append(max(row.values()))
    choice_correct = [prediction == label for prediction, label in zip(normalized_selected, normalized_labels)]
    calibration_labels = choice_correct
    return {
        "count": count,
        "accuracy": _rounded(sum(correct) / count),
        "top2_accuracy": _rounded(sum(top2) / count),
        "macro_f1": _rounded(sum(per_class_f1) / len(per_class_f1)) if per_class_f1 else 0.0,
        "brier": _rounded(brier / count),
        "nll": _rounded(negative_log_likelihood / count),
        "adaptive_ece": _rounded(_adaptive_calibration_error(calibration_labels, confidence, 10)),
        "risk_coverage": _selective_curve(choice_correct, confidence),
    }


def score_metrics(
    targets: Sequence[float],
    values: Sequence[float],
    intervals: Sequence[Sequence[float]],
) -> dict[str, Any]:
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
    ranked_targets = _average_ranks(normalized_targets)
    ranked_values = _average_ranks(normalized_values)
    target_mean = sum(ranked_targets) / count
    value_mean = sum(ranked_values) / count
    numerator = sum((target - target_mean) * (value - value_mean) for target, value in zip(ranked_targets, ranked_values))
    target_norm = math.sqrt(sum((target - target_mean) ** 2 for target in ranked_targets))
    value_norm = math.sqrt(sum((value - value_mean) ** 2 for value in ranked_values))
    spearman = numerator / (target_norm * value_norm) if target_norm and value_norm else None
    return {
        "count": count,
        "mae": _rounded(sum(absolute_errors) / count),
        "rmse": _rounded(math.sqrt(sum(squared_errors) / count)),
        "interval_90_coverage": _rounded(coverage / count),
        "interval_90_width": _rounded(sum(widths) / count),
        "spearman": _rounded(spearman) if spearman is not None else None,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = rank
        cursor = end
    return ranks
