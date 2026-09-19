"""Dependency-free temperature calibration primitives for Student logits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "temperature": self.temperature,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "sample_count": self.sample_count,
        }


def _validate(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
    if not logits or len(logits) != len(labels):
        raise ValueError("logits and labels must be non-empty and have equal length")
    class_count = len(logits[0])
    if class_count < 2 or any(len(row) != class_count for row in logits):
        raise ValueError("all logits rows must have at least two equal-length classes")
    if any(label < 0 or label >= class_count for label in labels):
        raise ValueError("labels must be valid class indices")


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for row, label in zip(logits, labels):
        scaled = [float(value) / temperature for value in row]
        maximum = max(scaled)
        log_sum_exp = maximum + math.log(sum(math.exp(value - maximum) for value in scaled))
        total += log_sum_exp - scaled[label]
    return total / len(labels)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    minimum: float = 0.25,
    maximum: float = 4.0,
    steps: int = 76,
) -> CalibrationResult:
    """Select temperature by deterministic grid search on held-out logits."""

    _validate(logits, labels)
    if minimum <= 0 or maximum < minimum or steps < 2:
        raise ValueError("invalid temperature search range")
    before = _nll(logits, labels, 1.0)
    candidates = [minimum + (maximum - minimum) * index / (steps - 1) for index in range(steps)]
    temperature = min(candidates, key=lambda candidate: (_nll(logits, labels, candidate), candidate))
    after = _nll(logits, labels, temperature)
    return CalibrationResult(
        temperature=round(temperature, 6),
        nll_before=round(before, 6),
        nll_after=round(after, 6),
        sample_count=len(labels),
    )


def probabilities(logits: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Convert logits to a numerically stable calibrated probability tuple."""

    if temperature <= 0 or len(logits) < 2:
        raise ValueError("temperature must be positive and logits must contain two classes")
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)
