"""Raw-input-free traffic and drift monitoring primitives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .contracts import BooleanDecision, ChoiceDecision, DecisionResponse, ScoreDecision


def _confidence(result: Any) -> float:
    if isinstance(result, BooleanDecision):
        return result.probability
    if isinstance(result, ChoiceDecision):
        return result.probabilities.get(result.selected, 0.0)
    if isinstance(result, ScoreDecision):
        return max(0.0, 1.0 - (result.interval_90[1] - result.interval_90[0]))
    return 0.0


@dataclass
class TrafficMonitor:
    """Aggregate response metadata without retaining request text."""

    total_requests: int = 0
    total_questions: int = 0
    abstained_questions: int = 0
    confidence_sum: float = 0.0
    route_counts: Counter[str] = field(default_factory=Counter)

    def observe(self, response: DecisionResponse) -> None:
        self.total_requests += 1
        self.route_counts[response.route] += 1
        for result in response.results.values():
            self.total_questions += 1
            self.abstained_questions += int(result.abstained)
            self.confidence_sum += _confidence(result)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_questions": self.total_questions,
            "abstained_questions": self.abstained_questions,
            "abstain_rate": round(self.abstained_questions / self.total_questions, 6)
            if self.total_questions
            else 0.0,
            "mean_confidence": round(self.confidence_sum / self.total_questions, 6)
            if self.total_questions
            else 0.0,
            "route_counts": dict(sorted(self.route_counts.items())),
        }


def compare_snapshots(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    threshold: float = 0.10,
) -> dict[str, Any]:
    """Report material rate shifts; no automatic retraining or promotion."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    baseline_questions = max(int(baseline.get("total_questions", 0)), 1)
    current_questions = max(int(current.get("total_questions", 0)), 1)
    baseline_abstain = float(baseline.get("abstained_questions", 0)) / baseline_questions
    current_abstain = float(current.get("abstained_questions", 0)) / current_questions
    baseline_routes = baseline.get("route_counts", {})
    current_routes = current.get("route_counts", {})
    route_names = sorted(set(baseline_routes) | set(current_routes))
    route_shifts = {
        route: round(
            float(current_routes.get(route, 0)) / max(int(current.get("total_requests", 0)), 1)
            - float(baseline_routes.get(route, 0)) / max(int(baseline.get("total_requests", 0)), 1),
            6,
        )
        for route in route_names
    }
    max_route_shift = max((abs(value) for value in route_shifts.values()), default=0.0)
    abstain_shift = round(current_abstain - baseline_abstain, 6)
    drifted = abs(abstain_shift) > threshold or max_route_shift > threshold
    return {
        "drifted": drifted,
        "threshold": threshold,
        "abstain_rate_shift": abstain_shift,
        "route_shifts": route_shifts,
        "max_route_shift": round(max_route_shift, 6),
        "action": "review_dataset_and_calibration" if drifted else "observe",
    }
