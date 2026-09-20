"""Low-latency, safety-bounded control contracts for HyperJev integrations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ControlContractError(ValueError):
    """Raised when an observation or action violates the control contract."""


CONTROL_SKILLS = frozenset(
    {
        "STOP",
        "HOLD",
        "MOVE",
        "ROTATE",
        "APPROACH",
        "RETREAT",
        "INTERACT",
        "RECOVER",
    }
)
CONTROL_QUESTION = "Select the next safe high-level control skill."


@dataclass(frozen=True)
class ControlObservation:
    """Bounded state snapshot supplied to a real-time decision loop."""

    observation_id: str
    state: str
    domain: str
    timestamp_ms: float
    emergency_stop: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.state.strip() or not self.domain.strip():
            raise ControlContractError("observation_id, state, and domain must not be empty")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ControlContractError("timestamp_ms must be a finite non-negative number")
        if not isinstance(self.emergency_stop, bool):
            raise ControlContractError("emergency_stop must be a boolean")
        if not isinstance(self.metadata, dict):
            raise ControlContractError("metadata must be an object")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ControlObservation:
        if not isinstance(raw, dict):
            raise ControlContractError("observation must be an object")
        try:
            return cls(
                observation_id=str(raw["observation_id"]),
                state=str(raw["state"]),
                domain=str(raw["domain"]),
                timestamp_ms=float(raw["timestamp_ms"]),
                emergency_stop=raw.get("emergency_stop", False),
                metadata=raw.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlContractError(f"invalid observation: {exc}") from exc


@dataclass(frozen=True)
class ControlAction:
    """Abstract skill command; it is not a motor/PWM command."""

    skill: str
    parameters: dict[str, float] = field(default_factory=dict)
    ttl_ms: int = 100
    confidence: float = 0.0
    source: str = "hyperjev"
    abstained: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.skill not in CONTROL_SKILLS:
            raise ControlContractError(f"unsupported control skill: {self.skill}")
        if isinstance(self.ttl_ms, bool) or self.ttl_ms < 1:
            raise ControlContractError("ttl_ms must be positive")
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ControlContractError("confidence must be between 0 and 1")
        for name, value in self.parameters.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ControlContractError(f"parameter {name!r} must be numeric") from exc
            if (
                isinstance(value, bool)
                or not str(name).strip()
                or not math.isfinite(numeric_value)
                or not -1.0 <= numeric_value <= 1.0
            ):
                raise ControlContractError(f"parameter {name!r} must be finite and within [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "parameters": dict(self.parameters),
            "ttl_ms": self.ttl_ms,
            "confidence": self.confidence,
            "source": self.source,
            "abstained": self.abstained,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ControlSafetyPolicy:
    """Deterministic policy that runs after every model decision."""

    minimum_confidence: float = 0.90
    max_observation_age_ms: float = 100.0
    max_action_ttl_ms: int = 100

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ControlContractError("minimum_confidence must be between 0 and 1")
        if (
            isinstance(self.max_observation_age_ms, bool)
            or not math.isfinite(self.max_observation_age_ms)
            or self.max_observation_age_ms < 0
            or isinstance(self.max_action_ttl_ms, bool)
            or self.max_action_ttl_ms < 1
        ):
            raise ControlContractError("safety limits must be non-negative/positive")


def safe_stop(*, reason: str, ttl_ms: int = 50) -> ControlAction:
    """Create the only action allowed when the loop cannot safely continue."""

    return ControlAction(
        skill="STOP",
        ttl_ms=max(1, ttl_ms),
        confidence=1.0,
        source="safety",
        abstained=True,
        reason=reason,
    )


def apply_safety_policy(
    observation: ControlObservation,
    action: ControlAction,
    *,
    now_ms: float,
    policy: ControlSafetyPolicy | None = None,
) -> ControlAction:
    """Return an executable high-level action or deterministic safe STOP."""

    selected = policy or ControlSafetyPolicy()
    if not math.isfinite(now_ms) or now_ms < observation.timestamp_ms:
        return safe_stop(reason="invalid_clock")
    if observation.emergency_stop:
        return safe_stop(reason="emergency_stop")
    if now_ms - observation.timestamp_ms > selected.max_observation_age_ms:
        return safe_stop(reason="stale_observation")
    if action.abstained:
        return safe_stop(reason=action.reason or "model_abstained")
    if action.confidence < selected.minimum_confidence:
        return safe_stop(reason="confidence_below_control_threshold")
    if action.ttl_ms > selected.max_action_ttl_ms:
        return safe_stop(reason="action_ttl_exceeded")
    return action


class ControlStudentClient:
    """Adapt a typed Student control head to the safety-bounded action contract."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        registry: Any,
        *,
        policy: ControlSafetyPolicy | None = None,
        device: str = "cpu",
    ) -> None:
        from .student_client import StudentClient

        self.policy = policy or ControlSafetyPolicy()
        self._client = StudentClient(
            checkpoint_path,
            registry,
            minimum_confidence=0.0,
            device=device,
        )
        self._task = registry.get("control.skill", 1)

    def decide(self, observation: ControlObservation, *, now_ms: float) -> ControlAction:
        """Return a safe skill decision; runtime failures fail closed to STOP."""

        if observation.emergency_stop:
            return safe_stop(reason="emergency_stop")
        if not math.isfinite(now_ms) or now_ms < observation.timestamp_ms:
            return safe_stop(reason="invalid_clock")
        if now_ms - observation.timestamp_ms > self.policy.max_observation_age_ms:
            return safe_stop(reason="stale_observation")
        try:
            completion = self._client.complete_decision(
                self._task,
                state=observation.state,
                question=CONTROL_QUESTION,
                candidates=list(self._task.output["candidates"]),
            )
            from .contracts import ChoiceDecision, parse_decision_result, validate_result_for_task
            from .teachers import parse_json_object

            result = parse_decision_result(parse_json_object(completion.content))
            validate_result_for_task(self._task, result)
            if not isinstance(result, ChoiceDecision):
                return safe_stop(reason="control_head_output_type_error")
            confidence = result.probabilities.get(result.selected, 0.0)
            action = ControlAction(
                skill=result.selected,
                confidence=confidence,
                source="hyperjev-control",
                abstained=result.abstained,
                reason="student_abstained" if result.abstained else None,
            )
            return apply_safety_policy(
                observation,
                action,
                now_ms=now_ms,
                policy=self.policy,
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return safe_stop(reason=f"control_inference_error:{type(exc).__name__}")
