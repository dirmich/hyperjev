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
MAX_MEMORY_CONTEXT_ITEMS = 8
MAX_MEMORY_CONTEXT_CHARS = 2048
_EXPLICIT_STOP_SIGNALS = (
    "obstacle is directly ahead",
    "obstacle is inside the safety zone",
    "immediate collision risk",
    "collision risk is immediate",
    "collision risk is inside the safety zone",
    "emergency collision risk",
    "emergency hazard is detected inside the collision zone",
    "sensor reports an emergency collision risk",
    "collision may occur",
    "imminent impact",
    "stopping distance",
    "wall blocks the vehicle",
    "brake before contact",
    "충돌 위험이 즉시 발생해 정지가 필수다",
    "충돌 구역 안에서 비상 위험이 감지됐다",
    "장애물이 바로 앞",
    "즉시 충돌 위험",
    "비상 충돌 위험",
)

# These are intentionally compound, phrase-level signals. A control fast path
# must be more specific than a single keyword because ``target``, ``safe``, and
# ``wait`` occur in several skills. Ambiguous text is left to the typed model.
_CONTROL_FAST_PATHS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "APPROACH",
        (
            ("target is ahead", "approached safely"),
            ("reachable target", "getting closer"),
            ("reachable target", "locked directly ahead"),
            ("object is still far", "approached first"),
            ("close the gap", "visible marker"),
            ("move nearer", "selected object"),
            ("destination is visible", "still distant"),
            ("reduce distance", "goal"),
            ("go toward", "locked target"),
            ("목표가 앞에 있고", "접근"),
            ("접근 가능한 목표", "가까워지고"),
            ("물체가 아직 멀어", "먼저 접근"),
        ),
    ),
    (
        "HOLD",
        (
            ("pose is stable", "no new command"),
            ("pose is stable", "no hazard is present"),
            ("wait safely", "next sensor update"),
            ("stable pose", "maintained"),
            ("remain stationary", "scene is stable"),
            ("keep the current pose", "another command"),
            ("wait in place", "sensors refresh"),
            ("no motion is needed", "stable waypoint"),
            ("stay still", "no immediate hazard"),
            ("자세가 안정적이고", "새 명령이 없다"),
            ("센서 업데이트까지", "안전하게 기다린다"),
            ("안정된 자세", "유지"),
            ("위험이 없고 자세가 안정적이므로", "안전하게 잠시 멈춘다"),
        ),
    ),
    (
        "MOVE",
        (
            ("corridor is clear", "forward motion"),
            ("corridor is clear", "continue forward"),
            ("free space is available", "route"),
            ("heading is aligned", "open route", "straight ahead"),
            ("advance through", "unobstructed hallway"),
            ("continue forward", "open route"),
            ("clear path",),
            ("forward navigation", "free space"),
            ("proceed straight", "corridor"),
            ("통로가 비어 있어", "앞으로 움직일"),
            ("경로에 자유 공간",),
        ),
    ),
    (
        "ROTATE",
        (
            ("heading is wrong", "turn space is clear"),
            ("turn toward", "next waypoint"),
            ("heading is wrong", "clear turn"),
            ("reorient toward", "corridor"),
            ("turn left", "waypoint"),
            ("change heading", "junction"),
            ("route requires", "turn"),
            ("rotate in place", "face the goal"),
            ("방향이 틀렸고", "회전 공간"),
            ("웨이포인트 방향으로", "회전"),
        ),
    ),
    (
        "RETREAT",
        (
            ("hazard is approaching", "move away"),
            ("safe space behind",),
            ("reverse away", "blocked area"),
            ("back up", "moving obstacle"),
            ("increase distance", "approaching hazard"),
            ("reverse into", "safe rear"),
            ("withdraw from", "blocked front"),
            ("move backward", "escape the danger"),
            ("위험이 다가오므로", "멀어져야"),
            ("뒤쪽에 안전한 공간",),
            ("위험이 다가오지만", "뒤의 빈 경로는 안전하다"),
        ),
    ),
    (
        "INTERACT",
        (
            ("aligned object", "within reach"),
            ("nearby switch", "ready to activate"),
            ("button is aligned", "within hand reach"),
            ("press", "button"),
            ("grasp", "handle"),
            ("activate", "switch"),
            ("touch", "reachable object"),
            ("pick up", "selected item"),
            ("정렬된 물체", "손이 닿는 거리"),
            ("근처 스위치", "작동할 준비"),
            ("버튼이 정렬됐고", "손이 닿는 거리에 있다"),
        ),
    ),
    (
        "RECOVER",
        (
            ("localization is lost", "recovery is needed"),
            ("localization is lost", "no collision risk is present"),
            ("controller reports", "balance fault"),
            ("reinitialize", "pose estimation failure"),
            ("restore balance", "fall"),
            ("lost localization",),
            ("reset", "navigation fault"),
            ("recover from", "unstable state"),
            ("robot needs to regain balance",),
            ("위치를 잃어", "복구가 필요"),
            ("제어기가 균형 오류",),
        ),
    ),
)

_CONTROL_FAST_PATH_BLOCKERS: dict[str, tuple[str, ...]] = {
    "APPROACH": (
        "cannot be approached",
        "can't be approached",
        "do not approach",
        "don't approach",
        "접근할 수 없다",
        "접근할 수 없어",
        "접근하지",
    ),
    "HOLD": (
        "pose is unstable",
        "an emergency hazard is present",
        "the hazard is present",
        "emergency",
        "자세가 불안정",
        "위험이 있다",
        "비상",
    ),
    "MOVE": (
        "path is blocked",
        "corridor is blocked",
        "cannot move",
        "can't move",
        "경로가 막혀",
        "움직일 수 없다",
    ),
    "ROTATE": (
        "cannot turn",
        "can't turn",
        "turn space is blocked",
        "회전할 수 없다",
        "회전 공간이 막혀",
    ),
    "RETREAT": (
        "no safe space behind",
        "cannot retreat",
        "can't retreat",
        "뒤에 안전한 공간이 없다",
        "후퇴할 수 없다",
    ),
    "INTERACT": (
        "not within reach",
        "not aligned",
        "cannot activate",
        "can't activate",
        "손이 닿지 않는다",
        "정렬되지",
        "작동할 수 없다",
    ),
    "RECOVER": (
        "no recovery is needed",
        "recovery is not needed",
        "복구가 필요 없다",
    ),
}


@dataclass(frozen=True)
class ControlMemoryContext:
    """Bounded summaries prefetched from Hyper Memory outside the motor loop."""

    summaries: tuple[str, ...]
    source: str = "hypermemory"

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ControlContractError("memory context source must not be empty")
        if len(self.summaries) > MAX_MEMORY_CONTEXT_ITEMS:
            raise ControlContractError("memory context contains too many summaries")
        total_chars = 0
        for summary in self.summaries:
            if not isinstance(summary, str) or not summary.strip():
                raise ControlContractError("memory summaries must be non-empty strings")
            total_chars += len(summary)
        if total_chars > MAX_MEMORY_CONTEXT_CHARS:
            raise ControlContractError("memory context exceeds the bounded character limit")

    @classmethod
    def from_text(cls, text: str, *, source: str = "hypermemory") -> ControlMemoryContext:
        """Convert a remote context response to one bounded model summary."""

        if not isinstance(text, str) or not text.strip():
            raise ControlContractError("memory context text must not be empty")
        return cls((text.strip()[:MAX_MEMORY_CONTEXT_CHARS],), source=source)

    def render(self) -> str:
        return "\n".join(f"- {summary}" for summary in self.summaries)


@dataclass(frozen=True)
class ControlObservation:
    """Bounded state snapshot supplied to a real-time decision loop."""

    observation_id: str
    state: str
    domain: str
    timestamp_ms: float
    emergency_stop: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_context: ControlMemoryContext | None = None

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.state.strip() or not self.domain.strip():
            raise ControlContractError("observation_id, state, and domain must not be empty")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ControlContractError("timestamp_ms must be a finite non-negative number")
        if not isinstance(self.emergency_stop, bool):
            raise ControlContractError("emergency_stop must be a boolean")
        if not isinstance(self.metadata, dict):
            raise ControlContractError("metadata must be an object")
        if self.memory_context is not None and not isinstance(self.memory_context, ControlMemoryContext):
            raise ControlContractError("memory_context must be a ControlMemoryContext")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ControlObservation:
        if not isinstance(raw, dict):
            raise ControlContractError("observation must be an object")
        try:
            raw_memory_context = raw.get("memory_context")
            if raw_memory_context is None:
                memory_context = None
            elif isinstance(raw_memory_context, dict):
                summaries = raw_memory_context.get("summaries", [])
                if not isinstance(summaries, list):
                    raise ControlContractError("memory_context.summaries must be a list")
                memory_context = ControlMemoryContext(
                    summaries=tuple(summaries),
                    source=str(raw_memory_context.get("source", "hypermemory")),
                )
            elif isinstance(raw_memory_context, list):
                memory_context = ControlMemoryContext(summaries=tuple(raw_memory_context))
            else:
                raise ControlContractError("memory_context must be an object or list")
            return cls(
                observation_id=str(raw["observation_id"]),
                state=str(raw["state"]),
                domain=str(raw["domain"]),
                timestamp_ms=float(raw["timestamp_ms"]),
                emergency_stop=raw.get("emergency_stop", False),
                metadata=raw.get("metadata", {}),
                memory_context=memory_context,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlContractError(f"invalid observation: {exc}") from exc

    def model_state(self) -> str:
        """Render bounded state plus prefetched memory without raw frame history."""

        if self.memory_context is None:
            return self.state
        return f"{self.state}\nRelevant memory context ({self.memory_context.source}):\n{self.memory_context.render()}"


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


def explicit_stop_signal(state: str) -> bool:
    """Recognize only unambiguous collision phrases before model inference."""

    normalized = " ".join(state.casefold().split())
    return any(signal in normalized for signal in _EXPLICIT_STOP_SIGNALS)


def explicit_stop_action() -> ControlAction:
    """Return a planned STOP skill for an explicit collision signal."""

    return ControlAction(
        skill="STOP",
        ttl_ms=50,
        confidence=1.0,
        source="safety-rule",
        abstained=False,
        reason="explicit_collision_signal",
    )


def deterministic_control_action(state: str) -> ControlAction | None:
    """Return one high-precision non-STOP action, or defer to the model.

    The matcher deliberately requires a complete phrase pattern and rejects
    collisions between patterns. STOP remains a separate safety rule and is
    evaluated before this helper by :class:`ControlStudentClient`.
    """

    normalized = " ".join(state.casefold().split())
    matches: list[str] = []
    for skill, patterns in _CONTROL_FAST_PATHS:
        if any(blocker in normalized for blocker in _CONTROL_FAST_PATH_BLOCKERS.get(skill, ())):
            continue
        if any(all(phrase in normalized for phrase in pattern) for pattern in patterns):
            matches.append(skill)
    if len(matches) != 1:
        return None
    return ControlAction(
        skill=matches[0],
        ttl_ms=50,
        confidence=1.0,
        source="control-rule",
        abstained=False,
        reason=f"high_precision_{matches[0].lower()}_signal",
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
        calibration_path: str | Path | None = None,
        enable_fast_path: bool = True,
        device: str = "cpu",
    ) -> None:
        from .student_client import StudentClient

        self.policy = policy or ControlSafetyPolicy()
        if not isinstance(enable_fast_path, bool):
            raise ControlContractError("enable_fast_path must be a boolean")
        self.enable_fast_path = enable_fast_path
        self._client = StudentClient(
            checkpoint_path,
            registry,
            minimum_confidence=0.0,
            calibration_path=calibration_path,
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
        if self.enable_fast_path:
            if explicit_stop_signal(observation.state):
                return explicit_stop_action()
            fast_path_action = deterministic_control_action(observation.state)
            if fast_path_action is not None:
                return apply_safety_policy(
                    observation,
                    fast_path_action,
                    now_ms=now_ms,
                    policy=self.policy,
                )
        try:
            completion = self._client.complete_decision(
                self._task,
                state=observation.model_state(),
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
