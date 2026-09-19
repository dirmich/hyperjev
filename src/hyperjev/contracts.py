"""Typed decision request/response contracts for the HyperJev API."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .registry import TaskDefinition


class ContractError(ValueError):
    """Raised when an API or teacher payload violates the typed contract."""


TASK_REFERENCE_PATTERN = re.compile(r"^(?P<id>[a-z][a-z0-9_.-]+)@(?P<version>[1-9][0-9]*)$")
QUESTION_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
MAX_QUESTIONS = 128
MAX_STATE_CHARS = 12000


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{label} must be a number between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a number between 0 and 1") from exc
    if not 0.0 <= result <= 1.0:
        raise ContractError(f"{label} must be a number between 0 and 1")
    return result


def parse_task_reference(value: str) -> tuple[str, int]:
    """Parse the public ``task.id@version`` reference form."""

    match = TASK_REFERENCE_PATTERN.fullmatch(value)
    if not match:
        raise ContractError(f"task must use id@version syntax: {value!r}")
    return match.group("id"), int(match.group("version"))


@dataclass(frozen=True)
class Question:
    id: str
    task: str
    candidates: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Question:
        raw = _require_mapping(value, "question")
        question_id = str(raw.get("id", ""))
        if not QUESTION_ID_PATTERN.fullmatch(question_id):
            raise ContractError(f"question.id is invalid: {question_id!r}")
        task = str(raw.get("task", ""))
        parse_task_reference(task)
        raw_candidates = raw.get("candidates", [])
        if raw_candidates is None:
            raw_candidates = []
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise ContractError(f"question {question_id}: candidates must be a list")
        candidates = tuple(str(candidate) for candidate in raw_candidates)
        if len(set(candidates)) != len(candidates):
            raise ContractError(f"question {question_id}: candidates must be unique")
        return cls(id=question_id, task=task, candidates=candidates)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "task": self.task}
        if self.candidates:
            result["candidates"] = list(self.candidates)
        return result


@dataclass(frozen=True)
class DecisionOptions:
    allow_fallback: bool = True
    return_evidence: bool = False
    deadline_ms: int = 500

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> DecisionOptions:
        raw = _require_mapping(value or {}, "options")
        deadline_ms = int(raw.get("deadline_ms", 500))
        if deadline_ms < 1 or deadline_ms > 300_000:
            raise ContractError("options.deadline_ms must be between 1 and 300000")
        return cls(
            allow_fallback=bool(raw.get("allow_fallback", True)),
            return_evidence=bool(raw.get("return_evidence", False)),
            deadline_ms=deadline_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_fallback": self.allow_fallback,
            "return_evidence": self.return_evidence,
            "deadline_ms": self.deadline_ms,
        }


@dataclass(frozen=True)
class DecisionRequest:
    state: str
    questions: tuple[Question, ...]
    context: Mapping[str, Any]
    options: DecisionOptions

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionRequest:
        raw = _require_mapping(value, "request")
        state = raw.get("state")
        if not isinstance(state, str) or not state.strip():
            raise ContractError("state must be a non-empty string")
        if len(state) > MAX_STATE_CHARS:
            raise ContractError(f"state must not exceed {MAX_STATE_CHARS} characters")
        raw_questions = raw.get("questions")
        if not isinstance(raw_questions, Sequence) or isinstance(raw_questions, (str, bytes)):
            raise ContractError("questions must be a list")
        if not 1 <= len(raw_questions) <= MAX_QUESTIONS:
            raise ContractError(f"questions must contain 1 to {MAX_QUESTIONS} items")
        questions = tuple(Question.from_dict(item) for item in raw_questions)
        question_ids = [question.id for question in questions]
        if len(set(question_ids)) != len(question_ids):
            raise ContractError("question ids must be unique within a request")
        context = _require_mapping(raw.get("context", {}), "context")
        return cls(
            state=state,
            questions=questions,
            context=dict(context),
            options=DecisionOptions.from_dict(raw.get("options")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "context": dict(self.context),
            "questions": [question.to_dict() for question in self.questions],
            "options": self.options.to_dict(),
        }


@dataclass(frozen=True)
class BooleanDecision:
    value: bool
    probability: float
    abstained: bool = False
    evidence: tuple[Mapping[str, Any], ...] = ()

    @property
    def type(self) -> str:
        return "boolean"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "value": self.value,
            "probability": self.probability,
            "abstained": self.abstained,
        }
        if self.evidence:
            result["evidence"] = [dict(item) for item in self.evidence]
        return result


@dataclass(frozen=True)
class ChoiceDecision:
    selected: str
    probabilities: Mapping[str, float]
    abstained: bool = False
    evidence: tuple[Mapping[str, Any], ...] = ()

    @property
    def type(self) -> str:
        return "choice"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "selected": self.selected,
            "probabilities": dict(self.probabilities),
            "abstained": self.abstained,
        }
        if self.evidence:
            result["evidence"] = [dict(item) for item in self.evidence]
        return result


@dataclass(frozen=True)
class ScoreDecision:
    value: float
    interval_90: tuple[float, float]
    abstained: bool = False
    evidence: tuple[Mapping[str, Any], ...] = ()

    @property
    def type(self) -> str:
        return "score"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "value": self.value,
            "interval_90": list(self.interval_90),
            "abstained": self.abstained,
        }
        if self.evidence:
            result["evidence"] = [dict(item) for item in self.evidence]
        return result


DecisionResult = BooleanDecision | ChoiceDecision | ScoreDecision


def _evidence(raw: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    value = raw.get("evidence", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError("result.evidence must be a list")
    return tuple(_require_mapping(item, "result.evidence[]") for item in value)


def parse_decision_result(value: Mapping[str, Any]) -> DecisionResult:
    """Parse a typed result, accepting the legacy ``noul`` boolean alias."""

    raw = _require_mapping(value, "result")
    result_type = str(raw.get("type", ""))
    evidence = _evidence(raw)
    abstained = bool(raw.get("abstained", False))
    if result_type in {"boolean", "noul"}:
        if not isinstance(raw.get("value"), bool):
            raise ContractError("boolean result.value must be bool")
        return BooleanDecision(
            value=raw["value"],
            probability=_probability(raw.get("probability"), "boolean result.probability"),
            abstained=abstained,
            evidence=evidence,
        )
    if result_type == "choice":
        selected = raw.get("selected")
        probabilities = _require_mapping(raw.get("probabilities"), "choice result.probabilities")
        if not isinstance(selected, str) or not selected:
            raise ContractError("choice result.selected must be a non-empty string")
        if not probabilities:
            raise ContractError("choice result.probabilities must not be empty")
        parsed_probabilities = {
            str(key): _probability(probability, f"choice probability {key}")
            for key, probability in probabilities.items()
        }
        if selected not in parsed_probabilities:
            raise ContractError("choice result.selected must be present in probabilities")
        if abs(sum(parsed_probabilities.values()) - 1.0) > 0.02:
            raise ContractError("choice result.probabilities must sum to 1 within 0.02")
        return ChoiceDecision(
            selected=selected,
            probabilities=parsed_probabilities,
            abstained=abstained,
            evidence=evidence,
        )
    if result_type == "score":
        value_number = _probability(raw.get("value"), "score result.value")
        interval = raw.get("interval_90")
        if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)) or len(interval) != 2:
            raise ContractError("score result.interval_90 must contain two numbers")
        lower = _probability(interval[0], "score interval lower")
        upper = _probability(interval[1], "score interval upper")
        if lower > upper or not lower <= value_number <= upper:
            raise ContractError("score result.interval_90 must contain value in ascending order")
        return ScoreDecision(
            value=value_number,
            interval_90=(lower, upper),
            abstained=abstained,
            evidence=evidence,
        )
    raise ContractError(f"unsupported result type: {result_type!r}")


def validate_result_for_task(task: TaskDefinition, result: DecisionResult) -> None:
    """Validate that a typed result matches its registered task schema."""

    if task.output_type == "boolean" and not isinstance(result, BooleanDecision):
        raise ContractError(f"{task.id}: expected boolean result")
    if task.output_type == "choice":
        if not isinstance(result, ChoiceDecision):
            raise ContractError(f"{task.id}: expected choice result")
        candidates = tuple(str(candidate) for candidate in task.output.get("candidates", []))
        if result.selected not in candidates:
            raise ContractError(f"{task.id}: selected value is not a registered candidate")
        if set(result.probabilities) != set(candidates):
            raise ContractError(f"{task.id}: probabilities must cover all registered candidates")
    if task.output_type == "score" and not isinstance(result, ScoreDecision):
        raise ContractError(f"{task.id}: expected score result")


@dataclass(frozen=True)
class DecisionResponse:
    request_id: str
    model: str
    calibration: str
    results: Mapping[str, DecisionResult]
    route: str
    latency_ms: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionResponse:
        raw = _require_mapping(value, "response")
        request_id = str(raw.get("request_id", ""))
        model = str(raw.get("model", ""))
        calibration = str(raw.get("calibration", ""))
        route = str(raw.get("route", ""))
        if not request_id or not model or not calibration:
            raise ContractError("response request_id, model, and calibration are required")
        if route not in {"hyperjev", "rule", "qwen", "gemma", "human", "mixed", "mock"}:
            raise ContractError(f"unsupported response route: {route!r}")
        try:
            latency_ms = float(raw.get("latency_ms"))
        except (TypeError, ValueError) as exc:
            raise ContractError("response.latency_ms must be a non-negative number") from exc
        if latency_ms < 0:
            raise ContractError("response.latency_ms must be non-negative")
        raw_results = _require_mapping(raw.get("results"), "response.results")
        results = {str(key): parse_decision_result(result) for key, result in raw_results.items()}
        if not results:
            raise ContractError("response.results must not be empty")
        return cls(
            request_id=request_id,
            model=model,
            calibration=calibration,
            results=results,
            route=route,
            latency_ms=latency_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "calibration": self.calibration,
            "results": {key: result.to_dict() for key, result in self.results.items()},
            "route": self.route,
            "latency_ms": self.latency_ms,
        }
