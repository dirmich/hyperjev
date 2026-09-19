"""Phase 1 rule and teacher decision router."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any

from .config import Phase0Config
from .contracts import (
    BooleanDecision,
    ChoiceDecision,
    ContractError,
    DecisionRequest,
    DecisionResponse,
    DecisionResult,
    ScoreDecision,
    parse_decision_result,
    parse_task_reference,
    validate_result_for_task,
)
from .prompts import messages_for_question
from .registry import TaskDefinition, TaskRegistry
from .review import ReviewStore
from .rules import match_rule
from .teachers import TeacherClient, parse_json_object, response_hash


@dataclass(frozen=True)
class DecisionTrace:
    question_id: str
    task: str
    state_sha256: str
    attempts: tuple[dict[str, Any], ...]
    final_route: str
    review_required: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "task": self.task,
            "state_sha256": self.state_sha256,
            "attempts": [dict(attempt) for attempt in self.attempts],
            "final_route": self.final_route,
            "review_required": self.review_required,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class RouterOutcome:
    response: DecisionResponse
    traces: tuple[DecisionTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = self.response.to_dict()
        payload["traces"] = [trace.to_dict() for trace in self.traces]
        return payload


def _state_hash(state: str) -> str:
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def _request_id(request: DecisionRequest) -> str:
    serialized = json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"req-{hashlib.sha256(serialized).hexdigest()[:24]}"


def _accepted(task: TaskDefinition, result: DecisionResult) -> bool:
    if result.abstained:
        return False
    thresholds = task.thresholds
    if isinstance(result, BooleanDecision):
        if result.value:
            return result.probability >= float(thresholds.get("auto_true", 0.92))
        return result.probability >= 1.0 - float(thresholds.get("auto_false", 0.08))
    if isinstance(result, ChoiceDecision):
        return result.probabilities.get(result.selected, 0.0) >= float(
            thresholds.get("minimum_probability", 0.9)
        )
    return bool(isinstance(result, ScoreDecision))


def _abstain(task: TaskDefinition, result: DecisionResult | None) -> DecisionResult:
    if result is not None:
        return replace(result, abstained=True)
    if task.output_type == "boolean":
        return BooleanDecision(value=False, probability=0.5, abstained=True)
    if task.output_type == "choice":
        candidates = tuple(str(candidate) for candidate in task.output.get("candidates", []))
        probability = 1.0 / len(candidates)
        return ChoiceDecision(
            selected=candidates[0],
            probabilities={candidate: probability for candidate in candidates},
            abstained=True,
        )
    return ScoreDecision(value=0.5, interval_90=(0.0, 1.0), abstained=True)


class DecisionRouter:
    """Route each question through rules, Qwen, Gemma, and human review."""

    def __init__(
        self,
        config: Phase0Config,
        registry: TaskRegistry,
        *,
        clients: dict[str, Any] | None = None,
        review_store: ReviewStore | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.clients: dict[str, Any] = {
            "qwen": TeacherClient(config.teachers["qwen"]),
            "gemma": TeacherClient(config.teachers["gemma"]),
        }
        if clients:
            self.clients.update(clients)
        self.review_store = review_store or ReviewStore(config.review_path)

    def _teacher_attempt(
        self,
        provider: str,
        task: TaskDefinition,
        request: DecisionRequest,
        question_id: str,
        candidates: list[str],
    ) -> tuple[DecisionResult | None, dict[str, Any]]:
        client = self.clients[provider]
        started = time.perf_counter()
        attempt: dict[str, Any] = {
            "provider": provider,
            "model": self.config.teachers[provider].model,
            "status": "error",
            "schema_valid": False,
            "accepted": False,
            "latency_ms": None,
            "response_sha256": None,
            "reason": None,
        }
        try:
            completion = client.complete(
                messages_for_question(
                    provider,
                    task,
                    state=request.state,
                    question=question_id,
                    candidates=candidates,
                )
            )
            attempt["latency_ms"] = round(float(completion.elapsed_ms), 3)
            attempt["model"] = completion.model
            attempt["response_sha256"] = response_hash(completion.content)
            parsed = parse_decision_result(parse_json_object(completion.content))
            validate_result_for_task(task, parsed)
            attempt["schema_valid"] = True
            if _accepted(task, parsed):
                attempt["status"] = "accepted"
                attempt["accepted"] = True
                return parsed, attempt
            attempt["status"] = "uncertain"
            attempt["reason"] = "valid_result_below_acceptance_threshold"
            return parsed, attempt
        except (ContractError, OSError, TypeError, ValueError) as exc:
            # One unavailable or malformed teacher must not break the chain.
            attempt["reason"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            attempt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return None, attempt

    def decide(self, request: DecisionRequest) -> RouterOutcome:
        started = time.perf_counter()
        request_id = _request_id(request)
        state_sha256 = _state_hash(request.state)
        results: dict[str, DecisionResult] = {}
        traces: list[DecisionTrace] = []
        routes: list[str] = []

        for question in request.questions:
            task_id, task_version = parse_task_reference(question.task)
            task = self.registry.get(task_id, task_version)
            attempts: list[dict[str, Any]] = []
            selected: DecisionResult | None = None
            final_route = "human"
            fallback_reason: str | None = None
            last_valid: DecisionResult | None = None

            rule = match_rule(task, state=request.state, question=question.id)
            if rule is not None:
                validate_result_for_task(task, rule.result)
                rule_attempt = {
                    "provider": "rule",
                    "rule_id": rule.rule_id,
                    "status": "accepted" if _accepted(task, rule.result) else "uncertain",
                    "schema_valid": True,
                    "accepted": _accepted(task, rule.result),
                    "latency_ms": 0.0,
                    "response_sha256": None,
                    "reason": rule.reason,
                }
                attempts.append(rule_attempt)
                if _accepted(task, rule.result):
                    selected = rule.result
                    final_route = "rule"

            if selected is None and request.options.allow_fallback:
                candidates = list(question.candidates or task.output.get("candidates", []))
                for provider in ("qwen", "gemma"):
                    if provider == "gemma" and (
                        (time.perf_counter() - started) * 1000 >= request.options.deadline_ms
                    ):
                        fallback_reason = "deadline_exceeded_before_gemma"
                        break
                    result, attempt = self._teacher_attempt(
                        provider, task, request, question.id, candidates
                    )
                    attempts.append(attempt)
                    if result is not None:
                        last_valid = result
                    if result is not None and attempt["accepted"]:
                        selected = result
                        final_route = provider
                        break
                    if attempt["status"] == "uncertain":
                        fallback_reason = f"{provider}_uncertain"
                    elif attempt["status"] == "error":
                        fallback_reason = f"{provider}_error"
            elif selected is None:
                fallback_reason = "fallback_disabled"

            if selected is None:
                selected = _abstain(task, last_valid)
                final_route = "human"
                fallback_reason = fallback_reason or "teachers_exhausted"
            review_required = final_route == "human"
            results[question.id] = selected
            routes.append(final_route)
            trace = DecisionTrace(
                question_id=question.id,
                task=f"{task.id}@{task.version}",
                state_sha256=state_sha256,
                attempts=tuple(attempts),
                final_route=final_route,
                review_required=review_required,
                fallback_reason=fallback_reason,
            )
            traces.append(trace)
            if review_required:
                self.review_store.append(
                    request_id=request_id,
                    question_id=question.id,
                    task=task,
                    state_sha256=state_sha256,
                    result=selected,
                    route=final_route,
                    trace=trace.to_dict(),
                    reason=fallback_reason or "human_review_required",
                )

        route = routes[0] if routes and len(set(routes)) == 1 else "mixed"
        response = DecisionResponse(
            request_id=request_id,
            model="phase1-router",
            calibration="phase1-router-uncalibrated",
            results=results,
            route=route,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return RouterOutcome(response=response, traces=tuple(traces))
