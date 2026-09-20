"""Hyper Memory ingestion gate and REST adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any
from urllib.request import Request, urlopen

from .contracts import BooleanDecision, DecisionRequest
from .routing import DecisionRouter, RouterOutcome


@dataclass(frozen=True)
class IngestionDecision:
    """A source-bound gate result safe to pass between Hyper Memory services."""

    status: str
    reason: str
    request_id: str | None
    route: str
    state_sha256: str
    remember: BooleanDecision | None
    memory_response: Mapping[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "request_id": self.request_id,
            "route": self.route,
            "state_sha256": self.state_sha256,
            "remember": self.remember.to_dict() if self.remember else None,
            "memory_response": dict(self.memory_response) if self.memory_response else None,
            "error": self.error,
        }


class HyperMemoryClient:
    """Minimal client for Hyper Memory's documented ``/v1/documents`` API."""

    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def ingest_document(
        self,
        content: str,
        *,
        container: str = "default",
        title: str | None = None,
        source_type: str = "conversation",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "container": container,
            "content": content,
            "source_type": source_type,
        }
        if title:
            payload["title"] = title
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/v1/documents",
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise TypeError("Hyper Memory response must be an object")
        return result

    def compile_context(
        self,
        query: str,
        *,
        container: str = "default",
        max_tokens: int = 256,
    ) -> str:
        """Fetch compact relevant context for a slower skill-decision tick."""

        if not query.strip() or not container.strip():
            raise ValueError("query and container must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        payload = {
            "container": container,
            "query": query,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/v1/context",
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("context"), str):
            raise TypeError("Hyper Memory context response must contain a string context")
        return result["context"]

    def control_context(
        self,
        query: str,
        *,
        container: str = "default",
        max_tokens: int = 256,
    ) -> Any:
        """Return the bounded control context contract used by ControlObservation."""

        from .control import ControlMemoryContext

        return ControlMemoryContext.from_text(
            self.compile_context(query, container=container, max_tokens=max_tokens)
        )


def _hash_state(state: str) -> str:
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


class IngestionGate:
    """Apply deterministic filtering and typed remember-worthiness routing."""

    def __init__(self, router: DecisionRouter, client: HyperMemoryClient | None = None) -> None:
        self.router = router
        self.client = client

    def evaluate(
        self,
        state: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> IngestionDecision:
        state_sha256 = _hash_state(state)
        if not state.strip():
            return IngestionDecision(
                status="ignored",
                reason="empty_input",
                request_id=None,
                route="rule",
                state_sha256=state_sha256,
                remember=None,
            )
        if len(state) > 12000:
            return IngestionDecision(
                status="retain_raw",
                reason="input_exceeds_decision_limit",
                request_id=None,
                route="rule",
                state_sha256=state_sha256,
                remember=None,
            )
        request = DecisionRequest.from_dict(
            {
                "state": state,
                "context": dict(context or {}),
                "questions": [
                    {"id": "remember", "task": "memory.remember_worthy@1"},
                ],
            }
        )
        outcome = self.router.decide(request)
        return self._decision_from_outcome(outcome, state_sha256)

    def ingest(
        self,
        state: str,
        *,
        container: str = "default",
        title: str | None = None,
        source_type: str = "conversation",
        context: Mapping[str, Any] | None = None,
    ) -> IngestionDecision:
        decision = self.evaluate(state, context=context)
        if decision.status != "extract" or self.client is None:
            return decision
        try:
            memory_response = self.client.ingest_document(
                state,
                container=container,
                title=title,
                source_type=source_type,
            )
        except (OSError, TypeError, ValueError) as exc:
            return replace(
                decision,
                status="review",
                reason="hypermemory_unavailable",
                error=f"{type(exc).__name__}: {str(exc)[:240]}",
            )
        return replace(decision, memory_response=memory_response)

    @staticmethod
    def _decision_from_outcome(outcome: RouterOutcome, state_sha256: str) -> IngestionDecision:
        result = outcome.response.results["remember"]
        if not isinstance(result, BooleanDecision):
            raise TypeError("ingestion gate requires a boolean remember_worthy result")
        trace = outcome.traces[0]
        if result.abstained or trace.review_required:
            status = "review"
            reason = trace.fallback_reason or "human_review_required"
        elif result.value:
            status = "extract"
            reason = "remember_worthy_accepted"
        else:
            status = "retain_raw"
            reason = "remember_worthy_rejected"
        return IngestionDecision(
            status=status,
            reason=reason,
            request_id=outcome.response.request_id,
            route=outcome.response.route,
            state_sha256=state_sha256,
            remember=result,
        )
