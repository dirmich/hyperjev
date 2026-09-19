"""Dependency-free Phase 0 mock API for contract and integration tests."""

from __future__ import annotations

import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

from .config import Phase0Config
from .contracts import (
    BooleanDecision,
    ChoiceDecision,
    ContractError,
    DecisionRequest,
    DecisionResponse,
    ScoreDecision,
    parse_task_reference,
)
from .model_registry import ModelRegistry, ModelRegistryError
from .registry import RegistryError, TaskRegistry
from .routing import DecisionRouter


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _mock_result(task: Any, candidates: tuple[str, ...]):
    if task.output_type == "boolean":
        return BooleanDecision(value=False, probability=0.5, abstained=True)
    if task.output_type == "choice":
        task_candidates = tuple(str(item) for item in task.output.get("candidates", []))
        selected_candidates = candidates or task_candidates
        probabilities = {candidate: 1.0 / len(selected_candidates) for candidate in selected_candidates}
        return ChoiceDecision(
            selected=selected_candidates[0],
            probabilities=probabilities,
            abstained=True,
        )
    if task.output_type == "score":
        return ScoreDecision(value=0.5, interval_90=(0.0, 1.0), abstained=True)
    raise ContractError(f"unsupported mock task type: {task.output_type}")


def _decision_payload(
    registry: TaskRegistry,
    request: DecisionRequest,
    raw: bytes,
    router: DecisionRouter | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if router is not None:
        outcome = router.decide(request)
        payload = outcome.response.to_dict()
        if request.options.return_evidence:
            payload["traces"] = [trace.to_dict() for trace in outcome.traces]
        return payload
    results = {}
    for question in request.questions:
        task_id, version = parse_task_reference(question.task)
        task = registry.get(task_id, version)
        results[question.id] = _mock_result(task, question.candidates)
    response = DecisionResponse(
        request_id=hashlib.sha256(raw).hexdigest()[:16] or str(uuid4()),
        model="phase0-mock",
        calibration="phase0-none",
        results=results,
        route="mock",
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    return response.to_dict()


def handler_for(
    registry: TaskRegistry,
    router: DecisionRouter | None = None,
    model_registry: ModelRegistry | None = None,
):
    """Create a request handler bound to a registry and optional Phase 1 router."""

    class MockHandler(BaseHTTPRequestHandler):
        server_version = "HyperJevPhase1Router/0.9" if router else "HyperJevPhase0Mock/0.7"

        def _send(self, status: int, payload: Any) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send(status, {"error": message})

        def do_GET(self) -> None:
            if self.path == "/health/live":
                self._send(200, {"status": "ok"})
                return
            if self.path == "/health":
                self._send(200, {"status": "ok"})
                return
            if self.path == "/health/ready":
                self._send(
                    200,
                    {
                        "status": "ready",
                        "phase": 1 if router else 0,
                        "model": "phase1-router" if router else "phase0-mock",
                    },
                )
                return
            if self.path == "/v1/tasks":
                self._send(200, {"tasks": [registry.get(task_id).to_dict() for task_id in registry.ids()]})
                return
            if self.path == "/v1/models":
                self._send(200, {"models": list(model_registry.list()) if model_registry else []})
                return
            if self.path == "/metrics":
                self._send(200, {"hyperjev_requests_total": 0})
                return
            self._error(404, "not found")

        def do_POST(self) -> None:
            is_decision = self.path in {"/v1/decide", "/v1/batch/decide"}
            is_task_validation = self.path == "/v1/tasks/validate"
            is_activation = self.path.startswith("/v1/models/") and self.path.endswith("/activate")
            if not is_decision and not is_task_validation and not is_activation:
                self._error(404, "not found")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length)
                if is_activation:
                    if model_registry is None:
                        raise ModelRegistryError("model registry is not configured")
                    model_id = self.path[len("/v1/models/") : -len("/activate")].strip("/")
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                    reason = str(payload.get("reason", "api activation")) if isinstance(payload, dict) else "api activation"
                    self._send(200, model_registry.transition(model_id, "active", reason=reason))
                    return
                decoded = json.loads(raw.decode("utf-8"))
                if is_task_validation:
                    if not isinstance(decoded, dict) or not isinstance(decoded.get("tasks"), list):
                        raise ContractError("task validation requires a tasks list")
                    validated = []
                    for task_reference in decoded["tasks"]:
                        task_id, version = parse_task_reference(str(task_reference))
                        validated.append(registry.get(task_id, version).to_dict())
                    self._send(200, {"valid": True, "tasks": validated})
                    return
                if self.path == "/v1/batch/decide":
                    if not isinstance(decoded, dict) or not isinstance(decoded.get("requests"), list):
                        raise ContractError("batch request must contain a requests list")
                    responses = []
                    for item in decoded["requests"]:
                        if not isinstance(item, dict):
                            raise ContractError("batch requests must contain objects")
                        item_raw = json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
                        responses.append(
                            _decision_payload(
                                registry,
                                DecisionRequest.from_dict(item),
                                item_raw,
                                router,
                            )
                        )
                    self._send(200, {"responses": responses})
                    return
                request = DecisionRequest.from_dict(decoded)
                self._send(200, _decision_payload(registry, request, raw, router))
            except (ContractError, ModelRegistryError, RegistryError, json.JSONDecodeError, ValueError) as exc:
                self._error(400, str(exc))

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return MockHandler


def create_server(
    host: str,
    port: int,
    registry: TaskRegistry,
    *,
    router: DecisionRouter | None = None,
    model_registry: ModelRegistry | None = None,
) -> ThreadingHTTPServer:
    """Create, but do not start, a mock or Phase 1 router server."""

    return ThreadingHTTPServer((host, port), handler_for(registry, router, model_registry))


def serve(
    config: Phase0Config,
    host: str | None = None,
    port: int | None = None,
    *,
    mode: str = "mock",
) -> None:
    """Run the local mock or Phase 1 router server until interrupted."""

    registry = TaskRegistry.load(config.registry_path)
    if mode not in {"mock", "router"}:
        raise ValueError(f"unsupported server mode: {mode}")
    router = DecisionRouter(config, registry) if mode == "router" else None
    model_registry = ModelRegistry(config.model_registry_path)
    server = create_server(
        host or config.server_host,
        port or config.server_port,
        registry,
        router=router,
        model_registry=model_registry,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
