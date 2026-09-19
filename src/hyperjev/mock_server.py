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
from .registry import RegistryError, TaskRegistry


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


def handler_for(registry: TaskRegistry):
    """Create a request handler bound to a task registry."""

    class MockHandler(BaseHTTPRequestHandler):
        server_version = "HyperJevPhase0Mock/0.7"

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
            if self.path == "/health/ready":
                self._send(200, {"status": "ready", "phase": 0, "model": "phase0-mock"})
                return
            if self.path == "/v1/tasks":
                self._send(200, {"tasks": [registry.get(task_id).to_dict() for task_id in registry.ids()]})
                return
            if self.path == "/metrics":
                self._send(200, {"hyperjev_phase0_mock_requests_total": 0})
                return
            self._error(404, "not found")

        def do_POST(self) -> None:
            if self.path != "/v1/decide":
                self._error(404, "not found")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length)
                request = DecisionRequest.from_dict(json.loads(raw.decode("utf-8")))
                started = time.perf_counter()
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
                self._send(200, response.to_dict())
            except (ContractError, RegistryError, json.JSONDecodeError, ValueError) as exc:
                self._error(400, str(exc))

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return MockHandler


def create_server(host: str, port: int, registry: TaskRegistry) -> ThreadingHTTPServer:
    """Create, but do not start, a Phase 0 mock server."""

    return ThreadingHTTPServer((host, port), handler_for(registry))


def serve(config: Phase0Config, host: str | None = None, port: int | None = None) -> None:
    """Run the Phase 0 mock server until interrupted."""

    registry = TaskRegistry.load(config.registry_path)
    server = create_server(host or config.server_host, port or config.server_port, registry)
    try:
        server.serve_forever()
    finally:
        server.server_close()
