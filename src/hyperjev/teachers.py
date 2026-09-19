"""OpenAI-compatible teacher probes and completion client."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import TeacherSettings


@dataclass(frozen=True)
class TeacherProbe:
    name: str
    base_url: str
    configured_model: str
    roles: tuple[str, ...]
    ok: bool
    status_code: int | None
    elapsed_ms: float
    available_models: tuple[str, ...]
    model_found: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "configured_model": self.configured_model,
            "roles": list(self.roles),
            "ok": self.ok,
            "status_code": self.status_code,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "available_models": list(self.available_models),
            "model_found": self.model_found,
            "error": self.error,
        }


@dataclass(frozen=True)
class TeacherCompletion:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    elapsed_ms: float


def _model_ids(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    raw_models = payload.get("data", payload.get("models", []))
    if not isinstance(raw_models, list):
        return ()
    ids: list[str] = []
    for item in raw_models:
        if isinstance(item, dict):
            value = item.get("id") or item.get("name")
            if value:
                ids.append(str(value))
    return tuple(dict.fromkeys(ids))


def probe_teacher(settings: TeacherSettings, timeout_s: float = 3.0) -> TeacherProbe:
    """Probe ``GET /models`` and verify the configured alias is advertised."""

    started = time.perf_counter()
    url = f"{settings.base_url}/models"
    status_code: int | None = None
    models: tuple[str, ...] = ()
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_s) as response:
            status_code = int(response.status)
            payload = json.loads(response.read().decode("utf-8"))
        models = _model_ids(payload)
        ok = 200 <= status_code < 300 and settings.model in models
        error = None if ok else f"configured model not advertised: {settings.model}"
    except HTTPError as exc:
        status_code = int(exc.code)
        ok = False
        error = f"HTTP {exc.code}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        ok = False
        error = str(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return TeacherProbe(
        name=settings.name,
        base_url=settings.base_url,
        configured_model=settings.model,
        roles=settings.roles,
        ok=ok,
        status_code=status_code,
        elapsed_ms=elapsed_ms,
        available_models=models,
        model_found=settings.model in models,
        error=error,
    )


class TeacherClient:
    """Small dependency-free client for llama.cpp/OpenAI-compatible endpoints."""

    def __init__(self, settings: TeacherSettings, timeout_s: float | None = None) -> None:
        self.settings = settings
        self.timeout_s = timeout_s if timeout_s is not None else settings.request_timeout_s

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        seed: int = 0,
    ) -> TeacherCompletion:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "max_tokens": max_tokens,
            "response_format": {"type": self.settings.response_format},
            "stream": False,
        }
        if self.settings.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.settings.base_url}/chat/completions",
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - started) * 1000
        choices = result.get("choices", []) if isinstance(result, dict) else []
        if not choices or not isinstance(choices[0], dict):
            raise ValueError("teacher response does not contain choices[0]")
        choice = choices[0]
        content = _message_content(choice)
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        return TeacherCompletion(
            content=content,
            model=str(result.get("model", self.settings.model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            elapsed_ms=elapsed_ms,
        )


def response_hash(content: str) -> str:
    """Hash teacher output for provenance without storing raw text by default."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _message_content(choice: dict[str, Any]) -> str:
    """Read normal, multimodal, and function-call compatible message output."""

    message = choice.get("message", {})
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            joined = "".join(part for part in parts if isinstance(part, str))
            if joined.strip():
                return joined
        tool_calls = message.get("tool_calls", [])
        if isinstance(tool_calls, list) and tool_calls:
            function = tool_calls[0].get("function", {})
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str) and arguments.strip():
                return arguments
    content = choice.get("text")
    if isinstance(content, str) and content.strip():
        return content
    return ""
