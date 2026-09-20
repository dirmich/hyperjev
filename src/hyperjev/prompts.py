"""Deterministic Phase 0 teacher prompts."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = 2


def system_prompt(provider: str) -> str:
    if provider == "qwen":
        role = "Qwen is the primary data-generator and fallback labeler."
    elif provider == "gemma":
        role = "Gemma is an independent cross-validator and judge; do not copy another model's answer."
    else:
        raise ValueError(f"unsupported teacher provider: {provider}")
    return (
        "You are a deterministic HyperJev Phase 0 teacher. "
        f"{role} Return exactly one JSON object and no markdown. "
        "The object must match the requested typed result schema. "
        "Use probability values in [0, 1] and abstain when uncertain. "
        "For a choice result, include every candidate exactly once and make "
        "the probabilities sum to exactly 1.0."
    )


def user_prompt(
    *,
    task_id: str,
    task_version: int,
    state: str,
    question: str,
    output_type: str = "boolean",
    candidates: list[str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "task": f"{task_id}@{task_version}",
        "state": state,
        "question": question,
        "output_schema": {
            "type": output_type,
            "required": (
                ["type", "value", "probability", "abstained"]
                if output_type == "boolean"
                else ["type", "selected", "probabilities", "abstained"]
                if output_type == "choice"
                else ["type", "value", "interval_90", "abstained"]
            ),
        },
    }
    if candidates:
        payload["candidates"] = candidates
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def messages_for_sample(provider: str, sample: Any, task: Any) -> list[dict[str, str]]:
    return messages_for_question(
        provider,
        task,
        state=sample.state,
        question=sample.question,
        candidates=list(task.output.get("candidates", [])),
    )


def messages_for_question(
    provider: str,
    task: Any,
    *,
    state: str,
    question: str,
    candidates: list[str] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(provider)},
        {
            "role": "user",
            "content": user_prompt(
                task_id=task.id,
                task_version=task.version,
                state=state,
                question=question,
                output_type=task.output_type,
                candidates=candidates or list(task.output.get("candidates", [])),
            ),
        },
    ]
