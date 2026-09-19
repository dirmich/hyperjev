"""Canonical sample validation for Phase 0 fixtures and later datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import TaskRegistry


class SampleError(ValueError):
    """Raised when a canonical dataset sample is malformed."""


@dataclass(frozen=True)
class CanonicalSample:
    sample_id: str
    task_id: str
    task_version: int
    state: str
    question: str
    target: Any
    language: str
    domain: str
    source: Mapping[str, Any]
    labels: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], registry: TaskRegistry) -> CanonicalSample:
        required = {
            "sample_id",
            "task_id",
            "task_version",
            "state",
            "question",
            "target",
            "language",
            "domain",
            "source",
            "labels",
            "provenance",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise SampleError(f"sample is missing fields: {', '.join(missing)}")
        sample_id = str(value["sample_id"])
        task_id = str(value["task_id"])
        task_version = int(value["task_version"])
        task = registry.get(task_id, task_version)
        state = str(value["state"])
        question = str(value["question"])
        language = str(value["language"])
        domain = str(value["domain"])
        if not sample_id or not state or not question or not domain:
            raise SampleError("sample_id, state, question, and domain must not be empty")
        if language not in task.languages:
            raise SampleError(f"{task_id}: language {language!r} is not registered")
        for field in ("source", "labels", "provenance"):
            if not isinstance(value[field], Mapping):
                raise SampleError(f"sample.{field} must be an object")
        if not isinstance(value["provenance"].get("prompt_version"), int):
            raise SampleError("sample.provenance.prompt_version must be an integer")
        return cls(
            sample_id=sample_id,
            task_id=task_id,
            task_version=task_version,
            state=state,
            question=question,
            target=value["target"],
            language=language,
            domain=domain,
            source=value["source"],
            labels=value["labels"],
            provenance=value["provenance"],
        )


def load_jsonl(path: str | Path, registry: TaskRegistry) -> tuple[CanonicalSample, ...]:
    """Load and validate a JSONL sample file, rejecting duplicate IDs."""

    source_path = Path(path)
    samples: list[CanonicalSample] = []
    seen: set[str] = set()
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SampleError(f"cannot read sample file {source_path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SampleError(f"{source_path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(raw, Mapping):
            raise SampleError(f"{source_path}:{line_number}: sample must be an object")
        sample = CanonicalSample.from_dict(raw, registry)
        if sample.sample_id in seen:
            raise SampleError(f"duplicate sample_id: {sample.sample_id}")
        seen.add(sample.sample_id)
        samples.append(sample)
    if not samples:
        raise SampleError(f"sample file is empty: {source_path}")
    return tuple(samples)
