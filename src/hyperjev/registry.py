"""Versioned task registry loader and validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class RegistryError(ValueError):
    """Raised when a task definition violates the registry contract."""


TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]+$")
OUTPUT_TYPES = {"boolean", "choice", "score"}


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    version: int
    input: Mapping[str, Any]
    output: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    cost: Mapping[str, Any]
    languages: tuple[str, ...]
    teacher_prompt_version: int
    source_path: Path

    @property
    def output_type(self) -> str:
        return str(self.output["type"])

    @classmethod
    def from_file(cls, path: str | Path) -> TaskDefinition:
        source_path = Path(path).resolve()
        try:
            with source_path.open(encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except OSError as exc:
            raise RegistryError(f"cannot read task definition {source_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RegistryError(f"task definition must be a mapping: {source_path}")
        task = cls(
            id=str(raw.get("id", "")),
            version=int(raw.get("version", 0)),
            input=raw.get("input", {}),
            output=raw.get("output", {}),
            thresholds=raw.get("thresholds", {}),
            cost=raw.get("cost", {}),
            languages=tuple(str(language) for language in raw.get("languages", [])),
            teacher_prompt_version=int(raw.get("teacher_prompt_version", 0)),
            source_path=source_path,
        )
        task.validate()
        return task

    def validate(self) -> None:
        if not TASK_ID_PATTERN.fullmatch(self.id):
            raise RegistryError(f"invalid task id: {self.id!r} ({self.source_path})")
        if self.version < 1:
            raise RegistryError(f"task version must be positive: {self.id}")
        if not isinstance(self.input, Mapping) or int(self.input.get("max_chars", 0)) <= 0:
            raise RegistryError(f"{self.id}: input.max_chars must be positive")
        if not isinstance(self.output, Mapping) or self.output.get("type") not in OUTPUT_TYPES:
            raise RegistryError(f"{self.id}: output.type must be one of {sorted(OUTPUT_TYPES)}")
        if self.output_type == "choice":
            candidates = self.output.get("candidates", [])
            if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates):
                raise RegistryError(f"{self.id}: choice candidates must be a non-empty unique list")
        if not self.languages:
            raise RegistryError(f"{self.id}: languages must not be empty")
        if self.teacher_prompt_version < 1:
            raise RegistryError(f"{self.id}: teacher_prompt_version must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "input": dict(self.input),
            "output": dict(self.output),
            "thresholds": dict(self.thresholds),
            "cost": dict(self.cost),
            "languages": list(self.languages),
            "teacher_prompt_version": self.teacher_prompt_version,
        }


class TaskRegistry:
    """In-memory view of all task definitions in a registry directory."""

    def __init__(self, tasks: Mapping[str, TaskDefinition]) -> None:
        self._tasks = dict(tasks)

    @classmethod
    def load(cls, directory: str | Path) -> TaskRegistry:
        root = Path(directory).resolve()
        if not root.is_dir():
            raise RegistryError(f"task registry directory not found: {root}")
        tasks: dict[str, TaskDefinition] = {}
        files = sorted((*root.glob("*.yaml"), *root.glob("*.yml")))
        if not files:
            raise RegistryError(f"task registry is empty: {root}")
        for path in files:
            task = TaskDefinition.from_file(path)
            if task.id in tasks:
                raise RegistryError(f"duplicate task id: {task.id}")
            tasks[task.id] = task
        return cls(tasks)

    def get(self, task_id: str, version: int | None = None) -> TaskDefinition:
        try:
            task = self._tasks[task_id]
        except KeyError as exc:
            raise RegistryError(f"unknown task: {task_id}") from exc
        if version is not None and task.version != version:
            raise RegistryError(f"task {task_id} has version {task.version}, requested {version}")
        return task

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))

    def __len__(self) -> int:
        return len(self._tasks)
