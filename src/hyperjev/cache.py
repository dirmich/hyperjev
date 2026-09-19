"""Privacy-safe cache keys and a bounded in-memory decision cache."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def normalize_state(value: str) -> str:
    """Normalize whitespace only; semantic text is never lowercased or stored as a key."""

    return " ".join(value.split())


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def state_cache_key(model_version: str, state: str) -> str:
    return _digest({"kind": "state", "model_version": model_version, "state": normalize_state(state)})


def question_cache_key(task_reference: str, question: str, candidates: tuple[str, ...] = ()) -> str:
    return _digest(
        {
            "kind": "question",
            "task": task_reference,
            "question": normalize_state(question),
            "candidates": list(candidates),
        }
    )


def result_cache_key(
    model_version: str,
    calibration_version: str,
    state_key: str,
    question_key: str,
) -> str:
    return _digest(
        {
            "kind": "result",
            "model_version": model_version,
            "calibration_version": calibration_version,
            "state_key": state_key,
            "question_key": question_key,
        }
    )


@dataclass
class BoundedCache(Generic[T]):
    """Small LRU cache whose keys are already privacy-safe digests."""

    max_entries: int = 1024

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._items: OrderedDict[str, T] = OrderedDict()

    def get(self, key: str) -> T | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: str, value: T) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
