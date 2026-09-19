"""Local model manifest registry with explicit promotion and rollback gates."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistryError(ValueError):
    """Raised when a model manifest or lifecycle transition is invalid."""


STATUSES = ("trained", "evaluated", "calibrated", "candidate", "canary", "active", "retired")
ALLOWED_TRANSITIONS = {
    "trained": {"evaluated"},
    "evaluated": {"calibrated"},
    "calibrated": {"candidate"},
    "candidate": {"canary", "retired"},
    "canary": {"active", "candidate", "retired"},
    "active": {"retired"},
    "retired": {"candidate", "active"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _empty_state() -> dict[str, Any]:
    return {"models": {}, "active_model": None, "events": []}


class ModelRegistry:
    """A small atomic JSON registry for local model lifecycle control."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_state()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelRegistryError(f"cannot read model registry: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("models", {}), dict):
            raise ModelRegistryError("model registry state is malformed")
        return value

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f"{self.path.name}.", dir=self.path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
            Path(temporary).replace(self.path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any]) -> None:
        required = {
            "model_id",
            "base_model",
            "dataset_hash",
            "git_commit",
            "tasks",
            "calibration_version",
            "runtime",
            "precision",
            "status",
        }
        missing = sorted(required.difference(manifest))
        if missing:
            raise ModelRegistryError(f"model manifest missing fields: {', '.join(missing)}")
        model_id = str(manifest["model_id"])
        if not model_id or manifest["status"] not in STATUSES:
            raise ModelRegistryError("model_id and status are invalid")
        if not isinstance(manifest["tasks"], dict) or not manifest["tasks"]:
            raise ModelRegistryError("model manifest tasks must be a non-empty object")

    def register(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._validate_manifest(manifest)
        state = self._read()
        model_id = str(manifest["model_id"])
        if model_id in state["models"]:
            raise ModelRegistryError(f"model already registered: {model_id}")
        record = dict(manifest)
        record["registered_at"] = _now()
        state["models"][model_id] = record
        state["events"].append({"at": record["registered_at"], "action": "register", "model_id": model_id})
        self._write(state)
        return record

    def get(self, model_id: str) -> dict[str, Any]:
        state = self._read()
        try:
            return dict(state["models"][model_id])
        except KeyError as exc:
            raise ModelRegistryError(f"unknown model: {model_id}") from exc

    def list(self) -> tuple[dict[str, Any], ...]:
        state = self._read()
        return tuple(dict(state["models"][model_id]) for model_id in sorted(state["models"]))

    def transition(self, model_id: str, target: str, *, reason: str = "") -> dict[str, Any]:
        if target not in STATUSES:
            raise ModelRegistryError(f"unsupported target status: {target}")
        state = self._read()
        if model_id not in state["models"]:
            raise ModelRegistryError(f"unknown model: {model_id}")
        record = dict(state["models"][model_id])
        current = str(record["status"])
        if target not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ModelRegistryError(f"invalid transition {current} -> {target}")
        if target == "active":
            previous = state.get("active_model")
            if previous and previous != model_id:
                old = dict(state["models"][previous])
                old["status"] = "retired"
                old["updated_at"] = _now()
                state["models"][previous] = old
            state["active_model"] = model_id
        record["status"] = target
        record["updated_at"] = _now()
        state["models"][model_id] = record
        state["events"].append(
            {
                "at": record["updated_at"],
                "action": "transition",
                "model_id": model_id,
                "from": current,
                "to": target,
                "reason": reason,
            }
        )
        self._write(state)
        return record

    def rollback(self, model_id: str, *, reason: str = "rollback") -> dict[str, Any]:
        """Explicitly activate a known model; no implicit artifact selection."""

        record = self.get(model_id)
        if record["status"] not in {"retired", "candidate", "canary"}:
            raise ModelRegistryError("rollback target must be retired, candidate, or canary")
        return self.transition(model_id, "active", reason=reason)
