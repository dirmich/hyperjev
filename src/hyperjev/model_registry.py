"""Local model manifest registry with explicit promotion and rollback gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
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


def build_model_manifest(
    training_plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    git_commit: str,
    runtime: str = "pytorch",
    status: str = "trained",
    model_id: str | None = None,
) -> dict[str, Any]:
    """Build a registry manifest from validated training and calibration artifacts."""

    if not isinstance(training_plan, Mapping) or not isinstance(calibration, Mapping):
        raise ModelRegistryError("training plan and calibration must be objects")
    if training_plan.get("record_type") != "training_plan":
        raise ModelRegistryError("training plan must have record_type=training_plan")
    if calibration.get("record_type") != "calibration_manifest":
        raise ModelRegistryError("calibration must have record_type=calibration_manifest")
    if not git_commit.strip():
        raise ModelRegistryError("git_commit must not be empty")
    if not runtime.strip():
        raise ModelRegistryError("runtime must not be empty")
    if status not in STATUSES:
        raise ModelRegistryError(f"unsupported manifest status: {status}")
    dataset = training_plan.get("dataset")
    student = training_plan.get("student")
    training = training_plan.get("training")
    if not isinstance(dataset, Mapping) or not isinstance(student, Mapping) or not isinstance(training, Mapping):
        raise ModelRegistryError("training plan must contain dataset and student objects")
    dataset_hash = str(dataset.get("sha256", ""))
    if len(dataset_hash) != 64 or any(character not in "0123456789abcdef" for character in dataset_hash.lower()):
        raise ModelRegistryError("training plan dataset sha256 is invalid")
    heads = student.get("heads")
    if not dataset_hash or not isinstance(heads, list) or not heads:
        raise ModelRegistryError("training plan dataset hash and student heads are required")
    tasks: dict[str, list[int]] = {}
    for head in heads:
        if not isinstance(head, Mapping):
            raise ModelRegistryError("student heads must contain objects")
        task_id = str(head.get("task_id", ""))
        try:
            version = int(head.get("task_version", 0))
        except (TypeError, ValueError) as exc:
            raise ModelRegistryError("student head task_version must be an integer") from exc
        if not task_id or version < 1:
            raise ModelRegistryError("student head task_id and task_version are required")
        if task_id in tasks:
            raise ModelRegistryError(f"duplicate student head task_id: {task_id}")
        tasks[task_id] = [version]
    calibration_version = str(calibration.get("calibration_version", ""))
    if not calibration_version:
        raise ModelRegistryError("calibration_version is required")
    precision = str(training.get("precision", ""))
    if not precision:
        raise ModelRegistryError("training precision is required")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise ModelRegistryError(f"checkpoint not found: {checkpoint}")
    try:
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    except OSError as exc:
        raise ModelRegistryError(f"cannot read checkpoint: {checkpoint}") from exc
    selected_model_id = model_id or str(student.get("model_id", ""))
    base_model = str(student.get("backbone", ""))
    if not selected_model_id or not base_model:
        raise ModelRegistryError("model_id is required")
    return {
        "model_id": selected_model_id,
        "base_model": base_model,
        "dataset_hash": dataset_hash,
        "git_commit": git_commit,
        "tasks": tasks,
        "calibration_version": calibration_version,
        "runtime": runtime,
        "precision": precision,
        "status": status,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
    }


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

    @staticmethod
    def validate_control_quality_report(
        record: Mapping[str, Any],
        quality_report: Mapping[str, Any],
    ) -> None:
        """Require a passed, hash-bound human-gated control quality report."""

        if not isinstance(quality_report, Mapping):
            raise ModelRegistryError("quality report must be an object")
        if quality_report.get("record_type") != "control_quality_gate":
            raise ModelRegistryError("quality report must have record_type=control_quality_gate")
        if quality_report.get("passed") is not True:
            raise ModelRegistryError("quality report has not passed all quality gates")
        if quality_report.get("production_ready") is not True:
            raise ModelRegistryError("quality report is not production_ready")
        if quality_report.get("human_label_gate") is not True:
            raise ModelRegistryError("quality report is missing the full human-label gate")
        if quality_report.get("human_test_gate") is not True:
            raise ModelRegistryError("quality report is missing the human test gate")
        if quality_report.get("checkpoint_sha256") != record.get("checkpoint_sha256"):
            raise ModelRegistryError("quality report checkpoint hash does not match the model")
        if quality_report.get("dataset_sha256") != record.get("dataset_hash"):
            raise ModelRegistryError("quality report dataset hash does not match the model")
        safety = quality_report.get("safety")
        if not isinstance(safety, Mapping):
            raise ModelRegistryError("quality report safety section is missing")
        try:
            safe_stop_recall = float(safety.get("safe_stop_recall", 0.0))
        except (TypeError, ValueError) as exc:
            raise ModelRegistryError("quality report safe STOP recall is invalid") from exc
        if not math.isfinite(safe_stop_recall) or not 0.0 <= safe_stop_recall <= 1.0 or safe_stop_recall < 1.0:
            raise ModelRegistryError("quality report safe STOP recall is below 100%")
        confidence_interval = safety.get("safe_stop_recall_ci95")
        if not isinstance(confidence_interval, Mapping):
            raise ModelRegistryError("quality report safe STOP confidence interval is missing")
        lower_bound = confidence_interval.get("lower")
        try:
            lower_bound_value = float(lower_bound)
        except (TypeError, ValueError) as exc:
            raise ModelRegistryError("quality report safe STOP Wilson lower bound is invalid") from exc
        if not math.isfinite(lower_bound_value) or not 0.0 <= lower_bound_value <= 1.0 or lower_bound_value < 0.99:
            raise ModelRegistryError("quality report safe STOP Wilson lower bound is below 99%")

    def rollback(self, model_id: str, *, reason: str = "rollback") -> dict[str, Any]:
        """Explicitly activate a known model; no implicit artifact selection."""

        record = self.get(model_id)
        if record["status"] not in {"retired", "candidate", "canary"}:
            raise ModelRegistryError("rollback target must be retired, candidate, or canary")
        return self.transition(model_id, "active", reason=reason)
