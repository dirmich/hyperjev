"""Dataset validation and reproducible training-plan generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import BooleanDecision, ChoiceDecision, ScoreDecision, validate_result_for_task
from .registry import TaskRegistry
from .samples import CanonicalSample, load_jsonl
from .student import StudentConfig, student_manifest


class TrainingDataError(ValueError):
    """Raised when a normalized dataset is not safe to use for training."""


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 7
    gradient_accumulation_steps: int = 1
    precision: str = "bf16"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise TrainingDataError("epochs, batch_size, and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise TrainingDataError("learning_rate must be positive and weight_decay must be non-negative")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise TrainingDataError(f"unsupported training precision: {self.precision}")


@dataclass(frozen=True)
class TrainingDataset:
    path: Path
    samples: tuple[CanonicalSample, ...]
    split_counts: dict[str, int]
    task_counts: dict[str, int]
    dataset_hash: str


def _read_records(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingDataError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise TrainingDataError(f"{path}:{line_number}: dataset record must be an object")
        sample_id = str(raw.get("sample_id", ""))
        if not sample_id:
            raise TrainingDataError(f"{path}:{line_number}: sample_id is required")
        if sample_id in records:
            raise TrainingDataError(f"duplicate sample_id: {sample_id}")
        records[sample_id] = raw
    if not records:
        raise TrainingDataError(f"dataset is empty: {path}")
    return records


def _validate_target(sample: CanonicalSample, registry: TaskRegistry) -> None:
    task = registry.get(sample.task_id, sample.task_version)
    if task.output_type == "boolean":
        if not isinstance(sample.target, bool):
            raise TrainingDataError(f"{sample.sample_id}: boolean target must be bool")
        result = BooleanDecision(value=sample.target, probability=1.0)
    elif task.output_type == "choice":
        candidates = tuple(str(candidate) for candidate in task.output.get("candidates", []))
        if not isinstance(sample.target, str) or sample.target not in candidates:
            raise TrainingDataError(f"{sample.sample_id}: target is not a registered choice")
        result = ChoiceDecision(
            selected=sample.target,
            probabilities={candidate: float(candidate == sample.target) for candidate in candidates},
        )
    else:
        try:
            target = float(sample.target)
        except (TypeError, ValueError) as exc:
            raise TrainingDataError(f"{sample.sample_id}: score target must be numeric") from exc
        result = ScoreDecision(value=target, interval_90=(target, target))
    try:
        validate_result_for_task(task, result)
    except ValueError as exc:
        raise TrainingDataError(f"{sample.sample_id}: invalid target: {exc}") from exc


def _validate_soft_target(raw: Any, sample: CanonicalSample, registry: TaskRegistry) -> None:
    if raw is None:
        return
    task = registry.get(sample.task_id, sample.task_version)
    if task.output_type == "boolean":
        try:
            probability = float(raw)
        except (TypeError, ValueError) as exc:
            raise TrainingDataError(f"{sample.sample_id}: boolean soft_target must be numeric") from exc
        if isinstance(raw, bool) or not 0.0 <= probability <= 1.0:
            raise TrainingDataError(f"{sample.sample_id}: boolean soft_target must be between 0 and 1")
    elif task.output_type == "choice":
        if not isinstance(raw, dict):
            raise TrainingDataError(f"{sample.sample_id}: choice soft_target must be an object")
        candidates = {str(candidate) for candidate in task.output.get("candidates", [])}
        if set(raw) != candidates:
            raise TrainingDataError(f"{sample.sample_id}: choice soft_target must cover all candidates")
        probabilities = [float(value) for value in raw.values()]
        if any(value < 0 or value > 1 for value in probabilities) or abs(sum(probabilities) - 1.0) > 0.02:
            raise TrainingDataError(f"{sample.sample_id}: choice soft_target probabilities are invalid")
    else:
        try:
            float(raw)
        except (TypeError, ValueError) as exc:
            raise TrainingDataError(f"{sample.sample_id}: score soft_target must be numeric") from exc


def load_training_dataset(path: str | Path, registry: TaskRegistry) -> TrainingDataset:
    """Validate normalized dataset records before a training backend consumes them."""

    source = Path(path)
    raw_records = _read_records(source)
    samples = load_jsonl(source, registry)
    if set(raw_records) != {sample.sample_id for sample in samples}:
        raise TrainingDataError("dataset record IDs do not match canonical samples")
    split_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for sample in samples:
        raw = raw_records[sample.sample_id]
        split = str(sample.provenance.get("split", ""))
        if split not in {"train", "validation", "test"}:
            raise TrainingDataError(f"{sample.sample_id}: invalid provenance.split")
        if bool(sample.provenance.get("privacy_raw_inputs_stored", False)):
            raise TrainingDataError(f"{sample.sample_id}: raw inputs are not allowed in training data")
        _validate_target(sample, registry)
        _validate_soft_target(raw.get("soft_target"), sample, registry)
        split_counts[split] += 1
        task_counts[sample.task_id] += 1
    return TrainingDataset(
        path=source.resolve(),
        samples=samples,
        split_counts=dict(sorted(split_counts.items())),
        task_counts=dict(sorted(task_counts.items())),
        dataset_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def build_training_plan(
    dataset_path: str | Path,
    registry: TaskRegistry,
    *,
    student: StudentConfig | None = None,
    training: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Build a manifest for a future training run without fabricating a checkpoint."""

    selected_student = student or StudentConfig()
    selected_training = training or TrainingConfig()
    selected_student.validate()
    selected_training.validate()
    dataset = load_training_dataset(dataset_path, registry)
    return {
        "record_type": "training_plan",
        "plan_version": "training-plan-v1",
        "status": "planned",
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.dataset_hash,
            "sample_count": len(dataset.samples),
            "split_counts": dataset.split_counts,
            "task_counts": dataset.task_counts,
        },
        "student": student_manifest(registry, selected_student),
        "training": asdict(selected_training),
        "runtime": {
            "torch_available": importlib.util.find_spec("torch") is not None,
            "cuda_runtime_probe": "deferred_to_training_image",
        },
        "checkpoint": {"status": "not_created", "path": None},
    }


def write_training_plan(
    dataset_path: str | Path,
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    student: StudentConfig | None = None,
    training: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Write a JSON training plan and return it."""

    plan = build_training_plan(
        dataset_path,
        registry,
        student=student,
        training=training,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan | {"output_path": str(output.resolve())}
