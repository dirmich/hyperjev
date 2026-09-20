"""Dataset validation and reproducible training-plan generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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


class TrainingDependencyError(RuntimeError):
    """Raised when the optional PyTorch training runtime is unavailable."""


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


def _encode_reference_sample(
    sample: CanonicalSample,
    *,
    vocab_size: int,
    max_length: int,
    pad_to_max: bool = True,
    backbone: str = "reference-byte-encoder",
) -> tuple[list[int], list[int]]:
    """Encode text deterministically for the dependency-light reference trainer."""

    if vocab_size <= 2:
        raise TrainingDataError("vocab_size must be greater than 2")
    text = f"{sample.state}\n{sample.question}"
    if backbone == "reference-token-encoder":
        words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)[:max_length]
        token_ids = [
            2
            + (
                int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "big")
                % (vocab_size - 2)
            )
            for word in words
        ]
    else:
        raw = text.encode()[:max_length]
        token_ids = [2 + (byte % (vocab_size - 2)) for byte in raw]
    attention = [1] * len(token_ids)
    if pad_to_max:
        token_ids.extend([0] * (max_length - len(token_ids)))
        attention.extend([0] * (max_length - len(attention)))
    return token_ids, attention


def run_reference_training(
    dataset_path: str | Path,
    output_path: str | Path,
    registry: TaskRegistry,
    *,
    student: StudentConfig | None = None,
    training: TrainingConfig | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Train the small registry-derived reference Student when PyTorch is installed.

    This deliberately uses a deterministic byte-hash encoder and is a contract
    smoke trainer, not the production multilingual backbone from the PRD.
    """

    selected_student = student or StudentConfig()
    selected_training = training or TrainingConfig()
    selected_student.validate()
    selected_training.validate()
    dataset = load_training_dataset(dataset_path, registry)
    train_samples = tuple(
        sample
        for sample in dataset.samples
        if sample.provenance.get("split") == "train"
    )
    if not train_samples:
        raise TrainingDataError("training dataset has no train split")
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise TrainingDependencyError("PyTorch is required for train run") from exc

    if device not in {"auto", "cpu", "cuda"}:
        raise TrainingDataError("device must be one of auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise TrainingDependencyError("CUDA was requested but is unavailable")
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    torch.manual_seed(selected_training.seed)
    from .student import build_torch_model

    model = build_torch_model(registry, selected_student).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=selected_training.learning_rate,
        weight_decay=selected_training.weight_decay,
    )
    losses: list[float] = []
    model.train()
    for _epoch in range(selected_training.epochs):
        epoch_loss_total = 0.0
        epoch_sample_count = 0
        generator = torch.Generator(device="cpu").manual_seed(selected_training.seed + _epoch)
        order = torch.randperm(len(train_samples), generator=generator).tolist()
        task_batches: dict[str, list[CanonicalSample]] = {}
        task_versions: dict[str, int] = {}
        for index in order:
            sample = train_samples[index]
            task_batches.setdefault(sample.task_id, []).append(sample)
            task_versions[sample.task_id] = sample.task_version
        for task_id, task_samples in task_batches.items():
            task = registry.get(task_id, task_versions[task_id])
            for start in range(0, len(task_samples), selected_training.batch_size):
                batch = task_samples[start : start + selected_training.batch_size]
                encoded = [
                    _encode_reference_sample(
                        sample,
                        vocab_size=selected_student.vocab_size,
                        max_length=selected_student.max_sequence_length,
                        backbone=selected_student.backbone,
                    )
                    for sample in batch
                ]
                input_ids = torch.tensor(
                    [tokens for tokens, _attention in encoded],
                    dtype=torch.long,
                    device=selected_device,
                )
                attention_mask = torch.tensor(
                    [attention for _tokens, attention in encoded],
                    dtype=torch.long,
                    device=selected_device,
                )
                output = model(task_id, input_ids, attention_mask)
                if task.output_type == "boolean":
                    target = torch.tensor(
                        [int(sample.target) for sample in batch],
                        dtype=torch.long,
                        device=selected_device,
                    )
                    loss = nn.functional.cross_entropy(output["logits"], target)
                elif task.output_type == "choice":
                    candidates = [str(candidate) for candidate in task.output.get("candidates", [])]
                    target = torch.tensor(
                        [candidates.index(str(sample.target)) for sample in batch],
                        dtype=torch.long,
                        device=selected_device,
                    )
                    loss = nn.functional.cross_entropy(output["logits"], target)
                else:
                    target = torch.tensor(
                        [float(sample.target) for sample in batch],
                        dtype=torch.float32,
                        device=selected_device,
                    )
                    loss = nn.functional.mse_loss(output["parameters"][:, 0], target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_size = len(batch)
                epoch_loss_total += float(loss.detach().cpu().item()) * batch_size
                epoch_sample_count += batch_size
        losses.append(epoch_loss_total / epoch_sample_count)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "artifact_kind": "reference_student_checkpoint",
        "student": student_manifest(registry, selected_student),
        "training": asdict(selected_training),
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.dataset_hash,
            "sample_count": len(dataset.samples),
        },
        "runtime": {"device": str(selected_device), "torch_version": torch.__version__},
        "epochs_completed": selected_training.epochs,
        "mean_train_loss": losses,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, output)
    return {
        "record_type": "training_run",
        "status": "completed",
        "artifact_kind": "reference_student_checkpoint",
        "output_path": str(output.resolve()),
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "dataset_sha256": dataset.dataset_hash,
        "sample_count": len(dataset.samples),
        "train_count": len(train_samples),
        "validation_count": dataset.split_counts.get("validation", 0),
        "device": str(selected_device),
        "epochs_completed": selected_training.epochs,
        "mean_train_loss": losses,
    }
