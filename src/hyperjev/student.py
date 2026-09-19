"""Encoder + typed decision-head contract for the HyperJev Student."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .registry import TaskDefinition, TaskRegistry


class StudentDependencyError(RuntimeError):
    """Raised when the optional PyTorch Student backend is unavailable."""


@dataclass(frozen=True)
class HeadSpec:
    task_id: str
    task_version: int
    output_type: str
    candidates: tuple[str, ...] = ()

    @classmethod
    def from_task(cls, task: TaskDefinition) -> HeadSpec:
        return cls(
            task_id=task.id,
            task_version=task.version,
            output_type=task.output_type,
            candidates=tuple(str(candidate) for candidate in task.output.get("candidates", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "task_version": self.task_version,
            "output_type": self.output_type,
        }
        if self.candidates:
            result["candidates"] = list(self.candidates)
        return result


@dataclass(frozen=True)
class StudentConfig:
    model_id: str = "hyperjev-student-dev"
    backbone: str = "small-multilingual-encoder"
    hidden_size: int = 256
    vocab_size: int = 32768
    max_sequence_length: int = 1024
    precision: str = "bf16"

    def validate(self) -> None:
        if not self.model_id.strip() or not self.backbone.strip():
            raise ValueError("model_id and backbone must not be empty")
        if self.hidden_size < 8 or self.vocab_size < 128 or self.max_sequence_length < 1:
            raise ValueError("student dimensions are invalid")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"unsupported precision: {self.precision}")


def head_specs(registry: TaskRegistry) -> tuple[HeadSpec, ...]:
    """Build the typed-head inventory from the versioned task registry."""

    return tuple(HeadSpec.from_task(registry.get(task_id)) for task_id in registry.ids())


def student_manifest(
    registry: TaskRegistry,
    config: StudentConfig | None = None,
) -> dict[str, Any]:
    selected = config or StudentConfig()
    selected.validate()
    specs = head_specs(registry)
    return {
        "model_id": selected.model_id,
        "track": "encoder-typed-heads",
        "diffusion_track": "HyperJev-D",
        "backbone": selected.backbone,
        "hidden_size": selected.hidden_size,
        "vocab_size": selected.vocab_size,
        "max_sequence_length": selected.max_sequence_length,
        "precision": selected.precision,
        "heads": [spec.to_dict() for spec in specs],
    }


def build_torch_model(registry: TaskRegistry, config: StudentConfig | None = None) -> Any:
    """Build a minimal shared encoder with registry-derived typed heads.

    PyTorch is deliberately optional in the repository environment. The DGX
    Spark training image can install it and call this function; contract and
    registry tests do not need to import a GPU framework.
    """

    try:
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise StudentDependencyError("PyTorch is required to build the Student backend") from exc

    selected = config or StudentConfig()
    selected.validate()
    specs = head_specs(registry)

    class HyperJevTypedHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(selected.vocab_size, selected.hidden_size)
            self.encoder = nn.Sequential(
                nn.LayerNorm(selected.hidden_size),
                nn.Linear(selected.hidden_size, selected.hidden_size),
                nn.GELU(),
                nn.LayerNorm(selected.hidden_size),
            )
            self.boolean_heads = nn.ModuleDict(
                {
                    spec.task_id: nn.Linear(selected.hidden_size, 2)
                    for spec in specs
                    if spec.output_type == "boolean"
                }
            )
            self.choice_heads = nn.ModuleDict(
                {
                    spec.task_id: nn.Linear(selected.hidden_size, len(spec.candidates))
                    for spec in specs
                    if spec.output_type == "choice"
                }
            )
            self.score_heads = nn.ModuleDict(
                {
                    spec.task_id: nn.Linear(selected.hidden_size, 2)
                    for spec in specs
                    if spec.output_type == "score"
                }
            )

        def encode(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            embedded = self.embedding(input_ids)
            if attention_mask is None:
                return self.encoder(embedded.mean(dim=1))
            mask = attention_mask.to(dtype=embedded.dtype).unsqueeze(-1)
            pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return self.encoder(pooled)

        def forward(
            self,
            task_id: str,
            input_ids: Any,
            attention_mask: Any | None = None,
        ) -> Mapping[str, Any]:
            hidden = self.encode(input_ids, attention_mask)
            if task_id in self.boolean_heads:
                return {"type": "boolean", "logits": self.boolean_heads[task_id](hidden)}
            if task_id in self.choice_heads:
                return {"type": "choice", "logits": self.choice_heads[task_id](hidden)}
            if task_id in self.score_heads:
                return {"type": "score", "parameters": self.score_heads[task_id](hidden)}
            raise KeyError(f"unknown task head: {task_id}")

    return HyperJevTypedHeads()
