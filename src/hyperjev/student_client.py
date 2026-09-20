"""In-process Student provider for the guarded router path."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import Phase0Config
from .registry import TaskDefinition, TaskRegistry
from .samples import CanonicalSample
from .student import StudentDependencyError, build_torch_model
from .student_inference import _load_torch, _student_config
from .teachers import TeacherCompletion
from .training import _encode_reference_sample

_QUESTION_TEXT = {
    "memory.remember_worthy": "Is this worth long-term memory?",
    "memory.type": "What type of memory is this?",
    "memory.importance": "Score the importance of this memory from 0 to 1.",
    "query.route": "What retrieval route should be used?",
    "memory.relation": "What is the relation between these memories?",
    "wiki.semantic_change": "Did the factual meaning change?",
}


class StudentClient:
    """Load one checkpoint and expose a router-compatible typed decision call."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        registry: TaskRegistry,
        *,
        minimum_confidence: float = 0.95,
        allow_score: bool = False,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if device not in {"cpu", "cuda"}:
            raise ValueError("Student device must be cpu or cuda")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Student checkpoint not found: {checkpoint}")
        torch = _load_torch()
        if device == "cuda" and not torch.cuda.is_available():
            raise StudentDependencyError("CUDA was requested for Student but is unavailable")
        raw = torch.load(checkpoint, map_location=device, weights_only=False)
        if not isinstance(raw, dict) or not isinstance(raw.get("student"), dict):
            raise TypeError("Student checkpoint does not contain a student manifest")
        config = _student_config(raw["student"])
        model = build_torch_model(registry, config).to(device)
        model.load_state_dict(raw["model_state_dict"])
        model.eval()
        self.checkpoint_path = checkpoint.resolve()
        self.registry = registry
        self.model = model
        self.config = config
        self.device = device
        self.minimum_confidence = minimum_confidence
        self.allow_score = allow_score
        self.model_name = config.model_id
        self._torch = torch

    def _sample(self, task: TaskDefinition, state: str, question: str) -> CanonicalSample:
        question_text = question if (" " in question or len(question) > 20) else _QUESTION_TEXT.get(
            task.id, question
        )
        return CanonicalSample(
            sample_id="router-request",
            task_id=task.id,
            task_version=task.version,
            state=state,
            question=question_text,
            target=None,
            language="en",
            domain="serving",
            source={},
            labels={},
            provenance={"prompt_version": 1, "split": "inference", "privacy_raw_inputs_stored": False},
        )

    def complete_decision(
        self,
        task: TaskDefinition,
        *,
        state: str,
        question: str,
        candidates: list[str],
    ) -> TeacherCompletion:
        started = time.perf_counter()
        sample = self._sample(task, state, question)
        token_ids, attention = _encode_reference_sample(
            sample,
            vocab_size=self.config.vocab_size,
            max_length=self.config.max_sequence_length,
            pad_to_max=False,
        )
        input_ids = self._torch.tensor([token_ids], dtype=self._torch.long, device=self.device)
        attention_mask = self._torch.tensor([attention], dtype=self._torch.long, device=self.device)
        with self._torch.inference_mode():
            output = self.model(task.id, input_ids, attention_mask)
        result: dict[str, Any]
        if task.output_type == "boolean":
            probabilities = self._torch.softmax(output["logits"], dim=-1)[0].detach().cpu().tolist()
            index = int(self._torch.argmax(output["logits"], dim=-1)[0].item())
            confidence = float(probabilities[index])
            result = {
                "type": "boolean",
                "value": bool(index),
                "probability": confidence,
                "abstained": confidence < self.minimum_confidence,
            }
        elif task.output_type == "choice":
            registered_candidates = [str(item) for item in task.output.get("candidates", [])]
            selected_candidates = (
                candidates
                if candidates and candidates == registered_candidates
                else registered_candidates
            )
            probabilities = self._torch.softmax(output["logits"], dim=-1)[0].detach().cpu().tolist()
            index = int(self._torch.argmax(output["logits"], dim=-1)[0].item())
            confidence = float(probabilities[index])
            result = {
                "type": "choice",
                "selected": selected_candidates[index],
                "probabilities": {
                    candidate: float(probability)
                    for candidate, probability in zip(selected_candidates, probabilities)
                },
                "abstained": confidence < self.minimum_confidence,
            }
        else:
            raw_value = float(output["parameters"][0, 0].detach().cpu().item())
            value = min(1.0, max(0.0, raw_value))
            width = float(self._torch.sigmoid(output["parameters"][0, 1]).detach().cpu().item()) * 0.5
            confidence = max(0.0, 1.0 - width)
            result = {
                "type": "score",
                "value": value,
                "interval_90": [max(0.0, value - width), min(1.0, value + width)],
                "abstained": not self.allow_score or confidence < self.minimum_confidence,
            }
        return TeacherCompletion(
            content=json.dumps(result, ensure_ascii=False),
            model=self.model_name,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def student_client_from_config(config: Phase0Config, registry: TaskRegistry) -> StudentClient | None:
    """Build the optional Student provider when a checkpoint is configured."""

    if config.student_checkpoint is None:
        return None
    return StudentClient(
        config.student_checkpoint,
        registry,
        minimum_confidence=config.student_minimum_confidence,
        device=config.student_device,
    )
