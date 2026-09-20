import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperjev.config import load_config
from hyperjev.control_data import generate_control_review_queue
from hyperjev.golden import generate_review_queue
from hyperjev.golden_draft import (
    _repair_choice_probabilities,
    adjudicate_teacher_drafts,
    generate_gemma_draft,
    generate_teacher_draft,
)
from hyperjev.registry import TaskRegistry
from hyperjev.teachers import TeacherCompletion

ROOT = Path(__file__).resolve().parents[2]


class _FakeGemma:
    def __init__(self, _settings, timeout_s=None) -> None:
        self.timeout_s = timeout_s

    def complete(self, _messages, *, max_tokens=256):
        return TeacherCompletion(
            content='{"type":"boolean","value":true,"probability":0.99,"abstained":false}',
            model="fake-gemma-4",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            elapsed_ms=12.5,
        )


class _FakeQwen(_FakeGemma):
    def complete(self, messages, *, max_tokens=256):
        self.messages = messages
        return TeacherCompletion(
            content='{"type":"boolean","value":true,"probability":0.98,"abstained":false}',
            model="fake-qwen38fn",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            elapsed_ms=8.5,
        )


class _RetryInvalidQwen(_FakeQwen):
    calls = 0

    def complete(self, messages, *, max_tokens=256):
        type(self).calls += 1
        if type(self).calls == 1:
            return TeacherCompletion(
                content='{"type":"boolean","value":true,"probability":1.5,"abstained":false}',
                model="fake-qwen38fn",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                elapsed_ms=8.5,
            )
        return super().complete(messages, max_tokens=max_tokens)


class GoldenDraftTests(unittest.TestCase):
    def test_choice_probability_repair_is_explicit_and_bounded(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        task = registry.get("control.skill", 1)
        raw = {
            "type": "choice",
            "selected": "STOP",
            "probabilities": {
                "STOP": 0.85,
                "HOLD": 0.05,
                "MOVE": 0.0,
                "ROTATE": 0.0,
                "APPROACH": 0.0,
                "RETREAT": 0.05,
                "INTERACT": 0.0,
                "RECOVER": 0.0,
            },
            "abstained": False,
        }
        repaired = _repair_choice_probabilities(raw, task)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertAlmostEqual(sum(repaired["probabilities"].values()), 1.0, places=6)
        self.assertIsNone(_repair_choice_probabilities({**raw, "probabilities": {"STOP": 3.0}}, task))

    def test_gemma_draft_is_queue_bound_and_does_not_store_raw_state(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            output = Path(directory) / "draft.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            with patch("hyperjev.golden_draft.TeacherClient", _FakeGemma):
                report = generate_gemma_draft(config, registry, queue, output, limit=1)
            output_text = output.read_text(encoding="utf-8")
            lines = output_text.splitlines()

        manifest = json.loads(lines[0])
        record = json.loads(lines[1])
        self.assertEqual(report["manifest"]["schema_valid_count"], 1)
        self.assertEqual(manifest["queue_sha256"], report["manifest"]["queue_sha256"])
        self.assertEqual(record["normalized_result"]["type"], "boolean")
        self.assertNotIn("다음 분기부터", output_text)
        self.assertNotIn('"target"', output_text)

    def test_qwen_draft_uses_selected_provider_and_prompt(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            output = Path(directory) / "draft.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            with patch("hyperjev.golden_draft.TeacherClient", _FakeQwen):
                report = generate_teacher_draft(
                    config, registry, queue, output, provider="qwen", limit=1, timeout_s=17
                )

        record = report["records"][0]
        self.assertEqual(report["manifest"]["provider"], "qwen")
        self.assertEqual(report["manifest"]["model"], "qwen38fn")
        self.assertEqual(record["provider"], "qwen")
        self.assertEqual(record["model"], "fake-qwen38fn")

    def test_adjudication_keeps_disagreement_for_human_review(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.jsonl"
            qwen = root / "qwen.jsonl"
            gemma = root / "gemma.jsonl"
            output = root / "adjudicated.jsonl"
            generate_control_review_queue(queue, registry, count_per_skill=1, seed=7)
            samples = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
            queue_hash = __import__("hashlib").sha256(queue.read_bytes()).hexdigest()
            candidates = ["STOP", "HOLD", "MOVE", "ROTATE", "APPROACH", "RETREAT", "INTERACT", "RECOVER"]

            def draft(path: Path, *, disagree: bool) -> None:
                records = [
                    {
                        "record_type": "golden_teacher_draft_manifest",
                        "queue_sha256": queue_hash,
                        "provider": path.stem,
                        "model": path.stem,
                    }
                ]
                for index, sample in enumerate(samples):
                    selected = sample["target"]
                    if disagree and index == 0:
                        selected = next(candidate for candidate in candidates if candidate != selected)
                    records.append(
                        {
                            "record_type": "golden_teacher_draft",
                            "sample_id": sample["sample_id"],
                            "model": path.stem,
                            "schema_valid": True,
                            "status": "completed",
                            "normalized_result": {
                                "type": "choice",
                                "selected": selected,
                                "probabilities": {
                                    candidate: float(candidate == selected) for candidate in candidates
                                },
                                "abstained": False,
                            },
                        }
                    )
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            draft(qwen, disagree=False)
            draft(gemma, disagree=True)
            report = adjudicate_teacher_drafts(config, registry, queue, qwen, gemma, output)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["manifest"]["agreement_count"], 7)
        self.assertEqual(report["manifest"]["disagreement_count"], 1)
        self.assertEqual(records[1]["status"], "disagreement")
        self.assertIsNone(records[1]["normalized_result"])
        self.assertIn("qwen", records[1]["teacher_comparison"])

    def test_teacher_draft_resume_skips_completed_records(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)

        class _CountingQwen(_FakeQwen):
            calls = 0

            def complete(self, messages, *, max_tokens=256):
                type(self).calls += 1
                return super().complete(messages, max_tokens=max_tokens)

        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            output = Path(directory) / "draft.jsonl"
            generate_review_queue(queue, registry, count=2, seed=7)
            with patch("hyperjev.golden_draft.TeacherClient", _CountingQwen):
                generate_teacher_draft(config, registry, queue, output, provider="qwen", limit=1)
                report = generate_teacher_draft(
                    config, registry, queue, output, provider="qwen", resume=True
                )
            records = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(_CountingQwen.calls, 2)
        self.assertEqual(report["manifest"]["sample_count"], 2)
        self.assertEqual(len(records), 3)

    def test_teacher_draft_resume_retries_invalid_records(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        _RetryInvalidQwen.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            output = Path(directory) / "draft.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            with patch("hyperjev.golden_draft.TeacherClient", _RetryInvalidQwen):
                first = generate_teacher_draft(config, registry, queue, output, provider="qwen")
                report = generate_teacher_draft(
                    config, registry, queue, output, provider="qwen", resume=True
                )

        self.assertEqual(first["manifest"]["schema_valid_count"], 0)
        self.assertEqual(_RetryInvalidQwen.calls, 2)
        self.assertEqual(report["manifest"]["schema_valid_count"], 1)


if __name__ == "__main__":
    unittest.main()
