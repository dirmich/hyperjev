import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.control_data import generate_control_review_queue
from hyperjev.registry import TaskRegistry

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "evaluate_control_teacher_draft.py"
SPEC = importlib.util.spec_from_file_location("evaluate_control_teacher_draft", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControlTeacherDraftTests(unittest.TestCase):
    def test_draft_quality_keeps_synthetic_and_human_gates_separate(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            draft = Path(directory) / "draft.jsonl"
            generate_control_review_queue(queue, registry, count_per_skill=1, seed=7)
            samples = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
            queue_hash = hashlib.sha256(queue.read_bytes()).hexdigest()
            records = [
                {
                    "record_type": "golden_teacher_draft_manifest",
                    "queue_sha256": queue_hash,
                    "provider": "qwen",
                    "model": "qwen38fn",
                }
            ]
            candidates = registry.get("control.skill", 1).output["candidates"]
            for sample in samples:
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "status": "completed",
                        "schema_valid": True,
                        "latency_ms": 10.0,
                        "normalized_result": {
                            "type": "choice",
                            "selected": sample["target"],
                            "probabilities": {
                                candidate: float(candidate == sample["target"])
                                for candidate in candidates
                            },
                        },
                    }
                )
            draft.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            report = MODULE.evaluate_control_teacher_draft(queue, draft)

        self.assertEqual(report["synthetic_target_accuracy_overall"], 1.0)
        self.assertEqual(report["processed_sample_count"], 8)
        self.assertEqual(report["synthetic_target_accuracy_on_processed"], 1.0)
        self.assertFalse(report["human_label_gate"])
        self.assertFalse(report["production_ready"])

    def test_partial_draft_does_not_report_full_queue_accuracy(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            draft = Path(directory) / "partial-draft.jsonl"
            generate_control_review_queue(queue, registry, count_per_skill=1, seed=7)
            samples = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
            queue_hash = hashlib.sha256(queue.read_bytes()).hexdigest()
            sample = samples[0]
            candidates = registry.get("control.skill", 1).output["candidates"]
            records = [
                {
                    "record_type": "golden_teacher_draft_manifest",
                    "queue_sha256": queue_hash,
                    "provider": "qwen",
                    "model": "qwen38fn",
                },
                {
                    "sample_id": sample["sample_id"],
                    "status": "completed",
                    "schema_valid": True,
                    "latency_ms": 10.0,
                    "normalized_result": {
                        "type": "choice",
                        "selected": sample["target"],
                        "probabilities": {
                            candidate: float(candidate == sample["target"])
                            for candidate in candidates
                        },
                    },
                },
            ]
            draft.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            report = MODULE.evaluate_control_teacher_draft(queue, draft)

        self.assertEqual(report["sample_count"], 8)
        self.assertEqual(report["processed_sample_count"], 1)
        self.assertIsNone(report["synthetic_target_accuracy_overall"])
        self.assertEqual(report["synthetic_target_accuracy_on_processed"], 1.0)
        self.assertFalse(report["complete_and_valid"])


if __name__ == "__main__":
    unittest.main()
