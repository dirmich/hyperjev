import tempfile
import unittest
from pathlib import Path

from hyperjev.golden import append_golden_feedback, apply_golden_feedback, generate_review_queue
from hyperjev.registry import TaskRegistry
from hyperjev.samples import load_jsonl

ROOT = Path(__file__).resolve().parents[2]


class GoldenQueueTests(unittest.TestCase):
    def test_generation_is_deterministic_and_valid(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            report_first = generate_review_queue(first, registry, count=24, seed=7)
            report_second = generate_review_queue(second, registry, count=24, seed=7)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(report_first["sha256"], report_second["sha256"])
            self.assertEqual(report_first["review_status"], "pending")
            samples = load_jsonl(first, registry)
        self.assertEqual(len(samples), 24)
        self.assertTrue(all(sample.labels["human"] is None for sample in samples))
        self.assertEqual({sample.task_id for sample in samples}, set(report_first["task_counts"]))

    def test_non_positive_count_is_rejected(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with self.assertRaises(ValueError):
            generate_review_queue("/tmp/invalid-hyperjev-queue.jsonl", registry, count=0)

    def test_feedback_is_append_only_and_apply_creates_reviewed_copy(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            feedback = Path(directory) / "feedback.jsonl"
            reviewed = Path(directory) / "reviewed.jsonl"
            generate_review_queue(queue, registry, count=6, seed=7)
            original = queue.read_bytes()
            report = append_golden_feedback(
                queue,
                feedback,
                registry,
                sample_id="phase0-synthetic-0001",
                correction={
                    "type": "boolean",
                    "value": False,
                    "probability": 1.0,
                    "abstained": False,
                },
                reviewer="reviewer@example.test",
                reason="checked against source policy",
            )
            applied = apply_golden_feedback(queue, feedback, reviewed, registry)
            samples = load_jsonl(reviewed, registry)
            source_samples = load_jsonl(queue, registry)
            source_unchanged = queue.read_bytes() == original

        self.assertTrue(source_unchanged)
        self.assertEqual(report["sample_id"], "phase0-synthetic-0001")
        self.assertEqual(applied["human_reviewed_count"], 1)
        self.assertEqual(applied["pending_count"], 5)
        reviewed_sample = samples[0]
        self.assertEqual(reviewed_sample.labels["human"]["value"], False)
        self.assertEqual(reviewed_sample.labels["human"]["type"], "boolean")
        self.assertEqual(reviewed_sample.labels["human"]["probability"], 1.0)
        self.assertEqual(reviewed_sample.labels["human"]["abstained"], False)
        self.assertEqual(source_samples[0].labels["human"], None)

    def test_feedback_rejects_a_result_for_the_wrong_task(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            feedback = Path(directory) / "feedback.jsonl"
            generate_review_queue(queue, registry, count=6, seed=7)
            with self.assertRaises(ValueError):
                append_golden_feedback(
                    queue,
                    feedback,
                    registry,
                    sample_id="phase0-synthetic-0001",
                    correction={
                        "type": "choice",
                        "selected": "fact",
                        "probabilities": {"fact": 1.0},
                        "abstained": False,
                    },
                    reviewer="reviewer@example.test",
                )


if __name__ == "__main__":
    unittest.main()
