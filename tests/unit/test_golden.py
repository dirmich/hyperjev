import tempfile
import unittest
from pathlib import Path

from hyperjev.golden import generate_review_queue
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


if __name__ == "__main__":
    unittest.main()
