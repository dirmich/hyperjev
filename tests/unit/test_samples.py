import unittest
from pathlib import Path

from hyperjev.registry import TaskRegistry
from hyperjev.samples import SampleError, load_jsonl

ROOT = Path(__file__).resolve().parents[2]


class SampleTests(unittest.TestCase):
    def test_phase0_smoke_samples_load(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        samples = load_jsonl(ROOT / "tests" / "golden" / "phase0_smoke.jsonl", registry)
        self.assertEqual(len(samples), 6)
        self.assertEqual(len({sample.sample_id for sample in samples}), 6)

    def test_missing_provenance_is_rejected(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        sample = {
            "sample_id": "sample-1",
            "task_id": "memory.remember_worthy",
            "task_version": 1,
            "state": "x",
            "question": "y",
            "target": True,
            "language": "ko",
            "domain": "conversation",
            "source": {"kind": "synthetic"},
            "labels": {"human": None},
        }
        with self.assertRaises(SampleError):
            from hyperjev.samples import CanonicalSample

            CanonicalSample.from_dict(sample, registry)


if __name__ == "__main__":
    unittest.main()
