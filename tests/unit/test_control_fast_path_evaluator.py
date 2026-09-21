import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_control_fast_path import evaluate_fast_path, evaluate_runtime

from hyperjev.registry import TaskRegistry
from hyperjev.student import StudentConfig, build_torch_model, student_manifest


class ControlFastPathEvaluatorTests(unittest.TestCase):
    def test_report_separates_synthetic_and_human_targets(self) -> None:
        rows = [
            {
                "sample_id": "stop",
                "state": "obstacle is directly ahead",
                "target": "STOP",
                "labels": {"human": {"selected": "STOP"}},
            },
            {
                "sample_id": "hold",
                "state": "pause safely while the pose is stable and no hazard is present",
                "target": "HOLD",
                "labels": {"human": None},
            },
            {
                "sample_id": "ambiguous",
                "state": "target ahead",
                "target": "APPROACH",
                "labels": {"human": None},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            queue.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = evaluate_fast_path(queue)

        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["resolved_count"], 2)
        self.assertEqual(report["unresolved_count"], 1)
        self.assertEqual(report["synthetic_target"]["accuracy"], 1.0)
        self.assertEqual(report["human_target"]["labeled_count"], 1)
        self.assertFalse(report["human_label_gate"])
        self.assertFalse(report["production_ready"])
        self.assertEqual(report["source_counts"], {"control-rule": 1, "model": 1, "safety-rule": 1})
        self.assertIsNotNone(report["latency_us"]["p95"])

    def test_runtime_report_replays_rules_before_student(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        config = StudentConfig(
            model_id="fast-path-evaluator-test",
            backbone="reference-ngram-encoder",
            precision="fp32",
        )
        model = build_torch_model(registry, config)
        rows = [
            {
                "sample_id": "stop",
                "state": "obstacle is directly ahead",
                "target": "STOP",
                "domain": "simulation",
            },
            {
                "sample_id": "approach",
                "state": "the reachable target is locked directly ahead",
                "target": "APPROACH",
                "domain": "simulation",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.jsonl"
            checkpoint = root / "checkpoint.pt"
            queue.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            torch.save(
                {"student": student_manifest(registry, config), "model_state_dict": model.state_dict()},
                checkpoint,
            )
            report = evaluate_runtime(
                queue,
                checkpoint,
                ROOT / "registry" / "control_tasks",
                warmup_count=1,
            )

        self.assertEqual(report["source_counts"], {"control-rule": 1, "safety-rule": 1})
        self.assertEqual(report["warmup_count"], 1)
        self.assertFalse(report["cuda_synchronized"])
        self.assertEqual(report["synthetic_target"]["accuracy"], 1.0)
        self.assertEqual(report["synthetic_target"]["stop_recall"], 1.0)
        self.assertEqual(
            set(report["latency_by_source_us"]),
            {"control-rule", "safety-rule"},
        )
        self.assertFalse(report["production_ready"])


if __name__ == "__main__":
    unittest.main()
