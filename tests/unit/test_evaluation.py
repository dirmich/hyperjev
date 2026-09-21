import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.evaluation import evaluate_run, validate_golden_set
from hyperjev.registry import TaskRegistry

ROOT = Path(__file__).resolve().parents[2]


class EvaluationTests(unittest.TestCase):
    def test_golden_gate_reports_smoke_fixture_as_incomplete(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        report = validate_golden_set(ROOT / "tests/golden/phase0_smoke.jsonl", registry)
        self.assertFalse(report["ready"])
        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(report["missing_count"], 994)

    def test_evaluate_run_summarizes_schema_and_latency(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "run.jsonl"
            run_path.write_text(
                "\n".join(
                    [
                        json.dumps({"record_type": "manifest", "run_id": "test", "sample_count": 1}),
                        json.dumps(
                            {
                                "record_type": "sample",
                                "sample_id": "smoke-remember-001",
                                "provider": "qwen",
                                "status": "completed",
                                "schema_valid": True,
                                "latency_ms": 10,
                                "total_tokens": 12,
                                "normalized_result": {
                                    "type": "boolean",
                                    "value": True,
                                    "probability": 0.9,
                                    "abstained": False,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = evaluate_run(run_path, ROOT / "tests/golden/phase0_smoke.jsonl", registry)
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["providers"]["qwen"]["schema_valid_rate"], 1.0)
        self.assertEqual(report["providers"]["qwen"]["latency_ms"]["p95"], 10.0)
        self.assertEqual(report["providers"]["qwen"]["quality"]["memory.remember_worthy"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
