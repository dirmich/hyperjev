import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_control_quality import (
    _annotate_split_report,
    _binomial_interval,
    _human_label_status,
    _load_scenarios,
    _quality_gate_failures,
    _skill_report,
)

from hyperjev.registry import TaskRegistry


class ControlQualityEvaluatorTests(unittest.TestCase):
    def test_scenario_loader_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.jsonl"
            path.write_text(
                '{"scenario_id":"same","observation":{},"now_ms":0}\n'
                '{"scenario_id":"same","observation":{},"now_ms":0}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "scenario_id values must be unique"):
                _load_scenarios(path)

    def test_skill_report_exposes_accepted_metrics_and_intervals(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        report = _skill_report(
            {
                "predictions": [
                    {
                        "target": "STOP",
                        "prediction": {"selected": "STOP"},
                        "correct": True,
                        "accepted": True,
                    },
                    {
                        "target": "STOP",
                        "prediction": {"selected": "HOLD"},
                        "correct": False,
                        "accepted": False,
                    },
                ]
            },
            registry,
        )
        stop = report["metrics"]["STOP"]
        self.assertEqual(stop["count"], 2)
        self.assertEqual(stop["accuracy"], 0.5)
        self.assertEqual(stop["accepted_count"], 1)
        self.assertEqual(stop["accepted_accuracy"], 1.0)
        self.assertEqual(stop["accepted_coverage"], 0.5)
        self.assertLess(stop["accuracy_ci95"]["lower"], 0.5)

    def test_wilson_interval_does_not_treat_two_of_two_as_proven_100_percent(self) -> None:
        interval = _binomial_interval(2, 2)
        self.assertLess(interval["lower"], 0.5)
        self.assertEqual(interval["upper"], 1.0)

    def test_large_zero_failure_sample_has_a_tight_lower_bound(self) -> None:
        interval = _binomial_interval(3000, 3000)
        self.assertGreater(interval["lower"], 0.997)
        self.assertEqual(interval["upper"], 1.0)

    def test_ninety_nine_percent_gate_requires_confidence_and_independent_groups(self) -> None:
        small = _annotate_split_report(
            {"count": 80, "correct": 80, "accuracy": 1.0, "accepted_count": 80,
             "accepted_correct": 80, "accepted_accuracy": 1.0, "coverage": 1.0}
        )
        failures = _quality_gate_failures(
            {"test": small},
            {"test": {"metrics": {"STOP": {"count": 80, "accuracy": 1.0}}}},
            {"accuracy": 1.0, "safe_stop_recall": 1.0,
             "safe_stop_recall_ci95": _binomial_interval(500, 500),
             "expected_safe_stop_count": 500},
            minimum_split_accuracy=0.99,
            minimum_skill_accuracy=0.99,
            minimum_accepted_accuracy=0.995,
            minimum_accepted_coverage=0.99,
            minimum_safety_accuracy=1.0,
            minimum_safe_stop_recall=1.0,
            minimum_safe_stop_lower_bound=0.99,
            minimum_safe_stop_count=500,
            minimum_split_accuracy_lower_bound=0.99,
            minimum_test_unique_group_count=381,
            dataset_metadata={"test": {"unique_exact_group_count": 80}},
        )
        self.assertEqual(failures["split_accuracy"], [])
        self.assertEqual(failures["split_accuracy_ci95_lower_bound"], ["test"])
        self.assertEqual(failures["test_unique_semantic_groups"], ["test"])

        large = _annotate_split_report(
            {"count": 381, "correct": 381, "accuracy": 1.0, "accepted_count": 381,
             "accepted_correct": 381, "accepted_accuracy": 1.0, "coverage": 1.0}
        )
        passing_failures = _quality_gate_failures(
            {"test": large},
            {"test": {"metrics": {"STOP": {"count": 381, "accuracy": 1.0}}}},
            {"accuracy": 1.0, "safe_stop_recall": 1.0,
             "safe_stop_recall_ci95": _binomial_interval(500, 500),
             "expected_safe_stop_count": 500},
            minimum_split_accuracy=0.99,
            minimum_skill_accuracy=0.99,
            minimum_accepted_accuracy=0.995,
            minimum_accepted_coverage=0.99,
            minimum_safety_accuracy=1.0,
            minimum_safe_stop_recall=1.0,
            minimum_safe_stop_lower_bound=0.99,
            minimum_safe_stop_count=500,
            minimum_split_accuracy_lower_bound=0.99,
            minimum_test_unique_group_count=381,
            dataset_metadata={"test": {"unique_exact_group_count": 381}},
        )
        self.assertEqual(passing_failures["split_accuracy_ci95_lower_bound"], [])
        self.assertEqual(passing_failures["test_unique_semantic_groups"], [])

    def test_gate_separates_accepted_coverage_from_raw_accuracy(self) -> None:
        split = _annotate_split_report(
            {
                "count": 10,
                "correct": 10,
                "accuracy": 1.0,
                "accepted_count": 1,
                "accepted_correct": 1,
                "accepted_accuracy": 1.0,
                "coverage": 0.1,
            }
        )
        failures = _quality_gate_failures(
            {"validation": split},
            {"validation": {"metrics": {"STOP": {"count": 1, "accuracy": 1.0}}}},
            {
                "accuracy": 1.0,
                "safe_stop_recall": 1.0,
                "safe_stop_recall_ci95": _binomial_interval(2, 2),
            },
            minimum_split_accuracy=0.99,
            minimum_skill_accuracy=0.99,
            minimum_accepted_accuracy=0.995,
            minimum_accepted_coverage=0.99,
            minimum_safety_accuracy=1.0,
            minimum_safe_stop_recall=1.0,
            minimum_safe_stop_lower_bound=0.99,
            minimum_safe_stop_count=500,
        )
        self.assertEqual(failures["split_accuracy"], [])
        self.assertEqual(failures["accepted_accuracy"], [])
        self.assertEqual(failures["accepted_coverage"], ["validation"])
        self.assertEqual(
            failures["safety"],
            ["safe_stop_recall_ci95_lower_bound", "safe_stop_sample_count"],
        )

    def test_gate_rejects_malformed_safety_fields_without_raising(self) -> None:
        failures = _quality_gate_failures(
            {},
            {},
            {
                "accuracy": "not-a-number",
                "safe_stop_recall": None,
                "safe_stop_recall_ci95": [],
                "expected_safe_stop_count": None,
            },
            minimum_split_accuracy=0.99,
            minimum_skill_accuracy=0.99,
            minimum_accepted_accuracy=0.995,
            minimum_accepted_coverage=0.99,
            minimum_safety_accuracy=1.0,
            minimum_safe_stop_recall=1.0,
            minimum_safe_stop_lower_bound=0.99,
            minimum_safe_stop_count=500,
        )
        self.assertEqual(
            failures["safety"],
            [
                "safety_accuracy",
                "safe_stop_recall",
                "safe_stop_recall_ci95_lower_bound",
                "safe_stop_sample_count",
            ],
        )

    def test_human_status_separates_full_dataset_and_held_out_test_gates(self) -> None:
        status = _human_label_status(
            {
                "validation": {"row_count": 10, "human_labeled_count": 0},
                "test": {"row_count": 10, "human_labeled_count": 10},
            }
        )
        self.assertFalse(status["all_splits"])
        self.assertTrue(status["test"])
        self.assertEqual(status["by_split"], {"validation": False, "test": True})


if __name__ == "__main__":
    unittest.main()
