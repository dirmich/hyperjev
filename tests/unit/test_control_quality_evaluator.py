import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_control_quality import (
    _annotate_split_report,
    _binomial_interval,
    _human_label_status,
    _quality_gate_failures,
    _skill_report,
)

from hyperjev.registry import TaskRegistry


class ControlQualityEvaluatorTests(unittest.TestCase):
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
        )
        self.assertEqual(failures["split_accuracy"], [])
        self.assertEqual(failures["accepted_accuracy"], [])
        self.assertEqual(failures["accepted_coverage"], ["validation"])
        self.assertEqual(failures["safety"], ["safe_stop_recall_ci95_lower_bound"])

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
