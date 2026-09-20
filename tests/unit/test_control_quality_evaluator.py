import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_control_quality import (
    _annotate_split_report,
    _binomial_interval,
    _quality_gate_failures,
)


class ControlQualityEvaluatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
