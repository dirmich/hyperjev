import unittest

from hyperjev.metrics import MetricError, threshold_risk_coverage


class RiskCoverageTests(unittest.TestCase):
    def test_threshold_report_separates_accepted_quality_from_coverage(self) -> None:
        report = threshold_risk_coverage(
            [True, False, True, False],
            [0.99, 0.95, 0.75, 0.40],
            thresholds=(0.50, 0.90, 0.99),
        )
        self.assertEqual(report[0]["coverage"], 0.75)
        self.assertEqual(report[0]["accepted_accuracy"], 0.666667)
        self.assertEqual(report[0]["accepted_risk"], 0.333333)
        self.assertEqual(report[1]["accepted_count"], 2)
        self.assertEqual(report[1]["accepted_accuracy"], 0.5)
        self.assertEqual(report[2]["accepted_count"], 1)
        self.assertEqual(report[2]["accepted_accuracy"], 1.0)

    def test_threshold_report_rejects_empty_or_invalid_inputs(self) -> None:
        with self.assertRaises(MetricError):
            threshold_risk_coverage([True], [0.9], thresholds=())
        with self.assertRaises(MetricError):
            threshold_risk_coverage([True], [0.9], thresholds=(1.1,))


if __name__ == "__main__":
    unittest.main()
