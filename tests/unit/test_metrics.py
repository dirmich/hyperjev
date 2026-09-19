import unittest

from hyperjev.metrics import binary_metrics, choice_metrics, score_metrics


class MetricsTests(unittest.TestCase):
    def test_binary_metrics_include_quality_and_calibration(self) -> None:
        report = binary_metrics([True, False, True, False], [0.9, 0.8, 0.6, 0.3])
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["precision"], 0.666667)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 0.8)
        self.assertEqual(report["brier"], 0.225)
        self.assertEqual(report["ece"], 0.4)
        self.assertEqual(report["nll"], 0.645575)
        self.assertEqual(report["auroc"], 0.75)
        self.assertEqual(report["auprc"], 0.833333)
        self.assertEqual(len(report["risk_coverage"]), 4)

    def test_choice_metrics_report_top2_and_macro_f1(self) -> None:
        report = choice_metrics(
            ["a", "b", "c"],
            ["a", "c", "b"],
            [
                {"a": 0.8, "b": 0.1, "c": 0.1},
                {"a": 0.45, "b": 0.4, "c": 0.15},
                {"a": 0.2, "b": 0.6, "c": 0.2},
            ],
        )
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["accuracy"], 0.333333)
        self.assertEqual(report["top2_accuracy"], 0.666667)
        self.assertEqual(report["macro_f1"], 0.333333)
        self.assertEqual(report["nll"], 0.916291)
        self.assertEqual(len(report["risk_coverage"]), 3)

    def test_score_metrics_report_error_and_interval_coverage(self) -> None:
        report = score_metrics([0.5, 0.9], [0.4, 0.8], [(0.2, 0.6), (0.85, 0.95)])
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["mae"], 0.1)
        self.assertEqual(report["rmse"], 0.1)
        self.assertEqual(report["interval_90_coverage"], 1.0)
        self.assertEqual(report["interval_90_width"], 0.25)
        self.assertEqual(report["spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
