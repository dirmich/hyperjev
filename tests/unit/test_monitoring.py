import unittest

from hyperjev.contracts import BooleanDecision, DecisionResponse
from hyperjev.monitoring import TrafficMonitor, compare_snapshots


class MonitoringTests(unittest.TestCase):
    def test_snapshot_has_no_raw_input_and_drift_is_explicit(self) -> None:
        baseline_monitor = TrafficMonitor()
        baseline_monitor.observe(
            DecisionResponse(
                request_id="a",
                model="model-a",
                calibration="cal-a",
                results={"remember": BooleanDecision(value=True, probability=0.95)},
                route="hyperjev",
                latency_ms=2.0,
            )
        )
        current_monitor = TrafficMonitor()
        current_monitor.observe(
            DecisionResponse(
                request_id="b",
                model="qwen",
                calibration="none",
                results={"remember": BooleanDecision(value=False, probability=0.5, abstained=True)},
                route="human",
                latency_ms=40.0,
            )
        )
        baseline = baseline_monitor.snapshot()
        current = current_monitor.snapshot()
        drift = compare_snapshots(baseline, current)
        self.assertTrue(drift["drifted"])
        self.assertEqual(drift["action"], "review_dataset_and_calibration")
        self.assertNotIn("request_id", current)
        self.assertEqual(current["abstained_questions"], 1)


if __name__ == "__main__":
    unittest.main()
