import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_control_safety_scenarios import (
    build_safety_near_miss_scenarios,
    build_safety_scenarios,
)


class ControlSafetyScenarioTests(unittest.TestCase):
    def test_default_matrix_has_distinct_safe_stop_scenarios(self) -> None:
        scenarios = build_safety_scenarios()
        self.assertEqual(len(scenarios), 500)
        self.assertEqual(len({scenario["scenario_id"] for scenario in scenarios}), 500)
        self.assertEqual(
            len({scenario["source"]["episode_id"] for scenario in scenarios}),
            500,
        )
        self.assertTrue(all(scenario["expected_safe_stop"] for scenario in scenarios))
        self.assertEqual(
            {scenario["expected_reason"] for scenario in scenarios},
            {"emergency_stop", "stale_observation", "invalid_clock"},
        )

    def test_matrix_rejects_too_few_safety_cases(self) -> None:
        with self.assertRaises(ValueError):
            build_safety_scenarios(2)

    def test_near_miss_matrix_contains_only_fresh_non_stop_controls(self) -> None:
        scenarios = build_safety_near_miss_scenarios(21)
        self.assertEqual(len(scenarios), 21)
        self.assertEqual(len({scenario["scenario_id"] for scenario in scenarios}), 21)
        self.assertTrue(all(not scenario["expected_safe_stop"] for scenario in scenarios))
        self.assertEqual(
            {scenario["source"]["near_miss_reason"] for scenario in scenarios},
            {"emergency_stop", "stale_observation", "invalid_clock"},
        )
        self.assertTrue(all(scenario["observation"]["emergency_stop"] is False for scenario in scenarios))


if __name__ == "__main__":
    unittest.main()
