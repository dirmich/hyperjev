import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_control_safety_scenarios import build_safety_scenarios


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


if __name__ == "__main__":
    unittest.main()
