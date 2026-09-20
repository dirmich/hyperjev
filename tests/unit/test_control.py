import unittest

from hyperjev.control import (
    ControlAction,
    ControlContractError,
    ControlObservation,
    ControlSafetyPolicy,
    apply_safety_policy,
)


class ControlContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = ControlObservation(
            observation_id="frame-1",
            state="target ahead",
            domain="simulation",
            timestamp_ms=1000.0,
        )
        self.action = ControlAction(
            skill="APPROACH",
            parameters={"speed": 0.4},
            ttl_ms=50,
            confidence=0.98,
        )

    def test_valid_action_passes_safety_policy(self) -> None:
        result = apply_safety_policy(self.observation, self.action, now_ms=1010.0)
        self.assertEqual(result, self.action)

    def test_stale_observation_becomes_safe_stop(self) -> None:
        result = apply_safety_policy(self.observation, self.action, now_ms=1101.0)
        self.assertEqual(result.skill, "STOP")
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "stale_observation")

    def test_emergency_stop_precedes_model_action(self) -> None:
        observation = ControlObservation(
            observation_id="frame-2",
            state="target ahead",
            domain="simulation",
            timestamp_ms=1000.0,
            emergency_stop=True,
        )
        result = apply_safety_policy(observation, self.action, now_ms=1001.0)
        self.assertEqual(result.reason, "emergency_stop")
        self.assertEqual(result.skill, "STOP")

    def test_low_confidence_and_long_ttl_are_rejected(self) -> None:
        policy = ControlSafetyPolicy(minimum_confidence=0.95, max_action_ttl_ms=100)
        low_confidence = ControlAction(skill="MOVE", confidence=0.8, ttl_ms=50)
        long_action = ControlAction(skill="MOVE", confidence=0.99, ttl_ms=101)
        self.assertEqual(
            apply_safety_policy(self.observation, low_confidence, now_ms=1001.0, policy=policy).reason,
            "confidence_below_control_threshold",
        )
        self.assertEqual(
            apply_safety_policy(self.observation, long_action, now_ms=1001.0, policy=policy).reason,
            "action_ttl_exceeded",
        )

    def test_motor_like_unbounded_parameter_is_rejected(self) -> None:
        with self.assertRaises(ControlContractError):
            ControlAction(skill="MOVE", parameters={"pwm": 255.0}, confidence=0.99)


if __name__ == "__main__":
    unittest.main()
