import tempfile
import unittest
from pathlib import Path

from hyperjev.control import (
    CONTROL_SKILLS,
    ControlAction,
    ControlContractError,
    ControlMemoryContext,
    ControlObservation,
    ControlSafetyPolicy,
    ControlStudentClient,
    apply_safety_policy,
)
from hyperjev.registry import TaskRegistry
from hyperjev.student import StudentConfig, build_torch_model, student_manifest

ROOT = Path(__file__).resolve().parents[2]


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

    def test_observation_rejects_non_boolean_emergency_flag(self) -> None:
        with self.assertRaises(ControlContractError):
            ControlObservation.from_dict(
                {
                    "observation_id": "frame-1",
                    "state": "clear",
                    "domain": "simulation",
                    "timestamp_ms": 1000,
                    "emergency_stop": "false",
                }
            )

    def test_observation_renders_bounded_hypermemory_context(self) -> None:
        observation = ControlObservation.from_dict(
            {
                "observation_id": "frame-1",
                "state": "target is ahead",
                "domain": "simulation",
                "timestamp_ms": 1000,
                "memory_context": {
                    "source": "hypermemory",
                    "summaries": ["The robot prefers the north route."],
                },
            }
        )
        self.assertIn("Relevant memory context (hypermemory)", observation.model_state())
        self.assertIn("north route", observation.model_state())

    def test_memory_context_is_bounded(self) -> None:
        context = ControlMemoryContext.from_text("x" * 2049)
        self.assertEqual(len(context.summaries[0]), 2048)

    def test_boolean_parameters_are_rejected(self) -> None:
        with self.assertRaises(ControlContractError):
            ControlAction(skill="MOVE", parameters={"enabled": True}, confidence=0.99)

    def test_control_student_maps_typed_head_to_registered_skill(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        config = StudentConfig(
            model_id="control-test",
            backbone="reference-ngram-encoder",
            precision="fp32",
        )
        model = build_torch_model(registry, config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "control.pt"
            torch.save(
                {
                    "student": student_manifest(registry, config),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint,
            )
            client = ControlStudentClient(
                checkpoint,
                registry,
                policy=ControlSafetyPolicy(minimum_confidence=0.0),
            )
            action = client.decide(self.observation, now_ms=1001.0)
        self.assertIn(action.skill, CONTROL_SKILLS)
        self.assertEqual(action.source, "hyperjev-control")


if __name__ == "__main__":
    unittest.main()
