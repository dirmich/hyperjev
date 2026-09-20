import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from hyperjev.cli import main
from hyperjev.control import (
    CONTROL_SKILLS,
    ControlAction,
    ControlContractError,
    ControlMemoryContext,
    ControlObservation,
    ControlSafetyPolicy,
    ControlStudentClient,
    apply_safety_policy,
    explicit_stop_signal,
    safe_stop,
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

    def test_explicit_collision_signal_is_a_planned_stop(self) -> None:
        self.assertTrue(explicit_stop_signal("Obstacle is directly ahead"))
        self.assertFalse(explicit_stop_signal("obstacle is far behind"))

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

    def test_control_simulation_cli_reports_action_and_safety_metrics(self) -> None:
        class _FakeClient:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def decide(self, observation, *, now_ms):
                if observation.emergency_stop:
                    return safe_stop(reason="emergency_stop")
                if now_ms - observation.timestamp_ms > 100:
                    return safe_stop(reason="stale_observation")
                return ControlAction(skill="APPROACH", confidence=1.0)

        with tempfile.TemporaryDirectory() as directory:
            scenarios = Path(directory) / "scenarios.jsonl"
            output = Path(directory) / "simulation.json"
            scenarios.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "scenario_id": "normal",
                                "observation": {
                                    "observation_id": "normal",
                                    "state": "target ahead",
                                    "domain": "simulation",
                                    "timestamp_ms": 1000,
                                },
                                "now_ms": 1001,
                                "expected_skill": "APPROACH",
                                "expected_safe_stop": False,
                            }
                        ),
                        json.dumps(
                            {
                                "scenario_id": "stale",
                                "observation": {
                                    "observation_id": "stale",
                                    "state": "target ahead",
                                    "domain": "simulation",
                                    "timestamp_ms": 1000,
                                },
                                "now_ms": 1101,
                                "expected_skill": "STOP",
                                "expected_reason": "stale_observation",
                                "expected_safe_stop": True,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = StringIO()
            with patch("hyperjev.cli.ControlStudentClient", _FakeClient), redirect_stdout(stdout):
                exit_code = main(
                    [
                        "control",
                        "simulate",
                        "--checkpoint",
                        "unused.pt",
                        "--scenarios",
                        str(scenarios),
                        "--output",
                        str(output),
                        "--fail-on-mismatch",
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["safe_stop_recall"], 1.0)
        self.assertEqual(json.loads(stdout.getvalue())["count"], 2)

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

    def test_control_student_short_circuits_explicit_collision_signal(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        config = StudentConfig(model_id="control-stop-test", backbone="reference-ngram-encoder", precision="fp32")
        model = build_torch_model(registry, config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "control.pt"
            torch.save(
                {"student": student_manifest(registry, config), "model_state_dict": model.state_dict()},
                checkpoint,
            )
            client = ControlStudentClient(checkpoint, registry)
            observation = ControlObservation(
                observation_id="stop-frame",
                state="obstacle is directly ahead",
                domain="simulation",
                timestamp_ms=1000.0,
            )
            action = client.decide(observation, now_ms=1001.0)
        self.assertEqual(action.skill, "STOP")
        self.assertEqual(action.source, "safety-rule")
        self.assertFalse(action.abstained)


if __name__ == "__main__":
    unittest.main()
