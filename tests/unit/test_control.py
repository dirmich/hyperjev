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
    deterministic_control_action,
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
        self.assertTrue(explicit_stop_signal("Emergency hazard is detected inside the collision zone"))
        self.assertFalse(explicit_stop_signal("obstacle is far behind"))

    def test_control_fast_path_requires_compound_semantics(self) -> None:
        action = deterministic_control_action("the reachable target is locked directly ahead")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.skill, "APPROACH")
        self.assertEqual(action.source, "control-rule")
        self.assertEqual(action.confidence, 1.0)
        self.assertIsNone(deterministic_control_action("target ahead"))
        self.assertIsNone(
            deterministic_control_action(
                "the target is ahead but cannot be approached safely"
            )
        )
        action = deterministic_control_action("target is close and directly ahead")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.skill, "APPROACH")

    def test_control_fast_path_covers_hold_without_confusing_stop(self) -> None:
        action = deterministic_control_action("pause safely while the pose is stable and no hazard is present")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.skill, "HOLD")
        self.assertFalse(explicit_stop_signal("pause safely while the pose is stable and no hazard is present"))
        self.assertIsNone(
            deterministic_control_action(
                "pause safely while the pose is stable but an emergency hazard is present"
            )
        )

    def test_control_fast_path_covers_compositional_ood_phrases(self) -> None:
        cases = {
            "STOP": (
                "a collision may occur within one meter",
                "the front lidar reports an imminent impact",
                "danger is inside the stopping distance",
                "a sudden wall blocks the vehicle",
            ),
            "HOLD": (
                "remain stationary because the scene is stable",
                "keep the current pose until another command arrives",
                "wait in place while sensors refresh",
                "no motion is needed at this stable waypoint",
                "stay still with no immediate hazard",
            ),
            "MOVE": (
                "advance through the unobstructed hallway",
                "continue forward along the open route",
                "travel toward the next waypoint on the clear path",
                "forward navigation is safe in free space",
                "proceed straight through the corridor",
            ),
            "ROTATE": (
                "reorient toward the east corridor",
                "turn left to align with the waypoint",
                "change heading at the junction",
                "the next route requires a ninety degree turn",
                "rotate in place to face the goal",
            ),
            "APPROACH": (
                "close the gap to the visible marker",
                "move nearer to the selected object",
                "the destination is visible but still distant",
                "reduce distance to the goal safely",
                "go toward the locked target",
            ),
            "RETREAT": (
                "back up from the moving obstacle",
                "increase distance from the approaching hazard",
                "reverse into the safe rear area",
                "withdraw from the blocked front",
                "move backward to escape the danger",
            ),
            "INTERACT": (
                "press the illuminated button",
                "grasp the aligned handle",
                "activate the switch beside the robot",
                "touch the reachable object",
                "pick up the selected item",
            ),
            "RECOVER": (
                "reinitialize after pose estimation failure",
                "restore balance after a fall",
                "the controller lost localization",
                "reset the navigation fault",
                "recover from the unstable state",
            ),
        }
        for expected, states in cases.items():
            for state in states:
                with self.subTest(expected=expected, state=state):
                    if expected == "STOP":
                        self.assertTrue(explicit_stop_signal(state))
                        continue
                    action = deterministic_control_action(state)
                    self.assertIsNotNone(action)
                    assert action is not None
                    self.assertEqual(action.skill, expected)

    def test_control_fast_path_covers_korean_compositional_phrases(self) -> None:
        cases = {
            "STOP": ("전방 장애물이 제동 거리 안으로 들어왔다", "충돌할 위험이 있다"),
            "HOLD": ("자세가 안정되어 현재 위치를 그대로 유지한다", "위험이 없어 다음 명령을 기다린다"),
            "MOVE": ("앞쪽 통로가 비어 있어 직진한다", "열린 경로를 따라 계속 앞으로 이동한다"),
            "ROTATE": ("다음 복도를 향해 로봇의 방향을 돌린다", "목표 방향에 맞도록 제자리에서 회전한다"),
            "APPROACH": ("눈앞의 표지판까지 거리를 줄인다", "선택된 물체에 가까워지도록 이동한다"),
            "RETREAT": ("다가오는 장애물에서 멀어지도록 후진한다", "뒤쪽의 안전 구역으로 물러난다"),
            "INTERACT": ("손이 닿는 버튼을 눌러 장치를 작동한다", "정렬된 손잡이를 잡는다"),
            "RECOVER": ("위치 추적이 끊겨 복구 절차를 시작한다", "넘어진 뒤 균형을 되찾아야 한다"),
        }
        for expected, states in cases.items():
            for state in states:
                with self.subTest(expected=expected, state=state):
                    if expected == "STOP":
                        self.assertTrue(explicit_stop_signal(state))
                        continue
                    action = deterministic_control_action(state)
                    self.assertIsNotNone(action)
                    assert action is not None
                    self.assertEqual(action.skill, expected)

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
                        "--max-p95-ms",
                        "1000",
                        "--max-p99-ms",
                        "1000",
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["safe_stop_recall"], 1.0)
        self.assertIn("p99", report["latency_ms"])
        self.assertIn("max", report["latency_ms"])
        self.assertTrue(report["latency_gate"]["passed"])
        self.assertEqual(json.loads(stdout.getvalue())["count"], 2)

    def test_control_simulation_fails_latency_gate(self) -> None:
        class _FakeClient:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def decide(self, observation, *, now_ms):
                return ControlAction(skill="APPROACH", confidence=1.0)

        with tempfile.TemporaryDirectory() as directory:
            scenarios = Path(directory) / "scenarios.jsonl"
            scenarios.write_text(
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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("hyperjev.cli.ControlStudentClient", _FakeClient):
                exit_code = main(
                    [
                        "control",
                        "simulate",
                        "--checkpoint",
                        "unused.pt",
                        "--scenarios",
                        str(scenarios),
                        "--max-p95-ms",
                        "0.000000001",
                    ]
                )
        self.assertEqual(exit_code, 1)

    def test_control_simulation_rejects_duplicate_scenario_ids(self) -> None:
        class _FakeClient:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def decide(self, observation, *, now_ms):
                return ControlAction(skill="APPROACH", confidence=1.0)

        with tempfile.TemporaryDirectory() as directory:
            scenarios = Path(directory) / "duplicate-scenarios.jsonl"
            scenario = {
                "scenario_id": "same",
                "observation": {
                    "observation_id": "same",
                    "state": "target ahead",
                    "domain": "simulation",
                    "timestamp_ms": 1000,
                },
                "now_ms": 1001,
                "expected_skill": "APPROACH",
            }
            scenarios.write_text(
                json.dumps(scenario) + "\n" + json.dumps(scenario) + "\n",
                encoding="utf-8",
            )
            with patch("hyperjev.cli.ControlStudentClient", _FakeClient):
                exit_code = main(
                    [
                        "control",
                        "simulate",
                        "--checkpoint",
                        "unused.pt",
                        "--scenarios",
                        str(scenarios),
                    ]
                )
        self.assertEqual(exit_code, 2)

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

    def test_control_student_uses_fast_path_before_model(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        config = StudentConfig(model_id="control-fast-path-test", backbone="reference-ngram-encoder", precision="fp32")
        model = build_torch_model(registry, config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "control.pt"
            torch.save(
                {"student": student_manifest(registry, config), "model_state_dict": model.state_dict()},
                checkpoint,
            )
            client = ControlStudentClient(checkpoint, registry)
            called = False

            def fail_if_called(*_args, **_kwargs):
                nonlocal called
                called = True
                raise AssertionError("model should not run for a high-precision fast path")

            client._client.complete_decision = fail_if_called
            observation = ControlObservation(
                observation_id="approach-frame",
                state="the reachable target is locked directly ahead",
                domain="simulation",
                timestamp_ms=1000.0,
            )
            action = client.decide(observation, now_ms=1001.0)
        self.assertFalse(called)
        self.assertEqual(action.skill, "APPROACH")
        self.assertEqual(action.source, "control-rule")

    def test_control_student_can_disable_fast_path_for_model_only_measurement(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        config = StudentConfig(model_id="control-model-only-test", backbone="reference-ngram-encoder", precision="fp32")
        model = build_torch_model(registry, config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "control.pt"
            torch.save(
                {"student": student_manifest(registry, config), "model_state_dict": model.state_dict()},
                checkpoint,
            )
            client = ControlStudentClient(checkpoint, registry, enable_fast_path=False)
            observation = ControlObservation(
                observation_id="model-only-frame",
                state="the reachable target is locked directly ahead",
                domain="simulation",
                timestamp_ms=1000.0,
            )
            action = client.decide(observation, now_ms=1001.0)
        self.assertNotEqual(action.source, "control-rule")

    def test_model_only_measurement_keeps_explicit_stop_interlock(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        config = StudentConfig(model_id="control-model-only-stop-test", backbone="reference-ngram-encoder", precision="fp32")
        model = build_torch_model(registry, config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "control.pt"
            torch.save(
                {"student": student_manifest(registry, config), "model_state_dict": model.state_dict()},
                checkpoint,
            )
            client = ControlStudentClient(checkpoint, registry, enable_fast_path=False)
            observation = ControlObservation(
                observation_id="model-only-stop-frame",
                state="the front lidar reports an imminent impact",
                domain="simulation",
                timestamp_ms=1000.0,
            )
            action = client.decide(observation, now_ms=1001.0)
        self.assertEqual(action.skill, "STOP")
        self.assertEqual(action.source, "safety-rule")


if __name__ == "__main__":
    unittest.main()
