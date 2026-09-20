import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from hyperjev.calibration import fit_temperature, probabilities
from hyperjev.cli import main
from hyperjev.config import load_config
from hyperjev.registry import TaskRegistry
from hyperjev.samples import CanonicalSample
from hyperjev.student import (
    StudentConfig,
    StudentDependencyError,
    build_torch_model,
    head_specs,
    student_manifest,
)
from hyperjev.student_client import StudentClient
from hyperjev.student_inference import (
    _deduplicate_exact,
    _exact_group_key,
    evaluate_student_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]


class StudentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        cls.registry = TaskRegistry.load(config.registry_path)

    def test_exact_group_evaluation_representative_is_deterministic(self) -> None:
        common = {
            "task_id": "memory.remember_worthy",
            "task_version": 1,
            "state": "같은 내용",
            "question": "기억할 가치가 있는가?",
            "target": True,
            "language": "ko",
            "domain": "conversation",
            "source": {},
            "labels": {},
            "provenance": {"prompt_version": 1, "split": "test", "privacy_raw_inputs_stored": False},
        }
        first = CanonicalSample(sample_id="first", **common)
        duplicate = CanonicalSample(sample_id="second", **common)
        selected = _deduplicate_exact([first, duplicate])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].sample_id, "first")
        self.assertEqual(_exact_group_key(first), _exact_group_key(duplicate))

    def test_manifest_has_six_registry_derived_typed_heads(self) -> None:
        specs = head_specs(self.registry)
        manifest = student_manifest(self.registry, StudentConfig(model_id="test-student"))
        self.assertEqual(len(specs), 6)
        self.assertEqual(manifest["track"], "encoder-typed-heads")
        self.assertEqual(manifest["diffusion_track"], "HyperJev-D")
        self.assertEqual(
            {head["output_type"] for head in manifest["heads"]},
            {"boolean", "choice", "score"},
        )

    def test_torch_backend_is_optional(self) -> None:
        try:
            model = build_torch_model(self.registry)
        except StudentDependencyError as exc:
            self.assertIn("PyTorch", str(exc))
        else:
            self.assertTrue(hasattr(model, "forward"))

    def test_torch_backend_supports_dotted_registry_task_ids(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        model = build_torch_model(self.registry)
        input_ids = torch.zeros((1, 8), dtype=torch.long)
        attention = torch.ones((1, 8), dtype=torch.long)
        for task_id in self.registry.ids():
            output = model(task_id, input_ids, attention)
            self.assertIn(output["type"], {"boolean", "choice", "score"})

    def test_reference_ngram_encoder_is_a_compatible_backbone(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        self.assertIsNotNone(torch)
        model = build_torch_model(
            self.registry,
            StudentConfig(backbone="reference-ngram-encoder", precision="fp32"),
        )
        output = model(
            "wiki.semantic_change",
            torch.zeros((1, 16), dtype=torch.long),
            torch.ones((1, 16), dtype=torch.long),
        )
        self.assertEqual(output["type"], "boolean")

    def test_reference_token_encoder_uses_local_feature_path(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        self.assertIsNotNone(torch)
        model = build_torch_model(
            self.registry,
            StudentConfig(backbone="reference-token-encoder", precision="fp32"),
        )
        self.assertTrue(model.use_ngram_encoder)

    def test_reference_bow_encoder_is_a_compatible_backbone(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        model = build_torch_model(
            self.registry,
            StudentConfig(backbone="reference-bow-encoder", precision="fp32"),
        )
        output = model(
            "memory.type",
            torch.tensor([[2, 17, 0, 0]], dtype=torch.long),
            torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        )
        self.assertTrue(model.use_bow_encoder)
        self.assertEqual(output["type"], "choice")

    def test_student_evaluation_reports_accuracy_and_guarded_coverage(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "student.pt"
            dataset_path = Path(directory) / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-bool",
                        "task_id": "memory.remember_worthy",
                        "task_version": 1,
                        "state": "stable deployment decision",
                        "question": "is this worth long-term memory?",
                        "target": True,
                        "language": "en",
                        "domain": "test",
                        "source": {"kind": "synthetic"},
                        "labels": {"human": None},
                        "provenance": {
                            "prompt_version": 1,
                            "split": "test",
                            "privacy_raw_inputs_stored": False,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = StudentConfig(model_id="eval-test", precision="fp32")
            model = build_torch_model(self.registry, config)
            torch.save(
                {
                    "student": student_manifest(self.registry, config),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            report = evaluate_student_checkpoint(
                checkpoint_path,
                dataset_path,
                self.registry,
                minimum_confidence=1.0,
            )

        self.assertEqual(report["overall"]["count"], 1)
        self.assertIn("accuracy", report["overall"])
        self.assertIn("coverage", report["overall"])
        self.assertEqual(report["overall"]["accepted_count"], 0)
        self.assertFalse(report["quality_gate"]["ready"])
        self.assertIn("human_labels_required", report["quality_gate"]["reasons"])

    def test_student_client_loads_checkpoint_and_returns_typed_result(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "student.pt"
            config = StudentConfig(
                model_id="client-test",
                backbone="reference-ngram-encoder",
                precision="fp32",
            )
            model = build_torch_model(self.registry, config)
            torch.save(
                {
                    "student": student_manifest(self.registry, config),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            client = StudentClient(checkpoint_path, self.registry, minimum_confidence=0.0)
            completion = client.complete_decision(
                self.registry.get("memory.remember_worthy"),
                state="A stable deployment decision.",
                question="remember",
                candidates=[],
            )
        payload = json.loads(completion.content)
        self.assertIn(payload["type"], {"boolean", "choice", "score"})
        self.assertEqual(completion.model, "client-test")

    def test_student_client_applies_matching_control_temperature(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        control_registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "control.pt"
            calibration_path = Path(directory) / "calibration.json"
            config = StudentConfig(
                model_id="calibrated-control-test",
                backbone="reference-ngram-encoder",
                precision="fp32",
            )
            model = build_torch_model(control_registry, config)
            torch.save(
                {
                    "student": student_manifest(control_registry, config),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            calibration_path.write_text(
                json.dumps(
                    {
                        "record_type": "control_calibration_manifest",
                        "calibration_version": "cal-test",
                        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                        "tasks": {"control.skill": {"status": "fitted", "temperature": 4.0}},
                    }
                ),
                encoding="utf-8",
            )
            plain = StudentClient(checkpoint_path, control_registry, minimum_confidence=0.0)
            calibrated = StudentClient(
                checkpoint_path,
                control_registry,
                minimum_confidence=0.0,
                calibration_path=calibration_path,
            )
            task = control_registry.get("control.skill")
            plain_payload = json.loads(
                plain.complete_decision(
                    task,
                    state="the robot is in a stable open area",
                    question="select next safe high-level control skill",
                    candidates=list(task.output["candidates"]),
                ).content
            )
            calibrated_payload = json.loads(
                calibrated.complete_decision(
                    task,
                    state="the robot is in a stable open area",
                    question="select next safe high-level control skill",
                    candidates=list(task.output["candidates"]),
                ).content
            )

        plain_confidence = max(plain_payload["probabilities"].values())
        calibrated_confidence = max(calibrated_payload["probabilities"].values())
        self.assertEqual(calibrated.calibration_version, "cal-test")
        self.assertLessEqual(calibrated_confidence, plain_confidence + 1e-6)

    def test_student_evaluation_uses_human_label_over_synthetic_target(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "student.pt"
            dataset_path = Path(directory) / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "sample_id": "human-overrides-target",
                        "task_id": "memory.remember_worthy",
                        "task_version": 1,
                        "state": "stable deployment decision",
                        "question": "is this worth long-term memory?",
                        "target": True,
                        "language": "en",
                        "domain": "test",
                        "source": {"kind": "synthetic"},
                        "labels": {
                            "human": {
                                "type": "boolean",
                                "value": False,
                                "probability": 1.0,
                                "abstained": False,
                            }
                        },
                        "provenance": {
                            "prompt_version": 1,
                            "split": "test",
                            "privacy_raw_inputs_stored": False,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = StudentConfig(model_id="human-label-test", precision="fp32")
            model = build_torch_model(self.registry, config)
            torch.save(
                {
                    "student": student_manifest(self.registry, config),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            report = evaluate_student_checkpoint(
                checkpoint_path,
                dataset_path,
                self.registry,
                minimum_confidence=1.0,
            )

        prediction = report["predictions"][0]
        self.assertEqual(prediction["target"], False)
        self.assertEqual(prediction["target_source"], "human")
        self.assertTrue(report["golden"]["human_labeled"])
        self.assertNotIn("human_labels_required", report["quality_gate"]["reasons"])


class CalibrationTests(unittest.TestCase):
    def test_temperature_fit_and_probability_are_deterministic(self) -> None:
        logits = ((3.0, 0.0), (2.0, 0.0), (0.2, 1.0), (0.0, 2.0))
        labels = (0, 0, 1, 1)
        result = fit_temperature(logits, labels)
        self.assertEqual(result.sample_count, 4)
        self.assertLessEqual(result.nll_after, result.nll_before)
        values = probabilities((3.0, 0.0), result.temperature)
        self.assertAlmostEqual(sum(values), 1.0, places=6)
        self.assertGreater(values[0], values[1])

    def test_calibrate_cli_writes_versioned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "heldout.json"
            output_path = Path(directory) / "calibration.json"
            input_path.write_text(
                json.dumps(
                    {
                        "logits": [[3.0, 0.0], [2.0, 0.0], [0.2, 1.0], [0.0, 2.0]],
                        "labels": [0, 0, 1, 1],
                    }
                ),
                encoding="utf-8",
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["calibrate", "--input", str(input_path), "--output", str(output_path)]
                )
            manifest = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["record_type"], "calibration_manifest")
        self.assertTrue(manifest["calibration_version"].startswith("cal-"))
        self.assertEqual(json.loads(stdout.getvalue())["result"], manifest["result"])


if __name__ == "__main__":
    unittest.main()
