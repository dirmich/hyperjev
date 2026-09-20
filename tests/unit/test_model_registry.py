import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hyperjev.cli import main
from hyperjev.model_registry import ModelRegistry, ModelRegistryError, build_model_manifest


def _manifest(model_id: str, status: str = "trained") -> dict[str, object]:
    return {
        "model_id": model_id,
        "base_model": "small-multilingual-encoder",
        "dataset_hash": "dataset-sha",
        "git_commit": "abc123",
        "tasks": {"memory.remember_worthy": [1]},
        "calibration_version": "cal-1",
        "runtime": "pytorch",
        "precision": "bf16",
        "status": status,
    }


def _training_plan() -> dict[str, object]:
    return {
        "record_type": "training_plan",
        "dataset": {"sha256": "a" * 64},
        "student": {
            "model_id": "hyperjev-test",
            "backbone": "small-multilingual-encoder",
            "heads": [
                {"task_id": "memory.remember_worthy", "task_version": 1},
                {"task_id": "memory.type", "task_version": 1},
            ],
        },
        "training": {"precision": "bf16"},
    }


def _calibration() -> dict[str, str]:
    return {
        "record_type": "calibration_manifest",
        "calibration_version": "cal-test",
    }


class ModelRegistryTests(unittest.TestCase):
    def test_build_manifest_binds_training_calibration_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "student.pt"
            checkpoint.write_bytes(b"checkpoint-v1")
            manifest = build_model_manifest(
                _training_plan(),
                _calibration(),
                checkpoint,
                git_commit="abc123",
            )
        self.assertEqual(manifest["model_id"], "hyperjev-test")
        self.assertEqual(manifest["tasks"], {"memory.remember_worthy": [1], "memory.type": [1]})
        self.assertEqual(manifest["calibration_version"], "cal-test")
        self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(b"checkpoint-v1").hexdigest())
        self.assertEqual(manifest["status"], "trained")

    def test_build_manifest_rejects_invalid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ModelRegistryError):
                build_model_manifest(_training_plan(), _calibration(), Path(directory) / "missing.pt", git_commit="abc")
            invalid_plan = _training_plan()
            invalid_plan["dataset"] = {"sha256": "not-a-sha"}
            checkpoint = Path(directory) / "student.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaises(ModelRegistryError):
                build_model_manifest(invalid_plan, _calibration(), checkpoint, git_commit="abc")

    def test_manifest_cli_writes_registry_compatible_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            calibration = root / "calibration.json"
            checkpoint = root / "student.pt"
            output = root / "manifest.json"
            plan.write_text(json.dumps(_training_plan()), encoding="utf-8")
            calibration.write_text(json.dumps(_calibration()), encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "model",
                        "manifest",
                        "--training-plan",
                        str(plan),
                        "--calibration",
                        str(calibration),
                        "--checkpoint",
                        str(checkpoint),
                        "--git-commit",
                        "abc123",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["model_id"], "hyperjev-test")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["checkpoint_sha256"], hashlib.sha256(b"checkpoint").hexdigest())

    def test_lifecycle_and_explicit_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory) / "models.json")
            registry.register(_manifest("model-a"))
            for status in ("evaluated", "calibrated", "candidate", "canary", "active"):
                registry.transition("model-a", status)
            registry.register(_manifest("model-b"))
            for status in ("evaluated", "calibrated", "candidate", "canary", "active"):
                registry.transition("model-b", status)
            restored = registry.rollback("model-a")
            models = {record["model_id"]: record["status"] for record in registry.list()}
        self.assertEqual(restored["status"], "active")
        self.assertEqual(models, {"model-a": "active", "model-b": "retired"})

    def test_invalid_transition_and_duplicate_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory) / "models.json")
            registry.register(_manifest("model-a"))
            with self.assertRaises(ModelRegistryError):
                registry.transition("model-a", "active")
            with self.assertRaises(ModelRegistryError):
                registry.register(_manifest("model-a"))


if __name__ == "__main__":
    unittest.main()
