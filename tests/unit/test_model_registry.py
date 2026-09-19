import tempfile
import unittest
from pathlib import Path

from hyperjev.model_registry import ModelRegistry, ModelRegistryError


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


class ModelRegistryTests(unittest.TestCase):
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
