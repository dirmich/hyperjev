import unittest
from pathlib import Path

from hyperjev.calibration import fit_temperature, probabilities
from hyperjev.config import load_config
from hyperjev.registry import TaskRegistry
from hyperjev.student import (
    StudentConfig,
    StudentDependencyError,
    build_torch_model,
    head_specs,
    student_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


class StudentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        cls.registry = TaskRegistry.load(config.registry_path)

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


if __name__ == "__main__":
    unittest.main()
