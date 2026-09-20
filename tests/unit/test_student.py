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
