import importlib.util
import unittest
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "calibrate_control_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("calibrate_control_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControlCalibrationTests(unittest.TestCase):
    def test_target_index_uses_registered_boolean_and_choice_order(self) -> None:
        class BooleanTask:
            output_type = "boolean"
            output: ClassVar[dict[str, object]] = {}

        class ChoiceTask:
            output_type = "choice"
            output: ClassVar[dict[str, object]] = {"candidates": ["STOP", "MOVE"]}

        self.assertEqual(MODULE._target_index(True, BooleanTask()), 1)
        self.assertEqual(MODULE._target_index("MOVE", ChoiceTask()), 1)


if __name__ == "__main__":
    unittest.main()
