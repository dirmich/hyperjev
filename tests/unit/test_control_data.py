import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.control_data import validate_control_dataset
from hyperjev.registry import TaskRegistry

ROOT = Path(__file__).resolve().parents[2]


class ControlDatasetQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")

    @staticmethod
    def _record(
        sample_id: str,
        split: str,
        *,
        state: str,
        episode_id: str,
        semantic_group_id: str,
        human: object = None,
    ) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "task_id": "control.skill",
            "task_version": 1,
            "state": state,
            "question": "select next safe high-level control skill",
            "target": "MOVE",
            "language": "en",
            "domain": "simulation",
            "source": {
                "kind": "human-golden",
                "scenario_id": f"scenario-{sample_id}",
                "episode_id": episode_id,
                "semantic_group_id": semantic_group_id,
            },
            "labels": {"human": human},
            "provenance": {
                "prompt_version": 1,
                "split": split,
                "privacy_raw_inputs_stored": False,
            },
        }

    def _write(self, records: list[dict[str, object]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "control.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_independent_provenance_groups_pass(self) -> None:
        path = self._write(
            [
                self._record("train-1", "train", state="clear path", episode_id="ep-1", semantic_group_id="sg-1"),
                self._record(
                    "validation-1",
                    "validation",
                    state="blocked path",
                    episode_id="ep-2",
                    semantic_group_id="sg-2",
                ),
                self._record(
                    "test-1",
                    "test",
                    state="target behind",
                    episode_id="ep-3",
                    semantic_group_id="sg-3",
                ),
            ]
        )
        report = validate_control_dataset(path, self.registry)
        self.assertTrue(report["passed"])
        self.assertEqual(report["unique_episode_count"], 3)
        self.assertEqual(report["unique_semantic_group_count"], 3)

    def test_cross_split_exact_and_episode_leak_fails(self) -> None:
        path = self._write(
            [
                self._record(
                    "train-1",
                    "train",
                    state="same state",
                    episode_id="ep-shared",
                    semantic_group_id="sg-train",
                ),
                self._record(
                    "test-1",
                    "test",
                    state="  SAME   STATE ",
                    episode_id="ep-shared",
                    semantic_group_id="sg-test",
                ),
            ]
        )
        report = validate_control_dataset(path, self.registry)
        self.assertFalse(report["passed"])
        reasons = {error["reason"] for error in report["errors"]}
        self.assertIn("cross_split_exact_leak", reasons)
        self.assertIn("cross_split_episode_leak", reasons)

    def test_human_label_requirement_is_explicit(self) -> None:
        path = self._write(
            [
                self._record(
                    "train-1",
                    "train",
                    state="clear path",
                    episode_id="ep-1",
                    semantic_group_id="sg-1",
                )
            ]
        )
        report = validate_control_dataset(path, self.registry, require_human_labels=True)
        self.assertFalse(report["passed"])
        self.assertEqual(report["human_labeled_count"], 0)
        self.assertIn("human_label_required", {error["reason"] for error in report["errors"]})

    def test_smoke_fixture_is_rejected_until_control_provenance_is_added(self) -> None:
        report = validate_control_dataset(ROOT / "tests" / "golden" / "control_smoke.jsonl", self.registry)
        self.assertFalse(report["passed"])
        self.assertGreater(len(report["errors"]), 0)
        self.assertEqual(report["human_labeled_count"], 0)


if __name__ == "__main__":
    unittest.main()
