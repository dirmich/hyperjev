import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from hyperjev.cli import main
from hyperjev.registry import TaskRegistry
from hyperjev.samples import CanonicalSample
from hyperjev.student import StudentConfig
from hyperjev.training import (
    TrainingConfig,
    TrainingDataError,
    TrainingDependencyError,
    _encode_reference_sample,
    build_training_plan,
    load_training_dataset,
    run_reference_training,
)

ROOT = Path(__file__).resolve().parents[2]


def _record(sample_id: str, task_id: str, target: object, split: str, soft_target: object) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "task_version": 1,
        "state": "테스트 상태",
        "question": "테스트 질문",
        "target": target,
        "soft_target": soft_target,
        "language": "ko",
        "domain": "test",
        "source": {"kind": "synthetic", "source_id": sample_id},
        "labels": {"qwen": None, "gemma": None, "human": None},
        "provenance": {
            "prompt_version": 1,
            "split": split,
            "privacy_raw_inputs_stored": False,
        },
    }


class TrainingTests(unittest.TestCase):
    def test_reference_encoder_supports_dynamic_inference_padding(self) -> None:
        sample = CanonicalSample(
            sample_id="sample",
            task_id="memory.remember_worthy",
            task_version=1,
            state="테스트 상태",
            question="테스트 질문",
            target=True,
            language="ko",
            domain="test",
            source={"kind": "synthetic"},
            labels={"human": None},
            provenance={"prompt_version": 1, "split": "train", "privacy_raw_inputs_stored": False},
        )
        padded, padded_mask = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=1024,
        )
        dynamic, dynamic_mask = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=1024,
            pad_to_max=False,
        )
        self.assertEqual(len(padded), 1024)
        self.assertEqual(len(padded_mask), 1024)
        self.assertEqual(len(dynamic), sum(dynamic_mask))
        self.assertLess(len(dynamic), len(padded))
        self.assertEqual(padded[: len(dynamic)], dynamic)

    def test_reference_token_encoder_is_deterministic_and_bounded(self) -> None:
        sample = CanonicalSample(
            sample_id="sample-token",
            task_id="memory.remember_worthy",
            task_version=1,
            state="Obstacle is ahead",
            question="Is this worth long-term memory?",
            target=True,
            language="en",
            domain="test",
            source={"kind": "synthetic"},
            labels={"human": None},
            provenance={"prompt_version": 1, "split": "train", "privacy_raw_inputs_stored": False},
        )
        first, first_mask = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=4,
            pad_to_max=False,
            backbone="reference-token-encoder",
        )
        second, second_mask = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=4,
            pad_to_max=False,
            backbone="reference-token-encoder",
        )
        self.assertEqual(first, second)
        self.assertEqual(first_mask, second_mask)
        self.assertLessEqual(len(first), 4)

    def test_dataset_validation_and_plan_are_reproducible(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        json.dumps(_record("sample-bool", "memory.remember_worthy", True, "train", 0.9)),
                        json.dumps(
                            _record(
                                "sample-choice",
                                "memory.type",
                                "decision",
                                "validation",
                                {
                                    "fact": 0.0,
                                    "preference": 0.0,
                                    "episode": 0.0,
                                    "decision": 1.0,
                                    "goal": 0.0,
                                    "task": 0.0,
                                    "relationship": 0.0,
                                    "temporary": 0.0,
                                    "none": 0.0,
                                },
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_training_dataset(dataset, registry)
            plan = build_training_plan(
                dataset,
                registry,
                student=StudentConfig(model_id="test-student"),
                training=TrainingConfig(epochs=2, batch_size=4),
            )

        self.assertEqual(len(loaded.samples), 2)
        self.assertEqual(loaded.split_counts, {"train": 1, "validation": 1})
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(plan["dataset"]["sha256"], loaded.dataset_hash)
        self.assertEqual(plan["student"]["model_id"], "test-student")
        self.assertEqual(plan["training"]["epochs"], 2)
        self.assertEqual(plan["checkpoint"], {"status": "not_created", "path": None})

    def test_raw_input_training_data_is_rejected(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            record = _record("sample-bool", "memory.remember_worthy", True, "train", 0.9)
            record["provenance"] = {"prompt_version": 1, "split": "train", "privacy_raw_inputs_stored": True}
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(TrainingDataError):
                load_training_dataset(dataset, registry)

    def test_train_plan_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            output = Path(directory) / "training-plan.json"
            dataset.write_text(
                json.dumps(_record("sample-bool", "memory.remember_worthy", True, "train", 0.9)) + "\n",
                encoding="utf-8",
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "train",
                        "plan",
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(output),
                    ]
                )
            plan = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(plan["record_type"], "training_plan")
        self.assertEqual(json.loads(stdout.getvalue())["checkpoint"]["status"], "not_created")

    def test_reference_training_has_an_explicit_torch_dependency_gate(self) -> None:
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("the host has torch; DGX execution is covered by the optional runtime")
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            checkpoint = Path(directory) / "student.pt"
            dataset.write_text(
                json.dumps(_record("sample-bool", "memory.remember_worthy", True, "train", 0.9)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TrainingDependencyError):
                run_reference_training(dataset, checkpoint, registry, training=TrainingConfig(epochs=1))
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
