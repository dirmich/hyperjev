import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from hyperjev.cli import build_parser, main
from hyperjev.registry import TaskRegistry
from hyperjev.samples import CanonicalSample
from hyperjev.student import StudentConfig
from hyperjev.training import (
    TrainingConfig,
    TrainingDataError,
    TrainingDependencyError,
    _balance_class_samples,
    _encode_reference_sample,
    _sample_loss_weight,
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
    def test_class_balancer_equalizes_target_counts(self) -> None:
        samples = [
            CanonicalSample(
                sample_id=f"sample-{index}",
                task_id="memory.remember_worthy",
                task_version=1,
                state="state",
                question="question",
                target=target,
                language="en",
                domain="test",
                source={},
                labels={},
                provenance={"prompt_version": 1, "split": "train"},
            )
            for index, target in enumerate([True, False, False])
        ]
        balanced = _balance_class_samples(samples)
        counts = {True: 0, False: 0}
        for sample in balanced:
            counts[sample.target] += 1
        self.assertEqual(counts, {True: 2, False: 2})

    def test_hard_negative_weight_defaults_to_one_and_validates(self) -> None:
        self.assertEqual(TrainingConfig().hard_negative_weight, 1.0)
        with self.assertRaises(TrainingDataError):
            TrainingConfig(hard_negative_weight=0.0).validate()

    def test_char_bow_reference_encoder_emits_deterministic_character_features(self) -> None:
        sample = CanonicalSample(
            sample_id="char-bow",
            task_id="memory.remember_worthy",
            task_version=1,
            state="목표에 접근한다",
            question="장기 기억에 저장할 가치가 있는가?",
            target=True,
            language="ko",
            domain="test",
            source={},
            labels={},
            provenance={"prompt_version": 1, "split": "train"},
        )
        first = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=64,
            backbone="reference-char-bow-encoder",
        )
        second = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=64,
            backbone="reference-char-bow-encoder",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 64)
        self.assertGreater(len(set(first[0])), 2)

    def test_hybrid_bow_reference_encoder_keeps_word_and_character_buckets_separate(self) -> None:
        sample = CanonicalSample(
            sample_id="hybrid-bow",
            task_id="memory.remember_worthy",
            task_version=1,
            state="stable deployment decision",
            question="is this worth long-term memory?",
            target=True,
            language="en",
            domain="test",
            source={},
            labels={},
            provenance={"prompt_version": 1, "split": "train"},
        )
        token_ids, _attention = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=256,
            backbone="reference-hybrid-bow-encoder",
        )
        self.assertTrue(any(token_id < 16384 for token_id in token_ids))
        self.assertTrue(any(token_id >= 16384 for token_id in token_ids))

    def test_control_bow_reference_encoder_excludes_question_text(self) -> None:
        first = CanonicalSample(
            sample_id="control-bow-1",
            task_id="memory.remember_worthy",
            task_version=1,
            state="stable deployment decision",
            question="is this worth long-term memory?",
            target=True,
            language="en",
            domain="test",
            source={},
            labels={},
            provenance={"prompt_version": 2, "split": "train"},
        )
        second = replace(first, sample_id="control-bow-2", question="should this be retained?")
        first_ids, first_attention = _encode_reference_sample(
            first,
            vocab_size=32768,
            max_length=64,
            backbone="reference-control-bow-encoder",
        )
        second_ids, second_attention = _encode_reference_sample(
            second,
            vocab_size=32768,
            max_length=64,
            backbone="reference-control-bow-encoder",
        )
        self.assertEqual((first_ids, first_attention), (second_ids, second_attention))

        question_aware_first, _ = _encode_reference_sample(
            first,
            vocab_size=32768,
            max_length=64,
            backbone="reference-bow-encoder",
        )
        question_aware_second, _ = _encode_reference_sample(
            second,
            vocab_size=32768,
            max_length=64,
            backbone="reference-bow-encoder",
        )
        self.assertNotEqual(question_aware_first, question_aware_second)

    def test_segmented_bow_reference_encoder_separates_state_and_question_buckets(self) -> None:
        sample = CanonicalSample(
            sample_id="segmented-bow",
            task_id="memory.remember_worthy",
            task_version=1,
            state="stable deployment decision",
            question="stable deployment decision",
            target=True,
            language="en",
            domain="test",
            source={},
            labels={},
            provenance={"prompt_version": 2, "split": "train"},
        )
        token_ids, attention = _encode_reference_sample(
            sample,
            vocab_size=32768,
            max_length=64,
            backbone="reference-segmented-bow-encoder",
        )
        self.assertEqual(len(token_ids), 64)
        self.assertEqual(len(attention), 64)
        self.assertTrue(any(2 <= token_id < 16384 for token_id in token_ids))
        self.assertTrue(any(16384 <= token_id < 32768 for token_id in token_ids))
        self.assertEqual(
            len({token_id for token_id in token_ids if token_id >= 2}),
            len({token_id for token_id in token_ids if token_id >= 2 and token_id < 16384})
            + len({token_id for token_id in token_ids if token_id >= 16384}),
        )

    def test_sample_loss_weight_only_targets_counterfactual_provenance(self) -> None:
        training = TrainingConfig(hard_negative_weight=3.0)
        hard = CanonicalSample(
            sample_id="hard",
            task_id="memory.remember_worthy",
            task_version=1,
            state="state",
            question="question",
            target=True,
            language="en",
            domain="test",
            source={"counterfactual_group_id": "pair-1"},
            labels={},
            provenance={"prompt_version": 1, "split": "train"},
        )
        ordinary = CanonicalSample(
            sample_id="ordinary",
            task_id="memory.remember_worthy",
            task_version=1,
            state="state",
            question="question",
            target=True,
            language="en",
            domain="test",
            source={},
            labels={},
            provenance={"prompt_version": 1, "split": "train"},
        )
        self.assertEqual(_sample_loss_weight(hard, training), 3.0)
        self.assertEqual(_sample_loss_weight(ordinary, training), 1.0)

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

    def test_control_human_training_gate_rejects_synthetic_targets(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "control.jsonl"
            dataset.write_text(
                json.dumps(_record("control-synthetic", "control.skill", "STOP", "train", None))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TrainingDataError, "human label is required"):
                load_training_dataset(dataset, registry, require_human_labels=True)

    def test_control_human_training_gate_accepts_materialized_label(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "control-human.jsonl"
            record = _record("control-human", "control.skill", "STOP", "train", None)
            candidates = ["STOP", "HOLD", "MOVE", "ROTATE", "APPROACH", "RETREAT", "INTERACT", "RECOVER"]
            record["labels"] = {
                "human": {
                    "type": "choice",
                    "selected": "STOP",
                    "probabilities": {candidate: float(candidate == "STOP") for candidate in candidates},
                    "abstained": False,
                }
            }
            record["provenance"]["target_source"] = "human_review"
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            loaded = load_training_dataset(dataset, registry, require_human_labels=True)
        self.assertEqual(len(loaded.samples), 1)

    def test_control_train_exposes_human_label_gate(self) -> None:
        args = build_parser().parse_args(
            [
                "control",
                "train",
                "--dataset",
                "dataset.jsonl",
                "--output",
                "checkpoint.pt",
                "--require-human-labels",
            ]
        )
        self.assertTrue(args.require_human_labels)

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
