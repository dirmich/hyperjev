import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.cli import build_parser
from hyperjev.config import load_config
from hyperjev.control_data import generate_control_hard_negative_queue
from hyperjev.golden import generate_review_queue
from hyperjev.registry import TaskRegistry
from hyperjev.review_pack import (
    _prioritize_review_items,
    _teacher_priority,
    export_review_pack,
    run_review_session,
)

ROOT = Path(__file__).resolve().parents[2]


class ReviewPackTests(unittest.TestCase):
    def test_control_review_pack_exposes_priority_flag(self) -> None:
        args = build_parser().parse_args(
            [
                "control",
                "review-pack",
                "--queue",
                "queue.jsonl",
                "--draft",
                "draft.jsonl",
                "--output",
                "pack.jsonl",
                "--allow-raw",
                "--prioritize",
            ]
        )
        self.assertTrue(args.prioritize)

    def test_golden_review_pack_exposes_priority_flag(self) -> None:
        args = build_parser().parse_args(
            [
                "golden",
                "review-pack",
                "--draft",
                "draft.jsonl",
                "--allow-raw",
                "--prioritize",
            ]
        )
        self.assertTrue(args.prioritize)

    def test_teacher_priority_puts_invalid_repaired_and_low_confidence_first(self) -> None:
        invalid = {"sample_id": "invalid", "teacher": {"schema_valid": False}}
        repaired = {
            "sample_id": "repaired",
            "teacher": {"schema_valid": True, "schema_repaired": True},
        }
        low = {
            "sample_id": "low",
            "teacher": {
                "schema_valid": True,
                "normalized_result": {
                    "type": "choice",
                    "probabilities": {"a": 0.55, "b": 0.45},
                },
            },
        }
        high = {
            "sample_id": "high",
            "teacher": {
                "schema_valid": True,
                "normalized_result": {
                    "type": "choice",
                    "probabilities": {"a": 0.99, "b": 0.01},
                },
            },
        }
        ordered = sorted([high, low, repaired, invalid], key=_teacher_priority)
        self.assertEqual([item["sample_id"] for item in ordered], ["invalid", "repaired", "low", "high"])

    def test_pair_collision_precedes_low_confidence_group(self) -> None:
        collision = [
            {
                "sample_id": "collision-a",
                "source": {"counterfactual_group_id": "collision"},
                "teacher": {
                    "schema_valid": True,
                    "normalized_result": {
                        "type": "choice",
                        "selected": "STOP",
                        "probabilities": {"STOP": 1.0, "HOLD": 0.0},
                    },
                },
            },
            {
                "sample_id": "collision-b",
                "source": {"counterfactual_group_id": "collision"},
                "teacher": {
                    "schema_valid": True,
                    "normalized_result": {
                        "type": "choice",
                        "selected": "STOP",
                        "probabilities": {"STOP": 1.0, "HOLD": 0.0},
                    },
                },
            },
        ]
        low = [
            {
                "sample_id": "low",
                "source": {"counterfactual_group_id": "low"},
                "teacher": {
                    "schema_valid": True,
                    "normalized_result": {
                        "type": "choice",
                        "selected": "MOVE",
                        "probabilities": {"MOVE": 0.55, "HOLD": 0.45},
                    },
                },
            }
        ]
        ordered = _prioritize_review_items(low + collision)
        self.assertEqual([item["sample_id"] for item in ordered], ["collision-a", "collision-b", "low"])

    def test_prioritized_pack_keeps_counterfactual_pair_adjacent_without_target(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "control_tasks")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "hard.jsonl"
            draft = root / "draft.jsonl"
            output = root / "review-pack.jsonl"
            generate_control_hard_negative_queue(queue, registry, pair_count=2, seed=7)
            samples = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
            queue_sha256 = __import__("hashlib").sha256(queue.read_bytes()).hexdigest()
            draft_records = [
                {
                    "record_type": "golden_teacher_draft_manifest",
                    "queue_sha256": queue_sha256,
                    "provider": "qwen",
                    "model": "qwen38fn",
                    "prompt_version": 2,
                }
            ]
            for index, sample in enumerate(samples):
                confidence = 0.55 if index == 0 else 0.99
                draft_records.append(
                    {
                        "record_type": "golden_teacher_draft",
                        "sample_id": sample["sample_id"],
                        "provider": "qwen",
                        "model": "qwen38fn",
                        "normalized_result": {
                            "type": "choice",
                            "selected": sample["target"],
                            "probabilities": {"STOP": confidence, "MOVE": 1.0 - confidence},
                        },
                        "schema_valid": True,
                        "status": "completed",
                        "error": None,
                        "response_sha256": "hash",
                    }
                )
            draft.write_text(
                "\n".join(json.dumps(record) for record in draft_records) + "\n", encoding="utf-8"
            )
            export_review_pack(queue, draft, output, registry, include_raw=True, prioritize=True)
            output_text = output.read_text(encoding="utf-8")
            records = [json.loads(line) for line in output_text.splitlines()]

        items = records[1:]
        group_ids = [item["source"]["counterfactual_group_id"] for item in items]
        self.assertEqual(group_ids[:2], [group_ids[0], group_ids[0]])
        self.assertEqual(group_ids[2:], [group_ids[2], group_ids[2]])
        self.assertEqual(records[0]["priority_order"], "uncertain_first")
        self.assertNotIn('"target"', output_text)

    def test_pack_includes_context_but_excludes_target(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.jsonl"
            draft = root / "draft.jsonl"
            output = root / "review-pack.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            sample = json.loads(queue.read_text(encoding="utf-8"))
            queue_sha256 = __import__("hashlib").sha256(queue.read_bytes()).hexdigest()
            draft.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "golden_teacher_draft_manifest",
                                "queue_sha256": queue_sha256,
                                "provider": "qwen",
                                "model": "qwen38fn",
                                "prompt_version": 2,
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "golden_teacher_draft",
                                "sample_id": sample["sample_id"],
                                "provider": "qwen",
                                "model": "qwen38fn",
                                "normalized_result": {"type": "boolean", "value": True},
                                "schema_valid": True,
                                "status": "completed",
                                "error": None,
                                "response_sha256": "hash",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = export_review_pack(queue, draft, output, registry, include_raw=True)
            text = output.read_text(encoding="utf-8")
            records = [json.loads(line) for line in text.splitlines()]

        self.assertEqual(report["records"], 1)
        self.assertEqual(records[0]["target_excluded"], True)
        self.assertEqual(records[1]["state"], sample["state"])
        self.assertNotIn('"target"', text)
        self.assertNotIn('"labels"', text)

    def test_raw_context_requires_explicit_opt_in(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.jsonl"
            draft = root / "draft.jsonl"
            output = root / "review-pack.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            draft.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                export_review_pack(queue, draft, output, registry)

    def test_review_session_supports_accept_previous_edit_and_next(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.jsonl"
            draft = root / "draft.jsonl"
            pack = root / "review-pack.jsonl"
            feedback = root / "feedback.jsonl"
            generate_review_queue(queue, registry, count=2, seed=7)
            samples = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
            queue_sha256 = __import__("hashlib").sha256(queue.read_bytes()).hexdigest()
            draft_records = [
                {
                    "record_type": "golden_teacher_draft_manifest",
                    "queue_sha256": queue_sha256,
                    "provider": "qwen",
                    "model": "qwen38fn",
                    "prompt_version": 2,
                }
            ]
            for sample in samples:
                if sample["task_id"] == "memory.type":
                    normalized_result = {
                        "type": "choice",
                        "selected": "decision",
                        "probabilities": {
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
                        "abstained": False,
                    }
                else:
                    normalized_result = {
                        "type": "boolean",
                        "value": True,
                        "probability": 0.95,
                        "abstained": False,
                    }
                draft_records.append(
                    {
                        "record_type": "golden_teacher_draft",
                        "sample_id": sample["sample_id"],
                        "provider": "qwen",
                        "model": "qwen38fn",
                        "normalized_result": normalized_result,
                        "schema_valid": True,
                        "status": "completed",
                        "error": None,
                        "response_sha256": "hash",
                    }
                )
            draft.write_text(
                "\n".join(json.dumps(record) for record in draft_records) + "\n", encoding="utf-8"
            )
            export_review_pack(queue, draft, pack, registry, include_raw=True)
            answers = iter(["a", "p", "e", "false", "p", "n", "e", "decision"])
            report = run_review_session(
                pack,
                queue,
                feedback,
                registry,
                reviewer="tester",
                input_fn=lambda _prompt: next(answers),
                output_fn=lambda _message: None,
            )
            reviewed = Path(root / "reviewed.jsonl")
            from hyperjev.golden import apply_golden_feedback

            applied = apply_golden_feedback(queue, feedback, reviewed, registry)
            feedback_lines = feedback.read_text(encoding="utf-8").splitlines()
            reviewed_records = [
                json.loads(line) for line in reviewed.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(report["reviewed_count"], 2)
        self.assertEqual(report["pending_count"], 0)
        self.assertEqual(report["saved_in_session"], 3)
        self.assertEqual(len(feedback_lines), 3)
        self.assertEqual(applied["human_reviewed_count"], 2)
        self.assertFalse(reviewed_records[0]["labels"]["human"]["value"])
        self.assertEqual(reviewed_records[1]["labels"]["human"]["selected"], "decision")

    def test_review_session_treats_eof_as_safe_quit(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.jsonl"
            draft = root / "draft.jsonl"
            pack = root / "review-pack.jsonl"
            feedback = root / "feedback.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            sample = json.loads(queue.read_text(encoding="utf-8"))
            queue_sha256 = __import__("hashlib").sha256(queue.read_bytes()).hexdigest()
            draft.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "golden_teacher_draft_manifest",
                                "queue_sha256": queue_sha256,
                                "provider": "qwen",
                                "model": "qwen38fn",
                                "prompt_version": 2,
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "golden_teacher_draft",
                                "sample_id": sample["sample_id"],
                                "provider": "qwen",
                                "model": "qwen38fn",
                                "normalized_result": {
                                    "type": "boolean",
                                    "value": True,
                                    "probability": 0.95,
                                    "abstained": False,
                                },
                                "schema_valid": True,
                                "status": "completed",
                                "error": None,
                                "response_sha256": "hash",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            export_review_pack(queue, draft, pack, registry, include_raw=True)

            def eof(_prompt: str) -> str:
                raise EOFError

            report = run_review_session(
                pack,
                queue,
                feedback,
                registry,
                reviewer="tester",
                input_fn=eof,
                output_fn=lambda _message: None,
            )

        self.assertTrue(report["stopped"])
        self.assertEqual(report["saved_in_session"], 0)

    def test_review_session_can_propagate_exact_duplicate_labels(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.jsonl"
            draft = root / "draft.jsonl"
            pack = root / "review-pack.jsonl"
            feedback = root / "feedback.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            first = json.loads(queue.read_text(encoding="utf-8"))
            duplicate = dict(first)
            duplicate["sample_id"] = "phase0-synthetic-0002"
            queue.write_text(
                "\n".join(json.dumps(sample) for sample in (first, duplicate)) + "\n",
                encoding="utf-8",
            )
            queue_sha256 = __import__("hashlib").sha256(queue.read_bytes()).hexdigest()
            draft_records = [
                {
                    "record_type": "golden_teacher_draft_manifest",
                    "queue_sha256": queue_sha256,
                    "provider": "qwen",
                    "model": "qwen38fn",
                    "prompt_version": 2,
                }
            ]
            for sample_id in (first["sample_id"], duplicate["sample_id"]):
                draft_records.append(
                    {
                        "record_type": "golden_teacher_draft",
                        "sample_id": sample_id,
                        "provider": "qwen",
                        "model": "qwen38fn",
                        "normalized_result": {
                            "type": "boolean",
                            "value": True,
                            "probability": 0.95,
                            "abstained": False,
                        },
                        "schema_valid": True,
                        "status": "completed",
                        "error": None,
                        "response_sha256": "hash",
                    }
                )
            draft.write_text(
                "\n".join(json.dumps(record) for record in draft_records) + "\n",
                encoding="utf-8",
            )
            export_review_pack(queue, draft, pack, registry, include_raw=True)
            answers = iter(["a"])
            report = run_review_session(
                pack,
                queue,
                feedback,
                registry,
                reviewer="tester",
                input_fn=lambda _prompt: next(answers),
                output_fn=lambda _message: None,
                deduplicate_exact=True,
            )
            reviewed = root / "reviewed.jsonl"
            from hyperjev.golden import apply_golden_feedback

            applied = apply_golden_feedback(queue, feedback, reviewed, registry)
            feedback_lines = feedback.read_text(encoding="utf-8").splitlines()
            reviewed_records = [
                json.loads(line) for line in reviewed.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(report["reviewed_count"], 2)
        self.assertEqual(report["pending_count"], 0)
        self.assertEqual(report["saved_in_session"], 2)
        self.assertEqual(report["decisions_in_session"], 1)
        self.assertEqual(report["review_group_count"], 1)
        self.assertTrue(report["deduplicated_exact"])
        self.assertEqual(applied["human_reviewed_count"], 2)
        self.assertEqual(len(feedback_lines), 2)
        self.assertTrue(all(record["labels"]["human"]["value"] for record in reviewed_records))


if __name__ == "__main__":
    unittest.main()
