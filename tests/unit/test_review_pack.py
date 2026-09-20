import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.config import load_config
from hyperjev.golden import generate_review_queue
from hyperjev.registry import TaskRegistry
from hyperjev.review_pack import export_review_pack, run_review_session

ROOT = Path(__file__).resolve().parents[2]


class ReviewPackTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
