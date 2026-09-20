import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.config import load_config
from hyperjev.golden import generate_review_queue
from hyperjev.registry import TaskRegistry
from hyperjev.review_pack import export_review_pack

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


if __name__ == "__main__":
    unittest.main()
