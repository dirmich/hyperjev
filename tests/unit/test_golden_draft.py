import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperjev.config import load_config
from hyperjev.golden import generate_review_queue
from hyperjev.golden_draft import generate_gemma_draft
from hyperjev.registry import TaskRegistry
from hyperjev.teachers import TeacherCompletion

ROOT = Path(__file__).resolve().parents[2]


class _FakeGemma:
    def __init__(self, _settings, timeout_s=None) -> None:
        self.timeout_s = timeout_s

    def complete(self, _messages):
        return TeacherCompletion(
            content='{"type":"boolean","value":true,"probability":0.99,"abstained":false}',
            model="fake-gemma-4",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            elapsed_ms=12.5,
        )


class GoldenDraftTests(unittest.TestCase):
    def test_gemma_draft_is_queue_bound_and_does_not_store_raw_state(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.jsonl"
            output = Path(directory) / "draft.jsonl"
            generate_review_queue(queue, registry, count=1, seed=7)
            with patch("hyperjev.golden_draft.TeacherClient", _FakeGemma):
                report = generate_gemma_draft(config, registry, queue, output, limit=1)
            output_text = output.read_text(encoding="utf-8")
            lines = output_text.splitlines()

        manifest = json.loads(lines[0])
        record = json.loads(lines[1])
        self.assertEqual(report["manifest"]["schema_valid_count"], 1)
        self.assertEqual(manifest["queue_sha256"], report["manifest"]["queue_sha256"])
        self.assertEqual(record["normalized_result"]["type"], "boolean")
        self.assertNotIn("다음 분기부터", output_text)
        self.assertNotIn('"target"', output_text)


if __name__ == "__main__":
    unittest.main()
