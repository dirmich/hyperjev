import tempfile
import unittest
from pathlib import Path

from hyperjev.cache import BoundedCache
from hyperjev.config import load_config
from hyperjev.contracts import DecisionRequest
from hyperjev.registry import TaskRegistry
from hyperjev.review import ReviewStore
from hyperjev.routing import DecisionRouter
from hyperjev.teachers import TeacherCompletion

ROOT = Path(__file__).resolve().parents[2]


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _messages: list[dict[str, str]]) -> TeacherCompletion:
        self.calls += 1
        return TeacherCompletion(
            content='{"type":"score","value":0.8,"interval_90":[0.7,0.9]}',
            model="fake",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            elapsed_ms=1.0,
        )


class RouterCacheTests(unittest.TestCase):
    def test_accepted_outcome_is_reused_without_repeating_teacher_call(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        qwen = _FakeClient()
        request = DecisionRequest.from_dict(
            {
                "state": "cacheable score",
                "questions": [{"id": "importance", "task": "memory.importance@1"}],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            router = DecisionRouter(
                config,
                registry,
                clients={"qwen": qwen, "gemma": _FakeClient()},
                review_store=ReviewStore(Path(directory) / "review.jsonl"),
                cache=BoundedCache(max_entries=4),
            )
            first = router.decide(request)
            second = router.decide(request)
        self.assertEqual(first.response.to_dict(), second.response.to_dict())
        self.assertEqual(qwen.calls, 1)


if __name__ == "__main__":
    unittest.main()
