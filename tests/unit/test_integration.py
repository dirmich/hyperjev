import tempfile
import unittest
from pathlib import Path

from hyperjev.config import load_config
from hyperjev.integration import HyperMemoryClient, IngestionGate
from hyperjev.registry import TaskRegistry
from hyperjev.review import ReviewStore
from hyperjev.routing import DecisionRouter
from hyperjev.teachers import TeacherCompletion

ROOT = Path(__file__).resolve().parents[2]


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, _messages: list[dict[str, str]]) -> TeacherCompletion:
        return TeacherCompletion(
            content=self.content,
            model="fake",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            elapsed_ms=1.0,
        )


class _FakeMemoryClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ingest_document(self, content: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"content": content, **kwargs})
        return {"document_id": "doc-test", "extracted_memories": 1}


class _FailingMemoryClient:
    def ingest_document(self, _content: str, **_kwargs: object) -> dict[str, object]:
        raise OSError("connection refused")


class IngestionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "phase0.toml")
        self.registry = TaskRegistry.load(self.config.registry_path)

    def _gate(self, qwen: _FakeClient, gemma: _FakeClient, client: object | None = None):
        directory = tempfile.TemporaryDirectory()
        router = DecisionRouter(
            self.config,
            self.registry,
            clients={"qwen": qwen, "gemma": gemma},
            review_store=ReviewStore(Path(directory.name) / "review.jsonl"),
        )
        return IngestionGate(router, client), directory

    def test_accepted_extract_is_the_only_external_write_path(self) -> None:
        memory = _FakeMemoryClient()
        gate, directory = self._gate(
            _FakeClient('{"type":"boolean","value":true,"probability":0.99}'),
            _FakeClient('{"type":"boolean","value":true,"probability":0.99}'),
            memory,
        )
        try:
            decision = gate.ingest(
                "프로젝트 DB를 PostgreSQL에서 ClickHouse로 변경하기로 했다.",
                container="hyperjev",
                title="database decision",
            )
        finally:
            directory.cleanup()
        self.assertEqual(decision.status, "extract")
        self.assertEqual(decision.route, "rule")
        self.assertEqual(len(memory.calls), 1)
        self.assertEqual(memory.calls[0]["container"], "hyperjev")

    def test_low_value_input_is_retained_without_external_write(self) -> None:
        memory = _FakeMemoryClient()
        gate, directory = self._gate(
            _FakeClient('{"type":"boolean","value":true,"probability":0.99}'),
            _FakeClient('{"type":"boolean","value":true,"probability":0.99}'),
            memory,
        )
        try:
            decision = gate.ingest("안녕, 고마워!", container="hyperjev")
        finally:
            directory.cleanup()
        self.assertEqual(decision.status, "retain_raw")
        self.assertEqual(len(memory.calls), 0)

    def test_uncertain_decision_becomes_review(self) -> None:
        memory = _FakeMemoryClient()
        gate, directory = self._gate(_FakeClient("not-json"), _FakeClient("not-json"), memory)
        try:
            decision = gate.ingest("판단이 어려운 입력", container="hyperjev")
        finally:
            directory.cleanup()
        self.assertEqual(decision.status, "review")
        self.assertEqual(len(memory.calls), 0)
        self.assertTrue(decision.remember.abstained)  # type: ignore[union-attr]

    def test_hypermemory_failure_does_not_claim_ingestion(self) -> None:
        gate, directory = self._gate(
            _FakeClient('{"type":"boolean","value":true,"probability":0.99}'),
            _FakeClient('{"type":"boolean","value":true,"probability":0.99}'),
            _FailingMemoryClient(),
        )
        try:
            decision = gate.ingest("결정: Rust로 구현하기로 했다.")
        finally:
            directory.cleanup()
        self.assertEqual(decision.status, "review")
        self.assertEqual(decision.reason, "hypermemory_unavailable")
        self.assertIn("connection refused", decision.error or "")


class HyperMemoryClientTests(unittest.TestCase):
    def test_client_uses_documents_contract(self) -> None:
        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"document_id":"doc-1"}'

        from unittest.mock import patch

        with patch("hyperjev.integration.urlopen", return_value=_Response()) as opener:
            result = HyperMemoryClient("http://memory:6767").ingest_document(
                "safe content", container="project", source_type="conversation"
            )
        request = opener.call_args.args[0]
        self.assertEqual(result["document_id"], "doc-1")
        self.assertIn(b'"content": "safe content"', request.data)
        self.assertIn("/v1/documents", request.full_url)


if __name__ == "__main__":
    unittest.main()
