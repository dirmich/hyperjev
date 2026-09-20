import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.config import load_config
from hyperjev.contracts import DecisionRequest
from hyperjev.registry import TaskRegistry
from hyperjev.review import FeedbackStore, ReviewStore
from hyperjev.routing import DecisionRouter
from hyperjev.rules import match_rule
from hyperjev.teachers import TeacherCompletion

ROOT = Path(__file__).resolve().parents[2]


class _FakeClient:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls = 0

    def complete(self, _messages: list[dict[str, str]]) -> TeacherCompletion:
        self.calls += 1
        content = self.contents[min(self.calls - 1, len(self.contents) - 1)]
        return TeacherCompletion(
            content=content,
            model=f"fake-{self.calls}",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            elapsed_ms=1.0,
        )


class _FakeStudent:
    model_name = "fake-student"

    def complete_decision(self, task, *, state: str, question: str, candidates: list[str]):
        del state, question
        if task.output_type == "boolean":
            content = '{"type":"boolean","value":true,"probability":0.99}'
        elif task.output_type == "choice":
            selected = candidates[0] if candidates else str(task.output["candidates"][0])
            values = candidates or [str(item) for item in task.output["candidates"]]
            probability = 1.0 / len(values)
            content = json.dumps(
                {
                    "type": "choice",
                    "selected": selected,
                    "probabilities": {value: probability for value in values},
                }
            )
        else:
            content = '{"type":"score","value":0.5,"interval_90":[0.4,0.6]}'
        return TeacherCompletion(
            content=content,
            model=self.model_name,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            elapsed_ms=0.1,
        )


class _AbstainingStudent(_FakeStudent):
    model_name = "fake-abstaining-student"

    def complete_decision(self, task, *, state: str, question: str, candidates: list[str]):
        del task, state, question, candidates
        return TeacherCompletion(
            content='{"type":"boolean","value":true,"probability":0.99,"abstained":true}',
            model=self.model_name,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            elapsed_ms=0.1,
        )


def _request(task: str, *, state: str = "중립적인 입력", question_id: str = "q") -> DecisionRequest:
    return DecisionRequest.from_dict(
        {
            "state": state,
            "questions": [{"id": question_id, "task": task}],
            "options": {"deadline_ms": 5000},
        }
    )


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "phase0.toml")
        self.registry = TaskRegistry.load(self.config.registry_path)

    def _router(self, qwen: _FakeClient, gemma: _FakeClient) -> tuple[DecisionRouter, tempfile.TemporaryDirectory[str]]:
        directory = tempfile.TemporaryDirectory()
        router = DecisionRouter(
            self.config,
            self.registry,
            clients={"qwen": qwen, "gemma": gemma},
            review_store=ReviewStore(Path(directory.name) / "review.jsonl"),
        )
        return router, directory

    def test_high_precision_rule_stops_before_teachers(self) -> None:
        qwen = _FakeClient('{"type":"boolean","value":false,"probability":0.99}')
        gemma = _FakeClient('{"type":"boolean","value":false,"probability":0.99}')
        router, directory = self._router(qwen, gemma)
        try:
            outcome = router.decide(
                _request(
                    "memory.remember_worthy@1",
                    state="안녕, 고마워!",
                    question_id="remember",
                )
            )
        finally:
            directory.cleanup()
        self.assertEqual(outcome.response.route, "rule")
        self.assertFalse(outcome.response.results["remember"].value)  # type: ignore[union-attr]
        self.assertEqual(qwen.calls, 0)
        self.assertEqual(gemma.calls, 0)
        self.assertEqual(outcome.traces[0].attempts[0]["rule_id"], "remember.greeting_only")

    def test_wiki_style_change_does_not_match_factual_change_keyword(self) -> None:
        task = self.registry.get("wiki.semantic_change", 1)
        match = match_rule(
            task,
            state="Only the document title style changed.",
            question="What changed?",
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.rule_id, "wiki.style_only")  # type: ignore[union-attr]
        self.assertFalse(match.result.value)  # type: ignore[union-attr]

    def test_qwen_acceptance_does_not_call_gemma(self) -> None:
        qwen = _FakeClient('{"type":"score","value":0.86,"interval_90":[0.8,0.9]}')
        gemma = _FakeClient('{"type":"score","value":0.2,"interval_90":[0.1,0.3]}')
        router, directory = self._router(qwen, gemma)
        try:
            outcome = router.decide(_request("memory.importance@1", question_id="importance"))
        finally:
            directory.cleanup()
        self.assertEqual(outcome.response.route, "qwen")
        self.assertEqual(qwen.calls, 1)
        self.assertEqual(gemma.calls, 0)
        self.assertFalse(outcome.traces[0].review_required)

    def test_student_acceptance_precedes_parent_teachers(self) -> None:
        qwen = _FakeClient('{"type":"boolean","value":false,"probability":0.99}')
        gemma = _FakeClient('{"type":"boolean","value":false,"probability":0.99}')
        directory = tempfile.TemporaryDirectory()
        router = DecisionRouter(
            self.config,
            self.registry,
            clients={"student": _FakeStudent(), "qwen": qwen, "gemma": gemma},
            review_store=ReviewStore(Path(directory.name) / "review.jsonl"),
        )
        try:
            outcome = router.decide(_request("memory.remember_worthy@1", state="일반적인 정보에 대한 문장"))
        finally:
            directory.cleanup()
        self.assertEqual(outcome.response.route, "student")
        self.assertEqual(outcome.traces[0].attempts[0]["provider"], "student")
        self.assertEqual(qwen.calls, 0)
        self.assertEqual(gemma.calls, 0)

    def test_student_abstention_falls_through_to_qwen(self) -> None:
        qwen = _FakeClient('{"type":"boolean","value":false,"probability":0.99}')
        gemma = _FakeClient('{"type":"boolean","value":false,"probability":0.99}')
        directory = tempfile.TemporaryDirectory()
        router = DecisionRouter(
            self.config,
            self.registry,
            clients={"student": _AbstainingStudent(), "qwen": qwen, "gemma": gemma},
            review_store=ReviewStore(Path(directory.name) / "review.jsonl"),
        )
        try:
            outcome = router.decide(_request("memory.remember_worthy@1", state="일반적인 정보에 대한 문장"))
        finally:
            directory.cleanup()
        self.assertEqual(outcome.response.route, "qwen")
        self.assertEqual(qwen.calls, 1)
        self.assertEqual(gemma.calls, 0)
        self.assertEqual(outcome.traces[0].attempts[0]["status"], "uncertain")

    def test_uncertain_qwen_is_cross_validated_by_gemma(self) -> None:
        qwen = _FakeClient(
            '{"type":"choice","selected":"decision","probabilities":'
            '{"fact":0.05,"preference":0.05,"episode":0.05,"decision":0.60,'
            '"goal":0.05,"task":0.05,"relationship":0.05,"temporary":0.05,"none":0.05}}'
        )
        gemma = _FakeClient(
            '{"type":"choice","selected":"decision","probabilities":'
            '{"fact":0.02,"preference":0.02,"episode":0.02,"decision":0.82,'
            '"goal":0.02,"task":0.02,"relationship":0.02,"temporary":0.02,"none":0.04}}'
        )
        router, directory = self._router(qwen, gemma)
        try:
            outcome = router.decide(_request("memory.type@1", question_id="type"))
        finally:
            directory.cleanup()
        self.assertEqual(outcome.response.route, "gemma")
        self.assertEqual(qwen.calls, 1)
        self.assertEqual(gemma.calls, 1)
        self.assertEqual(outcome.traces[0].attempts[0]["status"], "uncertain")
        self.assertTrue(outcome.traces[0].attempts[1]["accepted"])

    def test_invalid_qwen_falls_through_to_valid_gemma(self) -> None:
        qwen = _FakeClient("not-json")
        gemma = _FakeClient('{"type":"boolean","value":true,"probability":0.97}')
        router, directory = self._router(qwen, gemma)
        try:
            outcome = router.decide(
                _request("memory.remember_worthy@1", state="무난한 내용", question_id="remember")
            )
        finally:
            directory.cleanup()
        self.assertEqual(outcome.response.route, "gemma")
        self.assertFalse(outcome.traces[0].attempts[0]["schema_valid"])

    def test_exhausted_chain_creates_review_without_raw_state(self) -> None:
        qwen = _FakeClient("not-json")
        gemma = _FakeClient("also-not-json")
        directory = tempfile.TemporaryDirectory()
        review_path = Path(directory.name) / "review.jsonl"
        router = DecisionRouter(
            self.config,
            self.registry,
            clients={"qwen": qwen, "gemma": gemma},
            review_store=ReviewStore(review_path),
        )
        secret_state = "민감한 원문을 review 파일에 쓰면 안 된다"
        outcome = router.decide(
            _request("memory.remember_worthy@1", state=secret_state, question_id="remember")
        )
        review_text = review_path.read_text(encoding="utf-8")
        record = json.loads(review_text)
        directory.cleanup()
        self.assertEqual(outcome.response.route, "human")
        self.assertTrue(outcome.response.results["remember"].abstained)  # type: ignore[union-attr]
        self.assertNotIn(secret_state, review_text)
        self.assertEqual(record["state_sha256"], outcome.traces[0].state_sha256)
        self.assertTrue(record["trace"]["review_required"])


class FeedbackTests(unittest.TestCase):
    def test_feedback_is_schema_validated_and_append_only(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        task = registry.get("memory.remember_worthy", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            store = FeedbackStore(path)
            record = store.append(
                request_id="req-1",
                question_id="remember",
                task=task,
                correction={"type": "boolean", "value": True, "probability": 1.0},
                reviewer="human@example",
                reason="explicit decision",
            )
            self.assertEqual(record["correction"]["type"], "boolean")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
