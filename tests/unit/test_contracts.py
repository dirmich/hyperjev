import unittest

from hyperjev.contracts import (
    BooleanDecision,
    ContractError,
    DecisionRequest,
    DecisionResponse,
    parse_decision_result,
    parse_task_reference,
    validate_result_for_task,
)
from hyperjev.registry import TaskRegistry


class ContractTests(unittest.TestCase):
    def test_request_accepts_up_to_typed_question_batch(self) -> None:
        request = DecisionRequest.from_dict(
            {
                "state": "프로젝트 DB를 변경하기로 했다.",
                "context": {"workspace_id": "hyper-memory"},
                "questions": [
                    {"id": "remember", "task": "memory.remember_worthy@1"},
                    {"id": "type", "task": "memory.type@1"},
                ],
                "options": {"deadline_ms": 500},
            }
        )
        self.assertEqual(len(request.questions), 2)
        self.assertEqual(request.questions[0].task, "memory.remember_worthy@1")

    def test_request_rejects_duplicate_question_ids_and_invalid_task(self) -> None:
        base = {"state": "x", "questions": [{"id": "same", "task": "memory.type@1"}]}
        with self.assertRaises(ContractError):
            DecisionRequest.from_dict({**base, "questions": [base["questions"][0], base["questions"][0]]})
        with self.assertRaises(ContractError):
            DecisionRequest.from_dict({**base, "questions": [{"id": "x", "task": "memory.type"}]})

    def test_boolean_alias_normalizes_to_boolean(self) -> None:
        result = parse_decision_result(
            {"type": "noul", "value": True, "probability": 0.973, "abstained": False}
        )
        self.assertIsInstance(result, BooleanDecision)
        self.assertEqual(result.to_dict()["type"], "boolean")

    def test_choice_and_score_reject_malformed_probability_data(self) -> None:
        with self.assertRaises(ContractError):
            parse_decision_result(
                {
                    "type": "choice",
                    "selected": "decision",
                    "probabilities": {"fact": 0.9, "decision": 0.9},
                }
            )
        with self.assertRaises(ContractError):
            parse_decision_result(
                {"type": "score", "value": 0.9, "interval_90": [0.1, 0.8]}
            )

    def test_response_round_trip(self) -> None:
        response = DecisionResponse.from_dict(
            {
                "request_id": "req-1",
                "model": "phase0-mock",
                "calibration": "none",
                "results": {
                    "remember": {
                        "type": "boolean",
                        "value": True,
                        "probability": 0.99,
                        "abstained": False,
                    }
                },
                "route": "mock",
                "latency_ms": 1.5,
            }
        )
        self.assertEqual(DecisionResponse.from_dict(response.to_dict()), response)

    def test_task_result_matches_registry_type_and_candidates(self) -> None:
        registry = TaskRegistry.load("registry/tasks")
        task = registry.get("memory.type", 1)
        result = parse_decision_result(
            {
                "type": "choice",
                "selected": "decision",
                "probabilities": {
                    "fact": 0.05,
                    "preference": 0.05,
                    "episode": 0.05,
                    "decision": 0.80,
                    "goal": 0.01,
                    "task": 0.01,
                    "relationship": 0.01,
                    "temporary": 0.01,
                    "none": 0.01,
                },
            }
        )
        validate_result_for_task(task, result)
        with self.assertRaises(ContractError):
            validate_result_for_task(task, BooleanDecision(value=True, probability=0.9))

    def test_task_reference_parser(self) -> None:
        self.assertEqual(parse_task_reference("memory.type@1"), ("memory.type", 1))
        with self.assertRaises(ContractError):
            parse_task_reference("memory.type")


if __name__ == "__main__":
    unittest.main()
