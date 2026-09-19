import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.config import load_config
from hyperjev.dataset_factory import build_dataset, validate_dataset
from hyperjev.registry import TaskRegistry
from hyperjev.teachers import TeacherCompletion

ROOT = Path(__file__).resolve().parents[2]


class _FakeClient:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls = 0

    def complete(self, _messages: list[dict[str, str]]) -> TeacherCompletion:
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return TeacherCompletion(
            content=content,
            model="fake-teacher",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            elapsed_ms=1.0,
        )


def _boolean(value: bool, probability: float) -> str:
    return json.dumps(
        {"type": "boolean", "value": value, "probability": probability, "abstained": False}
    )


def _choice(selected: str, probability: float) -> str:
    candidates = ["DUPLICATE", "UPDATE", "EXTEND", "CONTRADICT", "RELATED", "NONE"]
    remainder = (1.0 - probability) / (len(candidates) - 1)
    return json.dumps(
        {
            "type": "choice",
            "selected": selected,
            "probabilities": {
                candidate: probability if candidate == selected else remainder
                for candidate in candidates
            },
            "abstained": False,
        }
    )


class DatasetFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "phase0.toml")
        self.registry = TaskRegistry.load(self.config.registry_path)

    def test_build_filters_redacts_deduplicates_and_splits(self) -> None:
        seeds = [
            {
                "sample_id": "silver-1",
                "task": "memory.remember_worthy@1",
                "state": "결정 내용은 owner@example.com, 전화 010-1234-5678",
                "question": "장기 기억으로 저장할 가치가 있는가?",
                "language": "ko",
                "domain": "conversation",
                "source": {"kind": "synthetic", "document_id": "doc-a"},
            },
            {
                "sample_id": "duplicate-1",
                "task": "memory.remember_worthy@1",
                "state": "결정 내용은 owner@example.com, 전화 010-1234-5678",
                "question": "장기 기억으로 저장할 가치가 있는가?",
                "language": "ko",
                "domain": "conversation",
                "source": {"kind": "synthetic", "document_id": "doc-b"},
            },
            {
                "sample_id": "disagreement-1",
                "task": "memory.relation@1",
                "state": "기존 DB를 새 DB로 교체했다.",
                "question": "두 기억의 관계는 무엇인가?",
                "language": "ko",
                "domain": "conversation",
                "source": {"kind": "synthetic", "document_id": "doc-c"},
            },
            {
                "sample_id": "human-1",
                "task": "memory.remember_worthy@1",
                "state": "사람이 검수한 결정",
                "question": "remember",
                "language": "ko",
                "domain": "conversation",
                "source": {"kind": "synthetic", "document_id": "doc-d"},
                "labels": {
                    "human": {"type": "boolean", "value": True, "probability": 1.0}
                },
            },
            {
                "sample_id": "private-1",
                "task": "memory.remember_worthy@1",
                "state": "사용자 입력",
                "question": "remember",
                "language": "ko",
                "domain": "conversation",
                "source": {"kind": "user", "document_id": "doc-private"},
            },
            {
                "sample_id": "low-1",
                "task": "memory.remember_worthy@1",
                "state": "애매한 입력",
                "question": "remember",
                "language": "ko",
                "domain": "conversation",
                "source": {"kind": "synthetic", "document_id": "doc-e"},
            },
        ]
        qwen = _FakeClient(
            _boolean(True, 0.98),
            _choice("UPDATE", 0.9),
            "not-json",
            _boolean(True, 0.4),
        )
        gemma = _FakeClient(
            _boolean(True, 0.96),
            _choice("CONTRADICT", 0.9),
            "not-json",
            _boolean(False, 0.4),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_path = root / "seed.jsonl"
            seed_path.write_text(
                "\n".join(json.dumps(seed, ensure_ascii=False) for seed in seeds) + "\n",
                encoding="utf-8",
            )
            output_path = root / "dataset.jsonl"
            report = build_dataset(
                self.config,
                self.registry,
                seed_path,
                output_path,
                clients={"qwen": qwen, "gemma": gemma},
            )
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            reviews = [
                json.loads(line)
                for line in report.review_path.read_text(encoding="utf-8").splitlines()
            ]
            validation = validate_dataset(output_path, self.registry)
        self.assertEqual(report.counts["accepted_silver"], 1)
        self.assertEqual(report.counts["accepted_human"], 1)
        self.assertEqual(report.counts["excluded_duplicate"], 1)
        self.assertEqual(report.counts["excluded_privacy"], 1)
        self.assertEqual(report.counts["review_disagreement"], 1)
        self.assertEqual(report.counts["excluded_low_confidence"], 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(reviews), 2)
        self.assertNotIn("owner@example.com", json.dumps(records, ensure_ascii=False))
        self.assertIn("[EMAIL]", json.dumps(records, ensure_ascii=False))
        self.assertTrue(all(record["provenance"]["split"] in {"train", "validation", "test"} for record in records))
        self.assertEqual(validation["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
