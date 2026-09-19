import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from hyperjev.config import load_config
from hyperjev.mock_server import create_server
from hyperjev.monitoring import TrafficMonitor
from hyperjev.registry import TaskRegistry
from hyperjev.review import ReviewStore
from hyperjev.routing import DecisionRouter

ROOT = Path(__file__).resolve().parents[2]


class MockServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        cls.monitor = TrafficMonitor()
        cls.server = create_server("127.0.0.1", 0, registry, monitor=cls.monitor)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def test_health_and_tasks(self) -> None:
        with urlopen(f"{self.base_url}/health/ready", timeout=2) as response:
            health = json.loads(response.read())
        with urlopen(f"{self.base_url}/v1/tasks", timeout=2) as response:
            tasks = json.loads(response.read())
        self.assertEqual(health["status"], "ready")
        self.assertEqual(len(tasks["tasks"]), 6)

    def test_v1_registry_validation_and_model_listing_endpoints(self) -> None:
        with urlopen(f"{self.base_url}/health", timeout=2) as response:
            health = json.loads(response.read())
        request = Request(
            f"{self.base_url}/v1/tasks/validate",
            data=json.dumps({"tasks": ["memory.type@1", "memory.importance@1"]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            validation = json.loads(response.read())
        with urlopen(f"{self.base_url}/v1/models", timeout=2) as response:
            models = json.loads(response.read())
        self.assertEqual(health["status"], "ok")
        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["tasks"]), 2)
        self.assertEqual(models["models"], [])

    def test_decide_returns_typed_abstentions(self) -> None:
        body = json.dumps(
            {
                "state": "새로운 상태",
                "questions": [
                    {"id": "remember", "task": "memory.remember_worthy@1"},
                    {"id": "type", "task": "memory.type@1"},
                    {"id": "importance", "task": "memory.importance@1"},
                ],
            }
        ).encode()
        request = Request(
            f"{self.base_url}/v1/decide",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["route"], "mock")
        self.assertEqual(payload["results"]["remember"]["type"], "boolean")
        self.assertEqual(payload["results"]["type"]["type"], "choice")
        self.assertEqual(payload["results"]["importance"]["type"], "score")
        self.assertTrue(payload["results"]["remember"]["abstained"])

    def test_router_mode_runs_rule_and_optional_provenance(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            router = DecisionRouter(
                config,
                registry,
                review_store=ReviewStore(Path(directory) / "review.jsonl"),
            )
            server = create_server("127.0.0.1", 0, registry, router=router)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                body = json.dumps(
                    {
                        "state": "우리는 Rust로 구현하기로 했다.",
                        "questions": [
                            {"id": "remember", "task": "memory.remember_worthy@1"},
                        ],
                        "options": {"return_evidence": True},
                    }
                ).encode()
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/v1/decide",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    payload = json.loads(response.read())
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()
        self.assertEqual(payload["route"], "rule")
        self.assertTrue(payload["results"]["remember"]["value"])
        self.assertEqual(payload["traces"][0]["final_route"], "rule")

    def test_batch_decide_reuses_one_http_contract_for_multiple_states(self) -> None:
        body = json.dumps(
            {
                "requests": [
                    {
                        "state": "첫 상태",
                        "questions": [{"id": "remember", "task": "memory.remember_worthy@1"}],
                    },
                    {
                        "state": "둘째 상태",
                        "questions": [{"id": "importance", "task": "memory.importance@1"}],
                    },
                ]
            }
        ).encode()
        request = Request(
            f"{self.base_url}/v1/batch/decide",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        self.assertEqual(len(payload["responses"]), 2)
        self.assertEqual(payload["responses"][0]["route"], "mock")
        self.assertEqual(payload["responses"][1]["results"]["importance"]["type"], "score")

        with urlopen(f"{self.base_url}/metrics", timeout=2) as response:
            metrics = json.loads(response.read())
        self.assertGreaterEqual(metrics["total_requests"], 1)
        self.assertGreaterEqual(metrics["total_questions"], 1)


if __name__ == "__main__":
    unittest.main()
