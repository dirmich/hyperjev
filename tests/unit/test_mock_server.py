import json
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from hyperjev.config import load_config
from hyperjev.mock_server import create_server
from hyperjev.registry import TaskRegistry

ROOT = Path(__file__).resolve().parents[2]


class MockServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        cls.server = create_server("127.0.0.1", 0, registry)
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


if __name__ == "__main__":
    unittest.main()
