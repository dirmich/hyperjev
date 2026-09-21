import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hyperjev.baseline import run_benchmark
from hyperjev.config import load_config
from hyperjev.registry import TaskRegistry
from hyperjev.teachers import TeacherClient, _message_content, _model_ids, probe_teacher

ROOT = Path(__file__).resolve().parents[2]


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"data": [{"id": "test-model"}]}).encode()


class TeacherTests(unittest.TestCase):
    def test_generation_payload_includes_optional_reasoning_effort(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        settings = config.teachers["gemma"]
        settings = settings.__class__(
            name=settings.name,
            base_url=settings.base_url,
            model=settings.model,
            roles=settings.roles,
            request_timeout_s=settings.request_timeout_s,
            disable_thinking=settings.disable_thinking,
            reasoning_effort="none",
            response_format=settings.response_format,
        )
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = json.dumps(
            {
                "model": settings.model,
                "choices": [{"message": {"content": '{"value":false}'}}],
                "usage": {},
            }
        ).encode()
        with patch("hyperjev.teachers.urlopen", return_value=response) as open_url:
            TeacherClient(settings).complete([{"role": "user", "content": "test"}])
        payload = json.loads(open_url.call_args.args[0].data.decode())
        self.assertEqual(payload["reasoning_effort"], "none")

    def test_model_ids_supports_openai_and_ollama_shapes(self) -> None:
        self.assertEqual(_model_ids({"data": [{"id": "a"}, {"id": "b"}]}), ("a", "b"))
        self.assertEqual(_model_ids({"models": [{"name": "a"}]}), ("a",))

    def test_message_content_supports_function_call_arguments(self) -> None:
        self.assertEqual(
            _message_content(
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"arguments": '{"value":true}'}}],
                    }
                }
            ),
            '{"value":true}',
        )
        self.assertEqual(_message_content({"message": {"content": ""}}), "")

    def test_probe_reports_model_match(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        settings = config.teachers["qwen"]
        with patch("hyperjev.teachers.urlopen", return_value=_Response()):
            probe = probe_teacher(settings)
        self.assertTrue(probe.ok is False)
        self.assertFalse(probe.model_found)
        self.assertIn("configured model", probe.error or "")

    def test_generation_client_uses_long_configured_timeout(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        client = TeacherClient(config.teachers["gemma"])
        self.assertEqual(client.timeout_s, 900.0)


class BaselineTests(unittest.TestCase):
    def test_dry_run_writes_reproducible_shape_without_raw_text(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        registry = TaskRegistry.load(config.registry_path)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            run = run_benchmark(
                config,
                registry,
                limit=1,
                providers=("qwen", "gemma"),
                dry_run=True,
                output_path=output,
            )
            output_text = output.read_text(encoding="utf-8")
            lines = output_text.splitlines()
        self.assertEqual(len(run.records), 2)
        self.assertEqual(len(lines), 3)
        self.assertEqual(run.manifest["teacher_options"]["gemma"]["reasoning_effort"], None)
        self.assertTrue(all(record["status"] == "dry_run" for record in run.records))
        self.assertNotIn("PostgreSQL", output_text)


if __name__ == "__main__":
    unittest.main()
