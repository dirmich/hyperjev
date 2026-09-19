import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperjev.config import ConfigError, load_config
from hyperjev.registry import RegistryError, TaskRegistry

ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_phase0_config_uses_observed_teacher_defaults(self) -> None:
        config = load_config(ROOT / "configs" / "phase0.toml")
        self.assertEqual(config.phase, 0)
        self.assertEqual(config.model_track, "encoder-typed-heads")
        self.assertEqual(config.diffusion_track, "HyperJev-D")
        self.assertEqual(config.teachers["qwen"].base_url, "http://127.0.0.1:8081/v1")
        self.assertEqual(config.teachers["gemma"].base_url, "http://macmini:11434/v1")
        self.assertEqual(config.teachers["qwen"].request_timeout_s, 300.0)
        self.assertEqual(config.teachers["gemma"].request_timeout_s, 300.0)
        self.assertTrue(config.teachers["qwen"].disable_thinking)
        self.assertFalse(config.teachers["gemma"].disable_thinking)
        self.assertEqual(config.teachers["qwen"].response_format, "json_object")
        self.assertEqual(config.teachers["gemma"].response_format, "text")
        self.assertEqual(config.teachers["qwen"].roles, ("data_generator", "labeler", "fallback"))
        self.assertEqual(config.teachers["gemma"].roles, ("cross_validator", "judge"))

    def test_teacher_environment_overrides_are_scoped(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HYPERJEV_QWEN_MODEL": "test-qwen",
                "HYPERJEV_GEMMA_BASE_URL": "http://example.test/v1",
            },
            clear=False,
        ):
            config = load_config(ROOT / "configs" / "phase0.toml")
        self.assertEqual(config.teachers["qwen"].model, "test-qwen")
        self.assertEqual(config.teachers["gemma"].base_url, "http://example.test/v1")

    def test_missing_config_is_explicit(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(ROOT / "configs" / "missing.toml")


class RegistryTests(unittest.TestCase):
    def test_six_phase0_tasks_load_and_are_sorted(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        self.assertEqual(len(registry), 6)
        self.assertEqual(
            registry.ids(),
            (
                "memory.importance",
                "memory.relation",
                "memory.remember_worthy",
                "memory.type",
                "query.route",
                "wiki.semantic_change",
            ),
        )
        self.assertEqual(registry.get("memory.type", version=1).output_type, "choice")

    def test_unknown_or_wrong_version_task_fails(self) -> None:
        registry = TaskRegistry.load(ROOT / "registry" / "tasks")
        with self.assertRaises(RegistryError):
            registry.get("missing.task")
        with self.assertRaises(RegistryError):
            registry.get("memory.type", version=2)


if __name__ == "__main__":
    unittest.main()
