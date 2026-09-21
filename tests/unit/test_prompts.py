import unittest

from hyperjev.prompts import PROMPT_VERSION, messages_for_question, system_prompt
from hyperjev.registry import TaskRegistry


class PromptTests(unittest.TestCase):
    def test_control_prompt_contains_task_semantics(self) -> None:
        prompt = system_prompt("qwen", task_id="control.skill")

        self.assertEqual(PROMPT_VERSION, 3)
        self.assertIn("APPROACH means reducing distance to a reachable target", prompt)
        self.assertIn("STOP means an imminent hazard or unsafe state", prompt)

    def test_non_control_prompt_does_not_receive_control_boundaries(self) -> None:
        prompt = system_prompt("qwen", task_id="memory.type")

        self.assertNotIn("APPROACH means reducing distance", prompt)

    def test_message_builder_passes_task_id_to_system_prompt(self) -> None:
        registry = TaskRegistry.load("registry/control_tasks")
        task = registry.get("control.skill", 1)

        messages = messages_for_question(
            "qwen",
            task,
            state="The route is clear.",
            question="What should the controller do next?",
        )

        self.assertIn("ROTATE means changing heading", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
