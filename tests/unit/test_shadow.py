import json
import tempfile
import unittest
from pathlib import Path

from hyperjev.contracts import BooleanDecision, DecisionResponse
from hyperjev.shadow import ShadowLogger


class ShadowLoggerTests(unittest.TestCase):
    def test_shadow_record_is_hash_only_and_tracks_agreement(self) -> None:
        primary = DecisionResponse(
            request_id="req-1",
            model="primary",
            calibration="cal-1",
            results={"remember": BooleanDecision(value=True, probability=0.9)},
            route="hyperjev",
            latency_ms=1.0,
        )
        shadow = DecisionResponse(
            request_id="req-1",
            model="shadow",
            calibration="cal-2",
            results={"remember": BooleanDecision(value=True, probability=0.9)},
            route="qwen",
            latency_ms=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.jsonl"
            record = ShadowLogger(path).append(
                request_id="req-1",
                state_sha256="abc123",
                primary=primary,
                shadow=shadow,
            )
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["comparisons"][0]["agree"], True)
        self.assertEqual(persisted["state_sha256"], "abc123")
        self.assertNotIn("probability", json.dumps(persisted))


if __name__ == "__main__":
    unittest.main()
