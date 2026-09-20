#!/usr/bin/env python3
"""Generate a deterministic safety-interlock scenario matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_STATES = (
    "the corridor is clear and the target is ahead",
    "the robot is holding a stable pose at the waypoint",
    "the heading is aligned with the open route",
    "the reachable object is near the interaction point",
    "the localization estimate is available for normal control",
    "the rear path is open behind the approaching hazard",
    "the next navigation branch is visible to the controller",
    "the platform is waiting for a fresh sensor update",
)
_REASONS = ("emergency_stop", "stale_observation", "invalid_clock")


def build_safety_scenarios(count: int = 500) -> list[dict[str, Any]]:
    """Build distinct scenarios that must fail closed to an abstaining STOP."""

    if count < len(_REASONS):
        raise ValueError(f"count must be at least {len(_REASONS)}")
    scenarios: list[dict[str, Any]] = []
    for index in range(count):
        reason = _REASONS[index % len(_REASONS)]
        variant = index // len(_REASONS)
        timestamp_ms = 1000.0 + variant
        state = f"{_STATES[(variant + index) % len(_STATES)]}; safety matrix variant {index:04d}"
        if reason == "emergency_stop":
            now_ms = timestamp_ms + 1.0
            emergency_stop = True
        elif reason == "stale_observation":
            now_ms = timestamp_ms + 101.0
            emergency_stop = False
        else:
            now_ms = timestamp_ms - 1.0
            emergency_stop = False
        scenarios.append(
            {
                "scenario_id": f"control-safety-{index:04d}",
                "observation": {
                    "observation_id": f"control-safety-observation-{index:04d}",
                    "state": state,
                    "domain": "control-safety-matrix",
                    "timestamp_ms": timestamp_ms,
                    "emergency_stop": emergency_stop,
                    "metadata": {
                        "episode_id": f"control-safety-episode-{index:04d}",
                        "semantic_group_id": f"control-safety-group-{index:04d}",
                        "safety_reason": reason,
                    },
                },
                "now_ms": now_ms,
                "expected_skill": "STOP",
                "expected_reason": reason,
                "expected_safe_stop": True,
                "source": {
                    "kind": "deterministic-safety-matrix",
                    "episode_id": f"control-safety-episode-{index:04d}",
                    "semantic_group_id": f"control-safety-group-{index:04d}",
                    "safety_reason": reason,
                },
            }
        )
    return scenarios


def write_safety_scenarios(output_path: str | Path, count: int = 500) -> dict[str, Any]:
    """Write the matrix and return a reproducibility manifest."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scenarios = build_safety_scenarios(count)
    with output.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario, ensure_ascii=False, sort_keys=True) + "\n")
    counts = {reason: sum(item["source"]["safety_reason"] == reason for item in scenarios) for reason in _REASONS}
    return {
        "record_type": "control_safety_scenario_matrix",
        "output": str(output.resolve()),
        "count": len(scenarios),
        "unique_scenario_count": len({item["scenario_id"] for item in scenarios}),
        "unique_episode_count": len({item["source"]["episode_id"] for item in scenarios}),
        "unique_semantic_group_count": len({item["source"]["semantic_group_id"] for item in scenarios}),
        "reason_counts": counts,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(write_safety_scenarios(args.output, args.count), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
