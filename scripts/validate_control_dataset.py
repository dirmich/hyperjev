#!/usr/bin/env python3
"""Validate control dataset provenance and split leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hyperjev.control_data import validate_control_dataset
from hyperjev.registry import TaskRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("registry/control_tasks"))
    parser.add_argument("--require-human-labels", action="store_true")
    args = parser.parse_args()
    report = validate_control_dataset(
        args.dataset,
        TaskRegistry.load(args.registry),
        require_human_labels=args.require_human_labels,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
