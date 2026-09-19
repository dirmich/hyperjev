"""Read-only local environment checks for DGX Spark Phase 0."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _command_detail(command: list[str], timeout_s: float = 3.0) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, f"{command[0]} not found on PATH"
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, output[0] if output else f"exit code {result.returncode}"


def system_checks(root: str | Path) -> tuple[SystemCheck, ...]:
    """Collect checks without changing the machine or starting services."""

    root_path = Path(root).resolve()
    checks: list[SystemCheck] = []
    architecture = platform.machine()
    checks.append(SystemCheck("architecture", architecture == "aarch64", architecture))
    docker_ok, docker_detail = _command_detail(["docker", "--version"])
    checks.append(SystemCheck("docker", docker_ok, docker_detail))
    gpu_ok, gpu_detail = _command_detail(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    checks.append(SystemCheck("nvidia", gpu_ok, gpu_detail))
    try:
        free_gib = shutil.disk_usage(root_path).free / (1024**3)
        checks.append(SystemCheck("disk_free", free_gib >= 1.0, f"{free_gib:.1f} GiB free"))
    except OSError as exc:
        checks.append(SystemCheck("disk_free", False, str(exc)))
    return tuple(checks)
