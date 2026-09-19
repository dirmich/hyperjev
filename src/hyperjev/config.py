"""Configuration loading for the Phase 0 local development environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 on the DGX Spark image
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(ValueError):
    """Raised when the HyperJev configuration is invalid."""


@dataclass(frozen=True)
class TeacherSettings:
    """OpenAI-compatible teacher endpoint and its product roles."""

    name: str
    base_url: str
    model: str
    roles: tuple[str, ...]
    request_timeout_s: float = 300.0
    disable_thinking: bool = False
    response_format: str = "json_object"

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, Any]) -> TeacherSettings:
        base_url = str(values.get("base_url", "")).rstrip("/")
        model = str(values.get("model", "")).strip()
        raw_roles = values.get("roles", ())
        roles = tuple(str(role) for role in raw_roles)
        request_timeout_s = float(values.get("request_timeout_s", 300.0))
        disable_thinking = bool(values.get("disable_thinking", False))
        response_format = str(values.get("response_format", "json_object"))
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"teachers.{name}.base_url must be an http(s) URL")
        if not model:
            raise ConfigError(f"teachers.{name}.model must not be empty")
        if not roles:
            raise ConfigError(f"teachers.{name}.roles must not be empty")
        if request_timeout_s <= 0:
            raise ConfigError(f"teachers.{name}.request_timeout_s must be positive")
        if response_format not in {"json_object", "json_schema", "text"}:
            raise ConfigError(f"teachers.{name}.response_format is unsupported: {response_format}")
        return cls(
            name=name,
            base_url=base_url,
            model=model,
            roles=roles,
            request_timeout_s=request_timeout_s,
            disable_thinking=disable_thinking,
            response_format=response_format,
        )


@dataclass(frozen=True)
class Phase0Config:
    """Validated configuration used by Phase 0 commands."""

    root: Path
    project_name: str
    phase: int
    model_track: str
    diffusion_track: str
    server_host: str
    server_port: int
    registry_path: Path
    smoke_set_path: Path
    runs_path: Path
    teachers: Mapping[str, TeacherSettings]

    @classmethod
    def from_file(cls, path: str | Path) -> Phase0Config:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigError(f"configuration file not found: {config_path}")
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        project = raw.get("project", {})
        server = raw.get("server", {})
        paths = raw.get("paths", {})
        raw_teachers = raw.get("teachers", {})
        teachers: dict[str, TeacherSettings] = {}
        for name, values in raw_teachers.items():
            teachers[name] = TeacherSettings.from_mapping(name, values)

        env_overrides = {
            "qwen": {
                "base_url": os.getenv("HYPERJEV_QWEN_BASE_URL"),
                "model": os.getenv("HYPERJEV_QWEN_MODEL"),
            },
            "gemma": {
                "base_url": os.getenv("HYPERJEV_GEMMA_BASE_URL"),
                "model": os.getenv("HYPERJEV_GEMMA_MODEL"),
            },
        }
        for name, overrides in env_overrides.items():
            if name not in teachers:
                continue
            current = teachers[name]
            values = {
                "base_url": overrides["base_url"] or current.base_url,
                "model": overrides["model"] or current.model,
                "roles": current.roles,
                "request_timeout_s": current.request_timeout_s,
                "disable_thinking": current.disable_thinking,
                "response_format": current.response_format,
            }
            teachers[name] = TeacherSettings.from_mapping(name, values)

        if not teachers.get("qwen") or not teachers.get("gemma"):
            raise ConfigError("both qwen and gemma teacher configurations are required")
        return cls(
            root=root,
            project_name=str(project.get("name", "hyperjev")),
            phase=int(project.get("phase", 0)),
            model_track=str(project.get("model_track", "encoder-typed-heads")),
            diffusion_track=str(project.get("diffusion_track", "HyperJev-D")),
            server_host=str(server.get("host", "127.0.0.1")),
            server_port=int(server.get("port", 6777)),
            registry_path=(root / str(paths.get("registry", "registry/tasks"))).resolve(),
            smoke_set_path=(root / str(paths.get("phase0_smoke_set", "tests/golden/phase0_smoke.jsonl"))).resolve(),
            runs_path=(root / str(paths.get("runs", "runs/phase0"))).resolve(),
            teachers=teachers,
        )


def default_config_path() -> Path:
    """Return the repository's default Phase 0 configuration path."""

    return Path(__file__).resolve().parents[2] / "configs" / "phase0.toml"


def load_config(path: str | Path | None = None) -> Phase0Config:
    """Load the configured environment, honoring ``HYPERJEV_CONFIG``."""

    selected = path or os.getenv("HYPERJEV_CONFIG") or default_config_path()
    return Phase0Config.from_file(selected)
