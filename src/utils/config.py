"""
Configuration loader for Google AI Platform Engineering.
Loads and validates gcp_config.yaml with environment variable substitution.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _substitute_env_vars(value: str) -> str:
    pattern = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    return pattern.sub(replacer, value)


def _process_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _process_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_process_dict(item) for item in data]
    elif isinstance(data, str):
        return _substitute_env_vars(data)
    return data


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and return GCP platform config with env var substitution."""
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "gcp_config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return _process_dict(raw)


class VertexAIConfig(BaseModel):
    location: str = Field(default="australia-southeast1")
    staging_bucket: str = Field(default="")
    models: dict[str, str] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)


class GCPPlatformConfig(BaseModel):
    project_id: str = Field(default="")
    region: str = Field(default="australia-southeast1")
    vertex_ai: VertexAIConfig = Field(default_factory=VertexAIConfig)

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "GCPPlatformConfig":
        raw = load_config(config_path)
        gcp = raw.get("gcp", {})
        vai = raw.get("vertex_ai", {})
        return cls(
            project_id=gcp.get("project_id", ""),
            region=gcp.get("region", "australia-southeast1"),
            vertex_ai=VertexAIConfig(
                location=vai.get("location", "australia-southeast1"),
                staging_bucket=vai.get("staging_bucket", ""),
                models=vai.get("models", {}),
                inference=vai.get("inference", {}),
            ),
        )

    def get_model(self, key: str) -> str:
        model = self.vertex_ai.models.get(key)
        if not model:
            available = list(self.vertex_ai.models.keys())
            raise KeyError(f"Model '{key}' not found. Available: {available}")
        return model


_config: GCPPlatformConfig | None = None

def get_config() -> GCPPlatformConfig:
    global _config
    if _config is None:
        _config = GCPPlatformConfig.from_yaml()
    return _config
