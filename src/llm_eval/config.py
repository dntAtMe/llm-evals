"""Settings from .env and YAML scenario loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

from llm_eval.models import Scenario


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a YAML scenario file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Scenario.model_validate(raw)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
