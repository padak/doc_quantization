"""Application configuration loaded from config/config.json.

All tunables live in the JSON file; nothing is hardcoded in the modules.
Required keys fail fast at load time with a clear error message.
The Anthropic API key is read from the ANTHROPIC_API_KEY environment
variable only at the moment an API call is about to be made (see
`require_api_key`), so offline commands work without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


class ConfigError(Exception):
    """Raised when required configuration is missing or malformed."""


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int
    encoding: str
    detection_margin_tokens: int


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path


@dataclass(frozen=True)
class AnthropicConfig:
    model: str
    effort: str
    max_tokens: int


@dataclass(frozen=True)
class RedactionConfig:
    person: str
    company: str


@dataclass(frozen=True)
class AppConfig:
    chunking: ChunkingConfig
    database: DatabaseConfig
    anthropic: AnthropicConfig
    redaction: RedactionConfig


def _require(section: dict, key: str, section_name: str):
    if key not in section:
        raise ConfigError(f"Missing required config key: {section_name}.{key}")
    return section[key]


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)

    for section in ("chunking", "database", "anthropic", "redaction"):
        if section not in raw:
            raise ConfigError(f"Missing required config section: {section}")

    chunking = ChunkingConfig(
        chunk_size_tokens=int(_require(raw["chunking"], "chunk_size_tokens", "chunking")),
        encoding=_require(raw["chunking"], "encoding", "chunking"),
        detection_margin_tokens=int(
            _require(raw["chunking"], "detection_margin_tokens", "chunking")
        ),
    )
    db_path = Path(_require(raw["database"], "path", "database"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    database = DatabaseConfig(path=db_path)
    anthropic_cfg = AnthropicConfig(
        model=_require(raw["anthropic"], "model", "anthropic"),
        effort=_require(raw["anthropic"], "effort", "anthropic"),
        max_tokens=int(_require(raw["anthropic"], "max_tokens", "anthropic")),
    )
    redaction = RedactionConfig(
        person=_require(raw["redaction"], "person", "redaction"),
        company=_require(raw["redaction"], "company", "redaction"),
    )
    return AppConfig(
        chunking=chunking,
        database=database,
        anthropic=anthropic_cfg,
        redaction=redaction,
    )


def require_api_key() -> str:
    """Fail fast with a clear error when the API key is missing."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ConfigError(
            "Missing required environment variable: ANTHROPIC_API_KEY "
            "(needed only for batch submission/fetch; offline commands work without it)"
        )
    return key
