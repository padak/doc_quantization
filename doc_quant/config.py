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
    # How far a chunk boundary may be pushed so that it never cuts through a
    # run of capitalized words; chunks are the detection unit, so a name has to
    # fit inside one of them.
    name_run_max_extension_tokens: int


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path


@dataclass(frozen=True)
class AnthropicConfig:
    model: str
    effort: str
    max_tokens: int
    # How many detection requests the synchronous (web app) path keeps in
    # flight. The Batches API path is unaffected: it hands the whole batch to
    # the provider at once and has nothing to parallelize.
    detect_concurrency: int


@dataclass(frozen=True)
class RedactionConfig:
    person: str
    company: str


@dataclass(frozen=True)
class SyntheticLLMConfig:
    """Local OpenAI-compatible endpoint used to phrase synthetic fragments.

    Any server speaking the OpenAI chat-completions protocol works: Ollama,
    LM Studio or llama.cpp's server. Nothing sensitive is sent there, but the
    endpoint is expected to stay on the machine running this tool.
    """

    enabled: bool
    base_url: str
    model: str
    temperature: float
    timeout_seconds: float


@dataclass(frozen=True)
class SyntheticConfig:
    """Canaries-and-chaff settings.

    honeytoken_rate, chaff_ratio and canaries_per_batch describe how many
    synthetic fragments ride along with a batch of real chunks; the three
    *_enabled flags switch the individual mechanisms off without removing the
    already generated fragments from the store.
    """

    honeytokens_enabled: bool
    chaff_enabled: bool
    canaries_enabled: bool
    chaff_ratio: float
    honeytoken_rate: float
    canary_set_size: int
    canaries_per_batch: int
    seed: int
    llm: SyntheticLLMConfig


@dataclass(frozen=True)
class AppConfig:
    chunking: ChunkingConfig
    database: DatabaseConfig
    anthropic: AnthropicConfig
    redaction: RedactionConfig
    synthetic: SyntheticConfig


def _require(section: dict, key: str, section_name: str):
    if key not in section:
        raise ConfigError(f"Missing required config key: {section_name}.{key}")
    return section[key]


def _require_bool(section: dict, key: str, section_name: str) -> bool:
    """Fetch a key that must be a real JSON boolean.

    Plain `bool()` would silently turn the string "false" into True, which is
    exactly the kind of quiet misreading these switches must not suffer from.
    """
    value = _require(section, key, section_name)
    if not isinstance(value, bool):
        raise ConfigError(
            f"Config key {section_name}.{key} must be a boolean, "
            f"got {type(value).__name__}"
        )
    return value


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)

    for section in ("chunking", "database", "anthropic", "redaction", "synthetic"):
        if section not in raw:
            raise ConfigError(f"Missing required config section: {section}")

    chunking = ChunkingConfig(
        chunk_size_tokens=int(_require(raw["chunking"], "chunk_size_tokens", "chunking")),
        encoding=_require(raw["chunking"], "encoding", "chunking"),
        name_run_max_extension_tokens=int(
            _require(raw["chunking"], "name_run_max_extension_tokens", "chunking")
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
        detect_concurrency=int(
            _require(raw["anthropic"], "detect_concurrency", "anthropic")
        ),
    )
    redaction = RedactionConfig(
        person=_require(raw["redaction"], "person", "redaction"),
        company=_require(raw["redaction"], "company", "redaction"),
    )
    synthetic = _load_synthetic(raw["synthetic"])
    return AppConfig(
        chunking=chunking,
        database=database,
        anthropic=anthropic_cfg,
        redaction=redaction,
        synthetic=synthetic,
    )


def _load_synthetic(raw: dict) -> SyntheticConfig:
    """Build the synthetic section, failing fast on any missing key."""
    raw_llm = _require(raw, "llm", "synthetic")
    if not isinstance(raw_llm, dict):
        raise ConfigError(
            f"Config key synthetic.llm must be an object, got {type(raw_llm).__name__}"
        )
    llm = SyntheticLLMConfig(
        enabled=_require_bool(raw_llm, "enabled", "synthetic.llm"),
        base_url=_require(raw_llm, "base_url", "synthetic.llm"),
        model=_require(raw_llm, "model", "synthetic.llm"),
        temperature=float(_require(raw_llm, "temperature", "synthetic.llm")),
        timeout_seconds=float(_require(raw_llm, "timeout_seconds", "synthetic.llm")),
    )
    return SyntheticConfig(
        honeytokens_enabled=_require_bool(raw, "honeytokens_enabled", "synthetic"),
        chaff_enabled=_require_bool(raw, "chaff_enabled", "synthetic"),
        canaries_enabled=_require_bool(raw, "canaries_enabled", "synthetic"),
        chaff_ratio=float(_require(raw, "chaff_ratio", "synthetic")),
        honeytoken_rate=float(_require(raw, "honeytoken_rate", "synthetic")),
        canary_set_size=int(_require(raw, "canary_set_size", "synthetic")),
        canaries_per_batch=int(_require(raw, "canaries_per_batch", "synthetic")),
        seed=int(_require(raw, "seed", "synthetic")),
        llm=llm,
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
