"""Application configuration loaded from config/config.json.

All tunables live in the JSON file; nothing is hardcoded in the modules.
Required keys fail fast at load time with a clear error message.
The Anthropic API key is read from the ANTHROPIC_API_KEY environment
variable only at the moment an API call is about to be made (see
`require_api_key`), so offline commands work without it. The CLI loads a
`.env` file from the project root at startup (see `load_env_file`); real
environment variables always win over the file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(path: Path | None = None) -> None:
    """Load variables from a .env file into the process environment.

    Variables already set in the real environment take precedence, so an
    exported ANTHROPIC_API_KEY always wins over the file. A missing file is
    silently ignored — the .env file is optional.
    """
    load_dotenv(dotenv_path=path or DEFAULT_ENV_PATH, override=False)


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
class ConversionConfig:
    """Where uploaded documents are turned into Markdown.

    `service_url` points at an optional external conversion service, kept in
    its own repository because the libraries that convert a PDF well are
    AGPL-licensed and this project is Apache-2.0. This project deliberately
    ships no converter of its own; an empty string means text-only mode, where
    only Markdown and plain text uploads are accepted.
    """

    service_url: str


@dataclass(frozen=True)
class AnthropicConfig:
    model: str
    effort: str
    max_tokens: int
    # How many detection requests the synchronous (web app) path keeps in
    # flight. The Batches API path is unaffected: it hands the whole batch to
    # the provider at once and has nothing to parallelize.
    detect_concurrency: int


DETECTION_PROVIDER_ANTHROPIC = "anthropic"
DETECTION_PROVIDER_LOCAL = "local"
VALID_DETECTION_PROVIDERS = frozenset(
    {DETECTION_PROVIDER_ANTHROPIC, DETECTION_PROVIDER_LOCAL}
)


@dataclass(frozen=True)
class LocalDetectionConfig:
    """Local OpenAI-compatible endpoint used when detection.provider is "local".

    Deliberately separate from synthetic.llm: the detection model and the
    prose model may differ, and switching the provider must not silently
    repoint the synthetic pipeline.
    """

    base_url: str
    model: str
    timeout_seconds: float
    concurrency: int


@dataclass(frozen=True)
class DetectionConfig:
    """Which backend detection requests go to."""

    provider: str
    local: LocalDetectionConfig


@dataclass(frozen=True)
class RedactionConfig:
    person: str
    company: str
    # Email addresses and URLs are regex-detectable, so they are replaced whole
    # and deterministically, without the detector ever seeing them.
    email: str
    url: str


@dataclass(frozen=True)
class LLMCatalogEntry:
    """One offered local model, with what it costs and how well it behaves.

    The figures are measurements, not promises: they were taken on one machine
    and live in the config so they can be re-measured and corrected without a
    code change.
    """

    model: str
    size: str
    seconds_per_fragment: float
    # Share of generations that pass validation on the first attempt; the rest
    # fall back to a deterministic template.
    first_try_validity: float
    note: str


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
    # The models the app offers to choose from, and one line about where the
    # numbers came from.
    catalog: tuple[LLMCatalogEntry, ...]
    catalog_note: str


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
    conversion: ConversionConfig
    anthropic: AnthropicConfig
    detection: DetectionConfig
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

    for section in (
        "chunking",
        "database",
        "conversion",
        "anthropic",
        "redaction",
        "synthetic",
        "detection",
    ):
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
    conversion = _load_conversion(raw["conversion"])
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
        email=_require(raw["redaction"], "email", "redaction"),
        url=_require(raw["redaction"], "url", "redaction"),
    )
    synthetic = _load_synthetic(raw["synthetic"])
    detection = _load_detection(raw["detection"])
    return AppConfig(
        chunking=chunking,
        database=database,
        conversion=conversion,
        anthropic=anthropic_cfg,
        detection=detection,
        redaction=redaction,
        synthetic=synthetic,
    )


def _load_detection(raw: dict) -> DetectionConfig:
    """Build the detection section, failing fast on any missing key.

    The provider is validated here rather than where it is first used, so a
    typo cannot select the remote backend by accident at detection time.
    """
    provider = _require(raw, "provider", "detection")
    if provider not in VALID_DETECTION_PROVIDERS:
        raise ConfigError(
            f"Config key detection.provider must be one of "
            f"{sorted(VALID_DETECTION_PROVIDERS)}, got {provider!r}"
        )
    raw_local = _require(raw, "local", "detection")
    if not isinstance(raw_local, dict):
        raise ConfigError(
            f"Config key detection.local must be an object, "
            f"got {type(raw_local).__name__}"
        )
    local = LocalDetectionConfig(
        base_url=_require(raw_local, "base_url", "detection.local"),
        model=_require(raw_local, "model", "detection.local"),
        timeout_seconds=float(
            _require(raw_local, "timeout_seconds", "detection.local")
        ),
        concurrency=int(_require(raw_local, "concurrency", "detection.local")),
    )
    return DetectionConfig(provider=provider, local=local)


def _load_conversion(raw: dict) -> ConversionConfig:
    """Build the conversion section.

    The key must be present - an absent one would silently decide that no
    service is in use - but an empty value is a legitimate answer meaning
    exactly that.
    """
    service_url = _require(raw, "service_url", "conversion")
    if not isinstance(service_url, str):
        raise ConfigError(
            f"Config key conversion.service_url must be a string, "
            f"got {type(service_url).__name__}"
        )
    return ConversionConfig(service_url=service_url.strip())


def _load_catalog(raw: object) -> tuple[LLMCatalogEntry, ...]:
    """Build the offered-model catalog, failing fast on any missing field.

    An entry short of a field would render as a blank figure in the settings
    view, which is worse than not offering the model at all.
    """
    if not isinstance(raw, list):
        raise ConfigError(
            f"Config key synthetic.llm.catalog must be a list, got {type(raw).__name__}"
        )
    entries = []
    for index, item in enumerate(raw):
        section_name = f"synthetic.llm.catalog[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(
                f"Config key {section_name} must be an object, "
                f"got {type(item).__name__}"
            )
        entries.append(
            LLMCatalogEntry(
                model=_require(item, "model", section_name),
                size=_require(item, "size", section_name),
                seconds_per_fragment=float(
                    _require(item, "seconds_per_fragment", section_name)
                ),
                first_try_validity=float(
                    _require(item, "first_try_validity", section_name)
                ),
                note=_require(item, "note", section_name),
            )
        )
    return tuple(entries)


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
        catalog=_load_catalog(_require(raw_llm, "catalog", "synthetic.llm")),
        catalog_note=_require(raw_llm, "catalog_note", "synthetic.llm"),
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
