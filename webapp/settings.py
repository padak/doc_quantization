"""User settings layered on top of `config/config.json`.

`config/config.json` is the checked-in default; `data/settings.json` holds what
the user changed in the web app. The two are combined into an ordinary
`AppConfig`, so every module downstream keeps reading its tunables from one
place and never has to know that an override existed.

The Anthropic API key is the one setting that is not part of `AppConfig`: it is
a secret, so it is carried separately, never written into any response, and
falls back to the ANTHROPIC_API_KEY environment variable when the settings file
does not carry one.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

from doc_quant.config import (
    PROJECT_ROOT,
    AnthropicConfig,
    AppConfig,
    ConfigError,
    SyntheticLLMConfig,
)

logger = logging.getLogger(__name__)

# The settings file lives inside the gitignored data/ directory: it may hold an
# API key, so it must never become a tracked file.
SETTINGS_DIR = PROJECT_ROOT / "data"
DEFAULT_SETTINGS_PATH = SETTINGS_DIR / "settings.json"

API_KEY_SETTING = "anthropic_api_key"
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

# Every key a user may override. Anything else found in the file is ignored
# rather than rejected, so a settings file written by a newer build does not
# break an older one.
SETTINGS_KEYS: tuple[str, ...] = (
    API_KEY_SETTING,
    "model",
    "effort",
    "llm_base_url",
    "llm_model",
)

# Structural constants of the mask, not tunables: how much of a key stays
# readable so a user can tell which key is stored, and the elision between the
# two ends. A key too short to mask this way is reported as fully elided.
MASK_VISIBLE_CHARS = 4
MASK_SEPARATOR = "..."

MISSING_API_KEY_MESSAGE = (
    "Missing Anthropic API key: store one via PUT /api/settings or set the "
    f"{API_KEY_ENV_VAR} environment variable."
)


def load_overrides(path: Path) -> dict[str, str]:
    """Read the user overrides from `path`.

    A missing file is normal and yields an empty mapping. Empty string values
    are dropped here as well as on write, so "cleared" and "never set" behave
    identically no matter how the file came to be.

    Raises:
        ConfigError: when the file exists but is not a JSON object of strings.
    """
    if not path.exists():
        logger.debug("No settings file at %s; using config defaults only", path)
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Settings file {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Settings file {path} must contain a JSON object, "
            f"got {type(raw).__name__}"
        )

    overrides: dict[str, str] = {}
    for key in SETTINGS_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ConfigError(
                f"Settings key {key} must be a string, got {type(value).__name__}"
            )
        if value:
            overrides[key] = value

    unknown = sorted(set(raw) - set(SETTINGS_KEYS))
    if unknown:
        logger.debug("Ignoring unknown settings keys in %s: %s", path, ", ".join(unknown))
    return overrides


def save_overrides(path: Path, updates: dict[str, str | None]) -> dict[str, str]:
    """Merge `updates` into the stored overrides and write them back.

    A key mapped to None or to the empty string is removed, which is how the
    user clears an override (including the API key) and falls back to the
    config default or the environment.

    Returns:
        The merged overrides as they were persisted.

    Raises:
        ConfigError: when `updates` carries a key that is not a known setting,
            or when the existing file cannot be read.
    """
    unknown = sorted(set(updates) - set(SETTINGS_KEYS))
    if unknown:
        raise ConfigError(f"Unknown settings keys: {', '.join(unknown)}")

    merged = load_overrides(path)
    for key, value in updates.items():
        if not value:
            merged.pop(key, None)
        else:
            merged[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Names only: the values include the API key and must not reach a log file.
    logger.info("Saved settings overrides to %s: %s", path, ", ".join(sorted(merged)))
    return merged


def effective_config(config: AppConfig, overrides: dict[str, str]) -> AppConfig:
    """Return `config` with the user overrides applied.

    Only the four non-secret settings are part of the returned config; the API
    key is handled by `effective_api_key`, so an `AppConfig` never carries a
    secret and can be logged or inspected freely.
    """
    anthropic_config = AnthropicConfig(
        model=overrides.get("model") or config.anthropic.model,
        effort=overrides.get("effort") or config.anthropic.effort,
        max_tokens=config.anthropic.max_tokens,
    )
    llm_config = SyntheticLLMConfig(
        base_url=overrides.get("llm_base_url") or config.synthetic.llm.base_url,
        model=overrides.get("llm_model") or config.synthetic.llm.model,
        temperature=config.synthetic.llm.temperature,
        timeout_seconds=config.synthetic.llm.timeout_seconds,
    )
    return replace(
        config,
        anthropic=anthropic_config,
        synthetic=replace(config.synthetic, llm=llm_config),
    )


def effective_api_key(overrides: dict[str, str]) -> str | None:
    """Return the API key to use, or None when none is configured.

    The stored setting wins over the environment: a key typed into the app is
    the more deliberate of the two.
    """
    stored = overrides.get(API_KEY_SETTING)
    if stored:
        return stored
    from_env = os.environ.get(API_KEY_ENV_VAR)
    return from_env or None


def mask_api_key(key: str | None) -> str | None:
    """Render a key as "sk-a...wxyz", or None when there is no key.

    Enough to recognise which key is stored, never enough to use it. A key too
    short to show both ends without overlapping is elided completely rather
    than partially revealed.
    """
    if not key:
        return None
    if len(key) < 2 * MASK_VISIBLE_CHARS:
        return MASK_SEPARATOR
    return f"{key[:MASK_VISIBLE_CHARS]}{MASK_SEPARATOR}{key[-MASK_VISIBLE_CHARS:]}"
