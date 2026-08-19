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

from doc_quant.config import PROJECT_ROOT, AppConfig, ConfigError

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
BOOL_SETTINGS_KEYS: tuple[str, ...] = ("llm_enabled",)

# String keys whose empty value is a decision rather than an absence, so they
# are stored by presence: an empty conversion service URL says "convert with the
# text-only mode", which is a different statement from "no override, take
# whatever config/config.json says". Every other string key falls back to the
# config once it is cleared, which is why an empty one is dropped there.
PRESENCE_SETTINGS_KEYS: tuple[str, ...] = ("conversion_service_url",)

SETTINGS_KEYS: tuple[str, ...] = (
    API_KEY_SETTING,
    "model",
    "effort",
    "llm_base_url",
    "llm_model",
    *PRESENCE_SETTINGS_KEYS,
    *BOOL_SETTINGS_KEYS,
)

# A setting value is a string, except for the switches, which are real
# booleans: "false" read as a string would be a true value, and a switch that
# silently means its opposite is worse than no switch at all.
SettingValue = str | bool

# Structural constants of the mask, not tunables: how much of a key stays
# readable so a user can tell which key is stored, and the elision between the
# two ends. A key too short to mask this way is reported as fully elided.
MASK_VISIBLE_CHARS = 4
MASK_SEPARATOR = "..."

MISSING_API_KEY_MESSAGE = (
    "Missing Anthropic API key: store one via PUT /api/settings or set the "
    f"{API_KEY_ENV_VAR} environment variable."
)


def load_overrides(path: Path) -> dict[str, SettingValue]:
    """Read the user overrides from `path`.

    A missing file is normal and yields an empty mapping. Empty string values
    are dropped here as well as on write, so "cleared" and "never set" behave
    identically no matter how the file came to be. A switch (see
    `BOOL_SETTINGS_KEYS`) is kept whichever way it points: False is a decision,
    not an absent value. So is an empty value of a presence key (see
    `PRESENCE_SETTINGS_KEYS`), which is kept for the same reason.

    Raises:
        ConfigError: when the file exists but is not a JSON object whose values
            have the type their key calls for.
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

    overrides: dict[str, SettingValue] = {}
    for key in SETTINGS_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if key in BOOL_SETTINGS_KEYS:
            if not isinstance(value, bool):
                raise ConfigError(
                    f"Settings key {key} must be a boolean, got {type(value).__name__}"
                )
            overrides[key] = value
            continue
        if not isinstance(value, str):
            raise ConfigError(
                f"Settings key {key} must be a string, got {type(value).__name__}"
            )
        if value or key in PRESENCE_SETTINGS_KEYS:
            overrides[key] = value

    unknown = sorted(set(raw) - set(SETTINGS_KEYS))
    if unknown:
        logger.debug("Ignoring unknown settings keys in %s: %s", path, ", ".join(unknown))
    return overrides


def save_overrides(
    path: Path, updates: dict[str, SettingValue | None]
) -> dict[str, SettingValue]:
    """Merge `updates` into the stored overrides and write them back.

    A key mapped to None or to the empty string is removed, which is how the
    user clears an override (including the API key) and falls back to the
    config default or the environment. A switch set to False is stored rather
    than removed: switching something off is an override like any other, and so
    is the empty value of a presence key (see `PRESENCE_SETTINGS_KEYS`).

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
        if value is None or (value == "" and key not in PRESENCE_SETTINGS_KEYS):
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


def effective_config(config: AppConfig, overrides: dict[str, SettingValue]) -> AppConfig:
    """Return `config` with the user overrides applied.

    Only the non-secret settings are part of the returned config; the API key
    is handled by `effective_api_key`, so an `AppConfig` never carries a secret
    and can be logged or inspected freely.

    The overridable fields are patched onto the loaded dataclasses rather than
    rebuilt from scratch, so a tunable that is not overridable here keeps
    whatever `config/config.json` said about it.
    """
    anthropic_config = replace(
        config.anthropic,
        model=_text(overrides, "model") or config.anthropic.model,
        effort=_text(overrides, "effort") or config.anthropic.effort,
    )
    llm_config = replace(
        config.synthetic.llm,
        enabled=effective_llm_enabled(config, overrides),
        base_url=_text(overrides, "llm_base_url") or config.synthetic.llm.base_url,
        model=_text(overrides, "llm_model") or config.synthetic.llm.model,
    )
    conversion_config = replace(
        config.conversion,
        service_url=effective_conversion_service_url(config, overrides),
    )
    return replace(
        config,
        anthropic=anthropic_config,
        conversion=conversion_config,
        synthetic=replace(config.synthetic, llm=llm_config),
    )


def effective_llm_enabled(config: AppConfig, overrides: dict[str, SettingValue]) -> bool:
    """Whether the local LLM may be contacted at all.

    With it off, `doc_quant.synthetic` phrases every fragment from its
    deterministic templates, so the pipeline runs with nothing installed
    locally.
    """
    stored = overrides.get("llm_enabled")
    if isinstance(stored, bool):
        return stored
    return config.synthetic.llm.enabled


def effective_conversion_service_url(
    config: AppConfig, overrides: dict[str, SettingValue]
) -> str:
    """Where documents are converted: an external service, or nothing.

    An empty result means text-only mode (no converter). The stored setting
    wins whenever it is present, empty included - a user who cleared the field
    asked for the built-in converter, which the config's own URL must not undo.
    """
    stored = overrides.get("conversion_service_url")
    if isinstance(stored, str):
        return stored
    return config.conversion.service_url


def effective_api_key(overrides: dict[str, SettingValue]) -> str | None:
    """Return the API key to use, or None when none is configured.

    The stored setting wins over the environment: a key typed into the app is
    the more deliberate of the two.
    """
    stored = _text(overrides, API_KEY_SETTING)
    if stored:
        return stored
    from_env = os.environ.get(API_KEY_ENV_VAR)
    return from_env or None


def _text(overrides: dict[str, SettingValue], key: str) -> str | None:
    """Return a string-valued override, or None when it is absent."""
    value = overrides.get(key)
    return value if isinstance(value, str) and value else None


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
