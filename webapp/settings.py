"""User settings layered on top of `config/config.json`.

`config/config.json` is the checked-in default; `data/settings.json` holds what
the user changed in the web app. The two are combined into an ordinary
`AppConfig`, so every module downstream keeps reading its tunables from one
place and never has to know that an override existed.

Overridable settings fall into four shapes, each validated and merged its own
way:

* free-text strings (`model`, `llm_base_url`, ...) - an empty value clears
  the override and falls back to the config;
* presence strings (see `PRESENCE_SETTINGS_KEYS`) - an empty value is itself a
  stored decision, not a clear;
* switches (see `BOOL_SETTINGS_KEYS`) - `True`/`False` are both stored
  decisions; there is no "empty" boolean;
* numbers (see `NUMERIC_SETTINGS_SPECS`) - range-checked on both read and
  write, and only `None` clears one; `0` is a legitimate stored value.

The Anthropic API key is the one setting that is not part of `AppConfig`: it is
a secret, so it is carried separately, never written into any response, and
falls back to the ANTHROPIC_API_KEY environment variable when the settings file
does not carry one.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
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
BOOL_SETTINGS_KEYS: tuple[str, ...] = (
    "llm_enabled",
    "honeytokens_enabled",
    "chaff_enabled",
    "canaries_enabled",
)

# String keys whose empty value is a decision rather than an absence, so they
# are stored by presence: an empty conversion service URL says "convert with the
# built-in markitdown", which is a different statement from "no override, take
# whatever config/config.json says". Every other string key falls back to the
# config once it is cleared, which is why an empty one is dropped there.
PRESENCE_SETTINGS_KEYS: tuple[str, ...] = ("conversion_service_url",)


@dataclass(frozen=True)
class NumericSpec:
    """Validation rule for one numeric override key.

    Read and write go through the same spec so a value that would be rejected
    on `PUT /api/settings` can never end up sitting in the settings file some
    other way (a hand-edited file, an older build) and only be discovered
    invalid the next time it is read - or worse, silently coerced.
    """

    kind: type  # int or float
    minimum: int | float


NUMERIC_SETTINGS_SPECS: dict[str, NumericSpec] = {
    "chunk_size_tokens": NumericSpec(kind=int, minimum=1),
    "canaries_per_batch": NumericSpec(kind=int, minimum=0),
    "chaff_ratio": NumericSpec(kind=float, minimum=0),
    "honeytoken_rate": NumericSpec(kind=float, minimum=0),
}
NUMERIC_SETTINGS_KEYS: tuple[str, ...] = tuple(NUMERIC_SETTINGS_SPECS)

SETTINGS_KEYS: tuple[str, ...] = (
    API_KEY_SETTING,
    "model",
    "effort",
    "llm_base_url",
    "llm_model",
    *PRESENCE_SETTINGS_KEYS,
    *BOOL_SETTINGS_KEYS,
    *NUMERIC_SETTINGS_KEYS,
)

# A setting value is a string for the free-text fields, a real boolean for the
# switches ("false" read as a string would be a true value, and a switch that
# silently means its opposite is worse than no switch at all), or an int/float
# for the numeric pipeline parameters.
SettingValue = str | bool | int | float


def _validate_numeric(key: str, value: object) -> int | float:
    """Coerce and range-check a numeric override, failing fast on either.

    `isinstance(True, int)` is True in Python, so a bool must be rejected
    before the int/float check would otherwise wave it through disguised as
    0 or 1. An int key rejects a float (including a whole-number float such
    as `2.0`) so that a fractional value typed for `chunk_size_tokens` is
    caught here instead of silently truncating. A float key accepts an int
    and normalizes it to float, since JSON does not distinguish `1` from
    `1.0` and both are legitimate spellings of the same ratio.
    """
    spec = NUMERIC_SETTINGS_SPECS[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"Settings key {key} must be a {spec.kind.__name__}, "
            f"got {type(value).__name__}"
        )
    if spec.kind is int and not isinstance(value, int):
        raise ConfigError(
            f"Settings key {key} must be an int, got {type(value).__name__}"
        )
    number = spec.kind(value)
    if number < spec.minimum:
        raise ConfigError(
            f"Settings key {key} must be >= {spec.minimum}, got {number}"
        )
    return number

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
        if key in NUMERIC_SETTINGS_KEYS:
            overrides[key] = _validate_numeric(key, value)
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
    is the empty value of a presence key (see `PRESENCE_SETTINGS_KEYS`) and a
    numeric value of 0 (see `NUMERIC_SETTINGS_SPECS`) - a number is never
    cleared by the empty-string convention, only by sending None.

    Every numeric value is run through the same `NUMERIC_SETTINGS_SPECS` check
    that `load_overrides` applies, so a value this function would refuse to
    store can never be discovered invalid only later, when it is read back:
    validation happens once, in one place, on both paths.

    Nothing is written to disk before every update in the batch has validated:
    a single bad field must leave the previously stored overrides untouched
    rather than partially applied.

    Returns:
        The merged overrides as they were persisted.

    Raises:
        ConfigError: when `updates` carries a key that is not a known setting,
            a value of the wrong type, or an out-of-range number; or when the
            existing file cannot be read.
    """
    unknown = sorted(set(updates) - set(SETTINGS_KEYS))
    if unknown:
        raise ConfigError(f"Unknown settings keys: {', '.join(unknown)}")

    merged = load_overrides(path)
    for key, value in updates.items():
        clearable_empty = key not in PRESENCE_SETTINGS_KEYS and key not in NUMERIC_SETTINGS_KEYS
        if value is None or (value == "" and clearable_empty):
            merged.pop(key, None)
        elif key in NUMERIC_SETTINGS_KEYS:
            merged[key] = _validate_numeric(key, value)
        elif key in BOOL_SETTINGS_KEYS and not isinstance(value, bool):
            raise ConfigError(
                f"Settings key {key} must be a boolean, got {type(value).__name__}"
            )
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
    chunking_config = replace(
        config.chunking,
        chunk_size_tokens=_number(
            overrides, "chunk_size_tokens", config.chunking.chunk_size_tokens
        ),
    )
    synthetic_config = replace(
        config.synthetic,
        llm=llm_config,
        chaff_ratio=_number(overrides, "chaff_ratio", config.synthetic.chaff_ratio),
        honeytoken_rate=_number(overrides, "honeytoken_rate", config.synthetic.honeytoken_rate),
        canaries_per_batch=_number(
            overrides, "canaries_per_batch", config.synthetic.canaries_per_batch
        ),
        honeytokens_enabled=_flag(
            overrides, "honeytokens_enabled", config.synthetic.honeytokens_enabled
        ),
        chaff_enabled=_flag(overrides, "chaff_enabled", config.synthetic.chaff_enabled),
        canaries_enabled=_flag(overrides, "canaries_enabled", config.synthetic.canaries_enabled),
    )
    return replace(
        config,
        chunking=chunking_config,
        anthropic=anthropic_config,
        conversion=conversion_config,
        synthetic=synthetic_config,
    )


def effective_llm_enabled(config: AppConfig, overrides: dict[str, SettingValue]) -> bool:
    """Whether the local LLM may be contacted at all.

    With it off, `doc_quant.synthetic` phrases every fragment from its
    deterministic templates, so the pipeline runs with nothing installed
    locally.
    """
    return _flag(overrides, "llm_enabled", config.synthetic.llm.enabled)


def effective_conversion_service_url(
    config: AppConfig, overrides: dict[str, SettingValue]
) -> str:
    """Where documents are converted: an external service, or nothing.

    An empty result means the built-in markitdown converter. The stored setting
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


def _flag(overrides: dict[str, SettingValue], key: str, default: bool) -> bool:
    """Return a boolean override, or `default` when none is stored.

    Checked by `isinstance` rather than truthiness so that an explicit `False`
    override wins over `default` instead of being treated as "nothing stored"
    - `bool(False)` and "absent" must not collapse into the same answer.
    """
    stored = overrides.get(key)
    if isinstance(stored, bool):
        return stored
    return default


def _number(
    overrides: dict[str, SettingValue], key: str, default: int | float
) -> int | float:
    """Return a numeric override, or `default` when none is stored.

    Checked by presence rather than truthiness: a stored 0 (an off-switch for
    `canaries_per_batch`, say) must win over the config default exactly as a
    stored positive number would, so `or`-chaining is wrong here - `0 or
    default` would silently discard the override. `bool` is excluded even
    though it is a subtype of `int`, since a switch key never carries a
    number and vice versa.
    """
    stored = overrides.get(key)
    if isinstance(stored, (int, float)) and not isinstance(stored, bool):
        return stored
    return default


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
