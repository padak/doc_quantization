"""Unit tests for the pipeline-parameter overrides in webapp.settings.

These exercise `load_overrides`, `save_overrides` and `effective_config`
directly, without going through the FastAPI layer, so the validation and
merge rules for chunk_size_tokens, chaff_ratio, honeytoken_rate,
canaries_per_batch and the three *_enabled switches are pinned independently
of the HTTP contract (see tests/test_webapp.py for the endpoint-level tests).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from doc_quant.config import ConfigError, load_config
from webapp.settings import effective_config, load_overrides, save_overrides


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    return tmp_path / "settings.json"


# ---------------------------------------------------------------------------
# save_overrides: persisting numeric and boolean pipeline overrides
# ---------------------------------------------------------------------------


def test_save_overrides_persists_numeric_pipeline_values(settings_path):
    merged = save_overrides(
        settings_path,
        {
            "chunk_size_tokens": 8,
            "chaff_ratio": 2.5,
            "honeytoken_rate": 0.1,
            "canaries_per_batch": 3,
        },
    )

    assert merged == {
        "chunk_size_tokens": 8,
        "chaff_ratio": 2.5,
        "honeytoken_rate": 0.1,
        "canaries_per_batch": 3,
    }
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored == merged


def test_save_overrides_normalizes_int_to_float_for_float_keys(settings_path):
    """A whole-number JSON int is a legitimate spelling of a ratio."""
    merged = save_overrides(settings_path, {"chaff_ratio": 2, "honeytoken_rate": 0})

    assert merged["chaff_ratio"] == 2.0
    assert isinstance(merged["chaff_ratio"], float)
    assert merged["honeytoken_rate"] == 0.0
    assert isinstance(merged["honeytoken_rate"], float)


def test_save_overrides_persists_a_stored_zero_for_canaries_per_batch(settings_path):
    """0 turns a mechanism's per-batch contribution off; it must not be dropped."""
    merged = save_overrides(settings_path, {"canaries_per_batch": 0})

    assert merged["canaries_per_batch"] == 0
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["canaries_per_batch"] == 0


def test_save_overrides_persists_a_stored_false_toggle(settings_path):
    merged = save_overrides(
        settings_path,
        {"honeytokens_enabled": False, "chaff_enabled": False, "canaries_enabled": False},
    )

    assert merged == {
        "honeytokens_enabled": False,
        "chaff_enabled": False,
        "canaries_enabled": False,
    }
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["honeytokens_enabled"] is False


def test_save_overrides_clears_a_numeric_override_via_none(settings_path):
    save_overrides(settings_path, {"chunk_size_tokens": 8})

    merged = save_overrides(settings_path, {"chunk_size_tokens": None})

    assert "chunk_size_tokens" not in merged
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "chunk_size_tokens" not in stored


@pytest.mark.parametrize(
    "updates",
    [
        {"chunk_size_tokens": 0},
        {"chunk_size_tokens": -1},
        {"chaff_ratio": -0.1},
        {"honeytoken_rate": -0.5},
        {"canaries_per_batch": -1},
        {"chunk_size_tokens": 2.5},  # int key rejects a float
        {"chunk_size_tokens": True},  # bool must not pass as an int
        {"chaff_ratio": True},  # bool must not pass as a float either
        {"honeytokens_enabled": 1},  # a number must not pass as a bool
    ],
)
def test_save_overrides_rejects_invalid_pipeline_values(settings_path, updates):
    with pytest.raises(ConfigError):
        save_overrides(settings_path, updates)


def test_save_overrides_rejects_a_batch_atomically(settings_path):
    """One bad field in a PUT must not partially apply the rest of it."""
    save_overrides(settings_path, {"chunk_size_tokens": 8})
    before = settings_path.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        save_overrides(settings_path, {"honeytoken_rate": 0.5, "chaff_ratio": -1})

    assert settings_path.read_text(encoding="utf-8") == before
    stored = json.loads(before)
    assert stored == {"chunk_size_tokens": 8}


# ---------------------------------------------------------------------------
# load_overrides: reading a settings file written by hand or an older build
# ---------------------------------------------------------------------------


def test_load_overrides_reads_back_numeric_and_boolean_values(settings_path):
    settings_path.write_text(
        json.dumps(
            {
                "chunk_size_tokens": 8,
                "chaff_ratio": 2.5,
                "honeytoken_rate": 0.1,
                "canaries_per_batch": 0,
                "honeytokens_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    overrides = load_overrides(settings_path)

    assert overrides["chunk_size_tokens"] == 8
    assert overrides["chaff_ratio"] == 2.5
    assert overrides["honeytoken_rate"] == 0.1
    assert overrides["canaries_per_batch"] == 0
    assert overrides["honeytokens_enabled"] is False


@pytest.mark.parametrize(
    "raw",
    [
        {"chunk_size_tokens": 0},
        {"chunk_size_tokens": -1},
        {"chunk_size_tokens": "8"},
        {"chunk_size_tokens": 2.5},
        {"chunk_size_tokens": True},
        {"chaff_ratio": -1},
        {"chaff_ratio": "high"},
        {"honeytoken_rate": True},
        {"canaries_per_batch": -1},
        {"canaries_enabled": 1},
    ],
)
def test_load_overrides_rejects_malformed_pipeline_values(settings_path, raw):
    settings_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_overrides(settings_path)


# ---------------------------------------------------------------------------
# effective_config: overrides win, including 0 and False
# ---------------------------------------------------------------------------


def test_effective_config_falls_back_to_config_defaults_with_no_overrides():
    config = load_config()

    effective = effective_config(config, {})

    assert effective.chunking.chunk_size_tokens == config.chunking.chunk_size_tokens
    assert effective.synthetic.chaff_ratio == config.synthetic.chaff_ratio
    assert effective.synthetic.honeytoken_rate == config.synthetic.honeytoken_rate
    assert effective.synthetic.canaries_per_batch == config.synthetic.canaries_per_batch
    assert effective.synthetic.honeytokens_enabled == config.synthetic.honeytokens_enabled
    assert effective.synthetic.chaff_enabled == config.synthetic.chaff_enabled
    assert effective.synthetic.canaries_enabled == config.synthetic.canaries_enabled


def test_effective_config_applies_overrides_including_zero_and_false():
    config = load_config()
    # Pick values that differ from the config defaults so the assertions are
    # not satisfied by coincidence.
    config = replace(
        config,
        synthetic=replace(
            config.synthetic,
            canaries_per_batch=5,
            honeytokens_enabled=True,
            chaff_enabled=True,
            canaries_enabled=True,
        ),
    )
    overrides = {
        "chunk_size_tokens": 4,
        "chaff_ratio": 0.0,
        "honeytoken_rate": 0.0,
        "canaries_per_batch": 0,
        "honeytokens_enabled": False,
        "chaff_enabled": False,
        "canaries_enabled": False,
    }

    effective = effective_config(config, overrides)

    assert effective.chunking.chunk_size_tokens == 4
    assert effective.synthetic.chaff_ratio == 0.0
    assert effective.synthetic.honeytoken_rate == 0.0
    assert effective.synthetic.canaries_per_batch == 0
    assert effective.synthetic.honeytokens_enabled is False
    assert effective.synthetic.chaff_enabled is False
    assert effective.synthetic.canaries_enabled is False
