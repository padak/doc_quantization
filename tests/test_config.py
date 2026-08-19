"""Tests for the detection section of doc_quant.config.

Every case builds a throwaway config file from the shipped one, so the tests
stay honest about what production actually ships: a test that invents its own
JSON can keep passing after config/config.json has drifted away from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_quant.config import (
    DEFAULT_CONFIG_PATH,
    DETECTION_PROVIDER_ANTHROPIC,
    DETECTION_PROVIDER_LOCAL,
    VALID_DETECTION_PROVIDERS,
    ConfigError,
    load_config,
)


def _write_config(tmp_path: Path, raw: dict) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return config_path


def test_detection_section_is_required(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.pop("detection")

    with pytest.raises(ConfigError, match="detection"):
        load_config(_write_config(tmp_path, raw))


def test_detection_provider_is_validated(tmp_path: Path) -> None:
    # A typo must not quietly fall through to the remote backend at detection
    # time; the load has to reject it here.
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["detection"]["provider"] = "remote"

    with pytest.raises(ConfigError, match="detection.provider"):
        load_config(_write_config(tmp_path, raw))


def test_detection_local_fields_load() -> None:
    config = load_config()

    assert config.detection.provider == DETECTION_PROVIDER_ANTHROPIC
    assert config.detection.local.base_url.startswith("http")
    assert config.detection.local.model
    assert config.detection.local.timeout_seconds > 0
    assert config.detection.local.concurrency >= 1


def test_valid_detection_providers_are_exactly_the_two_backends() -> None:
    assert VALID_DETECTION_PROVIDERS == frozenset(
        {DETECTION_PROVIDER_ANTHROPIC, DETECTION_PROVIDER_LOCAL}
    )


def test_missing_detection_provider_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["detection"]["provider"]

    with pytest.raises(ConfigError, match="detection.provider"):
        load_config(_write_config(tmp_path, raw))


def test_missing_detection_local_key_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["detection"]["local"]["base_url"]

    with pytest.raises(ConfigError, match="detection.local.base_url"):
        load_config(_write_config(tmp_path, raw))


def test_detection_local_that_is_not_an_object_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["detection"]["local"] = "http://localhost:11434/v1"

    with pytest.raises(ConfigError, match="must be an object"):
        load_config(_write_config(tmp_path, raw))


def test_local_detection_endpoint_is_independent_of_the_synthetic_llm(
    tmp_path: Path,
) -> None:
    # The prose model and the detection model may differ; repointing one must
    # never drag the other along.
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["detection"]["local"]["model"] = "detector-model"
    raw["synthetic"]["llm"]["model"] = "prose-model"

    config = load_config(_write_config(tmp_path, raw))

    assert config.detection.local.model == "detector-model"
    assert config.synthetic.llm.model == "prose-model"
