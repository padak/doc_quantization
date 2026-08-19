"""Tests for .env file loading in doc_quant.config."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import doc_quant.config
from doc_quant.cli import main
from doc_quant.config import DEFAULT_CONFIG_PATH, load_env_file, require_api_key


@pytest.fixture(autouse=True)
def clean_api_key_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_loads_api_key_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_file = tmp_path / ".env"
    env_file.write_text('ANTHROPIC_API_KEY="sk-ant-from-file"\n', encoding="utf-8")

    load_env_file(env_file)

    assert require_api_key() == "sk-ant-from-file"


def test_real_environment_wins_over_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8")

    load_env_file(env_file)

    assert require_api_key() == "sk-ant-from-shell"


def test_missing_env_file_is_a_noop(tmp_path: Path):
    load_env_file(tmp_path / ".env")

    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        require_api_key()


def test_cli_loads_env_file_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8"
    )
    # Point the CLI at a throwaway config so the test never touches the real
    # database; only the .env default path and db path differ from production.
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["database"]["path"] = str(tmp_path / "chunks.db")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(doc_quant.config, "DEFAULT_CONFIG_PATH", config_file)
    monkeypatch.setattr(doc_quant.config, "DEFAULT_ENV_PATH", tmp_path / ".env")

    exit_code = main(["docs"])

    assert exit_code == 0
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-from-file"
