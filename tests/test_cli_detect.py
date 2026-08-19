"""Tests for the CLI's local `detect` command and the provider guards.

Everything here runs offline: the local model server is an `httpx.MockTransport`
injected through the names `cli` imports, and the config is a real one loaded
from `config/config.json` with the database path and the detection provider
replaced. Going through `cli.main([...])` rather than the private command
functions keeps the guards - which live in the dispatcher - under test too.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import pytest

from doc_quant import cli, local_detector
from doc_quant.config import (
    DETECTION_PROVIDER_ANTHROPIC,
    DETECTION_PROVIDER_LOCAL,
    AppConfig,
    load_config,
)
from doc_quant.local_detector import (
    LOCAL_BATCH_PREFIX,
    LOCAL_BATCH_STATUS_COMPLETED,
)
from doc_quant.local_llm import LocalLLMError
from doc_quant.store import ChunkStore

# The one name planted in the fixture document; the fake server answers with it
# whenever it appears in the fragment it was given, and with nothing otherwise -
# an entity the fragment does not contain would be dropped as a hallucination
# and would make the assertions lie about what was detected.
KNOWN_NAME = "Jan Novak"

DOCUMENT = (
    "# Project Meridian\n"
    "\n"
    f"{KNOWN_NAME} signed the Project Meridian memo on Tuesday and sent it to "
    "the steering group for review. The rollout window stays unchanged, and "
    "the budget line is carried over from the previous quarter without any "
    "further adjustment.\n"
)


@dataclass(frozen=True)
class CliEnv:
    """Handle on a throwaway CLI environment: its config, db and server calls."""

    config: AppConfig
    db_path: Path
    doc_dir: Path
    # Thread names the fake local server saw, one per request; used to assert
    # that the HTTP work - and only the HTTP work - leaves the calling thread.
    request_threads: list[str]

    def open_store(self) -> ChunkStore:
        return ChunkStore(self.db_path)


def _answer(entities: list[dict]) -> httpx.Response:
    """Shape one OpenAI-compatible chat completion carrying `entities`."""
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps({"entities": entities})}}
            ]
        },
    )


def _make_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    *,
    ingest: bool = True,
    handler=None,
    concurrency: int | None = None,
) -> CliEnv:
    """Point the CLI at a tmp database and a fake local server."""
    base = load_config()
    local = base.detection.local
    if concurrency is not None:
        local = replace(local, concurrency=concurrency)
    config = replace(
        base,
        database=replace(base.database, path=tmp_path / "chunks.db"),
        detection=replace(base.detection, provider=provider, local=local),
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)

    request_threads: list[str] = []

    def default_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = payload["messages"][-1]["content"]
        found = [{"text": KNOWN_NAME, "type": "person"}] if KNOWN_NAME in text else []
        return _answer(found)

    chosen = handler or default_handler

    def recording_handler(request: httpx.Request) -> httpx.Response:
        request_threads.append(threading.current_thread().name)
        return chosen(request)

    monkeypatch.setattr(cli, "probe_local_server", lambda cfg: None)
    monkeypatch.setattr(
        cli,
        "get_local_client",
        lambda cfg: local_detector.get_local_client(
            cfg, transport=httpx.MockTransport(recording_handler)
        ),
    )

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    if ingest:
        (doc_dir / "memo.md").write_text(DOCUMENT, encoding="utf-8")
        assert cli.main(["ingest", str(doc_dir)]) == 0

    return CliEnv(
        config=config,
        db_path=config.database.path,
        doc_dir=doc_dir,
        request_threads=request_threads,
    )


@pytest.fixture
def local_cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliEnv:
    return _make_env(tmp_path, monkeypatch, DETECTION_PROVIDER_LOCAL)


@pytest.fixture
def local_cli_env_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliEnv:
    return _make_env(tmp_path, monkeypatch, DETECTION_PROVIDER_LOCAL, ingest=False)


@pytest.fixture
def anthropic_cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliEnv:
    return _make_env(tmp_path, monkeypatch, DETECTION_PROVIDER_ANTHROPIC)


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------


def test_detect_refuses_under_anthropic_provider(
    anthropic_cli_env: CliEnv, capsys: pytest.CaptureFixture
):
    assert cli.main(["detect"]) == 1
    assert "detection.provider" in capsys.readouterr().err


def test_submit_refuses_under_local_provider(
    local_cli_env: CliEnv, capsys: pytest.CaptureFixture
):
    assert cli.main(["submit"]) == 1
    assert "detection.provider is 'local'" in capsys.readouterr().err


def test_fetch_refuses_under_local_provider(
    local_cli_env: CliEnv, capsys: pytest.CaptureFixture
):
    assert cli.main(["fetch", "batch_x"]) == 1
    assert "detection.provider is 'local'" in capsys.readouterr().err


def test_status_with_batch_id_refuses_under_local_provider(
    local_cli_env: CliEnv, capsys: pytest.CaptureFixture
):
    assert cli.main(["status", "batch_x"]) == 1
    assert "detection.provider is 'local'" in capsys.readouterr().err


def test_bare_status_still_lists_under_local_provider(
    local_cli_env: CliEnv, capsys: pytest.CaptureFixture
):
    # Listing batches is offline, and a local run is recorded as a batch too:
    # refusing here would hide local runs from their own operator.
    assert cli.main(["status"]) == 0


def test_a_refused_command_leaves_the_chunks_untouched(local_cli_env: CliEnv):
    assert cli.main(["submit"]) == 1

    store = local_cli_env.open_store()
    try:
        assert store.get_unsubmitted_chunks()
    finally:
        store.close()


# ----------------------------------------------------------------------
# detect
# ----------------------------------------------------------------------


def test_detect_runs_locally_and_stores_entities(
    local_cli_env: CliEnv, capsys: pytest.CaptureFixture
):
    assert cli.main(["detect"]) == 0
    out = capsys.readouterr().out
    assert "entities=" in out

    store = local_cli_env.open_store()
    try:
        docs = store.list_documents()
        entities = store.get_document_entities(docs[0]["doc_id"])
        assert (KNOWN_NAME, "person") in entities

        chunks = store.get_document_chunks(docs[0]["doc_id"])
        assert chunks
        assert all(chunk["batch_id"].startswith(LOCAL_BATCH_PREFIX) for chunk in chunks)

        batches = store.list_batches()
        assert [batch["status"] for batch in batches] == [LOCAL_BATCH_STATUS_COMPLETED]
    finally:
        store.close()


def test_detect_sends_one_request_per_chunk(local_cli_env: CliEnv):
    store = local_cli_env.open_store()
    try:
        chunk_count = len(store.get_unsubmitted_chunks())
    finally:
        store.close()

    assert cli.main(["detect"]) == 0
    assert len(local_cli_env.request_threads) == chunk_count


def test_detect_is_a_no_op_the_second_time(
    local_cli_env: CliEnv, capsys: pytest.CaptureFixture
):
    assert cli.main(["detect"]) == 0
    capsys.readouterr()

    # Chunks are marked submitted by the first run, so the second finds nothing:
    # detection is never paid for twice.
    assert cli.main(["detect"]) == 0
    assert "nothing to detect" in capsys.readouterr().out


def test_detect_with_nothing_to_do(
    local_cli_env_empty: CliEnv, capsys: pytest.CaptureFixture
):
    assert cli.main(["detect"]) == 0
    assert "nothing to detect" in capsys.readouterr().out


def test_detect_reports_errored_fragments_without_failing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]})

    env = _make_env(
        tmp_path, monkeypatch, DETECTION_PROVIDER_LOCAL, handler=garbage
    )

    assert cli.main(["detect"]) == 0
    out = capsys.readouterr().out
    assert "ok=0" in out
    assert "entities=0" in out

    store = env.open_store()
    try:
        docs = store.list_documents()
        assert store.get_document_entities(docs[0]["doc_id"]) == []
        # Even a run where every fragment errored is closed out, so the batch
        # never looks like it is still in flight.
        assert [batch["status"] for batch in store.list_batches()] == [
            LOCAL_BATCH_STATUS_COMPLETED
        ]
    finally:
        store.close()


def test_detect_counts_hallucinated_entities_as_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    def hallucinating(request: httpx.Request) -> httpx.Response:
        return _answer([{"text": "Elvira Ghost", "type": "person"}])

    env = _make_env(
        tmp_path, monkeypatch, DETECTION_PROVIDER_LOCAL, handler=hallucinating
    )

    assert cli.main(["detect"]) == 0
    assert "dropped=" in capsys.readouterr().out

    store = env.open_store()
    try:
        docs = store.list_documents()
        assert store.get_document_entities(docs[0]["doc_id"]) == []
    finally:
        store.close()


def test_detect_aborts_before_any_store_write_when_the_server_is_down(
    local_cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    def unreachable(config):
        raise LocalLLMError("Local detection server unreachable at http://x")

    monkeypatch.setattr(cli, "probe_local_server", unreachable)

    assert cli.main(["detect"]) == 1
    assert "unreachable" in capsys.readouterr().err

    store = local_cli_env.open_store()
    try:
        # Nothing was submitted, so a retry after starting the server picks the
        # document up exactly where it was left.
        assert store.get_unsubmitted_chunks()
        assert store.list_batches() == []
    finally:
        store.close()


def test_detect_keeps_store_writes_on_the_calling_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The store is opened with its default thread guard on, so any write from a
    # pool worker would raise; the run succeeding while the requests demonstrably
    # left this thread is what proves the split.
    env = _make_env(tmp_path, monkeypatch, DETECTION_PROVIDER_LOCAL, concurrency=4)

    assert cli.main(["detect"]) == 0
    assert env.request_threads
    assert threading.current_thread().name not in env.request_threads
