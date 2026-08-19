"""Tests for the observability web app.

Everything runs offline: the Anthropic client and the synthetic fragment
generator are replaced through the module-level factories in `webapp.server`,
the store is pointed at a temporary database, and the settings file lives in
`tmp_path`. No test may depend on ANTHROPIC_API_KEY being set in the
environment, so the variable is removed and a dummy key is written into the
settings file instead.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from doc_quant.config import ConfigError, load_config
from doc_quant.detector import DETECTION_SYSTEM_PROMPT, ENTITY_SCHEMA
from doc_quant.store import ChunkStore
from doc_quant.synthetic import (
    CANARY_FACT_TEMPLATE,
    KIND_CANARY,
    KIND_CHAFF,
    KIND_HONEYTOKEN,
    SyntheticFragment,
)
from webapp import server
from webapp.server import (
    NDJSON_MEDIA_TYPE,
    PHASE_SYNTHETICS_TEMPLATE_DETAIL,
    VERIFY_LLM_DISABLED_DETAIL,
    VERIFY_NO_KEY_DETAIL,
)
from webapp.settings import MISSING_API_KEY_MESSAGE

DUMMY_API_KEY = "sk-ant-testing-0123456789wxyz"
MASKED_DUMMY_KEY = "sk-a...wxyz"

SAMPLE_MARKDOWN = """# Quarterly Review

Jan Novak met Petra Svobodova at the Keboola office in Prague to discuss the
migration plan for the reporting warehouse. The team agreed to ship the new
connector before the end of the quarter and to review the remaining risks.
"""

EMAIL_MARKDOWN = """# Contacts

Jan Novak <jan.novak+press@mail.keboola.com> met Petra Svobodova at the Keboola
office in Prague. Write to padak@keboola.com about the migration plan.
"""

SAMPLE_EMAILS = ("jan.novak+press@mail.keboola.com", "padak@keboola.com")

URL_MARKDOWN = """# Links

Jan Novak at Keboola forwarded the notice from
https://resolve.picrights.com/Home/Settlement/3979-4561-9198 and asked us to
check www.keboola.com/pricing. Reply to padak@keboola.com.
"""

SAMPLE_URLS = (
    "https://resolve.picrights.com/Home/Settlement/3979-4561-9198",
    "www.keboola.com/pricing",
)

MULTIBYTE_MARKDOWN = (
    "# Zápis z porady\n\n"
    "Petr Šimeček <petr@keboola.com> podepsal smlouvu s firmou Čerpadla Plzeň "
    "s.r.o. 東京の田中さんも参加しました 🎉 a všichni souhlasili.\n"
)

SAMPLE_HTML = (
    "<html><body><h1>Acme Report</h1>"
    "<p>Jan Novak signed the agreement with Keboola.</p></body></html>"
)

KNOWN_ENTITIES = [
    ("Jan Novak", "person"),
    ("Petra Svobodova", "person"),
    ("Keboola", "company"),
]

HONEYTOKEN_TEXT = (
    "Meeting notes: Hedda Honeywell signed the revised service "
    "agreement with Vorncorp Systems."
)
HONEYTOKEN_PLANTED = [("Hedda Honeywell", "person"), ("Vorncorp Systems", "company")]
# The fake provider reports only the person back, so recall is measurably
# partial rather than trivially 0 or 1.
HONEYTOKEN_REPORTED = [("Hedda Honeywell", "person")]

CANARY_PEOPLE = ("Alpha Canaryson", "Beta Canaryson")
CANARY_PLACES = ("Nonceville Flats", "Otherplace Flats")
CANARY_COUNT = len(CANARY_PEOPLE)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeAnthropicClient:
    """Stand-in for `anthropic.Anthropic` covering `messages.create` only.

    Behaviour is driven by the fragment text of a request, which is what the
    tests know about: a text may be configured to yield entities, to be
    refused, or to raise an API error. Probe questions (which carry no
    fragment) are answered from `raw_by_substring`.
    """

    def __init__(self) -> None:
        self.entities_by_text: dict[str, list[tuple[str, str]]] = {}
        self.refuse_texts: set[str] = set()
        self.error_texts: set[str] = set()
        self.raw_by_substring: dict[str, str] = {}
        self.calls: list[dict] = []
        # Answers are instant unless a test asks for jitter, which is how the
        # completion order is made to differ from the submission order.
        self.delay_seconds = 0.0
        self.retrieved: list[str] = []
        self.model_error: Exception | None = None
        self.messages = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(retrieve=self._retrieve)

    def _retrieve(self, model: str) -> SimpleNamespace:
        self.retrieved.append(model)
        if self.model_error is not None:
            raise self.model_error
        return SimpleNamespace(id=model)

    def _create(self, **params) -> SimpleNamespace:
        if self.delay_seconds:
            time.sleep(random.uniform(0, self.delay_seconds))
        # list.append is atomic, so several workers may record here at once.
        self.calls.append(params)
        text = params["messages"][0]["content"]

        if text in self.error_texts:
            raise anthropic.APIError(
                "upstream exploded", httpx.Request("POST", "http://test"), body=None
            )
        if text in self.refuse_texts:
            return _message("I cannot help with that.", stop_reason="refusal")

        for needle, answer in self.raw_by_substring.items():
            if needle in text:
                return _message(answer)

        entities = self.entities_by_text.get(text, [])
        return _message(
            json.dumps(
                {"entities": [{"text": t, "type": k} for t, k in entities]}
            )
        )


def _message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


class FakeGenerator:
    """Deterministic synthetic fragments, registered in the real store.

    Registration matters: the app records honeytoken results against the store,
    which rejects a fragment id it has never seen.
    """

    def __init__(self, store: ChunkStore, created: list[SyntheticFragment]) -> None:
        self._store = store
        self._created = created

    def make_honeytokens(self, count: int) -> list[SyntheticFragment]:
        fragments = [
            SyntheticFragment(
                fragment_id=uuid.uuid4().hex,
                kind=KIND_HONEYTOKEN,
                text=HONEYTOKEN_TEXT,
                planted=list(HONEYTOKEN_PLANTED),
                fact=None,
            )
            for _ in range(count)
        ]
        return self._register(fragments)

    def make_chaff(self, count: int) -> list[SyntheticFragment]:
        fragments = [
            SyntheticFragment(
                fragment_id=uuid.uuid4().hex,
                kind=KIND_CHAFF,
                text=f"Status report {index}: the delivery schedule was approved.",
                planted=[],
                fact=None,
            )
            for index in range(count)
        ]
        return self._register(fragments)

    def ensure_canaries(self) -> list[SyntheticFragment]:
        existing = self._store.list_synthetic_fragments(kind=KIND_CANARY)
        if existing:
            return [_fragment_from_row(row) for row in existing]
        return self._register(
            [
                _build_canary(person, place)
                for person, place in zip(CANARY_PEOPLE, CANARY_PLACES)
            ]
        )

    def _register(self, fragments: list[SyntheticFragment]) -> list[SyntheticFragment]:
        self._store.add_synthetic_fragments(
            [
                {
                    "fragment_id": fragment.fragment_id,
                    "kind": fragment.kind,
                    "text": fragment.text,
                    "planted": fragment.planted,
                    "fact": fragment.fact,
                }
                for fragment in fragments
            ]
        )
        self._created.extend(fragments)
        return fragments


class FakeHTTPClient:
    """Stand-in for `httpx.Client`: the preflight GETs and the conversion POST."""

    def __init__(self) -> None:
        self.response: SimpleNamespace | None = None
        self.error: Exception | None = None
        self.requested: list[str] = []
        self.closed = False
        self.timeouts: list[float] = []
        # The conversion endpoint answers separately from the GET endpoints.
        self.post_response: SimpleNamespace | None = None
        self.post_error: Exception | None = None
        self.posted: list[dict] = []

    def serve(self, model_ids: list[str], status_code: int = 200) -> None:
        """Answer as an OpenAI-compatible server offering `model_ids`."""
        self.error = None
        self.response = SimpleNamespace(
            status_code=status_code,
            json=lambda: {"data": [{"id": model_id} for model_id in model_ids]},
        )

    def serve_conversion(self, markdown: str, status_code: int = 200) -> None:
        """Answer as the external conversion service."""
        self.post_error = None
        self.post_response = SimpleNamespace(
            status_code=status_code, json=lambda: {"markdown": markdown}
        )

    def get(self, url: str) -> SimpleNamespace:
        self.requested.append(url)
        if self.error is not None:
            raise self.error
        return self.response

    def post(self, url: str, files: dict) -> SimpleNamespace:
        self.posted.append({"url": url, "files": files})
        if self.post_error is not None:
            raise self.post_error
        return self.post_response

    def close(self) -> None:
        self.closed = True


def _build_canary(person: str, place: str) -> SyntheticFragment:
    fact = CANARY_FACT_TEMPLATE.format(person=person, place=place)
    return SyntheticFragment(
        fragment_id=uuid.uuid4().hex,
        kind=KIND_CANARY,
        text=fact,
        planted=[(person, "person")],
        fact=fact,
    )


def _fragment_from_row(row: dict) -> SyntheticFragment:
    return SyntheticFragment(
        fragment_id=row["fragment_id"],
        kind=row["kind"],
        text=row["text"],
        planted=[(name, kind) for name, kind in row["planted"]],
        fact=row["fact"],
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@dataclass
class Harness:
    """Everything a test needs to drive the app and inspect its side effects."""

    client: TestClient
    provider: FakeAnthropicClient
    http: FakeHTTPClient
    db_path: Path
    settings_path: Path
    fragments: list[SyntheticFragment] = field(default_factory=list)

    def store(self) -> ChunkStore:
        return ChunkStore(self.db_path)

    def synthetic(self, kind: str) -> list[SyntheticFragment]:
        return [fragment for fragment in self.fragments if fragment.kind == kind]


@pytest.fixture
def harness(tmp_path, monkeypatch) -> Harness:
    """Wire the app to a temporary store, settings file and fake provider."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    db_path = tmp_path / "chunks.db"
    config = load_config()
    config = replace(config, database=replace(config.database, path=db_path))
    # The external conversion service is opt-in per test: a service_url left in
    # the developer's own config/config.json must not decide how an upload is
    # converted here, any more than their database path decides where it is
    # stored.
    config = replace(config, conversion=replace(config.conversion, service_url=""))
    monkeypatch.setattr(server, "get_config", lambda: config)

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"anthropic_api_key": DUMMY_API_KEY}), encoding="utf-8"
    )
    monkeypatch.setattr(server, "get_settings_path", lambda: settings_path)

    provider = FakeAnthropicClient()
    created: list[SyntheticFragment] = []

    def fake_get_client(api_key: str | None) -> FakeAnthropicClient:
        # The real factory's contract is kept, so the missing-key path stays
        # testable while nothing ever reaches the network.
        if not api_key:
            raise ConfigError(MISSING_API_KEY_MESSAGE)
        return provider

    monkeypatch.setattr(server, "get_anthropic_client", fake_get_client)
    monkeypatch.setattr(
        server, "get_generator", lambda config, store: FakeGenerator(store, created)
    )

    # The preflight's only network call is stubbed out too: by default the
    # local endpoint answers and serves exactly the configured model.
    http = FakeHTTPClient()
    http.serve([config.synthetic.llm.model])

    def fake_http_client(timeout: float = server.VERIFY_HTTP_TIMEOUT_SECONDS):
        # The timeout is part of the contract: the conversion call must not run
        # on the preflight's short one.
        http.timeouts.append(timeout)
        return http

    monkeypatch.setattr(server, "get_http_client", fake_http_client)

    with TestClient(server.app) as client:
        yield Harness(
            client=client,
            provider=provider,
            http=http,
            db_path=db_path,
            settings_path=settings_path,
            fragments=created,
        )


def upload_markdown(harness: Harness, name: str = "review.md", text: str = SAMPLE_MARKDOWN) -> dict:
    response = harness.client.post(
        "/api/documents", files={"file": (name, text.encode("utf-8"), "text/markdown")}
    )
    assert response.status_code == 200, response.text
    return response.json()


def arm_provider(harness: Harness, document: dict) -> dict[str, list[tuple[str, str]]]:
    """Teach the fake provider which chunk text carries which known names.

    Returns the real-chunk mapping only, so a test can compute how many
    entities the run should have stored without hardcoding where the chunker
    put its boundaries. The honeytoken answer is armed here too, keyed on the
    fixed text the fake generator plants it in.
    """
    mapping: dict[str, list[tuple[str, str]]] = {}
    for chunk in document["chunks"]:
        found = [(text, kind) for text, kind in KNOWN_ENTITIES if text in chunk["text"]]
        if found:
            mapping[chunk["text"]] = found
    harness.provider.entities_by_text.update(mapping)
    harness.provider.entities_by_text[HONEYTOKEN_TEXT] = list(HONEYTOKEN_REPORTED)
    return mapping


def with_concurrency(config, workers: int):
    """Return `config` with the detection concurrency set to `workers`."""
    return replace(config, anthropic=replace(config.anthropic, detect_concurrency=workers))


def set_concurrency(monkeypatch, workers: int) -> None:
    """Make the app run its detection requests `workers` at a time.

    Called from inside a test, so it patches the config the `harness` fixture
    already installed.
    """
    config = with_concurrency(server.get_config(), workers)
    monkeypatch.setattr(server, "get_config", lambda: config)


def set_conversion_service(monkeypatch, service_url: str) -> None:
    """Point the app at an external conversion service (empty = built-in)."""
    config = server.get_config()
    config = replace(config, conversion=replace(config.conversion, service_url=service_url))
    monkeypatch.setattr(server, "get_config", lambda: config)


def detect_events(harness: Harness, doc_id: str) -> list[dict]:
    """Run a detection and return its progress stream, one event per line."""
    response = harness.client.post("/api/detect", json={"doc_id": doc_id})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(NDJSON_MEDIA_TYPE)
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def detect(harness: Harness, doc_id: str) -> dict:
    """Run a detection and return the final `done` event.

    The `done` event repeats the whole run, so a test that is not about the
    streaming itself reads it exactly as it read the old JSON body.
    """
    events = detect_events(harness, doc_id)
    assert events, "the stream carried no events"
    assert events[-1]["type"] == "done", events[-1]
    return events[-1]


# ---------------------------------------------------------------------------
# frontend serving and per-request store threading
# ---------------------------------------------------------------------------


def test_frontend_files_demand_revalidation(harness):
    """Without this header browsers cache app.js heuristically for days and
    keep rendering a long-gone UI after a deploy."""
    for path in ("/", "/static/app.js"):
        response = harness.client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == server.STATIC_CACHE_CONTROL


def test_api_responses_are_not_marked_for_frontend_revalidation(harness):
    response = harness.client.get("/api/documents")
    assert response.status_code == 200
    assert "cache-control" not in response.headers


def test_request_store_survives_the_threadpool_handoff(tmp_path):
    """FastAPI may run a sync dependency and the endpoint it feeds on two
    different worker threads; the per-request store must tolerate the hop.

    Regression: with sqlite3's same-thread guard on, a request whose
    dependency and endpoint landed on different workers intermittently died
    with ProgrammingError.
    """
    config = SimpleNamespace(database=SimpleNamespace(path=tmp_path / "chunks.db"))
    store = server.get_store(config)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(store.list_documents).result() == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


def test_get_settings_masks_the_key_and_reports_config_defaults(harness):
    config = load_config()

    payload = harness.client.get("/api/settings").json()

    assert payload["has_api_key"] is True
    assert payload["anthropic_api_key_masked"] == MASKED_DUMMY_KEY
    assert payload["model"] == config.anthropic.model
    assert payload["effort"] == config.anthropic.effort
    assert payload["llm_base_url"] == config.synthetic.llm.base_url
    assert payload["llm_model"] == config.synthetic.llm.model
    assert payload["llm_enabled"] == config.synthetic.llm.enabled
    # The fixture pins the conversion service off, so this is the config value.
    assert payload["conversion_service_url"] == ""
    assert payload["chunk_size_tokens"] == config.chunking.chunk_size_tokens
    assert payload["chaff_ratio"] == config.synthetic.chaff_ratio
    assert payload["honeytoken_rate"] == config.synthetic.honeytoken_rate
    assert payload["canaries_per_batch"] == config.synthetic.canaries_per_batch


def test_settings_roundtrip_persists_overrides(harness):
    response = harness.client.put(
        "/api/settings",
        json={
            "model": "claude-test-model",
            "effort": "high",
            "llm_base_url": "http://localhost:9999/v1",
            "llm_model": "tiny-local",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "claude-test-model"
    assert payload["effort"] == "high"
    assert payload["llm_base_url"] == "http://localhost:9999/v1"
    assert payload["llm_model"] == "tiny-local"
    # Survives a fresh read, and the untouched key is still there.
    assert harness.client.get("/api/settings").json() == payload
    assert payload["has_api_key"] is True

    stored = json.loads(harness.settings_path.read_text(encoding="utf-8"))
    assert stored["model"] == "claude-test-model"
    assert stored["anthropic_api_key"] == DUMMY_API_KEY


def test_llm_enabled_survives_being_switched_off_and_on(harness):
    """False is a stored decision, not an absent setting."""
    assert harness.client.get("/api/settings").json()["llm_enabled"] is True

    payload = harness.client.put("/api/settings", json={"llm_enabled": False}).json()

    assert payload["llm_enabled"] is False
    assert harness.client.get("/api/settings").json()["llm_enabled"] is False
    stored = json.loads(harness.settings_path.read_text(encoding="utf-8"))
    assert stored["llm_enabled"] is False
    # The untouched key is still there: one field may be sent on its own.
    assert stored["anthropic_api_key"] == DUMMY_API_KEY

    payload = harness.client.put("/api/settings", json={"llm_enabled": True}).json()

    assert payload["llm_enabled"] is True
    assert harness.client.get("/api/settings").json()["llm_enabled"] is True


def test_conversion_service_url_roundtrips_and_keeps_an_explicit_empty(harness, monkeypatch):
    """An empty URL is a stored decision, not an absent setting.

    The config points at a service here, so falling back to it would be visible:
    only a stored empty value can keep the answer empty.
    """
    set_conversion_service(monkeypatch, "http://from-config:9000")
    assert harness.client.get("/api/settings").json()["conversion_service_url"] == (
        "http://from-config:9000"
    )

    payload = harness.client.put(
        "/api/settings", json={"conversion_service_url": "http://converter:9000"}
    ).json()

    assert payload["conversion_service_url"] == "http://converter:9000"
    assert harness.client.get("/api/settings").json() == payload
    stored = json.loads(harness.settings_path.read_text(encoding="utf-8"))
    assert stored["conversion_service_url"] == "http://converter:9000"
    # The untouched key is still there: one field may be sent on its own.
    assert stored["anthropic_api_key"] == DUMMY_API_KEY

    payload = harness.client.put(
        "/api/settings", json={"conversion_service_url": ""}
    ).json()

    assert payload["conversion_service_url"] == ""
    assert harness.client.get("/api/settings").json()["conversion_service_url"] == ""
    stored = json.loads(harness.settings_path.read_text(encoding="utf-8"))
    assert stored["conversion_service_url"] == ""


def test_conversion_service_url_setting_routes_the_upload(harness):
    """The stored URL is where the file actually goes."""
    harness.client.put(
        "/api/settings", json={"conversion_service_url": "http://converter:9000"}
    )
    harness.http.serve_conversion("# Converted\n\nJan Novak signed it.\n")

    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.pdf", b"%PDF-1.7 fake bytes", "application/pdf")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["markdown"] == "# Converted\n\nJan Novak signed it.\n"
    assert [call["url"] for call in harness.http.posted] == [
        "http://converter:9000/convert"
    ]


def test_empty_conversion_service_url_setting_runs_text_only(harness, monkeypatch):
    """A cleared field wins over a service_url left in the config."""
    set_conversion_service(monkeypatch, "http://from-config:9000")
    harness.client.put("/api/settings", json={"conversion_service_url": ""})

    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.html", SAMPLE_HTML.encode("utf-8"), "text/html")},
    )

    assert response.status_code == 422, response.text
    assert "no conversion service" in response.json()["detail"]
    assert harness.http.posted == []


def test_settings_never_echo_the_raw_key(harness):
    new_key = "sk-ant-secretvalue-abcdefgh"

    response = harness.client.put("/api/settings", json={"anthropic_api_key": new_key})

    assert new_key not in response.text
    assert response.json()["anthropic_api_key_masked"] == "sk-a...efgh"
    assert new_key not in harness.client.get("/api/settings").text


def test_empty_api_key_clears_the_stored_one(harness):
    payload = harness.client.put(
        "/api/settings", json={"anthropic_api_key": ""}
    ).json()

    assert payload["has_api_key"] is False
    assert payload["anthropic_api_key_masked"] is None
    assert "anthropic_api_key" not in json.loads(
        harness.settings_path.read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# settings: pipeline parameters
# ---------------------------------------------------------------------------


def test_pipeline_defaults_report_the_base_config_regardless_of_overrides(harness):
    config = load_config()

    before = harness.client.get("/api/settings").json()["pipeline_defaults"]
    assert before == {
        "chunk_size_tokens": config.chunking.chunk_size_tokens,
        "chaff_ratio": config.synthetic.chaff_ratio,
        "honeytoken_rate": config.synthetic.honeytoken_rate,
        "canaries_per_batch": config.synthetic.canaries_per_batch,
        "honeytokens_enabled": config.synthetic.honeytokens_enabled,
        "chaff_enabled": config.synthetic.chaff_enabled,
        "canaries_enabled": config.synthetic.canaries_enabled,
    }

    harness.client.put(
        "/api/settings",
        json={
            "chunk_size_tokens": 4,
            "chaff_ratio": 0.0,
            "honeytoken_rate": 0.0,
            "canaries_per_batch": 0,
            "honeytokens_enabled": False,
            "chaff_enabled": False,
            "canaries_enabled": False,
        },
    )

    after = harness.client.get("/api/settings").json()["pipeline_defaults"]
    assert after == before


def test_pipeline_parameters_roundtrip_including_zero_and_false(harness):
    """0 and False must persist and be reported as the effective value.

    `or`-chaining anywhere on the read path would silently discard these in
    favour of the config default, which is exactly what must not happen.
    """
    response = harness.client.put(
        "/api/settings",
        json={
            "chunk_size_tokens": 6,
            "chaff_ratio": 0.5,
            "honeytoken_rate": 0.0,
            "canaries_per_batch": 0,
            "honeytokens_enabled": True,
            "chaff_enabled": False,
            "canaries_enabled": False,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["chunk_size_tokens"] == 6
    assert payload["chaff_ratio"] == 0.5
    assert payload["honeytoken_rate"] == 0.0
    assert payload["canaries_per_batch"] == 0
    assert payload["honeytokens_enabled"] is True
    assert payload["chaff_enabled"] is False
    assert payload["canaries_enabled"] is False

    # Survives a fresh read.
    assert harness.client.get("/api/settings").json() == payload

    stored = json.loads(harness.settings_path.read_text(encoding="utf-8"))
    assert stored["canaries_per_batch"] == 0
    assert stored["chaff_enabled"] is False
    assert stored["canaries_enabled"] is False


def test_clearing_a_numeric_setting_falls_back_to_the_config_default(harness):
    config = load_config()
    harness.client.put("/api/settings", json={"chunk_size_tokens": 6})
    assert harness.client.get("/api/settings").json()["chunk_size_tokens"] == 6

    payload = harness.client.put(
        "/api/settings", json={"chunk_size_tokens": None}
    ).json()

    assert payload["chunk_size_tokens"] == config.chunking.chunk_size_tokens
    assert "chunk_size_tokens" not in json.loads(
        harness.settings_path.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "update",
    [
        {"chunk_size_tokens": 0},
        {"chunk_size_tokens": -1},
        {"chaff_ratio": -0.1},
        {"honeytoken_rate": -1.0},
        {"canaries_per_batch": -1},
    ],
)
def test_invalid_pipeline_values_are_rejected_and_leave_the_file_untouched(harness, update):
    # A prior, valid override must survive an update that fails validation.
    harness.client.put("/api/settings", json={"chunk_size_tokens": 6})
    before = harness.settings_path.read_text(encoding="utf-8")

    response = harness.client.put("/api/settings", json=update)

    assert response.status_code == 400
    assert harness.settings_path.read_text(encoding="utf-8") == before
    assert harness.client.get("/api/settings").json()["chunk_size_tokens"] == 6


def test_settings_file_with_a_malformed_pipeline_value_is_a_400_on_read(harness):
    stored = json.loads(harness.settings_path.read_text(encoding="utf-8"))
    stored["chunk_size_tokens"] = "eight"  # wrong type, written outside the app
    harness.settings_path.write_text(json.dumps(stored), encoding="utf-8")

    response = harness.client.get("/api/settings")

    assert response.status_code == 400
    assert "chunk_size_tokens" in response.json()["detail"]


def test_chunk_size_override_reaches_the_chunker(harness):
    default_document = upload_markdown(harness, name="default.md")
    default_count = len(default_document["chunks"])

    harness.client.put("/api/settings", json={"chunk_size_tokens": 3})
    small_document = upload_markdown(harness, name="small.md")

    assert len(small_document["chunks"]) > default_count


def test_chaff_and_canary_overrides_reach_the_detection_plan(harness):
    document = upload_markdown(harness)
    real_count = len(document["chunks"])
    arm_provider(harness, document)

    harness.client.put(
        "/api/settings", json={"chaff_ratio": 3.0, "canaries_per_batch": 1}
    )

    result = detect(harness, document["doc_id"])

    assert result["composition"]["chaff"] == round(real_count * 3.0)
    assert result["composition"]["canary"] == min(1, CANARY_COUNT)


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def test_upload_markdown_returns_chunks_whose_tokens_join_to_their_text(harness):
    document = upload_markdown(harness)

    assert document["filename"] == "review.md"
    # Markdown is taken verbatim, so the document survives the upload untouched.
    assert document["markdown"] == SAMPLE_MARKDOWN
    assert document["chunks"]

    chunk_size = load_config().chunking.chunk_size_tokens
    for chunk in document["chunks"]:
        assert "".join(chunk["tokens"]) == chunk["text"]
        # A display segment is one token unless a character spans several, so
        # the count is the real token count and never below the segments.
        assert chunk["token_count"] >= len(chunk["tokens"])
        assert chunk["extended"] == (chunk["token_count"] > chunk_size)

    # Reassembling the chunks reproduces the document byte for byte.
    assert "".join(chunk["text"] for chunk in document["chunks"]) == SAMPLE_MARKDOWN
    assert [chunk["seq"] for chunk in document["chunks"]] == list(
        range(len(document["chunks"]))
    )


def test_upload_shows_multibyte_characters_whole(harness):
    """A name whose letters span two tokens must not be shown as U+FFFD.

    This is the difference between decoding each token on its own and decoding
    the token bytes as they accumulate: "Šimeček" costs several tokens per
    accented letter, and the per-token rendering turns each of them into a
    replacement character.
    """
    document = upload_markdown(harness, name="cz.md", text=MULTIBYTE_MARKDOWN)

    assert document["markdown"] == MULTIBYTE_MARKDOWN
    for chunk in document["chunks"]:
        assert "".join(chunk["tokens"]) == chunk["text"]
        assert "�" not in "".join(chunk["tokens"])
    assert "".join(chunk["text"] for chunk in document["chunks"]) == MULTIBYTE_MARKDOWN
    # The name really did travel through the payload intact.
    rendered = "".join(
        segment for chunk in document["chunks"] for segment in chunk["tokens"]
    )
    assert "Petr Šimeček <petr@keboola.com>" in rendered


def test_upload_uses_the_conversion_service_when_one_is_configured(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")
    harness.http.serve_conversion("# Converted\n\nJan Novak signed it.\n")

    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.pdf", b"%PDF-1.7 fake bytes", "application/pdf")},
    )

    assert response.status_code == 200, response.text
    document = response.json()
    assert document["markdown"] == "# Converted\n\nJan Novak signed it.\n"
    assert len(harness.http.posted) == 1
    call = harness.http.posted[0]
    assert call["url"] == "http://converter:9000/convert"
    filename, content = call["files"]["file"]
    assert filename == "report.pdf"
    assert content == b"%PDF-1.7 fake bytes"
    # Converting is the slow call, so it must not run on the preflight timeout.
    assert harness.http.timeouts[-1] == server.CONVERT_TIMEOUT_SECONDS


def test_markdown_upload_never_reaches_the_conversion_service(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")

    document = upload_markdown(harness)

    assert document["markdown"] == SAMPLE_MARKDOWN
    assert harness.http.posted == []


def test_upload_without_a_service_is_rejected_with_guidance(harness):
    """The app ships no converter: non-text uploads need the companion service."""
    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.html", SAMPLE_HTML.encode("utf-8"), "text/html")},
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "no conversion service" in detail
    assert "doc_converter" in detail
    assert "Settings" in detail
    assert harness.http.posted == []


def test_upload_reports_an_unreachable_conversion_service(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")
    harness.http.post_error = httpx.ConnectError("connection refused")

    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.pdf", b"%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "http://converter:9000/convert" in detail
    assert "unreachable" in detail


def test_upload_reports_a_conversion_service_error_response(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")
    harness.http.serve_conversion("", status_code=500)

    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.pdf", b"%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 502
    assert "500" in response.json()["detail"]


def test_upload_reports_a_conversion_service_payload_without_markdown(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")
    harness.http.post_error = None
    harness.http.post_response = SimpleNamespace(
        status_code=200, json=lambda: {"text": "wrong field"}
    )

    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.pdf", b"%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 502
    assert "unexpected payload" in response.json()["detail"]


def test_upload_of_an_empty_document_is_rejected(harness):
    response = harness.client.post(
        "/api/documents", files={"file": ("empty.md", b"   \n", "text/markdown")}
    )

    assert response.status_code == 422
    assert "empty.md" in response.json()["detail"]


# ---------------------------------------------------------------------------
# pasted text
# ---------------------------------------------------------------------------


def test_pasted_text_is_stored_verbatim_with_lossless_chunks(harness):
    response = harness.client.post("/api/documents/text", json={"text": SAMPLE_MARKDOWN})

    assert response.status_code == 200, response.text
    document = response.json()
    assert document["filename"] == server.FALLBACK_PASTED_NAME
    # Pasted text takes the verbatim path: no conversion, byte-for-byte storage.
    assert document["markdown"] == SAMPLE_MARKDOWN
    assert harness.http.posted == []
    assert "".join(chunk["text"] for chunk in document["chunks"]) == SAMPLE_MARKDOWN

    listing = harness.client.get("/api/documents").json()
    assert len(listing) == 1
    assert listing[0]["doc_id"] == document["doc_id"]
    assert listing[0]["path"] == server.FALLBACK_PASTED_NAME


def test_pasted_text_honours_a_custom_name(harness):
    response = harness.client.post(
        "/api/documents/text",
        json={"text": SAMPLE_MARKDOWN, "name": "  board-memo.md  "},
    )

    assert response.status_code == 200, response.text
    assert response.json()["filename"] == "board-memo.md"


def test_pasted_text_with_a_blank_name_falls_back_to_the_default(harness):
    response = harness.client.post(
        "/api/documents/text", json={"text": SAMPLE_MARKDOWN, "name": "   "}
    )

    assert response.status_code == 200, response.text
    assert response.json()["filename"] == server.FALLBACK_PASTED_NAME


def test_pasted_blank_text_is_rejected(harness):
    response = harness.client.post("/api/documents/text", json={"text": "   \n\t"})

    assert response.status_code == 422
    assert "empty" in response.json()["detail"]


def test_pasted_multibyte_text_survives_chunk_display(harness):
    response = harness.client.post(
        "/api/documents/text", json={"text": MULTIBYTE_MARKDOWN}
    )

    assert response.status_code == 200, response.text
    document = response.json()
    for chunk in document["chunks"]:
        assert "".join(chunk["tokens"]) == chunk["text"]
        assert "�" not in "".join(chunk["tokens"])
    assert "".join(chunk["text"] for chunk in document["chunks"]) == MULTIBYTE_MARKDOWN


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


def test_documents_are_listed_and_readable(harness):
    document = upload_markdown(harness)

    listing = harness.client.get("/api/documents").json()
    assert len(listing) == 1
    assert listing[0]["doc_id"] == document["doc_id"]
    assert listing[0]["path"] == "review.md"
    assert listing[0]["chunk_count"] == len(document["chunks"])
    assert listing[0]["created_at"]

    detail = harness.client.get(f"/api/documents/{document['doc_id']}").json()
    assert detail["markdown"] == SAMPLE_MARKDOWN
    assert detail["path"] == "review.md"
    assert detail["chunks"] == document["chunks"]


def test_unknown_document_is_a_404(harness):
    for path in ("/api/documents/nope", "/api/documents/nope/redaction"):
        response = harness.client.get(path)
        assert response.status_code == 404
        assert "nope" in response.json()["detail"]


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_detect_mixes_shuffles_and_routes_results(harness):
    document = upload_markdown(harness)
    doc_id = document["doc_id"]
    real_count = len(document["chunks"])
    config = load_config()
    arm_provider(harness, document)

    result = detect(harness, doc_id)

    assert result["batch_id"].startswith("sync-")

    # --- composition follows the configured mixing math ---
    expected_chaff = round(real_count * config.synthetic.chaff_ratio)
    assert result["composition"] == {
        "real": real_count,
        "honeytoken": max(1, round(real_count * config.synthetic.honeytoken_rate)),
        "chaff": expected_chaff,
        "canary": min(config.synthetic.canaries_per_batch, CANARY_COUNT),
    }

    # --- the submission carries every id, real and synthetic alike ---
    total = sum(result["composition"].values())
    assert len(result["requests"]) == total
    assert len(result["results"]) == total
    chunk_ids = {chunk["chunk_id"] for chunk in document["chunks"]}
    fragment_ids = {fragment.fragment_id for fragment in harness.fragments}
    submitted = {item["custom_id"] for item in result["requests"]}
    assert submitted == chunk_ids | fragment_ids
    # Every request is answered exactly once. The requests run several at a
    # time, so the answers arrive in completion order rather than in submission
    # order; that ordering is pinned in the concurrency tests below.
    assert Counter(item["custom_id"] for item in result["results"]) == Counter(
        item["custom_id"] for item in result["requests"]
    )

    # Only real requests carry a sequence number; synthetic ones must not.
    for item in result["requests"]:
        assert (item["seq"] is None) == (item["kind"] != "real")

    # --- the payload is the batch path's payload ---
    template = result["payload_template"]
    assert template == {
        "model": config.anthropic.model,
        "max_tokens": config.anthropic.max_tokens,
        "system": DETECTION_SYSTEM_PROMPT,
        "output_config": {
            "effort": config.anthropic.effort,
            "format": {"type": "json_schema", "schema": ENTITY_SCHEMA},
        },
    }
    assert len(harness.provider.calls) == total
    for call in harness.provider.calls:
        assert call["model"] == config.anthropic.model
        assert call["max_tokens"] == config.anthropic.max_tokens
        assert call["system"] == DETECTION_SYSTEM_PROMPT
        assert call["output_config"] == template["output_config"]
        assert len(call["messages"]) == 1

    assert all(item["status"] == "ok" for item in result["results"])
    assert all(item["latency_ms"] >= 0 for item in result["results"])


def test_detect_stores_entities_for_real_chunks_only(harness):
    document = upload_markdown(harness)
    expected = arm_provider(harness, document)
    assert expected, "the sample document must contain at least one known name"

    result = detect(harness, document["doc_id"])

    assert result["entities_stored"] == sum(len(names) for names in expected.values())

    store = harness.store()
    try:
        stored = store.get_document_entities(document["doc_id"])
        expected_pairs = {pair for names in expected.values() for pair in names}
        assert set(stored) == expected_pairs
        # The honeytoken's planted names are synthetic: they belong to no
        # document and must never reach the entity tables.
        for name, _ in HONEYTOKEN_PLANTED:
            assert all(name != text for text, _ in stored)
    finally:
        store.close()


def test_detect_measures_honeytoken_recall(harness):
    document = upload_markdown(harness)
    arm_provider(harness, document)

    result = detect(harness, document["doc_id"])

    honeytokens = harness.synthetic(KIND_HONEYTOKEN)
    planted = len(honeytokens) * len(HONEYTOKEN_PLANTED)
    found = len(honeytokens) * len(HONEYTOKEN_REPORTED)
    assert result["honeytoken_recall"] == {
        "planted": planted,
        "found": found,
        "recall": found / planted,
    }

    store = harness.store()
    try:
        stats = store.honeytoken_stats()
        assert len(stats) == 1
        assert stats[0]["batch_id"] == result["batch_id"]
        assert stats[0]["planted_total"] == planted
        assert stats[0]["found_total"] == found
    finally:
        store.close()


def test_detect_marks_chunks_and_fragments_as_submitted(harness):
    document = upload_markdown(harness)
    arm_provider(harness, document)

    result = detect(harness, document["doc_id"])

    store = harness.store()
    try:
        assert all(
            chunk["batch_id"] == result["batch_id"]
            for chunk in store.get_document_chunks(document["doc_id"])
        )
        for fragment in harness.fragments:
            row = store.get_synthetic_fragment(fragment.fragment_id)
            assert row is not None
            assert row["batch_id"] == result["batch_id"]
        assert [batch["status"] for batch in store.list_batches()] == ["sync-completed"]
    finally:
        store.close()


def test_detect_twice_reports_that_nothing_is_left(harness):
    document = upload_markdown(harness)
    arm_provider(harness, document)
    detect(harness, document["doc_id"])

    response = harness.client.post("/api/detect", json={"doc_id": document["doc_id"]})

    assert response.status_code == 409
    assert response.json()["detail"] == "no unsubmitted chunks"


def test_detect_on_an_unknown_document_is_a_404(harness):
    response = harness.client.post("/api/detect", json={"doc_id": "nope"})

    assert response.status_code == 404


def test_detect_reports_refusals_and_errors_per_request(harness):
    document = upload_markdown(harness)
    expected = arm_provider(harness, document)
    refused = document["chunks"][0]["text"]
    errored = document["chunks"][-1]["text"]
    assert refused != errored
    harness.provider.refuse_texts.add(refused)
    harness.provider.error_texts.add(errored)

    result = detect(harness, document["doc_id"])

    by_id = {item["custom_id"]: item for item in result["results"]}
    refused_result = by_id[document["chunks"][0]["chunk_id"]]
    assert refused_result["status"] == "refusal"
    assert refused_result["entities"] == []
    assert refused_result["detail"] is None
    assert refused_result["raw_text"] == "I cannot help with that."

    errored_result = by_id[document["chunks"][-1]["chunk_id"]]
    assert errored_result["status"] == "error"
    assert errored_result["entities"] == []
    assert errored_result["raw_text"] is None
    assert "upstream exploded" in errored_result["detail"]

    # The rest of the run went out regardless.
    assert sum(1 for item in result["results"] if item["status"] == "ok") == (
        len(result["results"]) - 2
    )

    # Nothing was stored for the two fragments that produced no usable answer.
    assert result["entities_stored"] == sum(
        len(names)
        for text, names in expected.items()
        if text not in (refused, errored)
    )


def test_detect_streams_its_progress_in_order(harness):
    """The stream tells the story of the run, step by step."""
    document = upload_markdown(harness)
    arm_provider(harness, document)
    config = load_config()

    events = detect_events(harness, document["doc_id"])

    types = [event["type"] for event in events]
    assert types[0] == "phase"
    assert types.count("submitted") == 1
    assert types.count("done") == 1
    assert types[-1] == "done"
    assert "error" not in types

    # --- phases come first, then the fragments they generate, then the
    # submission, then one result per request ---
    submitted_at = types.index("submitted")
    done_at = types.index("done")
    first_synthetic = types.index("synthetic")
    assert types.index("phase") < first_synthetic < submitted_at
    assert all(index < submitted_at for index, kind in enumerate(types) if kind == "synthetic")
    assert all(
        submitted_at < index < done_at
        for index, kind in enumerate(types)
        if kind == "result"
    )

    phases = [event for event in events if event["type"] == "phase"]
    assert [phase["phase"] for phase in phases] == [
        "planning",
        "synthetics",
        "canaries",
    ]
    assert phases[0]["detail"] == f"{len(document['chunks'])} real fragments"
    assert config.synthetic.llm.model in phases[1]["detail"]
    assert config.synthetic.llm.base_url in phases[1]["detail"]
    assert phases[2]["detail"] == "ensuring canary set"

    # --- one synthetic event per generated fragment, counted as it goes ---
    done = events[-1]
    synthetic_events = [event for event in events if event["type"] == "synthetic"]
    by_kind = Counter(event["kind"] for event in synthetic_events)
    assert by_kind == Counter(
        {
            kind: count
            for kind, count in done["composition"].items()
            if kind != "real" and count
        }
    )
    for kind, count in by_kind.items():
        indexes = [event["index"] for event in synthetic_events if event["kind"] == kind]
        assert indexes == list(range(1, count + 1))
        assert all(
            event["total"] == count
            for event in synthetic_events
            if event["kind"] == kind
        )
    assert {event["fragment_id"] for event in synthetic_events} == {
        fragment.fragment_id for fragment in harness.fragments
    }

    # --- the submission is announced whole, before any answer arrives ---
    submitted = events[submitted_at]
    assert submitted["batch_id"] == done["batch_id"]
    assert submitted["composition"] == done["composition"]
    assert submitted["payload_template"] == done["payload_template"]
    assert submitted["requests"] == done["requests"]

    # --- one result event per request, in submission order ---
    results = [event for event in events if event["type"] == "result"]
    assert len(results) == len(done["requests"])
    assert [result["index"] for result in results] == list(
        range(1, len(results) + 1)
    )
    assert all(result["total"] == len(results) for result in results)
    assert Counter(result["custom_id"] for result in results) == Counter(
        item["custom_id"] for item in done["requests"]
    )
    for result, recorded in zip(results, done["results"]):
        assert {key: result[key] for key in recorded} == recorded


def test_detect_events_are_produced_one_at_a_time(harness):
    """Progress that only arrives at the end is not progress.

    Asserted on the generator behind the endpoint rather than through the test
    client, which buffers a streamed answer no matter how the server produced
    it. What matters here is that each event exists before the work that
    follows it has been done.
    """
    document = upload_markdown(harness)
    arm_provider(harness, document)

    store = harness.store()
    try:
        chunks = store.get_document_chunks(document["doc_id"])
        stream = server._detect_events(
            # One worker, so "one more event" means exactly one more call.
            with_concurrency(server.get_config(), 1),
            store,
            harness.provider,
            FakeGenerator(store, []),
            chunks,
        )
        try:
            events = []
            for line in stream:
                events.append(json.loads(line))
                if events[-1]["type"] == "submitted":
                    break

            # The submission was announced before a single fragment left.
            assert events[-1]["type"] == "submitted"
            assert harness.provider.calls == []

            # Pulling one more event costs exactly one provider call.
            assert json.loads(next(stream))["type"] == "result"
            assert len(harness.provider.calls) == 1
        finally:
            stream.close()
    finally:
        store.close()


def test_detect_reports_a_mid_stream_failure_as_the_last_event(harness, monkeypatch):
    """Once the status line is sent, a break can only be reported in-band."""
    document = upload_markdown(harness)
    arm_provider(harness, document)

    def explode(*args, **kwargs):
        raise RuntimeError("store went away")

    monkeypatch.setattr(ChunkStore, "record_batch", explode)

    events = detect_events(harness, document["doc_id"])

    assert events[-1] == {"type": "error", "detail": "store went away"}
    # The events produced before the break are still there, and no run was
    # ever reported as finished.
    assert events[0]["type"] == "phase"
    assert not any(event["type"] in {"submitted", "done"} for event in events)
    # Nothing was submitted, so the provider was never called.
    assert harness.provider.calls == []


def test_detect_streams_multibyte_fragments_unharmed(harness):
    """The stream carries the fragment texts; UTF-8 must survive the encoding."""
    document = upload_markdown(harness, name="cz.md", text=MULTIBYTE_MARKDOWN)

    events = detect_events(harness, document["doc_id"])

    done = events[-1]
    real = [item for item in done["requests"] if item["kind"] == "real"]
    assert "".join(
        item["text"] for item in sorted(real, key=lambda item: item["seq"])
    ) == MULTIBYTE_MARKDOWN
    assert "Petr Šimeček" in "".join(item["text"] for item in real)


@pytest.mark.parametrize("workers", [1, 0, -4], ids=["one", "zero", "negative"])
def test_detect_with_a_single_worker_answers_in_submission_order(
    harness, monkeypatch, workers
):
    """One worker is the old sequential run; a nonsensical count becomes one."""
    set_concurrency(monkeypatch, workers)
    document = upload_markdown(harness)
    expected = arm_provider(harness, document)

    result = detect(harness, document["doc_id"])

    assert [item["custom_id"] for item in result["results"]] == [
        item["custom_id"] for item in result["requests"]
    ]
    assert [call["messages"][0]["content"] for call in harness.provider.calls] == [
        item["text"] for item in result["requests"]
    ]
    assert result["entities_stored"] == sum(len(names) for names in expected.values())
    assert all(item["status"] == "ok" for item in result["results"])


def test_detect_in_parallel_answers_every_fragment_and_routes_it(harness, monkeypatch):
    """Six in flight, answers arriving out of order, same bookkeeping."""
    set_concurrency(monkeypatch, 6)
    document = upload_markdown(harness)
    expected = arm_provider(harness, document)
    # Tiny, uneven delays: the completion order stops mirroring the submission
    # order, which is what the routing must survive.
    harness.provider.delay_seconds = 0.01

    events = detect_events(harness, document["doc_id"])
    done = events[-1]

    assert done["type"] == "done"
    total = sum(done["composition"].values())
    assert len(done["requests"]) == total
    assert len(done["results"]) == total
    assert len(harness.provider.calls) == total

    # --- one result event per fragment, no repeats, no losses ---
    result_events = [event for event in events if event["type"] == "result"]
    assert len(result_events) == total
    assert [event["index"] for event in result_events] == list(range(1, total + 1))
    assert all(event["total"] == total for event in result_events)
    assert Counter(event["custom_id"] for event in result_events) == Counter(
        item["custom_id"] for item in done["requests"]
    )
    assert all(event["status"] == "ok" for event in result_events)

    # --- storage routing is unchanged by the concurrency ---
    assert done["entities_stored"] == sum(len(names) for names in expected.values())
    honeytokens = harness.synthetic(KIND_HONEYTOKEN)
    assert done["honeytoken_recall"] == {
        "planted": len(honeytokens) * len(HONEYTOKEN_PLANTED),
        "found": len(honeytokens) * len(HONEYTOKEN_REPORTED),
        "recall": len(HONEYTOKEN_REPORTED) / len(HONEYTOKEN_PLANTED),
    }

    store = harness.store()
    try:
        stored = store.get_document_entities(document["doc_id"])
        assert set(stored) == {pair for names in expected.values() for pair in names}
        for name, _ in HONEYTOKEN_PLANTED:
            assert all(name != text for text, _ in stored)
        assert [batch["status"] for batch in store.list_batches()] == ["sync-completed"]
    finally:
        store.close()


def test_detect_names_the_templates_when_the_local_llm_is_off(harness):
    harness.client.put("/api/settings", json={"llm_enabled": False})
    document = upload_markdown(harness)
    arm_provider(harness, document)

    events = detect_events(harness, document["doc_id"])

    phases = [event for event in events if event["type"] == "phase"]
    assert phases[1]["phase"] == "synthetics"
    assert phases[1]["detail"] == PHASE_SYNTHETICS_TEMPLATE_DETAIL
    assert events[-1]["type"] == "done"


def test_detect_without_an_api_key_changes_nothing(harness):
    document = upload_markdown(harness)
    harness.client.put("/api/settings", json={"anthropic_api_key": ""})

    response = harness.client.post("/api/detect", json={"doc_id": document["doc_id"]})

    assert response.status_code == 400
    assert response.json()["detail"] == MISSING_API_KEY_MESSAGE
    assert harness.provider.calls == []
    store = harness.store()
    try:
        # No batch was recorded and no chunk was marked, so the document can
        # still be detected once a key is configured.
        assert store.list_batches() == []
        assert all(
            chunk["batch_id"] is None
            for chunk in store.get_document_chunks(document["doc_id"])
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------


def test_redaction_replaces_every_detected_entity(harness):
    document = upload_markdown(harness)
    expected = arm_provider(harness, document)
    detect(harness, document["doc_id"])
    config = load_config()

    payload = harness.client.get(
        f"/api/documents/{document['doc_id']}/redaction"
    ).json()

    assert payload["original"] == SAMPLE_MARKDOWN
    expected_pairs = {pair for names in expected.values() for pair in names}
    assert {(item["text"], item["type"]) for item in payload["entities"]} == expected_pairs
    for text, kind in expected_pairs:
        assert text not in payload["redacted"]
        placeholder = (
            config.redaction.person if kind == "person" else config.redaction.company
        )
        assert placeholder in payload["redacted"]


def test_redaction_without_detection_returns_the_original(harness):
    document = upload_markdown(harness)

    payload = harness.client.get(
        f"/api/documents/{document['doc_id']}/redaction"
    ).json()

    assert payload["entities"] == []
    assert payload["redacted"] == payload["original"] == SAMPLE_MARKDOWN


def test_redaction_reports_and_removes_email_addresses(harness):
    document = upload_markdown(harness, name="contacts.md", text=EMAIL_MARKDOWN)
    arm_provider(harness, document)
    detect(harness, document["doc_id"])
    config = load_config()

    payload = harness.client.get(
        f"/api/documents/{document['doc_id']}/redaction"
    ).json()

    reported = {item["text"] for item in payload["entities"] if item["type"] == "email"}
    assert reported == set(SAMPLE_EMAILS)
    assert config.redaction.email in payload["redacted"]
    for address in SAMPLE_EMAILS:
        assert address not in payload["redacted"]
    # The local part is a person's name, and it must not survive next to a
    # redacted domain the way it used to.
    assert "jan.novak" not in payload["redacted"]


def test_redaction_reports_and_removes_urls(harness):
    document = upload_markdown(harness, name="links.md", text=URL_MARKDOWN)
    arm_provider(harness, document)
    detect(harness, document["doc_id"])
    config = load_config()

    payload = harness.client.get(
        f"/api/documents/{document['doc_id']}/redaction"
    ).json()

    reported = {item["text"] for item in payload["entities"] if item["type"] == "url"}
    assert reported == set(SAMPLE_URLS)
    assert config.redaction.url in payload["redacted"]
    for address in SAMPLE_URLS:
        assert address not in payload["redacted"]
    # The identifying path segment is gone, not just the host.
    assert "3979-4561-9198" not in payload["redacted"]
    # All three deterministic and detected kinds coexist in one document.
    assert config.redaction.email in payload["redacted"]
    assert config.redaction.person in payload["redacted"]


def test_redaction_removes_emails_without_any_detection(harness):
    # Emails need no detector: nothing has been sent to the provider here.
    document = upload_markdown(harness, name="contacts.md", text=EMAIL_MARKDOWN)

    payload = harness.client.get(
        f"/api/documents/{document['doc_id']}/redaction"
    ).json()

    assert [item["type"] for item in payload["entities"]] == ["email"] * len(SAMPLE_EMAILS)
    for address in SAMPLE_EMAILS:
        assert address in payload["original"]
        assert address not in payload["redacted"]


# ---------------------------------------------------------------------------
# stored run
# ---------------------------------------------------------------------------


def stored_run(harness: Harness, doc_id: str) -> dict:
    response = harness.client.get(f"/api/documents/{doc_id}/run")
    assert response.status_code == 200, response.text
    return response.json()


def sorted_entities(entities: list[dict]) -> list[dict]:
    return sorted(entities, key=lambda item: (item["text"], item["type"]))


def test_stored_run_is_empty_before_detection(harness):
    document = upload_markdown(harness)

    payload = stored_run(harness, document["doc_id"])

    assert payload["has_run"] is False
    assert payload["chunk_count"] == len(document["chunks"])
    assert payload["chunks_submitted"] == 0
    assert payload["batches"] == []
    assert payload["requests"] == []
    assert payload["composition"] is None
    assert payload["entities_stored"] == 0
    assert payload["honeytoken_recall"] is None


def test_stored_run_reconstructs_a_finished_run(harness):
    document = upload_markdown(harness)
    expected = arm_provider(harness, document)
    result = detect(harness, document["doc_id"])

    payload = stored_run(harness, document["doc_id"])

    assert payload["has_run"] is True
    assert payload["chunk_count"] == len(document["chunks"])
    assert payload["chunks_submitted"] == len(document["chunks"])
    assert [batch["batch_id"] for batch in payload["batches"]] == [result["batch_id"]]
    assert payload["batches"][0]["status"] == "sync-completed"
    assert payload["composition"] == result["composition"]
    assert payload["entities_stored"] == result["entities_stored"]
    assert payload["honeytoken_recall"] == result["honeytoken_recall"]

    # Real chunks come back in document order with the synthetics after them:
    # the provider-order shuffle of the live run is never persisted, so the
    # stored run makes no claim about it.
    requests = payload["requests"]
    real = [item for item in requests if item["kind"] == "real"]
    assert [item["seq"] for item in real] == list(range(len(document["chunks"])))

    by_id = {item["custom_id"]: item for item in requests}
    for chunk in document["chunks"]:
        entry = by_id[chunk["chunk_id"]]
        assert entry["text"] == chunk["text"]
        expected_entities = [
            {"text": text, "type": kind}
            for text, kind in expected.get(chunk["text"], [])
        ]
        assert sorted_entities(entry["entities"]) == sorted_entities(expected_entities)

    for fragment in harness.synthetic(KIND_HONEYTOKEN):
        entry = by_id[fragment.fragment_id]
        assert entry["kind"] == KIND_HONEYTOKEN
        assert entry["seq"] is None
        assert entry["entities"] == [
            {"text": text, "type": kind} for text, kind in HONEYTOKEN_REPORTED
        ]
    # Chaff dilutes and canaries wait to be probed; neither has an answer that
    # the run keeps, and the payload says so with null rather than [].
    for fragment in harness.synthetic(KIND_CHAFF) + harness.synthetic(KIND_CANARY):
        assert by_id[fragment.fragment_id]["entities"] is None


def test_stored_run_on_an_unknown_document_is_a_404(harness):
    response = harness.client.get("/api/documents/nope/run")

    assert response.status_code == 404


def test_redaction_reports_detection_state(harness):
    document = upload_markdown(harness)
    doc_id = document["doc_id"]

    before = harness.client.get(f"/api/documents/{doc_id}/redaction").json()
    assert before["has_detection"] is False
    assert before["chunk_count"] == len(document["chunks"])
    assert before["chunks_submitted"] == 0

    arm_provider(harness, document)
    detect(harness, doc_id)

    after = harness.client.get(f"/api/documents/{doc_id}/redaction").json()
    assert after["has_detection"] is True
    assert after["chunk_count"] == len(document["chunks"])
    assert after["chunks_submitted"] == len(document["chunks"])


# ---------------------------------------------------------------------------
# synthetic report
# ---------------------------------------------------------------------------


def test_report_is_empty_before_anything_is_generated(harness):
    payload = harness.client.get("/api/synthetic/report").json()

    assert payload["counts"] == {"honeytoken": 0, "chaff": 0, "canary": 0}
    assert payload["honeytoken_stats"] == []
    assert payload["canary_probes"] == []


def test_report_counts_fragments_and_recall_after_a_run(harness):
    document = upload_markdown(harness)
    arm_provider(harness, document)
    result = detect(harness, document["doc_id"])

    payload = harness.client.get("/api/synthetic/report").json()

    assert payload["counts"]["honeytoken"] == len(harness.synthetic(KIND_HONEYTOKEN))
    assert payload["counts"]["chaff"] == len(harness.synthetic(KIND_CHAFF))
    assert payload["counts"]["canary"] == CANARY_COUNT
    assert len(payload["honeytoken_stats"]) == 1
    assert payload["honeytoken_stats"][0]["batch_id"] == result["batch_id"]
    assert payload["honeytoken_stats"][0]["recall"] == pytest.approx(
        result["honeytoken_recall"]["recall"]
    )


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def verify(harness: Harness) -> dict[str, dict]:
    """Run the preflight and return its checks keyed by name."""
    response = harness.client.post("/api/verify")
    assert response.status_code == 200, response.text
    payload = response.json()
    checks = {check["name"]: check for check in payload["checks"]}
    assert set(checks) == {
        "anthropic",
        "local_llm",
        "conversion",
        "database",
    }
    for check in payload["checks"]:
        assert check["label"]
        assert isinstance(check["ok"], bool)
        assert check["detail"]
        assert check["latency_ms"] >= 0
    assert payload["all_ok"] == all(check["ok"] for check in payload["checks"])
    return {**checks, "all_ok": payload["all_ok"]}


def test_verify_passes_when_the_whole_environment_is_there(harness):
    config = load_config()

    checks = verify(harness)

    assert checks["all_ok"] is True
    assert checks["anthropic"]["ok"] is True
    assert config.anthropic.model in checks["anthropic"]["detail"]
    assert harness.provider.retrieved == [config.anthropic.model]
    assert checks["local_llm"]["ok"] is True
    assert harness.http.requested == [f"{config.synthetic.llm.base_url}/models"]
    assert harness.http.closed is True
    assert checks["conversion"]["ok"] is True
    assert checks["database"]["ok"] is True
    assert str(harness.db_path) in checks["database"]["detail"]


def test_verify_without_an_api_key_points_at_the_settings(harness):
    harness.client.put("/api/settings", json={"anthropic_api_key": ""})

    checks = verify(harness)

    assert checks["anthropic"]["ok"] is False
    assert checks["anthropic"]["detail"] == VERIFY_NO_KEY_DETAIL
    assert checks["all_ok"] is False
    # A key that is not there is never sent anywhere.
    assert harness.provider.retrieved == []
    # The other checks are independent and still ran.
    assert checks["local_llm"]["ok"] is True
    assert checks["database"]["ok"] is True


def test_verify_surfaces_the_api_error_behind_a_bad_key(harness):
    harness.provider.model_error = anthropic.APIError(
        "invalid x-api-key", httpx.Request("GET", "http://test"), body=None
    )

    checks = verify(harness)

    assert checks["anthropic"]["ok"] is False
    assert "invalid x-api-key" in checks["anthropic"]["detail"]
    assert checks["all_ok"] is False


def test_verify_reports_an_unreachable_local_llm_with_a_hint(harness):
    config = load_config()
    harness.http.error = httpx.ConnectError("connection refused")

    checks = verify(harness)

    assert checks["local_llm"]["ok"] is False
    assert f"ollama pull {config.synthetic.llm.model}" in checks["local_llm"]["detail"]
    assert checks["all_ok"] is False
    # The endpoint failing says nothing about the rest of the environment.
    assert checks["anthropic"]["ok"] is True
    assert checks["conversion"]["ok"] is True


def test_verify_lists_what_the_local_llm_serves_instead(harness):
    config = load_config()
    harness.http.serve(["other-model", "yet-another"])

    checks = verify(harness)

    assert checks["local_llm"]["ok"] is False
    assert config.synthetic.llm.model in checks["local_llm"]["detail"]
    assert "other-model, yet-another" in checks["local_llm"]["detail"]


def test_verify_accepts_the_implicit_ollama_latest_tag(harness):
    config = load_config()
    harness.http.serve([f"{config.synthetic.llm.model}:latest"])

    checks = verify(harness)

    assert checks["local_llm"]["ok"] is True


def test_verify_reports_a_local_llm_that_answers_with_an_error(harness):
    harness.http.serve([], status_code=503)

    checks = verify(harness)

    assert checks["local_llm"]["ok"] is False
    assert "503" in checks["local_llm"]["detail"]


def test_verify_skips_the_local_llm_when_it_is_switched_off(harness):
    harness.client.put("/api/settings", json={"llm_enabled": False})

    checks = verify(harness)

    assert checks["local_llm"]["ok"] is True
    assert checks["local_llm"]["detail"] == VERIFY_LLM_DISABLED_DETAIL
    # A component deliberately not in use is never probed and never fails.
    assert harness.http.requested == []
    assert checks["all_ok"] is True


def test_verify_passes_the_conversion_check_without_a_service(harness):
    checks = verify(harness)

    assert checks["conversion"]["ok"] is True
    assert checks["conversion"]["detail"].startswith("no conversion service configured")
    # Text-only mode is a deliberate state, not a broken environment.
    assert harness.http.posted == []


def test_verify_probes_a_configured_conversion_service(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000/")
    harness.http.serve_conversion("# Preflight sample")

    checks = verify(harness)

    assert checks["conversion"]["ok"] is True
    assert checks["conversion"]["detail"].startswith(
        "conversion service at http://converter:9000/"
    )
    assert "characters of markdown" in checks["conversion"]["detail"]
    # The check took the same route an upload takes: POST /convert.
    assert [entry["url"] for entry in harness.http.posted] == [
        "http://converter:9000/convert"
    ]


def test_verify_probes_the_conversion_service_from_the_settings(harness):
    """The preflight checks the service the user configured, not the config's."""
    harness.client.put(
        "/api/settings", json={"conversion_service_url": "http://converter:9000"}
    )
    harness.http.serve_conversion("# Preflight sample")

    checks = verify(harness)

    assert checks["conversion"]["ok"] is True
    assert "conversion service at http://converter:9000" in checks["conversion"]["detail"]
    assert [entry["url"] for entry in harness.http.posted] == [
        "http://converter:9000/convert"
    ]


def test_verify_follows_a_conversion_service_cleared_in_the_settings(harness, monkeypatch):
    """With the field emptied, nothing is probed even though the config has a URL."""
    set_conversion_service(monkeypatch, "http://from-config:9000")
    harness.client.put("/api/settings", json={"conversion_service_url": ""})

    checks = verify(harness)

    assert checks["conversion"]["ok"] is True
    assert checks["conversion"]["detail"].startswith("no conversion service configured")
    assert harness.http.posted == []


def test_verify_reports_an_unreachable_conversion_service(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")
    harness.http.post_error = httpx.ConnectError("connection refused")

    checks = verify(harness)

    assert checks["conversion"]["ok"] is False
    assert "http://converter:9000/convert" in checks["conversion"]["detail"]
    assert "Conversion service URL" in checks["conversion"]["detail"]
    assert checks["all_ok"] is False


def test_verify_reports_a_conversion_service_error_response(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")
    harness.http.serve_conversion("", status_code=503)

    checks = verify(harness)

    assert checks["conversion"]["ok"] is False
    assert "503" in checks["conversion"]["detail"]


def test_verify_reports_a_service_that_returns_no_markdown(harness, monkeypatch):
    set_conversion_service(monkeypatch, "http://converter:9000")
    harness.http.serve_conversion("   ")

    checks = verify(harness)

    assert checks["conversion"]["ok"] is False
    assert "returned no markdown" in checks["conversion"]["detail"]
    assert checks["all_ok"] is False


def test_verify_reports_an_unusable_database(harness, monkeypatch):
    def unusable(config) -> None:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(server, "get_store", unusable)

    checks = verify(harness)

    assert checks["database"]["ok"] is False
    assert "unable to open database file" in checks["database"]["detail"]
    assert checks["all_ok"] is False


# ---------------------------------------------------------------------------
# offered local models
# ---------------------------------------------------------------------------


def test_llm_models_reports_the_catalog_and_what_is_installed(harness):
    config = load_config()
    catalogued = [entry.model for entry in config.synthetic.llm.catalog]
    harness.http.serve([catalogued[0], "some-other-model"])

    payload = harness.client.get("/api/llm-models").json()

    assert payload["llm_reachable"] is True
    assert payload["available"] == [catalogued[0], "some-other-model"]
    assert payload["catalog_note"] == config.synthetic.llm.catalog_note
    assert [entry["model"] for entry in payload["catalog"]] == catalogued
    # Every measured figure travels with the entry: the view renders them.
    for entry, configured in zip(payload["catalog"], config.synthetic.llm.catalog):
        assert entry["size"] == configured.size
        assert entry["seconds_per_fragment"] == pytest.approx(
            configured.seconds_per_fragment
        )
        assert entry["first_try_validity"] == pytest.approx(
            configured.first_try_validity
        )
        assert entry["note"] == configured.note
    availability = {entry["model"]: entry["available"] for entry in payload["catalog"]}
    assert availability[catalogued[0]] is True
    assert all(value is False for model, value in availability.items() if model != catalogued[0])
    assert harness.http.requested == [f"{config.synthetic.llm.base_url}/models"]


def test_llm_models_accepts_the_implicit_ollama_latest_tag(harness):
    config = load_config()
    catalogued = [entry.model for entry in config.synthetic.llm.catalog]
    harness.http.serve([f"{catalogued[1]}:latest"])

    payload = harness.client.get("/api/llm-models").json()

    availability = {entry["model"]: entry["available"] for entry in payload["catalog"]}
    assert availability[catalogued[1]] is True


def test_llm_models_reports_an_unreachable_local_server(harness):
    harness.http.error = httpx.ConnectError("connection refused")

    payload = harness.client.get("/api/llm-models").json()

    assert payload["llm_reachable"] is False
    assert payload["available"] == []
    # The catalog is config data, so it is offered whatever the server does.
    assert payload["catalog"]
    assert all(entry["available"] is False for entry in payload["catalog"])


def test_llm_models_treats_an_error_response_as_nothing_installed(harness):
    harness.http.serve([], status_code=503)

    payload = harness.client.get("/api/llm-models").json()

    assert payload["llm_reachable"] is False
    assert payload["available"] == []


def test_llm_models_survives_an_unexpected_payload(harness):
    harness.http.response = SimpleNamespace(
        status_code=200, json=lambda: {"models": ["not-the-openai-shape"]}
    )
    harness.http.error = None

    payload = harness.client.get("/api/llm-models").json()

    assert payload["llm_reachable"] is False
    assert payload["available"] == []


def test_llm_models_follows_the_configured_base_url_override(harness):
    harness.client.put("/api/settings", json={"llm_base_url": "http://elsewhere:1234/v1"})
    harness.http.requested.clear()

    harness.client.get("/api/llm-models")

    assert harness.http.requested == ["http://elsewhere:1234/v1/models"]


# ---------------------------------------------------------------------------
# canary probe
# ---------------------------------------------------------------------------


def seed_canaries(harness: Harness) -> list[SyntheticFragment]:
    canaries = [
        _build_canary(person, place)
        for person, place in zip(CANARY_PEOPLE, CANARY_PLACES)
    ]
    store = harness.store()
    try:
        store.add_synthetic_fragments(
            [
                {
                    "fragment_id": canary.fragment_id,
                    "kind": canary.kind,
                    "text": canary.text,
                    "planted": canary.planted,
                    "fact": canary.fact,
                }
                for canary in canaries
            ]
        )
    finally:
        store.close()
    return canaries


def test_canary_probe_records_one_verdict_per_canary(harness):
    canaries = seed_canaries(harness)
    # Only the first canary's nonce comes back, so exactly one probe trips.
    harness.provider.raw_by_substring[CANARY_PEOPLE[0]] = (
        f"They run the records office in {CANARY_PLACES[0]}."
    )
    harness.provider.raw_by_substring[CANARY_PEOPLE[1]] = (
        "I have never heard of them."
    )

    payload = harness.client.post("/api/canary-probe", json={}).json()

    assert payload["total"] == len(canaries)
    assert payload["tripped"] == 1
    by_person = {item["person"]: item for item in payload["results"]}
    assert set(by_person) == set(CANARY_PEOPLE)
    assert by_person[CANARY_PEOPLE[0]]["tripped"] is True
    assert CANARY_PLACES[0] in by_person[CANARY_PEOPLE[0]]["excerpt"]
    assert by_person[CANARY_PEOPLE[1]]["tripped"] is False
    assert {item["fragment_id"] for item in payload["results"]} == {
        canary.fragment_id for canary in canaries
    }

    store = harness.store()
    try:
        probes = store.list_canary_probes()
        assert len(probes) == len(canaries)
        assert sum(1 for probe in probes if probe["tripped"]) == 1
    finally:
        store.close()


def test_canary_probe_uses_the_requested_model(harness):
    seed_canaries(harness)

    harness.client.post("/api/canary-probe", json={"model": "claude-probe-model"})

    assert harness.provider.calls
    assert all(call["model"] == "claude-probe-model" for call in harness.provider.calls)


def test_canary_probe_without_an_api_key_is_a_400(harness):
    seed_canaries(harness)
    harness.client.put("/api/settings", json={"anthropic_api_key": ""})

    response = harness.client.post("/api/canary-probe", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == MISSING_API_KEY_MESSAGE
    assert harness.provider.calls == []


def test_settings_model_override_reaches_the_detection_payload(harness):
    harness.client.put("/api/settings", json={"model": "claude-override", "effort": "high"})
    document = upload_markdown(harness)
    arm_provider(harness, document)

    result = detect(harness, document["doc_id"])

    assert result["payload_template"]["model"] == "claude-override"
    assert result["payload_template"]["output_config"]["effort"] == "high"
    assert all(call["model"] == "claude-override" for call in harness.provider.calls)
