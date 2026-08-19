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
import uuid
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
from webapp.settings import MISSING_API_KEY_MESSAGE

DUMMY_API_KEY = "sk-ant-testing-0123456789wxyz"
MASKED_DUMMY_KEY = "sk-a...wxyz"

SAMPLE_MARKDOWN = """# Quarterly Review

Jan Novak met Petra Svobodova at the Keboola office in Prague to discuss the
migration plan for the reporting warehouse. The team agreed to ship the new
connector before the end of the quarter and to review the remaining risks.
"""

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
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params) -> SimpleNamespace:
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

    with TestClient(server.app) as client:
        yield Harness(
            client=client,
            provider=provider,
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


def detect(harness: Harness, doc_id: str) -> dict:
    response = harness.client.post("/api/detect", json={"doc_id": doc_id})
    assert response.status_code == 200, response.text
    return response.json()


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
        assert chunk["token_count"] == len(chunk["tokens"])
        assert chunk["extended"] == (chunk["token_count"] > chunk_size)

    # Reassembling the chunks reproduces the document byte for byte.
    assert "".join(chunk["text"] for chunk in document["chunks"]) == SAMPLE_MARKDOWN
    assert [chunk["seq"] for chunk in document["chunks"]] == list(
        range(len(document["chunks"]))
    )


def test_upload_html_is_converted_to_markdown(harness):
    response = harness.client.post(
        "/api/documents",
        files={"file": ("report.html", SAMPLE_HTML.encode("utf-8"), "text/html")},
    )

    assert response.status_code == 200, response.text
    document = response.json()
    assert "# Acme Report" in document["markdown"]
    assert "<p>" not in document["markdown"]
    assert "".join(chunk["text"] for chunk in document["chunks"]) == document["markdown"]


def test_upload_of_an_empty_document_is_rejected(harness):
    response = harness.client.post(
        "/api/documents", files={"file": ("empty.md", b"   \n", "text/markdown")}
    )

    assert response.status_code == 422
    assert "empty.md" in response.json()["detail"]


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
    # Results are reported in the submission order.
    assert [item["custom_id"] for item in result["results"]] == [
        item["custom_id"] for item in result["requests"]
    ]

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
