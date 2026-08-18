"""Tests for the Batches API detection driver.

Everything is exercised against in-memory stand-ins: no network, no API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from doc_quant.detector import DETECTION_SYSTEM_PROMPT, ENTITY_SCHEMA, Detector

MODEL = "claude-opus-5"
EFFORT = "low"
MAX_TOKENS = 1024


def make_config() -> SimpleNamespace:
    """Config stand-in with every synthetic feature off.

    These tests cover the real-chunk path only; the synthetic fragments have
    their own file (test_detector_synthetic.py).
    """
    return SimpleNamespace(
        chunking=SimpleNamespace(
            chunk_size_tokens=22,
            encoding="cl100k_base",
            name_run_max_extension_tokens=12,
        ),
        anthropic=SimpleNamespace(model=MODEL, effort=EFFORT, max_tokens=MAX_TOKENS),
        synthetic=SimpleNamespace(
            honeytokens_enabled=False,
            chaff_enabled=False,
            canaries_enabled=False,
            chaff_ratio=1.0,
            honeytoken_rate=0.02,
            canary_set_size=5,
            canaries_per_batch=5,
        ),
    )


class FakeStore:
    """In-memory stand-in implementing only the methods Detector touches."""

    def __init__(
        self,
        unsubmitted: list[dict] | None = None,
        document_texts: dict[str, list[str]] | None = None,
    ) -> None:
        self._unsubmitted = unsubmitted or []
        self._document_texts = document_texts or {}
        self.recorded_batches: list[tuple[str, str]] = []
        self.submitted: list[tuple[list[str], str]] = []
        self.status_updates: list[tuple[str, str]] = []
        self.entities: dict[str, list[tuple[str, str]]] = {}
        self.document_text_calls: list[str] = []

    def get_unsubmitted_chunks(self) -> list[dict]:
        return list(self._unsubmitted)

    def get_document_chunk_texts(self, doc_id: str) -> list[str]:
        self.document_text_calls.append(doc_id)
        return list(self._document_texts[doc_id])

    def record_batch(self, batch_id: str, status: str) -> None:
        self.recorded_batches.append((batch_id, status))

    def mark_chunks_submitted(self, chunk_ids: list[str], batch_id: str) -> None:
        self.submitted.append((list(chunk_ids), batch_id))

    def set_batch_status(self, batch_id: str, status: str) -> None:
        self.status_updates.append((batch_id, status))

    def add_entities(self, chunk_id: str, entities: list[tuple[str, str]]) -> None:
        self.entities.setdefault(chunk_id, []).extend(entities)

    def get_synthetic_fragment(self, fragment_id: str) -> dict | None:
        # No synthetic fragments are generated in this file, so every id is real.
        return None


class FakeChunker:
    """Chunker stand-in.

    Submission no longer asks the chunker for anything: a request carries the
    stored chunk text verbatim. The parameter is kept so the wiring of the real
    Detector is exercised, and any call would be a regression.
    """

    def __getattr__(self, name: str):
        raise AssertionError(f"Detector must not call chunker.{name} during submission")


class FakeBatches:
    def __init__(self) -> None:
        self.create_calls: list[list[dict]] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []
        self.created_batch = SimpleNamespace(id="msgbatch_test", processing_status="in_progress")
        self.retrieved_batch = SimpleNamespace(id="msgbatch_test", processing_status="ended")
        self.result_items: list[SimpleNamespace] = []

    def create(self, requests: list[dict]) -> SimpleNamespace:
        self.create_calls.append(list(requests))
        return self.created_batch

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        self.retrieve_calls.append(batch_id)
        return self.retrieved_batch

    def results(self, batch_id: str):
        self.results_calls.append(batch_id)
        return iter(self.result_items)


class FakeClient:
    def __init__(self) -> None:
        self.batches = FakeBatches()
        self.messages = SimpleNamespace(batches=self.batches)


def succeeded_result(custom_id: str, text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    message = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=message),
    )


def failed_result(custom_id: str, result_type: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=result_type))


def entity_payload(*entities: tuple[str, str]) -> str:
    return json.dumps({"entities": [{"text": t, "type": k} for t, k in entities]})


def expected_counts(
    succeeded: int = 0, errored: int = 0, refused: int = 0, entities: int = 0
) -> dict:
    """Full fetch counts; the synthetic keys stay zero throughout this file."""
    return {
        "succeeded": succeeded,
        "errored": errored,
        "refused": refused,
        "entities": entities,
        "honeytokens_scored": 0,
        "synthetic_discarded": 0,
    }


@pytest.fixture
def two_chunk_store() -> FakeStore:
    return FakeStore(
        unsubmitted=[
            {"chunk_id": "chunk-a", "doc_id": "doc-1", "seq": 0, "text": "alpha"},
            {"chunk_id": "chunk-b", "doc_id": "doc-1", "seq": 1, "text": "beta"},
        ],
        document_texts={"doc-1": ["alpha", "beta"]},
    )


def make_detector(store: FakeStore, client: FakeClient) -> Detector:
    return Detector(make_config(), store, FakeChunker(), client=client)


# --- submit -----------------------------------------------------------------


def test_submit_builds_one_request_per_chunk(two_chunk_store):
    client = FakeClient()
    batch_id = make_detector(two_chunk_store, client).submit()

    assert batch_id == "msgbatch_test"
    assert len(client.batches.create_calls) == 1
    requests = client.batches.create_calls[0]
    assert len(requests) == 2
    # Order is shuffled, so identity is checked as a set.
    assert {request["custom_id"] for request in requests} == {"chunk-a", "chunk-b"}


def test_submit_uses_configured_model_effort_and_schema(two_chunk_store):
    client = FakeClient()
    make_detector(two_chunk_store, client).submit()

    for request in client.batches.create_calls[0]:
        params = request["params"]
        assert params["model"] == MODEL
        assert params["max_tokens"] == MAX_TOKENS
        assert params["system"] == DETECTION_SYSTEM_PROMPT
        assert params["output_config"]["effort"] == EFFORT
        assert params["output_config"]["format"] == {
            "type": "json_schema",
            "schema": ENTITY_SCHEMA,
        }


def test_submit_sends_the_stored_chunk_text_and_no_document_identity(two_chunk_store):
    client = FakeClient()
    make_detector(two_chunk_store, client).submit()

    by_id = {
        request["custom_id"]: request["params"]["messages"][0]["content"]
        for request in client.batches.create_calls[0]
    }
    # Exactly the stored text: no margin, so neighbouring requests share no
    # text that could be used to re-stitch the document.
    assert by_id["chunk-a"] == "alpha"
    assert by_id["chunk-b"] == "beta"

    # No request may leak the document id or the chunk's position.
    serialized = json.dumps(client.batches.create_calls[0], default=str)
    assert "doc-1" not in serialized
    assert "seq" not in serialized


def test_submit_never_reads_a_documents_ordered_texts():
    store = FakeStore(
        unsubmitted=[
            {"chunk_id": "a", "doc_id": "doc-1", "seq": 0, "text": "one"},
            {"chunk_id": "b", "doc_id": "doc-1", "seq": 1, "text": "two"},
            {"chunk_id": "c", "doc_id": "doc-2", "seq": 0, "text": "three"},
        ],
        document_texts={"doc-1": ["one", "two"], "doc-2": ["three"]},
    )
    make_detector(store, FakeClient()).submit()

    # Ordered texts were only ever needed to build windows.
    assert store.document_text_calls == []


def test_submit_records_batch_and_marks_chunks(two_chunk_store):
    client = FakeClient()
    make_detector(two_chunk_store, client).submit()

    assert two_chunk_store.recorded_batches == [("msgbatch_test", "in_progress")]
    assert len(two_chunk_store.submitted) == 1
    chunk_ids, batch_id = two_chunk_store.submitted[0]
    assert sorted(chunk_ids) == ["chunk-a", "chunk-b"]
    assert batch_id == "msgbatch_test"


def test_submit_with_no_chunks_returns_none_and_calls_nothing():
    store = FakeStore(unsubmitted=[])
    client = FakeClient()

    assert make_detector(store, client).submit() is None
    assert client.batches.create_calls == []
    assert store.recorded_batches == []
    assert store.submitted == []
    assert store.document_text_calls == []


# --- check ------------------------------------------------------------------


def test_check_returns_status_and_updates_store():
    store = FakeStore()
    client = FakeClient()

    status = make_detector(store, client).check("msgbatch_test")

    assert status == "ended"
    assert client.batches.retrieve_calls == ["msgbatch_test"]
    assert store.status_updates == [("msgbatch_test", "ended")]


# --- fetch ------------------------------------------------------------------


def test_fetch_stores_entities_from_succeeded_result():
    store = FakeStore()
    client = FakeClient()
    client.batches.result_items = [
        succeeded_result("chunk-a", entity_payload(("Jan Novak", "person"), ("Keboola", "company")))
    ]

    counts = make_detector(store, client).fetch("msgbatch_test")

    assert counts == expected_counts(succeeded=1, entities=2)
    assert store.entities == {"chunk-a": [("Jan Novak", "person"), ("Keboola", "company")]}
    assert store.status_updates == [("msgbatch_test", "fetched")]


def test_fetch_handles_empty_entity_list():
    store = FakeStore()
    client = FakeClient()
    client.batches.result_items = [succeeded_result("chunk-a", entity_payload())]

    counts = make_detector(store, client).fetch("msgbatch_test")

    assert counts == expected_counts(succeeded=1)
    assert store.entities == {"chunk-a": []}


def test_fetch_counts_refusal_without_parsing():
    store = FakeStore()
    client = FakeClient()
    client.batches.result_items = [
        succeeded_result("chunk-a", "I cannot help with that.", stop_reason="refusal")
    ]

    counts = make_detector(store, client).fetch("msgbatch_test")

    assert counts == expected_counts(refused=1)
    assert store.entities == {}


def test_fetch_counts_malformed_json_as_errored():
    store = FakeStore()
    client = FakeClient()
    client.batches.result_items = [succeeded_result("chunk-a", "{not json at all")]

    counts = make_detector(store, client).fetch("msgbatch_test")

    assert counts == expected_counts(errored=1)
    assert store.entities == {}


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({"wrong_key": []}),
        json.dumps({"entities": "not-a-list"}),
        json.dumps({"entities": [{"text": "Jan"}]}),
        json.dumps({"entities": [{"text": "Jan", "type": "location"}]}),
        json.dumps({"entities": [{"text": 42, "type": "person"}]}),
        json.dumps(["not", "an", "object"]),
    ],
)
def test_fetch_counts_schema_violations_as_errored(payload):
    store = FakeStore()
    client = FakeClient()
    client.batches.result_items = [succeeded_result("chunk-a", payload)]

    counts = make_detector(store, client).fetch("msgbatch_test")

    assert counts["errored"] == 1
    assert counts["succeeded"] == 0
    assert store.entities == {}


@pytest.mark.parametrize("result_type", ["errored", "canceled", "expired"])
def test_fetch_counts_failed_results_as_errored(result_type):
    store = FakeStore()
    client = FakeClient()
    client.batches.result_items = [failed_result("chunk-a", result_type)]

    counts = make_detector(store, client).fetch("msgbatch_test")

    assert counts == expected_counts(errored=1)
    assert store.entities == {}


def test_fetch_mixed_results_are_counted_independently():
    store = FakeStore()
    client = FakeClient()
    client.batches.result_items = [
        succeeded_result("chunk-a", entity_payload(("Jan", "person"))),
        succeeded_result("chunk-b", "refused", stop_reason="refusal"),
        succeeded_result("chunk-c", "}{"),
        failed_result("chunk-d", "expired"),
    ]

    counts = make_detector(store, client).fetch("msgbatch_test")

    assert counts == expected_counts(succeeded=1, errored=2, refused=1, entities=1)
    assert store.entities == {"chunk-a": [("Jan", "person")]}
    assert client.batches.results_calls == ["msgbatch_test"]
