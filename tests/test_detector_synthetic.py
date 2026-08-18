"""Tests for the synthetic fragments mixed into detection batches.

Everything runs against in-memory stand-ins: no network, no API key, and no
import of `doc_quant.synthetic` - the generator and the store are faked so that
only the contract between them and the detector is exercised here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from doc_quant.cli import (
    EXCERPT_ERROR,
    EXCERPT_REFUSAL,
    _cmd_synthetic_report,
    _fact_nonce,
    run_canary_probe,
)
from doc_quant.detector import (
    DETECTION_SYSTEM_PROMPT,
    ENTITY_SCHEMA,
    KIND_CANARY,
    KIND_CHAFF,
    KIND_HONEYTOKEN,
    Detector,
)

MODEL = "claude-opus-5"
EFFORT = "low"
MAX_TOKENS = 1024

HONEYTOKEN_RATE = 0.02
CHAFF_RATIO = 1.0
CANARY_SET_SIZE = 5
CANARIES_PER_BATCH = 5

BATCH_ID = "msgbatch_test"


def make_config(
    honeytokens_enabled: bool = True,
    chaff_enabled: bool = True,
    canaries_enabled: bool = True,
    canaries_per_batch: int = CANARIES_PER_BATCH,
) -> SimpleNamespace:
    """Config stand-in exposing only the sections the detector and CLI read."""
    return SimpleNamespace(
        chunking=SimpleNamespace(
            chunk_size_tokens=22,
            encoding="cl100k_base",
            name_run_max_extension_tokens=12,
        ),
        anthropic=SimpleNamespace(model=MODEL, effort=EFFORT, max_tokens=MAX_TOKENS),
        synthetic=SimpleNamespace(
            honeytokens_enabled=honeytokens_enabled,
            chaff_enabled=chaff_enabled,
            canaries_enabled=canaries_enabled,
            chaff_ratio=CHAFF_RATIO,
            honeytoken_rate=HONEYTOKEN_RATE,
            canary_set_size=CANARY_SET_SIZE,
            canaries_per_batch=canaries_per_batch,
            seed=1234,
            llm=SimpleNamespace(
                base_url="http://localhost:11434/v1",
                model="local-model",
                temperature=0.8,
                timeout_seconds=30,
            ),
        ),
    )


@dataclass(frozen=True)
class FakeFragment:
    """Local mirror of synthetic.SyntheticFragment; keeps this file independent."""

    fragment_id: str
    kind: str
    text: str
    planted: list[tuple[str, str]] = field(default_factory=list)
    fact: str | None = None


class FakeGenerator:
    """Stand-in for SyntheticGenerator handing out pre-built fragments."""

    def __init__(self, canary_count: int = CANARY_SET_SIZE) -> None:
        self.honeytoken_counts: list[int] = []
        self.chaff_counts: list[int] = []
        self.ensure_canaries_calls = 0
        self._canaries = [
            FakeFragment(
                fragment_id=f"canary-{index}",
                kind=KIND_CANARY,
                text=f"Petr Kanar {index} spent a winter in Zbrusnovice.",
                planted=[(f"Petr Kanar {index}", "person")],
                fact=f"Petr Kanar {index} spent a winter in Zbrusnovice.",
            )
            for index in range(canary_count)
        ]

    def make_honeytokens(self, count: int) -> list[FakeFragment]:
        self.honeytoken_counts.append(count)
        return [
            FakeFragment(
                fragment_id=f"honeytoken-{len(self.honeytoken_counts)}-{index}",
                kind=KIND_HONEYTOKEN,
                text="Ada Fikova signed for Nordwind s.r.o.",
                planted=[("Ada Fikova", "person"), ("Nordwind s.r.o.", "company")],
            )
            for index in range(count)
        ]

    def make_chaff(self, count: int) -> list[FakeFragment]:
        self.chaff_counts.append(count)
        return [
            FakeFragment(
                fragment_id=f"chaff-{len(self.chaff_counts)}-{index}",
                kind=KIND_CHAFF,
                text="The quarterly review moved to the larger room.",
            )
            for index in range(count)
        ]

    def ensure_canaries(self) -> list[FakeFragment]:
        self.ensure_canaries_calls += 1
        return list(self._canaries)

    @property
    def called(self) -> bool:
        return bool(self.honeytoken_counts or self.chaff_counts or self.ensure_canaries_calls)


class FakeStore:
    """In-memory store covering the real and the synthetic methods alike."""

    def __init__(
        self,
        unsubmitted: list[dict] | None = None,
        document_texts: dict[str, list[str]] | None = None,
        synthetic: dict[str, dict] | None = None,
        honeytoken_rows: list[dict] | None = None,
        probe_rows: list[dict] | None = None,
    ) -> None:
        self._unsubmitted = unsubmitted or []
        self._document_texts = document_texts or {}
        self._synthetic = dict(synthetic or {})
        self._honeytoken_rows = honeytoken_rows or []
        self._probe_rows = probe_rows or []
        self.recorded_batches: list[tuple[str, str]] = []
        self.submitted: list[tuple[list[str], str]] = []
        self.status_updates: list[tuple[str, str]] = []
        self.entities: dict[str, list[tuple[str, str]]] = {}
        self.document_text_calls: list[str] = []
        self.synthetic_submitted: list[tuple[list[str], str]] = []
        self.honeytoken_results: list[tuple[str, str, list[tuple[str, str]]]] = []
        self.canary_probes: list[tuple[str, str, bool, str]] = []

    # --- real chunk surface -------------------------------------------------

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

    # --- synthetic surface --------------------------------------------------

    def get_synthetic_fragment(self, fragment_id: str) -> dict | None:
        return self._synthetic.get(fragment_id)

    def mark_synthetic_submitted(self, fragment_ids: list[str], batch_id: str) -> None:
        self.synthetic_submitted.append((list(fragment_ids), batch_id))

    def record_honeytoken_result(
        self, fragment_id: str, batch_id: str, found: list[tuple[str, str]]
    ) -> None:
        self.honeytoken_results.append((fragment_id, batch_id, list(found)))

    def record_canary_probe(
        self, fragment_id: str, model: str, tripped: bool, response_excerpt: str
    ) -> None:
        self.canary_probes.append((fragment_id, model, tripped, response_excerpt))

    def list_synthetic_fragments(self, kind: str | None = None) -> list[dict]:
        rows = [
            {"fragment_id": fragment_id, **payload}
            for fragment_id, payload in self._synthetic.items()
        ]
        if kind is None:
            return rows
        return [row for row in rows if row["kind"] == kind]

    def honeytoken_stats(self) -> list[dict]:
        return list(self._honeytoken_rows)

    def list_canary_probes(self) -> list[dict]:
        return list(self._probe_rows)


class FakeChunker:
    """Chunker stand-in that fails loudly if submission asks it for anything.

    Real requests carry the stored chunk text verbatim, so there is nothing
    left for the chunker to contribute at submission time.
    """

    def __getattr__(self, name: str):
        raise AssertionError(f"Detector must not call chunker.{name} during submission")


class FakeBatches:
    def __init__(self) -> None:
        self.create_calls: list[list[dict]] = []
        self.results_calls: list[str] = []
        self.created_batch = SimpleNamespace(id=BATCH_ID, processing_status="in_progress")
        self.result_items: list[SimpleNamespace] = []

    def create(self, requests: list[dict]) -> SimpleNamespace:
        self.create_calls.append(list(requests))
        return self.created_batch

    def results(self, batch_id: str):
        self.results_calls.append(batch_id)
        return iter(self.result_items)


class FakeClient:
    def __init__(self) -> None:
        self.batches = FakeBatches()
        self.messages = SimpleNamespace(batches=self.batches)


def make_chunks(count: int) -> tuple[list[dict], dict[str, list[str]]]:
    texts = [f"real text {index}" for index in range(count)]
    chunks = [
        {"chunk_id": f"chunk-{index}", "doc_id": "doc-1", "seq": index, "text": texts[index]}
        for index in range(count)
    ]
    return chunks, {"doc-1": texts}


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


def custom_ids(requests: list[dict]) -> list[str]:
    return [request["custom_id"] for request in requests]


def sent_canary_ids(client: FakeClient) -> list[str]:
    """Ids of the canary requests inside the batch that was created."""
    return [
        ident
        for ident in custom_ids(client.batches.create_calls[0])
        if ident.startswith("canary-")
    ]


# --- submit -----------------------------------------------------------------


def test_submit_mixes_honeytokens_chaff_and_canaries_into_the_batch():
    chunks, texts = make_chunks(10)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()
    generator = FakeGenerator()

    batch_id = Detector(
        make_config(), store, FakeChunker(), client=client, generator=generator
    ).submit()

    assert batch_id == BATCH_ID
    requests = client.batches.create_calls[0]
    ids = custom_ids(requests)

    # round(10 * 0.02) is 0, and the floor of one honeytoken per batch lifts it.
    assert generator.honeytoken_counts == [1]
    assert generator.chaff_counts == [10]
    assert generator.ensure_canaries_calls == 1
    assert len(requests) == 10 + 1 + 10 + CANARIES_PER_BATCH
    assert len(set(ids)) == len(ids)

    kinds = [ident.split("-")[0] for ident in ids]
    assert kinds.count("chunk") == 10
    assert kinds.count("honeytoken") == 1
    assert kinds.count("chaff") == 10
    assert kinds.count("canary") == CANARIES_PER_BATCH


def test_submit_shuffles_real_and_synthetic_requests_together(monkeypatch):
    chunks, texts = make_chunks(10)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()
    shuffled: list[int] = []

    def record_shuffle(sequence):
        shuffled.append(len(sequence))

    monkeypatch.setattr("doc_quant.detector.random.shuffle", record_shuffle)

    Detector(
        make_config(), store, FakeChunker(), client=client, generator=FakeGenerator()
    ).submit()

    # The shuffle must see the combined list, not the real chunks alone.
    assert shuffled == [26]


def test_submit_marks_real_chunks_and_synthetic_fragments_separately():
    chunks, texts = make_chunks(10)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()

    Detector(
        make_config(), store, FakeChunker(), client=client, generator=FakeGenerator()
    ).submit()

    assert store.recorded_batches == [(BATCH_ID, "in_progress")]

    chunk_ids, chunk_batch = store.submitted[0]
    assert sorted(chunk_ids) == sorted(chunk["chunk_id"] for chunk in chunks)
    assert chunk_batch == BATCH_ID

    assert len(store.synthetic_submitted) == 1
    synthetic_ids, synthetic_batch = store.synthetic_submitted[0]
    assert synthetic_batch == BATCH_ID
    assert len(synthetic_ids) == 1 + 10 + CANARIES_PER_BATCH
    assert set(synthetic_ids).isdisjoint(chunk_ids)

    # Every synthetic id that was sent is also the id of a submitted request.
    sent = set(custom_ids(client.batches.create_calls[0]))
    assert set(synthetic_ids) <= sent


def test_submit_builds_identical_request_shape_for_real_and_synthetic():
    chunks, texts = make_chunks(4)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()

    Detector(
        make_config(), store, FakeChunker(), client=client, generator=FakeGenerator()
    ).submit()

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
        assert list(params["messages"][0]) == ["role", "content"]
        assert params["messages"][0]["role"] == "user"

    # The only structural difference between requests may be id and content.
    shapes = {
        tuple(sorted(request.keys())) + tuple(sorted(request["params"].keys()))
        for request in client.batches.create_calls[0]
    }
    assert len(shapes) == 1


def test_submit_sends_the_synthetic_fragment_text_verbatim():
    chunks, texts = make_chunks(2)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()
    generator = FakeGenerator()

    Detector(make_config(), store, FakeChunker(), client=client, generator=generator).submit()

    contents = {
        request["custom_id"]: request["params"]["messages"][0]["content"]
        for request in client.batches.create_calls[0]
    }
    canary_ids = [ident for ident in contents if ident.startswith("canary-")]
    assert canary_ids
    for canary_id in canary_ids:
        index = canary_id.split("-")[1]
        assert contents[canary_id] == f"Petr Kanar {index} spent a winter in Zbrusnovice."

    # A real chunk is sent the same way: its stored text, nothing added.
    for chunk in chunks:
        assert contents[chunk["chunk_id"]] == chunk["text"]


def test_submit_with_synthetic_disabled_sends_real_chunks_only():
    chunks, texts = make_chunks(10)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()
    generator = FakeGenerator()
    config = make_config(
        honeytokens_enabled=False, chaff_enabled=False, canaries_enabled=False
    )

    Detector(config, store, FakeChunker(), client=client, generator=generator).submit()

    assert generator.called is False
    ids = custom_ids(client.batches.create_calls[0])
    assert sorted(ids) == sorted(chunk["chunk_id"] for chunk in chunks)
    assert store.synthetic_submitted == []


def test_submit_with_no_real_chunks_generates_nothing():
    store = FakeStore(unsubmitted=[])
    client = FakeClient()
    generator = FakeGenerator()

    assert (
        Detector(
            make_config(), store, FakeChunker(), client=client, generator=generator
        ).submit()
        is None
    )
    assert generator.called is False
    assert client.batches.create_calls == []
    assert store.synthetic_submitted == []


def test_submit_scales_honeytokens_with_the_batch_size():
    chunks, texts = make_chunks(100)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    generator = FakeGenerator()

    Detector(
        make_config(), store, FakeChunker(), client=FakeClient(), generator=generator
    ).submit()

    assert generator.honeytoken_counts == [2]
    assert generator.chaff_counts == [100]


def test_submit_samples_at_most_canaries_per_batch_from_the_set():
    chunks, texts = make_chunks(3)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()
    generator = FakeGenerator(canary_count=9)

    Detector(
        make_config(canaries_per_batch=4), store, FakeChunker(), client=client, generator=generator
    ).submit()

    sent_canaries = sent_canary_ids(client)
    assert len(sent_canaries) == 4
    assert len(set(sent_canaries)) == 4


def test_submit_sends_the_whole_canary_set_when_it_is_smaller_than_the_sample():
    chunks, texts = make_chunks(3)
    store = FakeStore(unsubmitted=chunks, document_texts=texts)
    client = FakeClient()

    Detector(
        make_config(canaries_per_batch=8),
        store,
        FakeChunker(),
        client=client,
        generator=FakeGenerator(canary_count=2),
    ).submit()

    sent_canaries = sent_canary_ids(client)
    assert sorted(sent_canaries) == ["canary-0", "canary-1"]


# --- fetch ------------------------------------------------------------------


def mixed_batch_store() -> FakeStore:
    return FakeStore(
        synthetic={
            "honey-1": {
                "kind": KIND_HONEYTOKEN,
                "text": "Ada Fikova signed for Nordwind s.r.o.",
                "planted": [("Ada Fikova", "person"), ("Nordwind s.r.o.", "company")],
                "fact": None,
            },
            "chaff-1": {
                "kind": KIND_CHAFF,
                "text": "The quarterly review moved to the larger room.",
                "planted": [],
                "fact": None,
            },
            "canary-1": {
                "kind": KIND_CANARY,
                "text": "Petr Kanar spent a winter in Zbrusnovice.",
                "planted": [("Petr Kanar", "person")],
                "fact": "Petr Kanar spent a winter in Zbrusnovice.",
            },
        }
    )


def mixed_batch_results() -> list[SimpleNamespace]:
    return [
        succeeded_result(
            "chunk-a", entity_payload(("Jan Novak", "person"), ("Keboola", "company"))
        ),
        succeeded_result(
            "honey-1", entity_payload(("Ada Fikova", "person"), ("Nordwind s.r.o.", "company"))
        ),
        succeeded_result("chaff-1", entity_payload(("Someone Else", "person"))),
        succeeded_result("canary-1", entity_payload(("Petr Kanar", "person"))),
    ]


def test_fetch_routes_each_result_by_fragment_kind():
    store = mixed_batch_store()
    client = FakeClient()
    client.batches.result_items = mixed_batch_results()

    counts = Detector(make_config(), store, FakeChunker(), client=client).fetch(BATCH_ID)

    assert counts == {
        "succeeded": 1,
        "errored": 0,
        "refused": 0,
        "entities": 2,
        "honeytokens_scored": 1,
        "synthetic_discarded": 2,
    }
    assert store.entities == {"chunk-a": [("Jan Novak", "person"), ("Keboola", "company")]}
    assert store.honeytoken_results == [
        (
            "honey-1",
            BATCH_ID,
            [("Ada Fikova", "person"), ("Nordwind s.r.o.", "company")],
        )
    ]
    assert store.status_updates == [(BATCH_ID, "fetched")]


def test_fetch_never_stores_entities_for_a_synthetic_fragment():
    store = mixed_batch_store()
    client = FakeClient()
    client.batches.result_items = mixed_batch_results()

    Detector(make_config(), store, FakeChunker(), client=client).fetch(BATCH_ID)

    # The safety property of the whole extension: synthetic detections must not
    # be able to reach a document's redaction.
    assert set(store.entities) == {"chunk-a"}
    synthetic_ids = {row["fragment_id"] for row in store.list_synthetic_fragments()}
    assert synthetic_ids.isdisjoint(store.entities)


def test_fetch_scores_a_honeytoken_that_found_nothing():
    store = mixed_batch_store()
    client = FakeClient()
    client.batches.result_items = [succeeded_result("honey-1", entity_payload())]

    counts = Detector(make_config(), store, FakeChunker(), client=client).fetch(BATCH_ID)

    assert counts["honeytokens_scored"] == 1
    assert store.honeytoken_results == [("honey-1", BATCH_ID, [])]
    assert store.entities == {}


def test_fetch_counts_a_refused_honeytoken_without_scoring_it():
    store = mixed_batch_store()
    client = FakeClient()
    client.batches.result_items = [
        succeeded_result("honey-1", "I cannot help with that.", stop_reason="refusal")
    ]

    counts = Detector(make_config(), store, FakeChunker(), client=client).fetch(BATCH_ID)

    assert counts["refused"] == 1
    assert counts["honeytokens_scored"] == 0
    assert store.honeytoken_results == []


@pytest.mark.parametrize(
    "result_item",
    [
        succeeded_result("honey-1", "{not json at all"),
        failed_result("honey-1", "expired"),
    ],
)
def test_fetch_counts_a_broken_honeytoken_result_as_errored(result_item):
    store = mixed_batch_store()
    client = FakeClient()
    client.batches.result_items = [result_item]

    counts = Detector(make_config(), store, FakeChunker(), client=client).fetch(BATCH_ID)

    assert counts["errored"] == 1
    assert counts["honeytokens_scored"] == 0
    assert store.honeytoken_results == []


def test_fetch_discards_failed_chaff_and_canary_results_too():
    store = mixed_batch_store()
    client = FakeClient()
    client.batches.result_items = [
        failed_result("chaff-1", "expired"),
        succeeded_result("canary-1", "unparsable"),
    ]

    counts = Detector(make_config(), store, FakeChunker(), client=client).fetch(BATCH_ID)

    assert counts["synthetic_discarded"] == 2
    assert counts["errored"] == 0
    assert store.entities == {}


# --- canary probe -----------------------------------------------------------

CANARY_NAME = "Milada Vrbova"
CANARY_FACT = "Milada Vrbova ran the annual regatta at Zbrusnovice."
CANARY_NONCE = "Zbrusnovice"


def canary_store(fact: str | None = CANARY_FACT, planted=None) -> FakeStore:
    return FakeStore(
        synthetic={
            "canary-1": {
                "kind": KIND_CANARY,
                "text": fact or "",
                "planted": [(CANARY_NAME, "person")] if planted is None else planted,
                "fact": fact,
            }
        }
    )


class FakeMessages:
    """Stand-in for client.messages with a scripted single response."""

    def __init__(self, text: str | None = None, stop_reason: str = "end_turn", error=None) -> None:
        self._text = text
        self._stop_reason = stop_reason
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        content = [] if self._text is None else [SimpleNamespace(type="text", text=self._text)]
        return SimpleNamespace(stop_reason=self._stop_reason, content=content)


class FakeProbeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


def test_fact_nonce_picks_the_longest_word_outside_the_planted_names():
    assert _fact_nonce(CANARY_FACT, [(CANARY_NAME, "person")]) == CANARY_NONCE


def test_fact_nonce_ignores_words_belonging_to_a_planted_name():
    fact = "Vrbova was seen in Kremzovice."
    assert _fact_nonce(fact, [("Milada Vrbova", "person")]) == "Kremzovice"


def test_fact_nonce_returns_none_without_a_distinctive_word():
    assert _fact_nonce("Milada Vrbova was late.", [("Milada Vrbova", "person")]) is None


def test_fact_nonce_prefers_the_invented_place_over_the_template_wording():
    # The generator's fact template contributes long generic words such as
    # "coordinates"; picking one of those would trip on innocent answers.
    fact = "Torvald Grimsbury coordinates the records office in Bramwick Flats."
    assert _fact_nonce(fact, [("Torvald Grimsbury", "person")]) == "Bramwick"


def test_canary_probe_trips_when_the_answer_contains_the_nonce():
    store = canary_store()
    messages = FakeMessages(text=f"She is known for a regatta at {CANARY_NONCE}.")

    results = run_canary_probe(make_config(), store, FakeProbeClient(messages))

    assert [result["tripped"] for result in results] == [True]
    assert store.canary_probes[0][0] == "canary-1"
    assert store.canary_probes[0][1] == MODEL
    assert store.canary_probes[0][2] is True
    assert CANARY_NONCE in store.canary_probes[0][3]

    # A plain question, no structured output, configured limits.
    call = messages.calls[0]
    assert call["model"] == MODEL
    assert call["max_tokens"] == MAX_TOKENS
    assert call["output_config"] == {"effort": EFFORT}
    assert "format" not in call.get("output_config", {})
    assert CANARY_NAME in call["messages"][0]["content"]


def test_canary_probe_stays_clean_when_the_model_does_not_know_the_name():
    store = canary_store()
    messages = FakeMessages(text="I have never heard of this person.")

    results = run_canary_probe(make_config(), store, FakeProbeClient(messages))

    assert [result["tripped"] for result in results] == [False]
    assert store.canary_probes[0][2] is False
    assert store.canary_probes[0][3] == "I have never heard of this person."


def test_canary_probe_records_a_refusal_as_not_tripped():
    store = canary_store()
    messages = FakeMessages(text="I won't answer that.", stop_reason="refusal")

    run_canary_probe(make_config(), store, FakeProbeClient(messages))

    assert store.canary_probes == [("canary-1", MODEL, False, EXCERPT_REFUSAL)]


def test_canary_probe_records_an_api_error_as_not_tripped():
    store = canary_store()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    messages = FakeMessages(error=anthropic.APIConnectionError(request=request))

    run_canary_probe(make_config(), store, FakeProbeClient(messages))

    assert store.canary_probes == [("canary-1", MODEL, False, EXCERPT_ERROR)]


def test_canary_probe_truncates_a_long_answer():
    store = canary_store()
    long_answer = f"{CANARY_NONCE} " + "x" * 500
    messages = FakeMessages(text=long_answer)

    run_canary_probe(make_config(), store, FakeProbeClient(messages))

    assert len(store.canary_probes[0][3]) == 200


def test_canary_probe_honours_a_model_override():
    store = canary_store()
    messages = FakeMessages(text="No idea who that is.")

    results = run_canary_probe(
        make_config(), store, FakeProbeClient(messages), model="claude-haiku-test"
    )

    assert messages.calls[0]["model"] == "claude-haiku-test"
    assert results[0]["model"] == "claude-haiku-test"
    assert store.canary_probes[0][1] == "claude-haiku-test"


def test_canary_probe_skips_a_canary_without_a_testable_fact():
    store = canary_store(fact="He was late.")
    messages = FakeMessages(text="anything")

    results = run_canary_probe(make_config(), store, FakeProbeClient(messages))

    assert results == []
    assert messages.calls == []
    assert store.canary_probes == []


def test_canary_probe_skips_a_canary_without_a_planted_person():
    store = canary_store(planted=[("Nordwind s.r.o.", "company")])
    messages = FakeMessages(text="anything")

    assert run_canary_probe(make_config(), store, FakeProbeClient(messages)) == []
    assert store.canary_probes == []


def test_canary_probe_ignores_non_canary_fragments():
    store = mixed_batch_store()
    messages = FakeMessages(text="Zbrusnovice indeed.")

    results = run_canary_probe(make_config(), store, FakeProbeClient(messages))

    assert [result["fragment_id"] for result in results] == ["canary-1"]
    assert len(messages.calls) == 1


# --- synthetic report -------------------------------------------------------


def test_synthetic_report_runs_on_an_empty_store(capsys):
    assert _cmd_synthetic_report(FakeStore()) == 0

    out = capsys.readouterr().out
    assert "none generated yet" in out
    assert "no honeytoken results yet" in out
    assert "never probed" in out


def test_synthetic_report_shows_counts_recall_and_probes(capsys):
    store = FakeStore(
        synthetic=mixed_batch_store()._synthetic,
        honeytoken_rows=[
            {
                "batch_id": "msgbatch_one",
                "honeytokens_scored": 2,
                "planted_total": 4,
                "found_total": 3,
                "recall": 0.75,
            },
            {
                "batch_id": "msgbatch_two",
                "honeytokens_scored": 1,
                "planted_total": 4,
                "found_total": 1,
                "recall": 0.25,
            },
        ],
        probe_rows=[
            {
                "fragment_id": "canary-1",
                "model": MODEL,
                "tripped": True,
                "response_excerpt": "Zbrusnovice",
                "probed_at": "2026-08-18T10:00:00Z",
            }
        ],
    )

    assert _cmd_synthetic_report(store) == 0

    out = capsys.readouterr().out
    assert "honeytoken" in out and "chaff" in out and "canary" in out
    assert "75.0%" in out
    assert "25.0%" in out
    # Aggregate recall is recomputed from the totals, not averaged per batch.
    assert "50.0%" in out
    assert "TRIPPED" in out
    assert "2026-08-18T10:00:00Z" in out


def test_synthetic_report_marks_recall_as_not_available_without_planted_names(capsys):
    store = FakeStore(
        honeytoken_rows=[
            {
                "batch_id": "msgbatch_one",
                "honeytokens_scored": 1,
                "planted_total": 0,
                "found_total": 0,
                "recall": 0.0,
            }
        ]
    )

    assert _cmd_synthetic_report(store) == 0
    assert "n/a" in capsys.readouterr().out
