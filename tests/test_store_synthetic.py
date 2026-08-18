"""Tests for the synthetic-fragment ledger in doc_quant.store.ChunkStore.

Covers the registry itself, the honeytoken recall arithmetic, the canary probe
log, and the transparent upgrade of a database written before this extension.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from doc_quant.store import ChunkStore

CHUNKS = ["# Nadpis\n\n", "Petr Novák pracuje ", "pro Keboola s.r.o."]

HONEYTOKEN = {
    "fragment_id": "frag_honey",
    "kind": "honeytoken",
    "text": "Torvald Grimsbury approved the schedule for Quellandic Systems.",
    "planted": [("Torvald Grimsbury", "person"), ("Quellandic Systems", "company")],
    "fact": None,
}
CHAFF = {
    "fragment_id": "frag_chaff",
    "kind": "chaff",
    "text": "The quarterly review confirmed the revised delivery milestones.",
    "planted": [],
    "fact": None,
}
CANARY = {
    "fragment_id": "frag_canary",
    "kind": "canary",
    "text": (
        "Records were consolidated last spring. Yrsaolf Loftmere coordinates "
        "the records office in Krellinghausen Flats."
    ),
    "planted": [("Yrsaolf Loftmere", "person")],
    "fact": "Yrsaolf Loftmere coordinates the records office in Krellinghausen Flats.",
}

# The schema as it stood before the canaries-and-chaff extension, used to prove
# that an existing database upgrades on open.
_OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL REFERENCES documents(doc_id),
    seq      INTEGER NOT NULL,
    text     TEXT NOT NULL,
    batch_id TEXT,
    UNIQUE(doc_id, seq)
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id   TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
    text     TEXT NOT NULL,
    type     TEXT NOT NULL CHECK(type IN ('person','company'))
);
"""


@pytest.fixture
def store(tmp_path: Path):
    chunk_store = ChunkStore(tmp_path / "nested" / "chunks.db")
    yield chunk_store
    chunk_store.close()


def _planted_honeytokens(store: ChunkStore) -> None:
    """Register the hand-built scenario used by the recall tests.

    Three planted names across two honeytokens.
    """
    store.add_synthetic_fragments(
        [
            {
                "fragment_id": "h1",
                "kind": "honeytoken",
                "text": "Alpha One met Beta Two about the delivery schedule.",
                "planted": [("Alpha One", "person"), ("Beta Two", "person")],
                "fact": None,
            },
            {
                "fragment_id": "h2",
                "kind": "honeytoken",
                "text": "The invoice was issued to Gamma Corp last week.",
                "planted": [("Gamma Corp", "company")],
                "fact": None,
            },
        ]
    )


# ----------------------------------------------------------------------
# registry roundtrip
# ----------------------------------------------------------------------


def test_add_and_get_synthetic_fragment_roundtrip(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN])

    fragment = store.get_synthetic_fragment("frag_honey")
    assert fragment is not None
    assert set(fragment) == {
        "fragment_id",
        "kind",
        "text",
        "planted",
        "fact",
        "batch_id",
        "created_at",
    }
    assert fragment["kind"] == "honeytoken"
    assert fragment["text"] == HONEYTOKEN["text"]
    assert fragment["planted"] == HONEYTOKEN["planted"]
    assert fragment["fact"] is None
    assert fragment["batch_id"] is None
    assert fragment["created_at"]


def test_planted_pairs_come_back_as_tuples(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN])

    planted = store.get_synthetic_fragment("frag_honey")["planted"]
    assert all(isinstance(pair, tuple) for pair in planted)
    assert planted[0] == ("Torvald Grimsbury", "person")


def test_canary_fact_is_stored_verbatim(store: ChunkStore) -> None:
    store.add_synthetic_fragments([CANARY])

    fragment = store.get_synthetic_fragment("frag_canary")
    assert fragment["fact"] == CANARY["fact"]
    assert fragment["fact"] in fragment["text"]


def test_fragment_without_planted_names_roundtrips(store: ChunkStore) -> None:
    store.add_synthetic_fragments([CHAFF])
    assert store.get_synthetic_fragment("frag_chaff")["planted"] == []


def test_add_synthetic_fragments_with_empty_list_is_noop(store: ChunkStore) -> None:
    store.add_synthetic_fragments([])
    assert store.list_synthetic_fragments() == []


def test_add_synthetic_fragments_rejects_unknown_kind(store: ChunkStore) -> None:
    bad = dict(HONEYTOKEN, fragment_id="frag_bad", kind="decoy")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_synthetic_fragments([bad])


def test_add_synthetic_fragments_rejects_duplicate_id(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN])
    with pytest.raises(sqlite3.IntegrityError):
        store.add_synthetic_fragments([HONEYTOKEN])


def test_add_synthetic_fragments_requires_all_keys(store: ChunkStore) -> None:
    incomplete = {"fragment_id": "frag_x", "kind": "chaff", "text": "text"}
    with pytest.raises(KeyError):
        store.add_synthetic_fragments([incomplete])


# ----------------------------------------------------------------------
# lookup and listing
# ----------------------------------------------------------------------


def test_get_synthetic_fragment_is_none_for_unknown_id(store: ChunkStore) -> None:
    assert store.get_synthetic_fragment("nope") is None


def test_get_synthetic_fragment_is_none_for_a_real_chunk(store: ChunkStore) -> None:
    """A real chunk id must be distinguishable from a synthetic one."""
    doc_id = store.add_document("doc.md", CHUNKS)
    chunk_id = store.get_document_chunks(doc_id)[0]["chunk_id"]

    assert store.get_synthetic_fragment(chunk_id) is None


def test_list_synthetic_fragments_returns_every_kind(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN, CHAFF, CANARY])

    fragments = store.list_synthetic_fragments()
    assert {fragment["fragment_id"] for fragment in fragments} == {
        "frag_honey",
        "frag_chaff",
        "frag_canary",
    }


def test_list_synthetic_fragments_filters_by_kind(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN, CHAFF, CANARY])

    canaries = store.list_synthetic_fragments(kind="canary")
    assert [fragment["fragment_id"] for fragment in canaries] == ["frag_canary"]
    assert store.list_synthetic_fragments(kind="honeytoken")[0]["planted"] == (
        HONEYTOKEN["planted"]
    )
    assert store.list_synthetic_fragments(kind="nonexistent") == []


def test_list_synthetic_fragments_on_empty_store(store: ChunkStore) -> None:
    assert store.list_synthetic_fragments() == []


# ----------------------------------------------------------------------
# submission state
# ----------------------------------------------------------------------


def test_mark_synthetic_submitted_attaches_batch_id(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN, CHAFF])

    store.mark_synthetic_submitted(["frag_honey", "frag_chaff"], "batch_abc")

    assert store.get_synthetic_fragment("frag_honey")["batch_id"] == "batch_abc"
    assert store.get_synthetic_fragment("frag_chaff")["batch_id"] == "batch_abc"


def test_mark_synthetic_submitted_leaves_others_untouched(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN, CHAFF])

    store.mark_synthetic_submitted(["frag_honey"], "batch_abc")

    assert store.get_synthetic_fragment("frag_chaff")["batch_id"] is None


def test_mark_synthetic_submitted_with_empty_list_is_noop(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN])
    store.mark_synthetic_submitted([], "batch_abc")
    assert store.get_synthetic_fragment("frag_honey")["batch_id"] is None


def test_mark_synthetic_submitted_ignores_unknown_ids(store: ChunkStore) -> None:
    store.add_synthetic_fragments([HONEYTOKEN])

    store.mark_synthetic_submitted(["frag_honey", "missing"], "batch_abc")

    assert store.get_synthetic_fragment("frag_honey")["batch_id"] == "batch_abc"


def test_mark_synthetic_submitted_does_not_touch_chunks(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    store.add_synthetic_fragments([HONEYTOKEN])

    store.mark_synthetic_submitted(["frag_honey"], "batch_abc")

    assert len(store.get_unsubmitted_chunks()) == len(CHUNKS)
    assert store.reconstruct(doc_id) == "".join(CHUNKS)


# ----------------------------------------------------------------------
# honeytoken results and recall
# ----------------------------------------------------------------------


def test_honeytoken_stats_computes_recall(store: ChunkStore) -> None:
    """Two honeytokens, three planted names, two of them found: recall 2/3."""
    _planted_honeytokens(store)

    store.record_honeytoken_result("h1", "batch_abc", [("Alpha One", "person")])
    store.record_honeytoken_result("h2", "batch_abc", [("Gamma Corp", "company")])

    stats = store.honeytoken_stats()
    assert len(stats) == 1
    assert stats[0] == {
        "batch_id": "batch_abc",
        "honeytokens_scored": 2,
        "planted_total": 3,
        "found_total": 2,
        "recall": pytest.approx(2 / 3),
    }


def test_honeytoken_stats_ignores_the_reported_type(store: ChunkStore) -> None:
    """A person reported as a company still counts: the name was spotted."""
    _planted_honeytokens(store)

    store.record_honeytoken_result(
        "h1", "batch_abc", [("Alpha One", "company"), ("Beta Two", "company")]
    )

    stats = store.honeytoken_stats()[0]
    assert stats["found_total"] == 2
    assert stats["recall"] == pytest.approx(1.0)


def test_honeytoken_stats_ignores_names_that_were_not_planted(
    store: ChunkStore,
) -> None:
    _planted_honeytokens(store)

    store.record_honeytoken_result(
        "h2", "batch_abc", [("Gamma Corp", "company"), ("Somebody Else", "person")]
    )

    stats = store.honeytoken_stats()[0]
    assert stats["planted_total"] == 1
    assert stats["found_total"] == 1


def test_honeytoken_stats_with_nothing_found(store: ChunkStore) -> None:
    _planted_honeytokens(store)

    store.record_honeytoken_result("h1", "batch_abc", [])

    stats = store.honeytoken_stats()[0]
    assert stats["found_total"] == 0
    assert stats["recall"] == pytest.approx(0.0)


def test_honeytoken_stats_are_grouped_per_batch(store: ChunkStore) -> None:
    _planted_honeytokens(store)

    store.record_honeytoken_result("h1", "batch_abc", [("Alpha One", "person")])
    store.record_honeytoken_result("h2", "batch_def", [])

    stats = {row["batch_id"]: row for row in store.honeytoken_stats()}
    assert list(stats) == ["batch_abc", "batch_def"]
    assert stats["batch_abc"]["planted_total"] == 2
    assert stats["batch_abc"]["found_total"] == 1
    assert stats["batch_def"]["planted_total"] == 1
    assert stats["batch_def"]["found_total"] == 0


def test_honeytoken_stats_on_empty_store(store: ChunkStore) -> None:
    assert store.honeytoken_stats() == []


def test_record_honeytoken_result_unknown_fragment_raises_key_error(
    store: ChunkStore,
) -> None:
    with pytest.raises(KeyError):
        store.record_honeytoken_result("nope", "batch_abc", [])


# ----------------------------------------------------------------------
# canary probes
# ----------------------------------------------------------------------


def test_record_and_list_canary_probes(store: ChunkStore) -> None:
    store.add_synthetic_fragments([CANARY])

    store.record_canary_probe("frag_canary", "some-model", False, "I do not know.")
    store.record_canary_probe(
        "frag_canary", "other-model", True, "...in Krellinghausen Flats."
    )

    probes = store.list_canary_probes()
    assert len(probes) == 2
    assert set(probes[0]) == {
        "id",
        "fragment_id",
        "model",
        "tripped",
        "response_excerpt",
        "probed_at",
    }
    assert [probe["model"] for probe in probes] == ["some-model", "other-model"]
    assert probes[0]["tripped"] is False
    assert probes[1]["tripped"] is True
    assert probes[1]["response_excerpt"] == "...in Krellinghausen Flats."
    assert probes[0]["probed_at"]


def test_list_canary_probes_on_empty_store(store: ChunkStore) -> None:
    assert store.list_canary_probes() == []


def test_record_canary_probe_unknown_fragment_raises_key_error(
    store: ChunkStore,
) -> None:
    with pytest.raises(KeyError):
        store.record_canary_probe("nope", "some-model", True, "excerpt")


# ----------------------------------------------------------------------
# schema upgrade
# ----------------------------------------------------------------------


def test_reopening_store_keeps_synthetic_data(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    first = ChunkStore(db_path)
    first.add_synthetic_fragments([CANARY])
    first.close()

    second = ChunkStore(db_path)
    try:
        assert second.get_synthetic_fragment("frag_canary")["fact"] == CANARY["fact"]
    finally:
        second.close()


def test_database_from_the_old_schema_gains_the_new_tables(tmp_path: Path) -> None:
    """A database written before this extension must upgrade on open."""
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    try:
        legacy.executescript(_OLD_SCHEMA)
        legacy.execute(
            "INSERT INTO documents (doc_id, path, created_at) VALUES (?, ?, ?)",
            ("doc1", "legacy.md", "2026-01-01T00:00:00+00:00"),
        )
        legacy.execute(
            "INSERT INTO chunks (chunk_id, doc_id, seq, text) VALUES (?, ?, ?, ?)",
            ("chunk1", "doc1", 0, "Petr Novák pracuje pro Keboola s.r.o."),
        )
        legacy.execute(
            "INSERT INTO entities (chunk_id, text, type) VALUES (?, ?, ?)",
            ("chunk1", "Petr Novák", "person"),
        )
        legacy.commit()
    finally:
        legacy.close()

    store = ChunkStore(db_path)
    try:
        # Existing data survived untouched.
        assert store.reconstruct("doc1") == "Petr Novák pracuje pro Keboola s.r.o."
        assert store.get_document_entities("doc1") == [("Petr Novák", "person")]

        # And the new ledger is usable straight away.
        assert store.list_synthetic_fragments() == []
        store.add_synthetic_fragments([HONEYTOKEN])
        store.record_honeytoken_result(
            "frag_honey", "batch_abc", [("Torvald Grimsbury", "person")]
        )
        store.record_canary_probe("frag_honey", "some-model", False, "excerpt")

        assert store.get_synthetic_fragment("frag_honey") is not None
        assert store.honeytoken_stats()[0]["found_total"] == 1
        assert len(store.list_canary_probes()) == 1
    finally:
        store.close()
