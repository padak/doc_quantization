"""Tests for doc_quant.store.ChunkStore."""

from __future__ import annotations

import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from doc_quant.store import ChunkStore

CHUNKS = ["# Nadpis\n\n", "Petr Novák pracuje ", "pro Keboola s.r.o. 🎉"]
OTHER_CHUNKS = ["Second document, ", "written by Jane Doe."]


@pytest.fixture
def store(tmp_path: Path):
    chunk_store = ChunkStore(tmp_path / "nested" / "chunks.db")
    yield chunk_store
    chunk_store.close()


# ----------------------------------------------------------------------
# documents
# ----------------------------------------------------------------------


def test_init_creates_parent_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "a" / "b" / "chunks.db"
    chunk_store = ChunkStore(db_path)
    try:
        assert db_path.exists()
        assert chunk_store.list_documents() == []
    finally:
        chunk_store.close()


def test_reopening_existing_database_keeps_data(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    first = ChunkStore(db_path)
    doc_id = first.add_document("doc.md", CHUNKS)
    first.close()

    second = ChunkStore(db_path)
    try:
        assert second.reconstruct(doc_id) == "".join(CHUNKS)
    finally:
        second.close()


def test_default_store_refuses_use_from_another_thread(tmp_path: Path) -> None:
    """The same-thread guard stays on by default.

    It is what catches a parallel flow accidentally writing from a worker
    thread, so loosening it must remain an explicit opt-in.
    """
    chunk_store = ChunkStore(tmp_path / "chunks.db")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(sqlite3.ProgrammingError):
                pool.submit(chunk_store.list_documents).result()
    finally:
        chunk_store.close()


def test_cross_thread_store_can_be_handed_between_threads(tmp_path: Path) -> None:
    chunk_store = ChunkStore(tmp_path / "chunks.db", allow_cross_thread=True)
    try:
        doc_id = chunk_store.add_document("doc.md", CHUNKS)
        with ThreadPoolExecutor(max_workers=1) as pool:
            listed = pool.submit(chunk_store.list_documents).result()
            reconstructed = pool.submit(chunk_store.reconstruct, doc_id).result()
        assert [doc["doc_id"] for doc in listed] == [doc_id]
        assert reconstructed == "".join(CHUNKS)
    finally:
        chunk_store.close()


def test_add_document_and_reconstruct(store: ChunkStore) -> None:
    doc_id = store.add_document("docs/example.md", CHUNKS)

    assert isinstance(doc_id, str)
    assert len(doc_id) == 32
    assert store.reconstruct(doc_id) == "".join(CHUNKS)


def test_document_ids_are_random_and_unique(store: ChunkStore) -> None:
    first = store.add_document("a.md", CHUNKS)
    second = store.add_document("a.md", CHUNKS)
    assert first != second


def test_chunk_ids_are_opaque_random_identifiers(store: ChunkStore) -> None:
    doc_id = store.add_document("docs/example.md", CHUNKS)
    chunks = store.get_document_chunks(doc_id)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(set(chunk_ids)) == len(CHUNKS)
    for chunk_id in chunk_ids:
        # No document identity and no ordering hint may leak into the id.
        assert doc_id not in chunk_id
        assert uuid.UUID(hex=chunk_id).version == 4


def test_chunk_ids_do_not_encode_document_order(store: ChunkStore) -> None:
    """Chunk ids must not sort in seq order; only the seq column carries it."""
    documents = 20
    ascending = 0
    for index in range(documents):
        doc_id = store.add_document(f"doc_{index}.md", CHUNKS)
        chunk_ids = [chunk["chunk_id"] for chunk in store.get_document_chunks(doc_id)]
        if chunk_ids == sorted(chunk_ids):
            ascending += 1

    # Random ids line up with seq order by chance only; all 20 doing so would
    # mean the id encodes position.
    assert ascending < documents


def test_add_document_with_no_chunks(store: ChunkStore) -> None:
    doc_id = store.add_document("empty.md", [])
    assert store.reconstruct(doc_id) == ""
    assert store.get_document_chunks(doc_id) == []
    documents = store.list_documents()
    assert documents[0]["chunk_count"] == 0


def test_list_documents_reports_paths_and_counts(store: ChunkStore) -> None:
    first = store.add_document("first.md", CHUNKS)
    second = store.add_document("second.md", OTHER_CHUNKS)

    documents = {doc["doc_id"]: doc for doc in store.list_documents()}
    assert set(documents) == {first, second}
    assert documents[first]["path"] == "first.md"
    assert documents[first]["chunk_count"] == len(CHUNKS)
    assert documents[second]["chunk_count"] == len(OTHER_CHUNKS)
    assert documents[first]["created_at"]


def test_reconstruct_unknown_document_raises_key_error(store: ChunkStore) -> None:
    with pytest.raises(KeyError):
        store.reconstruct("does-not-exist")


def test_get_document_chunk_texts_is_ordered(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    assert store.get_document_chunk_texts(doc_id) == CHUNKS
    assert [chunk["seq"] for chunk in store.get_document_chunks(doc_id)] == [0, 1, 2]


# ----------------------------------------------------------------------
# chunks
# ----------------------------------------------------------------------


def test_get_chunk_returns_full_row(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    stored = store.get_document_chunks(doc_id)[1]

    chunk = store.get_chunk(stored["chunk_id"])
    assert set(chunk) == {"chunk_id", "doc_id", "seq", "text", "batch_id"}
    assert chunk["doc_id"] == doc_id
    assert chunk["seq"] == 1
    assert chunk["text"] == CHUNKS[1]
    assert chunk["batch_id"] is None


def test_get_chunk_unknown_id_raises_key_error(store: ChunkStore) -> None:
    with pytest.raises(KeyError):
        store.get_chunk("nope")


def test_update_chunk_text_roundtrip_and_reconstruct(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    target = store.get_document_chunks(doc_id)[1]
    redacted = "**PERSON** pracuje "

    store.update_chunk_text(target["chunk_id"], redacted)

    assert store.get_chunk(target["chunk_id"])["text"] == redacted
    assert store.reconstruct(doc_id) == CHUNKS[0] + redacted + CHUNKS[2]


def test_update_chunk_text_unknown_id_raises_key_error(store: ChunkStore) -> None:
    with pytest.raises(KeyError):
        store.update_chunk_text("nope", "text")


def test_get_document_chunks_of_unknown_document_is_empty(store: ChunkStore) -> None:
    assert store.get_document_chunks("nope") == []
    assert store.get_document_chunk_texts("nope") == []


# ----------------------------------------------------------------------
# submission state
# ----------------------------------------------------------------------


def test_unsubmitted_chunks_before_and_after_marking(store: ChunkStore) -> None:
    first = store.add_document("first.md", CHUNKS)
    second = store.add_document("second.md", OTHER_CHUNKS)

    pending = store.get_unsubmitted_chunks()
    assert len(pending) == len(CHUNKS) + len(OTHER_CHUNKS)
    assert all(chunk["batch_id"] is None for chunk in pending)

    first_ids = [chunk["chunk_id"] for chunk in store.get_document_chunks(first)]
    store.mark_chunks_submitted(first_ids, "batch_abc")

    remaining = store.get_unsubmitted_chunks()
    assert {chunk["doc_id"] for chunk in remaining} == {second}
    assert len(remaining) == len(OTHER_CHUNKS)
    for chunk_id in first_ids:
        assert store.get_chunk(chunk_id)["batch_id"] == "batch_abc"


def test_mark_chunks_submitted_with_empty_list_is_noop(store: ChunkStore) -> None:
    store.add_document("doc.md", CHUNKS)
    store.mark_chunks_submitted([], "batch_abc")
    assert len(store.get_unsubmitted_chunks()) == len(CHUNKS)


def test_mark_chunks_submitted_ignores_unknown_ids(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    known = store.get_document_chunks(doc_id)[0]["chunk_id"]

    store.mark_chunks_submitted([known, "missing"], "batch_abc")

    assert store.get_chunk(known)["batch_id"] == "batch_abc"
    assert len(store.get_unsubmitted_chunks()) == len(CHUNKS) - 1


# ----------------------------------------------------------------------
# batches
# ----------------------------------------------------------------------


def test_record_batch_and_list(store: ChunkStore) -> None:
    store.record_batch("batch_abc", "submitted")
    store.record_batch("batch_def", "submitted")

    batches = {batch["batch_id"]: batch for batch in store.list_batches()}
    assert set(batches) == {"batch_abc", "batch_def"}
    assert batches["batch_abc"]["status"] == "submitted"
    assert batches["batch_abc"]["created_at"]


def test_set_batch_status_updates_existing_batch(store: ChunkStore) -> None:
    store.record_batch("batch_abc", "submitted")
    created_at = store.list_batches()[0]["created_at"]

    store.set_batch_status("batch_abc", "ended")

    batch = store.list_batches()[0]
    assert batch["status"] == "ended"
    assert batch["created_at"] == created_at


def test_record_batch_twice_updates_status_without_duplicating(
    store: ChunkStore,
) -> None:
    store.record_batch("batch_abc", "submitted")
    store.record_batch("batch_abc", "ended")

    batches = store.list_batches()
    assert len(batches) == 1
    assert batches[0]["status"] == "ended"


def test_set_batch_status_unknown_batch_raises_key_error(store: ChunkStore) -> None:
    with pytest.raises(KeyError):
        store.set_batch_status("nope", "ended")


def test_list_batches_on_empty_store(store: ChunkStore) -> None:
    assert store.list_batches() == []


# ----------------------------------------------------------------------
# entities
# ----------------------------------------------------------------------


def test_add_entities_deduplicates(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    chunk_id = store.get_document_chunks(doc_id)[1]["chunk_id"]

    store.add_entities(chunk_id, [("Petr Novák", "person"), ("Keboola", "company")])
    store.add_entities(chunk_id, [("Petr Novák", "person")])

    entities = store.get_document_entities(doc_id)
    assert entities.count(("Petr Novák", "person")) == 1
    assert sorted(entities) == [("Keboola", "company"), ("Petr Novák", "person")]


def test_get_document_entities_aggregates_across_chunks(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    chunks = store.get_document_chunks(doc_id)

    store.add_entities(chunks[0]["chunk_id"], [("Petr Novák", "person")])
    store.add_entities(
        chunks[1]["chunk_id"],
        [("Petr Novák", "person"), ("Keboola s.r.o.", "company")],
    )
    store.add_entities(chunks[2]["chunk_id"], [("Jana Dvořáková", "person")])

    entities = store.get_document_entities(doc_id)
    assert sorted(entities) == [
        ("Jana Dvořáková", "person"),
        ("Keboola s.r.o.", "company"),
        ("Petr Novák", "person"),
    ]


def test_entities_are_scoped_per_document(store: ChunkStore) -> None:
    first = store.add_document("first.md", CHUNKS)
    second = store.add_document("second.md", OTHER_CHUNKS)

    store.add_entities(
        store.get_document_chunks(first)[0]["chunk_id"], [("Petr Novák", "person")]
    )
    store.add_entities(
        store.get_document_chunks(second)[0]["chunk_id"], [("Jane Doe", "person")]
    )

    assert store.get_document_entities(first) == [("Petr Novák", "person")]
    assert store.get_document_entities(second) == [("Jane Doe", "person")]


def test_add_entities_with_empty_list_is_noop(store: ChunkStore) -> None:
    doc_id = store.add_document("doc.md", CHUNKS)
    chunk_id = store.get_document_chunks(doc_id)[0]["chunk_id"]

    store.add_entities(chunk_id, [])

    assert store.get_document_entities(doc_id) == []


def test_add_entities_unknown_chunk_raises_key_error(store: ChunkStore) -> None:
    with pytest.raises(KeyError):
        store.add_entities("nope", [("Petr Novák", "person")])


def test_get_document_entities_of_unknown_document_is_empty(store: ChunkStore) -> None:
    assert store.get_document_entities("nope") == []
