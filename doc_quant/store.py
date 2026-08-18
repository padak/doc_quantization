"""SQLite key-value store for documents, chunks, batches and entities.

Chunks are addressed by a random `uuid4().hex` identifier on purpose: a chunk
id must not leak which document it belongs to nor where it sits inside that
document, because chunk ids travel out of context together with the chunk text.
The ordering information lives exclusively in the local `chunks.seq` column,
which never leaves this database.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
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

CREATE INDEX IF NOT EXISTS idx_chunks_doc_seq ON chunks(doc_id, seq);
CREATE INDEX IF NOT EXISTS idx_chunks_batch ON chunks(batch_id);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);
"""

_CHUNK_COLUMNS = "chunk_id, doc_id, seq, text, batch_id"


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class ChunkStore:
    """Persistent storage for chunked documents and their detection results."""

    def __init__(self, db_path: Path) -> None:
        """Open (and if needed create) the SQLite database at `db_path`.

        Missing parent directories and missing tables are created.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        logger.debug("Opened chunk store at %s", self.db_path)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()
        logger.debug("Closed chunk store at %s", self.db_path)

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------

    def add_document(self, path: str, chunk_texts: list[str]) -> str:
        """Store a document together with its ordered chunks.

        Args:
            path: source path of the document, kept for display purposes.
            chunk_texts: chunk texts in document order.

        Returns:
            The freshly generated random document id.
        """
        doc_id = uuid.uuid4().hex
        created_at = _utc_now()
        rows = [
            (uuid.uuid4().hex, doc_id, seq, text)
            for seq, text in enumerate(chunk_texts)
        ]
        with self._connection:
            self._connection.execute(
                "INSERT INTO documents (doc_id, path, created_at) VALUES (?, ?, ?)",
                (doc_id, path, created_at),
            )
            self._connection.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, seq, text) VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.info("Stored document %s (%d chunks) from %s", doc_id, len(rows), path)
        return doc_id

    def list_documents(self) -> list[dict]:
        """Return all documents with their chunk counts, newest last."""
        cursor = self._connection.execute(
            """
            SELECT d.doc_id AS doc_id,
                   d.path AS path,
                   d.created_at AS created_at,
                   COUNT(c.chunk_id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id, d.path, d.created_at
            ORDER BY d.created_at, d.doc_id
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def reconstruct(self, doc_id: str) -> str:
        """Concatenate the document's chunk texts in `seq` order.

        Raises:
            KeyError: if `doc_id` is unknown.
        """
        self._require_document(doc_id)
        return "".join(self.get_document_chunk_texts(doc_id))

    # ------------------------------------------------------------------
    # chunks
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> dict:
        """Return a single chunk row as a dict.

        Raises:
            KeyError: if `chunk_id` is unknown.
        """
        cursor = self._connection.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE chunk_id = ?", (chunk_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")
        return dict(row)

    def update_chunk_text(self, chunk_id: str, text: str) -> None:
        """Replace the stored text of a chunk (used by redaction).

        Raises:
            KeyError: if `chunk_id` is unknown.
        """
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE chunks SET text = ? WHERE chunk_id = ?", (text, chunk_id)
            )
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        """Return all chunks of a document ordered by `seq`."""
        cursor = self._connection.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE doc_id = ? ORDER BY seq",
            (doc_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_document_chunk_texts(self, doc_id: str) -> list[str]:
        """Return the texts of all chunks of a document ordered by `seq`."""
        cursor = self._connection.execute(
            "SELECT text FROM chunks WHERE doc_id = ? ORDER BY seq", (doc_id,)
        )
        return [row["text"] for row in cursor.fetchall()]

    def get_unsubmitted_chunks(self) -> list[dict]:
        """Return every chunk that has not been assigned to a batch yet."""
        cursor = self._connection.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks "
            "WHERE batch_id IS NULL ORDER BY doc_id, seq"
        )
        return [dict(row) for row in cursor.fetchall()]

    def mark_chunks_submitted(self, chunk_ids: list[str], batch_id: str) -> None:
        """Attach `batch_id` to the given chunks."""
        if not chunk_ids:
            logger.debug("mark_chunks_submitted called with no chunk ids")
            return
        with self._connection:
            cursor = self._connection.executemany(
                "UPDATE chunks SET batch_id = ? WHERE chunk_id = ?",
                [(batch_id, chunk_id) for chunk_id in chunk_ids],
            )
        if cursor.rowcount != len(chunk_ids):
            logger.warning(
                "Marked %d of %d chunks as submitted to batch %s; "
                "some chunk ids were unknown",
                cursor.rowcount,
                len(chunk_ids),
                batch_id,
            )

    # ------------------------------------------------------------------
    # batches
    # ------------------------------------------------------------------

    def record_batch(self, batch_id: str, status: str) -> None:
        """Register a batch, or update the status of an already known one."""
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO batches (batch_id, created_at, status)
                VALUES (?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET status = excluded.status
                """,
                (batch_id, _utc_now(), status),
            )
        logger.info("Recorded batch %s with status %s", batch_id, status)

    def set_batch_status(self, batch_id: str, status: str) -> None:
        """Update the status of an existing batch.

        Raises:
            KeyError: if `batch_id` is unknown.
        """
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?", (status, batch_id)
            )
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown batch_id: {batch_id}")

    def list_batches(self) -> list[dict]:
        """Return all batches ordered by creation time."""
        cursor = self._connection.execute(
            "SELECT batch_id, created_at, status FROM batches "
            "ORDER BY created_at, batch_id"
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # entities
    # ------------------------------------------------------------------

    def add_entities(self, chunk_id: str, entities: list[tuple[str, str]]) -> None:
        """Attach detected (text, type) entities to a chunk.

        Exact duplicates already stored for the same chunk are skipped, so
        re-processing a batch does not multiply rows.

        Raises:
            KeyError: if `chunk_id` is unknown.
        """
        self.get_chunk(chunk_id)
        if not entities:
            logger.debug("No entities to add for chunk %s", chunk_id)
            return

        inserted = 0
        with self._connection:
            for text, entity_type in entities:
                cursor = self._connection.execute(
                    "SELECT 1 FROM entities "
                    "WHERE chunk_id = ? AND text = ? AND type = ? LIMIT 1",
                    (chunk_id, text, entity_type),
                )
                if cursor.fetchone() is not None:
                    continue
                self._connection.execute(
                    "INSERT INTO entities (chunk_id, text, type) VALUES (?, ?, ?)",
                    (chunk_id, text, entity_type),
                )
                inserted += 1
        logger.debug("Added %d new entities to chunk %s", inserted, chunk_id)

    def get_document_entities(self, doc_id: str) -> list[tuple[str, str]]:
        """Return deduplicated (text, type) entity pairs across the document."""
        cursor = self._connection.execute(
            """
            SELECT DISTINCT e.text AS text, e.type AS type
            FROM entities e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.doc_id = ?
            ORDER BY e.type, e.text
            """,
            (doc_id,),
        )
        return [(row["text"], row["type"]) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_document(self, doc_id: str) -> None:
        """Raise KeyError when the document does not exist."""
        cursor = self._connection.execute(
            "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1", (doc_id,)
        )
        if cursor.fetchone() is None:
            raise KeyError(f"Unknown doc_id: {doc_id}")
