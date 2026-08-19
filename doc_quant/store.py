"""SQLite key-value store for documents, chunks, batches and entities.

Chunks are addressed by a random `uuid4().hex` identifier on purpose: a chunk
id must not leak which document it belongs to nor where it sits inside that
document, because chunk ids travel out of context together with the chunk text.
The ordering information lives exclusively in the local `chunks.seq` column,
which never leaves this database.

The store is also the ledger for the synthetic fragments (honeytokens, chaff
and canaries) that ride along with outbound batches: `synthetic_fragments`
records what was planted in each of them, `honeytoken_results` records what
the provider reported back, and `canary_probes` records later tripwire checks.
The tables are created with CREATE TABLE IF NOT EXISTS alongside the original
ones, so a database written before this extension upgrades on first open.
"""

from __future__ import annotations

import json
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

CREATE TABLE IF NOT EXISTS synthetic_fragments (
    fragment_id TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK(kind IN ('honeytoken','chaff','canary')),
    text        TEXT NOT NULL,
    planted     TEXT NOT NULL,
    fact        TEXT,
    batch_id    TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS honeytoken_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id TEXT NOT NULL REFERENCES synthetic_fragments(fragment_id),
    batch_id    TEXT NOT NULL,
    found       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canary_probes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id      TEXT NOT NULL REFERENCES synthetic_fragments(fragment_id),
    model            TEXT NOT NULL,
    tripped          INTEGER NOT NULL,
    response_excerpt TEXT NOT NULL,
    probed_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_seq ON chunks(doc_id, seq);
CREATE INDEX IF NOT EXISTS idx_chunks_batch ON chunks(batch_id);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_synthetic_kind ON synthetic_fragments(kind);
CREATE INDEX IF NOT EXISTS idx_synthetic_batch ON synthetic_fragments(batch_id);
CREATE INDEX IF NOT EXISTS idx_honeytoken_results_batch
    ON honeytoken_results(batch_id);
CREATE INDEX IF NOT EXISTS idx_canary_probes_fragment ON canary_probes(fragment_id);
"""

_CHUNK_COLUMNS = "chunk_id, doc_id, seq, text, batch_id"
_SYNTHETIC_COLUMNS = "fragment_id, kind, text, planted, fact, batch_id, created_at"


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _dump_pairs(pairs: list[tuple[str, str]]) -> str:
    """Serialize (name, type) pairs as a JSON list of two-element lists.

    Non-ASCII is kept verbatim so the stored column stays readable for names
    that the provider may report back with diacritics.
    """
    return json.dumps(
        [[name, entity_type] for name, entity_type in pairs], ensure_ascii=False
    )


def _load_pairs(payload: str) -> list[tuple[str, str]]:
    """Read back a column written by `_dump_pairs`."""
    return [(name, entity_type) for name, entity_type in json.loads(payload)]


def _synthetic_row_to_dict(row: sqlite3.Row) -> dict:
    """Turn a synthetic_fragments row into a dict with decoded `planted`."""
    fragment = dict(row)
    fragment["planted"] = _load_pairs(fragment["planted"])
    return fragment


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

    def get_document_entities_by_chunk(
        self, doc_id: str
    ) -> dict[str, list[tuple[str, str]]]:
        """Return each chunk's stored (text, type) entities, keyed by chunk id.

        Chunks without any stored entity are absent from the mapping. Within a
        chunk the pairs keep their insertion order, which is the order the
        provider reported them in.
        """
        cursor = self._connection.execute(
            """
            SELECT e.chunk_id AS chunk_id, e.text AS text, e.type AS type
            FROM entities e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.doc_id = ?
            ORDER BY e.chunk_id, e.id
            """,
            (doc_id,),
        )
        grouped: dict[str, list[tuple[str, str]]] = {}
        for row in cursor.fetchall():
            grouped.setdefault(row["chunk_id"], []).append((row["text"], row["type"]))
        return grouped

    # ------------------------------------------------------------------
    # synthetic fragments
    # ------------------------------------------------------------------

    def add_synthetic_fragments(self, fragments: list[dict]) -> None:
        """Register synthetic fragments before they may be submitted anywhere.

        Args:
            fragments: dicts with the keys `fragment_id`, `kind`, `text`,
                `planted` (a list of (name, type) pairs planted verbatim in the
                text) and `fact` (the unique canary fact, None for other kinds).

        Raises:
            KeyError: if a dict is missing one of the required keys.
            sqlite3.IntegrityError: on a duplicate id or an unknown kind.
        """
        if not fragments:
            logger.debug("add_synthetic_fragments called with no fragments")
            return

        created_at = _utc_now()
        rows = [
            (
                fragment["fragment_id"],
                fragment["kind"],
                fragment["text"],
                _dump_pairs(fragment["planted"]),
                fragment.get("fact"),
                created_at,
            )
            for fragment in fragments
        ]
        with self._connection:
            self._connection.executemany(
                "INSERT INTO synthetic_fragments "
                "(fragment_id, kind, text, planted, fact, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        logger.info("Registered %d synthetic fragments", len(rows))

    def get_synthetic_fragment(self, fragment_id: str) -> dict | None:
        """Return one synthetic fragment, or None when the id is not synthetic.

        Callers use the None result to tell a synthetic fragment id apart from
        a real chunk id when a batch result comes back, so an unknown id is a
        normal answer here rather than an error.
        """
        cursor = self._connection.execute(
            f"SELECT {_SYNTHETIC_COLUMNS} FROM synthetic_fragments "
            "WHERE fragment_id = ?",
            (fragment_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _synthetic_row_to_dict(row)

    def list_synthetic_fragments(self, kind: str | None = None) -> list[dict]:
        """Return all synthetic fragments, optionally restricted to one kind."""
        if kind is None:
            cursor = self._connection.execute(
                f"SELECT {_SYNTHETIC_COLUMNS} FROM synthetic_fragments "
                "ORDER BY created_at, fragment_id"
            )
        else:
            cursor = self._connection.execute(
                f"SELECT {_SYNTHETIC_COLUMNS} FROM synthetic_fragments "
                "WHERE kind = ? ORDER BY created_at, fragment_id",
                (kind,),
            )
        return [_synthetic_row_to_dict(row) for row in cursor.fetchall()]

    def mark_synthetic_submitted(self, fragment_ids: list[str], batch_id: str) -> None:
        """Attach `batch_id` to the given synthetic fragments."""
        if not fragment_ids:
            logger.debug("mark_synthetic_submitted called with no fragment ids")
            return
        with self._connection:
            cursor = self._connection.executemany(
                "UPDATE synthetic_fragments SET batch_id = ? WHERE fragment_id = ?",
                [(batch_id, fragment_id) for fragment_id in fragment_ids],
            )
        if cursor.rowcount != len(fragment_ids):
            logger.warning(
                "Marked %d of %d synthetic fragments as submitted to batch %s; "
                "some fragment ids were unknown",
                cursor.rowcount,
                len(fragment_ids),
                batch_id,
            )

    def get_batch_synthetic_fragments(self, batch_id: str) -> list[dict]:
        """Return the synthetic fragments submitted in `batch_id`, oldest first."""
        cursor = self._connection.execute(
            f"SELECT {_SYNTHETIC_COLUMNS} FROM synthetic_fragments "
            "WHERE batch_id = ? ORDER BY created_at, fragment_id",
            (batch_id,),
        )
        return [_synthetic_row_to_dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # honeytoken results
    # ------------------------------------------------------------------

    def get_batch_honeytoken_found(
        self, batch_id: str
    ) -> dict[str, list[tuple[str, str]]]:
        """Return what the provider reported per honeytoken of `batch_id`.

        Keyed by fragment id; a honeytoken the provider was never asked about
        (or answered outside this batch) is absent.
        """
        cursor = self._connection.execute(
            "SELECT fragment_id, found FROM honeytoken_results "
            "WHERE batch_id = ? ORDER BY id",
            (batch_id,),
        )
        return {
            row["fragment_id"]: _load_pairs(row["found"]) for row in cursor.fetchall()
        }

    def record_honeytoken_result(
        self, fragment_id: str, batch_id: str, found: list[tuple[str, str]]
    ) -> None:
        """Record which planted names the provider reported for a honeytoken.

        Args:
            fragment_id: the synthetic fragment that was scored.
            batch_id: the batch the fragment travelled in.
            found: the (name, type) pairs the provider returned for it.

        Raises:
            KeyError: if `fragment_id` is not a known synthetic fragment.
        """
        self._require_synthetic_fragment(fragment_id)
        with self._connection:
            self._connection.execute(
                "INSERT INTO honeytoken_results "
                "(fragment_id, batch_id, found, created_at) VALUES (?, ?, ?, ?)",
                (fragment_id, batch_id, _dump_pairs(found), _utc_now()),
            )
        logger.debug(
            "Recorded %d found names for honeytoken %s in batch %s",
            len(found),
            fragment_id,
            batch_id,
        )

    def honeytoken_stats(self) -> list[dict]:
        """Aggregate recall per batch over all recorded honeytoken results.

        A planted (name, type) pair counts as found when the same name text
        comes back, no matter which type the provider assigned to it: the
        measurement is about the name being spotted at all, and a person or
        company mix-up still means the name would have been redacted.

        Returns:
            One dict per batch with the keys `batch_id`, `honeytokens_scored`,
            `planted_total`, `found_total`, `recall`, `created_at`,
            `documents` and `fragments_total`, ordered by batch id. `recall` is
            0.0 for the degenerate case of nothing planted.

        The last three are context rather than measurement: a bare batch id
        says nothing about when the run happened, which document it carried or
        how much it carried, and a recall figure is only readable next to that.
        """
        cursor = self._connection.execute(
            """
            SELECT r.batch_id AS batch_id,
                   r.found AS found,
                   f.planted AS planted
            FROM honeytoken_results r
            JOIN synthetic_fragments f ON f.fragment_id = r.fragment_id
            ORDER BY r.batch_id, r.id
            """
        )

        stats: dict[str, dict] = {}
        for row in cursor.fetchall():
            entry = stats.setdefault(
                row["batch_id"],
                {
                    "batch_id": row["batch_id"],
                    "honeytokens_scored": 0,
                    "planted_total": 0,
                    "found_total": 0,
                },
            )
            planted = _load_pairs(row["planted"])
            found_names = {name for name, _ in _load_pairs(row["found"])}
            entry["honeytokens_scored"] += 1
            entry["planted_total"] += len(planted)
            entry["found_total"] += sum(
                1 for name, _ in planted if name in found_names
            )

        context = self._batch_context(list(stats))
        for batch_id, entry in stats.items():
            planted_total = entry["planted_total"]
            entry["recall"] = (
                entry["found_total"] / planted_total if planted_total else 0.0
            )
            entry.update(context[batch_id])
        return [stats[batch_id] for batch_id in sorted(stats)]

    def _batch_context(self, batch_ids: list[str]) -> dict[str, dict]:
        """When each batch ran, which documents it carried and how big it was.

        A batch recorded by an older build - or one whose chunks were since
        removed - still gets an entry, with the fields it has no answer for
        left empty: a missing row must not drop a measured batch from a report.
        """
        context = {
            batch_id: {"created_at": None, "documents": [], "fragments_total": 0}
            for batch_id in batch_ids
        }
        if not batch_ids:
            return context

        placeholders = ",".join("?" for _ in batch_ids)
        for row in self._connection.execute(
            f"SELECT batch_id, created_at FROM batches WHERE batch_id IN ({placeholders})",
            batch_ids,
        ):
            context[row["batch_id"]]["created_at"] = row["created_at"]

        for row in self._connection.execute(
            "SELECT c.batch_id AS batch_id, d.path AS path, COUNT(*) AS chunks "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE c.batch_id IN ({placeholders}) "
            "GROUP BY c.batch_id, d.path ORDER BY d.path",
            batch_ids,
        ):
            entry = context[row["batch_id"]]
            entry["documents"].append(row["path"])
            entry["fragments_total"] += row["chunks"]

        for row in self._connection.execute(
            "SELECT batch_id, COUNT(*) AS fragments FROM synthetic_fragments "
            f"WHERE batch_id IN ({placeholders}) GROUP BY batch_id",
            batch_ids,
        ):
            context[row["batch_id"]]["fragments_total"] += row["fragments"]

        return context

    # ------------------------------------------------------------------
    # canary probes
    # ------------------------------------------------------------------

    def record_canary_probe(
        self, fragment_id: str, model: str, tripped: bool, response_excerpt: str
    ) -> None:
        """Record the outcome of probing a model for a planted canary fact.

        Raises:
            KeyError: if `fragment_id` is not a known synthetic fragment.
        """
        self._require_synthetic_fragment(fragment_id)
        with self._connection:
            self._connection.execute(
                "INSERT INTO canary_probes "
                "(fragment_id, model, tripped, response_excerpt, probed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (fragment_id, model, int(tripped), response_excerpt, _utc_now()),
            )
        if tripped:
            logger.warning(
                "Canary %s tripped on model %s: the planted fact came back",
                fragment_id,
                model,
            )
        else:
            logger.debug("Canary %s did not trip on model %s", fragment_id, model)

    def list_canary_probes(self) -> list[dict]:
        """Return all canary probes, oldest first, with `tripped` as a bool."""
        cursor = self._connection.execute(
            "SELECT id, fragment_id, model, tripped, response_excerpt, probed_at "
            "FROM canary_probes ORDER BY probed_at, id"
        )
        probes = []
        for row in cursor.fetchall():
            probe = dict(row)
            probe["tripped"] = bool(probe["tripped"])
            probes.append(probe)
        return probes

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_synthetic_fragment(self, fragment_id: str) -> None:
        """Raise KeyError when the synthetic fragment does not exist."""
        cursor = self._connection.execute(
            "SELECT 1 FROM synthetic_fragments WHERE fragment_id = ? LIMIT 1",
            (fragment_id,),
        )
        if cursor.fetchone() is None:
            raise KeyError(f"Unknown synthetic fragment_id: {fragment_id}")

    def _require_document(self, doc_id: str) -> None:
        """Raise KeyError when the document does not exist."""
        cursor = self._connection.execute(
            "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1", (doc_id,)
        )
        if cursor.fetchone() is None:
            raise KeyError(f"Unknown doc_id: {doc_id}")
