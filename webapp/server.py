"""FastAPI observability layer over the `doc_quant` pipeline.

The app makes each step of the anonymization pipeline visible: what a document
was chunked into, which fragments were mixed into a submission, what the
provider actually received in which order, what it answered, and what the
resulting redaction looks like. None of the pipeline logic is reimplemented
here - chunking, synthetic mixing, answer parsing and redaction are imported
from `doc_quant`.

Detection is synchronous here, unlike the CLI's Batches API path: one
`messages.create` call per fragment, so a user watches the requests go out one
by one instead of waiting for a batch to finish. The payload is deliberately
identical to the batch path - same system prompt, schema, effort and
max_tokens, built from the very dict this endpoint reports back as
`payload_template` - so what is observed here is what the batch path sends. The
mixing (honeytokens, chaff, canaries) and the shuffle come from
`doc_quant.detector.plan_synthetic_fragments`, so the composition is the same
too; only the transport differs.

`requests[].kind` and `requests[].seq` in the detect response are the LOCAL
view. The provider never receives them: it sees ids and texts in the shuffled
submission order and nothing else, which is exactly what the frontend renders
by hiding those two fields.
"""

from __future__ import annotations

import json
import logging
import random
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markitdown import FileConversionException, MarkItDown, UnsupportedFormatException
from pydantic import BaseModel

from doc_quant.chunker import Chunker
from doc_quant.cli import run_canary_probe
from doc_quant.config import PROJECT_ROOT, AppConfig, ConfigError, load_config
from doc_quant.detector import (
    DETECTION_SYSTEM_PROMPT,
    ENTITY_SCHEMA,
    parse_entities,
    plan_synthetic_fragments,
)
from doc_quant.redactor import redact_text
from doc_quant.store import ChunkStore
from doc_quant.synthetic import (
    KIND_CANARY,
    KIND_CHAFF,
    KIND_HONEYTOKEN,
    SyntheticGenerator,
)
from webapp.settings import (
    DEFAULT_SETTINGS_PATH,
    MISSING_API_KEY_MESSAGE,
    effective_api_key,
    effective_config,
    load_overrides,
    mask_api_key,
    save_overrides,
)

logger = logging.getLogger(__name__)

STATIC_DIR = PROJECT_ROOT / "webapp" / "static"
INDEX_FILENAME = "index.html"

# Server logs are kept for backtesting; the directory is gitignored.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILENAME = "webapp.log"
LOG_FORMAT = (
    '{"time": "%(asctime)s", "level": "%(levelname)s", '
    '"logger": "%(name)s", "message": "%(message)s"}'
)
_LOG_HANDLER_MARKER = "doc_quant_webapp"

# Suffixes whose bytes are already Markdown-ish text: running them through a
# converter would only risk mangling what the user wrote.
RAW_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
UPLOAD_STEM = "upload"
FALLBACK_UPLOAD_NAME = "upload"

# A synchronous run is still recorded as a batch so that honeytoken results and
# chunk submission state share one ledger with the CLI's batch runs. The prefix
# and the trailing status make the two kinds of run distinguishable in it.
SYNC_BATCH_PREFIX = "sync-"
SYNC_BATCH_ID_CHARS = 12
BATCH_STATUS_SYNC = "sync"
BATCH_STATUS_SYNC_COMPLETED = "sync-completed"

KIND_REAL = "real"
SYNTHETIC_KINDS = (KIND_HONEYTOKEN, KIND_CHAFF, KIND_CANARY)

STATUS_OK = "ok"
STATUS_REFUSAL = "refusal"
STATUS_ERROR = "error"

REFUSAL_STOP_REASON = "refusal"

NO_UNSUBMITTED_CHUNKS_MESSAGE = "no unsubmitted chunks"

HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_UNPROCESSABLE = 422


def configure_logging() -> None:
    """Attach a JSON file handler for backtesting, once per process."""
    root = logging.getLogger()
    if any(getattr(handler, _LOG_HANDLER_MARKER, False) for handler in root.handlers):
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / LOG_FILENAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    setattr(handler, _LOG_HANDLER_MARKER, True)
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# factories - every external dependency is built through one of these, so a
# test can replace the whole of it by monkeypatching a single module attribute
# ---------------------------------------------------------------------------


def get_config() -> AppConfig:
    """Load the checked-in application config."""
    return load_config()


def get_settings_path() -> Path:
    """Return the path of the user settings file."""
    return DEFAULT_SETTINGS_PATH


def get_store(config: AppConfig) -> ChunkStore:
    """Open the chunk store for one request.

    A store per request rather than a shared one: its SQLite connection belongs
    to the thread that created it, and FastAPI runs these endpoints in a thread
    pool.
    """
    return ChunkStore(config.database.path)


def get_chunker(config: AppConfig) -> Chunker:
    """Build the chunker described by `config.chunking`."""
    return Chunker(
        config.chunking.encoding,
        config.chunking.chunk_size_tokens,
        config.chunking.name_run_max_extension_tokens,
    )


def get_anthropic_client(api_key: str | None) -> Any:
    """Build the Anthropic client.

    Raises:
        ConfigError: when no key is configured; the exception handler below
            turns that into a 400 rather than a 500, because it is the user's
            setting that is missing, not the server that is broken.
    """
    if not api_key:
        raise ConfigError(MISSING_API_KEY_MESSAGE)
    return anthropic.Anthropic(api_key=api_key)


def get_generator(config: AppConfig, store: ChunkStore) -> Any:
    """Build the synthetic fragment generator."""
    return SyntheticGenerator(config, store)


# ---------------------------------------------------------------------------
# request context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestContext:
    """Everything one request needs to know about how the app is configured."""

    config: AppConfig
    overrides: dict[str, str]
    api_key: str | None


def context_dependency() -> RequestContext:
    """Resolve config, user overrides and API key for the current request."""
    overrides = load_overrides(get_settings_path())
    return RequestContext(
        config=effective_config(get_config(), overrides),
        overrides=overrides,
        api_key=effective_api_key(overrides),
    )


def store_dependency(
    context: RequestContext = Depends(context_dependency),
) -> Iterator[ChunkStore]:
    """Open a store for the request and close it afterwards."""
    store = get_store(context.config)
    try:
        yield store
    finally:
        store.close()


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


class SettingsUpdate(BaseModel):
    """Partial settings update; only the fields actually sent are applied."""

    anthropic_api_key: str | None = None
    model: str | None = None
    effort: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None


class DetectRequest(BaseModel):
    doc_id: str


class CanaryProbeRequest(BaseModel):
    model: str | None = None


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

configure_logging()

app = FastAPI(
    title="doc_quant observability",
    description="Watch a document travel through the anonymization pipeline.",
)

# The frontend is built separately and may not exist yet; mounting a missing
# directory would fail at import, so it is created empty instead.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError) -> JSONResponse:
    """Report a configuration problem as a 400 carrying its own message."""
    logger.warning("Configuration error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=HTTP_BAD_REQUEST, content={"detail": str(exc)})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status and duration of every request."""
    started = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %d in %d ms",
        request.method,
        request.url.path,
        response.status_code,
        _elapsed_ms(started),
    )
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the frontend entry point."""
    index_path = STATIC_DIR / INDEX_FILENAME
    if not index_path.is_file():
        raise HTTPException(
            status_code=HTTP_NOT_FOUND,
            detail=f"{INDEX_FILENAME} not found in {STATIC_DIR}",
        )
    return FileResponse(index_path)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


@app.get("/api/settings")
def read_settings(context: RequestContext = Depends(context_dependency)) -> dict:
    """Return the effective settings; the API key only ever masked."""
    return _settings_payload(context)


@app.put("/api/settings")
def write_settings(update: SettingsUpdate) -> dict:
    """Persist the sent settings and return the new effective settings.

    Only fields present in the request body are touched, so a client may send
    one field without wiping the rest. An empty string clears a setting.
    """
    save_overrides(get_settings_path(), update.model_dump(exclude_unset=True))
    return _settings_payload(context_dependency())


def _settings_payload(context: RequestContext) -> dict:
    """Build the settings response.

    The raw API key is never part of it: a caller learns whether one is stored
    and which one, never its value.
    """
    config = context.config
    return {
        "has_api_key": context.api_key is not None,
        "anthropic_api_key_masked": mask_api_key(context.api_key),
        "model": config.anthropic.model,
        "effort": config.anthropic.effort,
        "llm_base_url": config.synthetic.llm.base_url,
        "llm_model": config.synthetic.llm.model,
        "chunk_size_tokens": config.chunking.chunk_size_tokens,
        "chaff_ratio": config.synthetic.chaff_ratio,
        "honeytoken_rate": config.synthetic.honeytoken_rate,
        "canaries_per_batch": config.synthetic.canaries_per_batch,
    }


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


@app.post("/api/documents")
def upload_document(
    file: UploadFile = File(...),
    context: RequestContext = Depends(context_dependency),
    store: ChunkStore = Depends(store_dependency),
) -> dict:
    """Convert an uploaded file to Markdown, chunk it and store it."""
    filename = file.filename or FALLBACK_UPLOAD_NAME
    markdown = _to_markdown(filename, file.file.read())
    if not markdown.strip():
        raise HTTPException(
            status_code=HTTP_UNPROCESSABLE,
            detail=f"Conversion of {filename} produced no text",
        )

    chunker = get_chunker(context.config)
    chunk_texts = chunker.chunk(markdown)
    doc_id = store.add_document(filename, chunk_texts)
    logger.info("Stored upload %s as document %s (%d chunks)", filename, doc_id, len(chunk_texts))
    return {
        "doc_id": doc_id,
        "filename": filename,
        "markdown": markdown,
        "chunks": _chunk_payloads(store, chunker, context.config, doc_id),
    }


@app.get("/api/documents")
def list_documents(store: ChunkStore = Depends(store_dependency)) -> list[dict]:
    """List the ingested documents."""
    return store.list_documents()


@app.get("/api/documents/{doc_id}")
def read_document(
    doc_id: str,
    context: RequestContext = Depends(context_dependency),
    store: ChunkStore = Depends(store_dependency),
) -> dict:
    """Return one document reassembled from its chunks, plus the chunks."""
    document = _require_document(store, doc_id)
    return {
        "doc_id": doc_id,
        "path": document["path"],
        "markdown": store.reconstruct(doc_id),
        "chunks": _chunk_payloads(store, get_chunker(context.config), context.config, doc_id),
    }


@app.get("/api/documents/{doc_id}/redaction")
def read_redaction(
    doc_id: str,
    context: RequestContext = Depends(context_dependency),
    store: ChunkStore = Depends(store_dependency),
) -> dict:
    """Return the document before and after placeholder substitution."""
    _require_document(store, doc_id)
    original = store.reconstruct(doc_id)
    entities = store.get_document_entities(doc_id)
    redacted = redact_text(
        original,
        entities,
        context.config.redaction.person,
        context.config.redaction.company,
    )
    return {
        "original": original,
        "redacted": redacted,
        "entities": _entity_payloads(entities),
    }


def _to_markdown(filename: str, raw: bytes) -> str:
    """Convert uploaded bytes to Markdown.

    Text formats are taken verbatim: they are already what the pipeline wants,
    and a round trip through a converter could only alter them. Everything else
    goes through markitdown, which needs a real file, so the bytes are written
    to a temporary one carrying the original suffix - that suffix is how the
    converter picks its reader.
    """
    suffix = Path(filename).suffix
    if suffix.lower() in RAW_TEXT_SUFFIXES:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=HTTP_UNPROCESSABLE,
                detail=f"{filename} is not valid UTF-8 text: {exc}",
            ) from exc

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / f"{UPLOAD_STEM}{suffix}"
        source.write_bytes(raw)
        try:
            result = MarkItDown().convert(source)
        except (
            FileConversionException,
            UnsupportedFormatException,
            ValueError,
            OSError,
        ) as exc:
            raise HTTPException(
                status_code=HTTP_UNPROCESSABLE,
                detail=f"Cannot convert {filename} to Markdown: {exc}",
            ) from exc

    # markitdown named the field `text_content` before it grew `markdown`;
    # accept either so the app works across both.
    text = getattr(result, "markdown", None)
    if not isinstance(text, str):
        text = getattr(result, "text_content", None)
    if not isinstance(text, str):
        raise HTTPException(
            status_code=HTTP_UNPROCESSABLE,
            detail=f"Conversion of {filename} returned no text",
        )
    return text


def _chunk_payloads(
    store: ChunkStore, chunker: Chunker, config: AppConfig, doc_id: str
) -> list[dict]:
    """Describe every chunk of a document, down to its individual tokens.

    `extended` marks a chunk that came out longer than the configured size:
    that is the visible trace of a boundary the chunker pushed forward to keep
    a name run whole.
    """
    payloads = []
    for chunk in store.get_document_chunks(doc_id):
        tokens = chunker.token_strings(chunk["text"])
        payloads.append(
            {
                "chunk_id": chunk["chunk_id"],
                "seq": chunk["seq"],
                "token_count": len(tokens),
                "text": chunk["text"],
                "tokens": tokens,
                "extended": len(tokens) > config.chunking.chunk_size_tokens,
            }
        )
    return payloads


def _entity_payloads(entities: list[tuple[str, str]]) -> list[dict]:
    """Render (text, type) pairs as JSON objects."""
    return [{"text": text, "type": entity_type} for text, entity_type in entities]


def _require_document(store: ChunkStore, doc_id: str) -> dict:
    """Return the document row, or raise a 404."""
    for document in store.list_documents():
        if document["doc_id"] == doc_id:
            return document
    raise HTTPException(status_code=HTTP_NOT_FOUND, detail=f"Unknown doc_id: {doc_id}")


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionOutcome:
    """What one synchronous detection call produced."""

    status: str
    entities: list[tuple[str, str]]
    raw_text: str | None
    latency_ms: int
    detail: str | None


@app.post("/api/detect")
def detect(
    request: DetectRequest,
    context: RequestContext = Depends(context_dependency),
    store: ChunkStore = Depends(store_dependency),
) -> dict:
    """Run detection over one document's not-yet-submitted chunks.

    The chunks are mixed with freshly generated synthetic fragments and
    shuffled exactly as the batch path mixes and shuffles them, then sent one
    request at a time. The response reports the submission order, so a reader
    can see the same sequence the provider saw.
    """
    _require_document(store, request.doc_id)
    chunks = [
        chunk
        for chunk in store.get_document_chunks(request.doc_id)
        if chunk["batch_id"] is None
    ]
    if not chunks:
        raise HTTPException(
            status_code=HTTP_CONFLICT, detail=NO_UNSUBMITTED_CHUNKS_MESSAGE
        )

    # Before anything is generated or marked as submitted: a missing key must
    # leave the document exactly as it was.
    client = get_anthropic_client(context.api_key)

    generator = get_generator(context.config, store)
    fragments = plan_synthetic_fragments(context.config, lambda: generator, len(chunks))

    planned = [
        {
            "custom_id": chunk["chunk_id"],
            "kind": KIND_REAL,
            "seq": chunk["seq"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]
    planned.extend(
        {
            "custom_id": fragment.fragment_id,
            "kind": fragment.kind,
            "seq": None,
            "text": fragment.text,
        }
        for fragment in fragments
    )
    # Order must not correlate with document identity, chunk sequence, or with
    # a request being real rather than synthetic.
    random.shuffle(planned)

    batch_id = f"{SYNC_BATCH_PREFIX}{uuid.uuid4().hex[:SYNC_BATCH_ID_CHARS]}"
    store.record_batch(batch_id, BATCH_STATUS_SYNC)
    store.mark_chunks_submitted([chunk["chunk_id"] for chunk in chunks], batch_id)
    if fragments:
        store.mark_synthetic_submitted(
            [fragment.fragment_id for fragment in fragments], batch_id
        )

    template = _payload_template(context.config)
    planted_by_id = {
        fragment.fragment_id: fragment.planted
        for fragment in fragments
        if fragment.kind == KIND_HONEYTOKEN
    }

    results: list[dict] = []
    entities_stored = 0
    honeytokens_found = 0
    for item in planned:
        outcome = _run_detection(client, template, item["text"])
        results.append(
            {
                "custom_id": item["custom_id"],
                "kind": item["kind"],
                "status": outcome.status,
                "entities": _entity_payloads(outcome.entities),
                "raw_text": outcome.raw_text,
                "latency_ms": outcome.latency_ms,
                "detail": outcome.detail,
            }
        )
        if outcome.status != STATUS_OK:
            continue
        if item["kind"] == KIND_REAL:
            store.add_entities(item["custom_id"], outcome.entities)
            entities_stored += len(outcome.entities)
        elif item["kind"] == KIND_HONEYTOKEN:
            store.record_honeytoken_result(item["custom_id"], batch_id, outcome.entities)
            found_names = {name for name, _ in outcome.entities}
            honeytokens_found += sum(
                1
                for name, _ in planted_by_id[item["custom_id"]]
                if name in found_names
            )
        # Chaff exists to dilute and a canary to be answered about later; what
        # the model reported about either of them is of no interest.

    store.set_batch_status(batch_id, BATCH_STATUS_SYNC_COMPLETED)

    by_kind = Counter(fragment.kind for fragment in fragments)
    composition = {KIND_REAL: len(chunks)}
    composition.update({kind: by_kind[kind] for kind in SYNTHETIC_KINDS})
    logger.info("Sync batch %s finished: %s", batch_id, composition)

    return {
        "batch_id": batch_id,
        "composition": composition,
        "payload_template": template,
        "requests": planned,
        "results": results,
        "honeytoken_recall": _honeytoken_recall(planted_by_id, honeytokens_found),
        "entities_stored": entities_stored,
    }


def _payload_template(config: AppConfig) -> dict:
    """Build the per-request payload minus the fragment text.

    This dict is both what is sent (spread into `messages.create`) and what the
    response reports, so the two can never drift apart.
    """
    return {
        "model": config.anthropic.model,
        "max_tokens": config.anthropic.max_tokens,
        "system": DETECTION_SYSTEM_PROMPT,
        "output_config": {
            "effort": config.anthropic.effort,
            "format": {"type": "json_schema", "schema": ENTITY_SCHEMA},
        },
    }


def _run_detection(client: Any, template: dict, text: str) -> DetectionOutcome:
    """Send one fragment and classify the answer.

    A refusal and an API error are both per-fragment outcomes rather than
    failures of the run: the remaining fragments still go out, which is what
    makes a partial result observable instead of lost.
    """
    started = time.perf_counter()
    try:
        message = client.messages.create(
            **template, messages=[{"role": "user", "content": text}]
        )
    except anthropic.APIError as exc:
        logger.warning("Detection request failed: %s", exc)
        return DetectionOutcome(STATUS_ERROR, [], None, _elapsed_ms(started), str(exc))

    raw_text = _message_text(message)
    if getattr(message, "stop_reason", None) == REFUSAL_STOP_REASON:
        # A safety classifier declined; the output need not match the schema,
        # so there is nothing to parse.
        return DetectionOutcome(STATUS_REFUSAL, [], raw_text, _elapsed_ms(started), None)

    try:
        entities = parse_entities(message)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Detection request returned unparsable output: %s", exc)
        return DetectionOutcome(
            STATUS_ERROR, [], raw_text, _elapsed_ms(started), str(exc)
        )

    return DetectionOutcome(STATUS_OK, entities, raw_text, _elapsed_ms(started), None)


def _message_text(message: Any) -> str | None:
    """Join the text blocks of a message, or None when it carries none."""
    content = getattr(message, "content", None) or []
    blocks = [
        block.text for block in content if getattr(block, "type", None) == "text"
    ]
    return "\n".join(blocks) if blocks else None


def _honeytoken_recall(
    planted_by_id: dict[str, list[tuple[str, str]]], found: int
) -> dict | None:
    """Summarise how many planted names came back, or None without honeytokens.

    A name counts as found no matter which type the provider assigned it: the
    measurement is about the name being spotted at all, and a person/company
    mix-up still means it would have been redacted. This mirrors
    `ChunkStore.honeytoken_stats`, which scores the persisted results the same
    way.
    """
    if not planted_by_id:
        return None
    planted = sum(len(names) for names in planted_by_id.values())
    return {
        "planted": planted,
        "found": found,
        "recall": found / planted if planted else 0.0,
    }


def _elapsed_ms(started: float) -> int:
    """Milliseconds since `started`, as measured by a monotonic clock."""
    return int((time.perf_counter() - started) * 1000)


# ---------------------------------------------------------------------------
# synthetic observability
# ---------------------------------------------------------------------------


@app.get("/api/synthetic/report")
def synthetic_report(store: ChunkStore = Depends(store_dependency)) -> dict:
    """Report what the synthetic fragments have measured so far. Offline."""
    counts = {kind: 0 for kind in SYNTHETIC_KINDS}
    for fragment in store.list_synthetic_fragments():
        kind = fragment["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "counts": counts,
        "honeytoken_stats": store.honeytoken_stats(),
        "canary_probes": store.list_canary_probes(),
    }


@app.post("/api/canary-probe")
def canary_probe(
    request: CanaryProbeRequest,
    context: RequestContext = Depends(context_dependency),
    store: ChunkStore = Depends(store_dependency),
) -> dict:
    """Ask a model about every canary person and record each verdict."""
    client = get_anthropic_client(context.api_key)
    results = run_canary_probe(context.config, store, client, model=request.model)
    return {
        "results": [
            {
                "fragment_id": result["fragment_id"],
                "person": result["name"],
                "tripped": result["tripped"],
                "excerpt": result["excerpt"],
            }
            for result in results
        ],
        "tripped": sum(1 for result in results if result["tripped"]),
        "total": len(results),
    }
