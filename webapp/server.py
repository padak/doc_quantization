"""FastAPI observability layer over the `doc_quant` pipeline.

The app makes each step of the anonymization pipeline visible: what a document
was chunked into, which fragments were mixed into a submission, what the
provider actually received in which order, what it answered, and what the
resulting redaction looks like. None of the pipeline logic is reimplemented
here - chunking, synthetic mixing, answer parsing and redaction are imported
from `doc_quant`.

Detection is synchronous here, unlike the CLI's Batches API path: one
`messages.create` call per fragment, a few of them in flight at a time
(`anthropic.detect_concurrency`), so a user watches the requests go out instead
of waiting for a batch to finish. It is also streamed: the
endpoint answers with newline-delimited JSON and emits an event as each step
completes, so the wait is observable rather than a blank several minutes. The
payload is deliberately
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
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from doc_quant.chunker import Chunker
from doc_quant.cli import run_canary_probe
from doc_quant.config import (
    PROJECT_ROOT,
    AppConfig,
    ConfigError,
    ConversionConfig,
    load_config,
)
from doc_quant.detector import (
    DETECTION_SYSTEM_PROMPT,
    ENTITY_SCHEMA,
    parse_entities,
    plan_synthetic_fragments,
)
from doc_quant.redactor import EMAIL, URL, find_emails, find_urls, redact_text
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
    SettingValue,
    effective_api_key,
    effective_config,
    load_overrides,
    mask_api_key,
    save_overrides,
)

logger = logging.getLogger(__name__)

STATIC_DIR = PROJECT_ROOT / "webapp" / "static"
INDEX_FILENAME = "index.html"
STATIC_URL_PREFIX = "/static/"

# Frontend files are served without a Cache-Control header by default, which
# lets browsers apply heuristic freshness and keep serving a stale app.js for
# days after a deploy without ever asking the server. "no-cache" means "store,
# but revalidate every time": with the ETag StaticFiles already sends, an
# unchanged file costs one cheap 304 and a changed one is picked up at once.
CACHE_CONTROL_HEADER = "Cache-Control"
STATIC_CACHE_CONTROL = "no-cache"

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

# The optional external conversion service (see `conversion` in
# config/config.json): one endpoint that takes a file and answers with
# Markdown, and one that says whether it is up.
CONVERT_PATH = "/convert"
CONVERT_FILE_FIELD = "file"
CONVERT_MARKDOWN_FIELD = "markdown"
# Converting a large PDF well takes time; this is not the preflight's timeout.
CONVERT_TIMEOUT_SECONDS = 120.0
FALLBACK_UPLOAD_NAME = "upload"
# Pasted text arrives with no filename at all; the suffix keeps the document
# on the raw-text path everywhere a name is inspected.
FALLBACK_PASTED_NAME = "pasted-text.md"

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

# Detection progress protocol: one JSON object per line, each carrying a
# "type". The frontend switches on these, so they are part of the API.
NDJSON_MEDIA_TYPE = "application/x-ndjson"
EVENT_PHASE = "phase"
EVENT_SYNTHETIC = "synthetic"
EVENT_SUBMITTED = "submitted"
EVENT_RESULT = "result"
EVENT_DONE = "done"
EVENT_ERROR = "error"

PHASE_PLANNING = "planning"
PHASE_SYNTHETICS = "synthetics"
PHASE_CANARIES = "canaries"
PHASE_CANARIES_DETAIL = "ensuring canary set"
PHASE_SYNTHETICS_TEMPLATE_DETAIL = "deterministic templates (local LLM disabled)"

# Preflight checks reported by /api/verify.
CHECK_ANTHROPIC = "anthropic"
CHECK_LOCAL_LLM = "local_llm"
CHECK_DATABASE = "database"
CHECK_CONVERSION = "conversion"
CHECK_LABELS = {
    CHECK_ANTHROPIC: "Anthropic API",
    CHECK_LOCAL_LLM: "Local LLM",
    CHECK_DATABASE: "Database",
    CHECK_CONVERSION: "Document conversion",
}
VERIFY_NO_KEY_DETAIL = (
    "No Anthropic API key stored: add one in Settings (or set the "
    "ANTHROPIC_API_KEY environment variable)."
)
# The preflight must answer quickly even when the local endpoint is dead, so it
# uses its own short timeout rather than the generation timeout from the config.
VERIFY_HTTP_TIMEOUT_SECONDS = 5.0
VERIFY_LLM_DISABLED_DETAIL = (
    "skipped - local LLM disabled, deterministic templates in use"
)
MODELS_PATH = "/models"
OLLAMA_DEFAULT_TAG = ":latest"
VERIFY_SAMPLE_NAME = "preflight.html"
VERIFY_SAMPLE_HTML = b"<html><body><h1>Preflight</h1><p>Conversion works.</p></body></html>"

HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_UNPROCESSABLE = 422
HTTP_BAD_GATEWAY = 502


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

    A store per request, and with the cross-thread guard off: FastAPI runs a
    sync dependency, the endpoint body and the dependency's cleanup each on
    whatever thread-pool worker is free, so one request's store hops threads.
    The hops are sequential - a request never runs two store calls at once -
    which is exactly the case `allow_cross_thread` exists for. With the guard
    on, a request whose dependency and endpoint landed on different workers
    died with sqlite3.ProgrammingError.
    """
    return ChunkStore(config.database.path, allow_cross_thread=True)


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


def get_http_client(timeout: float = VERIFY_HTTP_TIMEOUT_SECONDS) -> Any:
    """Build the HTTP client used to reach the local side services.

    Separate from `doc_quant.synthetic`'s own client. The default timeout is
    the preflight's: it only asks a server what it can do and must give up
    quickly when nothing answers. Converting a document is the other kind of
    call, and passes its own, far longer, timeout.
    """
    return httpx.Client(timeout=timeout)


# ---------------------------------------------------------------------------
# request context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestContext:
    """Everything one request needs to know about how the app is configured."""

    config: AppConfig
    overrides: dict[str, SettingValue]
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
    llm_enabled: bool | None = None
    # Stored verbatim, empty string included: an empty URL is the user asking
    # for the built-in converter rather than clearing an override.
    conversion_service_url: str | None = None


class TextDocumentRequest(BaseModel):
    """Text pasted straight into the UI, plus an optional display name."""

    text: str
    name: str | None = None


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
async def revalidate_frontend(request: Request, call_next):
    """Make browsers revalidate the frontend files on every load."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith(STATIC_URL_PREFIX):
        response.headers[CACHE_CONTROL_HEADER] = STATIC_CACHE_CONTROL
    return response


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
        "llm_enabled": config.synthetic.llm.enabled,
        "conversion_service_url": config.conversion.service_url,
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
    markdown = _to_markdown(filename, file.file.read(), context.config.conversion)
    if not markdown.strip():
        raise HTTPException(
            status_code=HTTP_UNPROCESSABLE,
            detail=f"Conversion of {filename} produced no text",
        )
    return _store_markdown(filename, markdown, context, store)


@app.post("/api/documents/text")
def ingest_text_document(
    request: TextDocumentRequest,
    context: RequestContext = Depends(context_dependency),
    store: ChunkStore = Depends(store_dependency),
) -> dict:
    """Chunk and store text pasted directly into the UI.

    The text is already what the pipeline wants, so it takes the same verbatim
    path as an uploaded ``.md``/``.txt`` file - no conversion service involved.
    """
    if not request.text.strip():
        raise HTTPException(
            status_code=HTTP_UNPROCESSABLE,
            detail="Pasted text is empty",
        )
    name = (request.name or "").strip() or FALLBACK_PASTED_NAME
    return _store_markdown(name, request.text, context, store)


def _store_markdown(
    filename: str, markdown: str, context: RequestContext, store: ChunkStore
) -> dict:
    """Chunk `markdown`, store it under `filename` and build the API payload."""
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
    """Return the document before and after placeholder substitution.

    Email addresses and URLs are not in the entities table - they are never
    sent to the detector - so they are found on the original here and reported
    alongside the stored entities, which is how the view can show what was
    removed.
    """
    _require_document(store, doc_id)
    original = store.reconstruct(doc_id)
    entities = store.get_document_entities(doc_id)
    redacted = redact_text(
        original,
        entities,
        context.config.redaction.person,
        context.config.redaction.company,
        email_placeholder=context.config.redaction.email,
        url_placeholder=context.config.redaction.url,
    )
    payloads = _entity_payloads(entities)
    known = {payload["text"] for payload in payloads}
    for found, entity_type in ((find_emails(original), EMAIL), (find_urls(original), URL)):
        payloads.extend(
            {"text": text, "type": entity_type}
            for text in found
            if text not in known
        )
        known.update(found)
    return {
        "original": original,
        "redacted": redacted,
        "entities": payloads,
    }


def _to_markdown(
    filename: str, raw: bytes, conversion: ConversionConfig | None = None
) -> str:
    """Convert uploaded bytes to Markdown.

    Text formats are taken verbatim and always locally: they are already what
    the pipeline wants, and a round trip through any converter could only alter
    them. Everything else requires the external conversion service - this app
    deliberately ships no converter of its own; conversion is the companion
    service's whole job (and keeping it behind an HTTP boundary is what keeps
    this repository free of the AGPL PDF stack).
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

    if conversion is not None and conversion.service_url:
        return _convert_via_service(filename, raw, conversion.service_url)

    raise HTTPException(
        status_code=HTTP_UNPROCESSABLE,
        detail=(
            f"Cannot convert {filename}: no conversion service is configured. "
            "Set the Conversion service URL in Settings to a running "
            "doc_converter instance (github.com/padak/doc_converter), or "
            "upload Markdown/plain text directly."
        ),
    )


def _convert_via_service(filename: str, raw: bytes, service_url: str) -> str:
    """Hand the file to the external conversion service and read its Markdown.

    A service that was configured but cannot be reached is a 502 rather than a
    silent fallback to the built-in converter: the user asked for the better
    conversion, and quietly giving them the worse one would be indistinguishable
    from it working.
    """
    url = f"{service_url.rstrip('/')}{CONVERT_PATH}"
    try:
        client = get_http_client(CONVERT_TIMEOUT_SECONDS)
        try:
            response = client.post(url, files={CONVERT_FILE_FIELD: (filename, raw)})
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - any transport failure is the same answer
        logger.warning("Conversion service unreachable at %s: %s", url, exc)
        raise HTTPException(
            status_code=HTTP_BAD_GATEWAY,
            detail=f"Conversion service unreachable at {url}: {_readable(exc)}",
        ) from exc

    status_code = getattr(response, "status_code", None)
    if status_code != 200:
        logger.warning("Conversion service at %s answered HTTP %s", url, status_code)
        raise HTTPException(
            status_code=HTTP_BAD_GATEWAY,
            detail=(
                f"Conversion service at {url} answered HTTP {status_code} "
                f"for {filename}"
            ),
        )

    try:
        markdown = response.json()[CONVERT_MARKDOWN_FIELD]
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Conversion service at %s returned an unexpected payload: %s", url, exc)
        raise HTTPException(
            status_code=HTTP_BAD_GATEWAY,
            detail=f"Conversion service at {url} returned an unexpected payload: {exc}",
        ) from exc

    if not isinstance(markdown, str):
        raise HTTPException(
            status_code=HTTP_BAD_GATEWAY,
            detail=f"Conversion service at {url} returned no text for {filename}",
        )
    return markdown


def _chunk_payloads(
    store: ChunkStore, chunker: Chunker, config: AppConfig, doc_id: str
) -> list[dict]:
    """Describe every chunk of a document, down to its individual tokens.

    `tokens` carries display segments rather than one string per token: a
    character whose bytes span two tokens has no faithful per-token rendering,
    and showing "Petr ??imecek" would misrepresent the very text the pipeline
    handles losslessly. The segments join back to the chunk text exactly.

    `token_count` stays the real token count - it is what the configured chunk
    size is expressed in - so it may exceed the number of segments on text with
    multi-byte characters.

    `extended` marks a chunk that came out longer than the configured size:
    that is the visible trace of a boundary the chunker pushed forward to keep
    a name run whole.
    """
    payloads = []
    for chunk in store.get_document_chunks(doc_id):
        segments = chunker.token_display_segments(chunk["text"])
        token_count = len(chunker.token_strings(chunk["text"]))
        payloads.append(
            {
                "chunk_id": chunk["chunk_id"],
                "seq": chunk["seq"],
                "token_count": token_count,
                "text": chunk["text"],
                "tokens": segments,
                "extended": token_count > config.chunking.chunk_size_tokens,
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


class _CountingProbe:
    """Stands in for the generator so a plan can be costed without running it.

    The mixing math must exist in exactly one place
    (`plan_synthetic_fragments`), yet the stream needs to know how many
    honeytokens and chaff fragments a submission of this size carries *before*
    generating them one at a time. So the plan is run against this probe, which
    records the counts it is asked for and generates nothing.
    """

    def __init__(self) -> None:
        self.honeytokens = 0
        self.chaff = 0

    def make_honeytokens(self, count: int) -> list:
        self.honeytokens = count
        return []

    def make_chaff(self, count: int) -> list:
        self.chaff = count
        return []

    def ensure_canaries(self) -> list:
        return []


class _CanaryProbe:
    """Runs only the canary half of a plan, so its sampling stays shared.

    The canary set is persistent and its per-batch sample size is decided by
    `plan_synthetic_fragments`; letting the plan do that work while the other
    two mechanisms answer with nothing keeps that decision in one place too.
    """

    def __init__(self, generator: Any) -> None:
        self._generator = generator

    def make_honeytokens(self, count: int) -> list:
        return []

    def make_chaff(self, count: int) -> list:
        return []

    def ensure_canaries(self) -> list:
        return self._generator.ensure_canaries()


@app.post("/api/detect")
def detect(
    request: DetectRequest,
    context: RequestContext = Depends(context_dependency),
    store: ChunkStore = Depends(store_dependency),
) -> StreamingResponse:
    """Run detection over one document's not-yet-submitted chunks, streaming.

    The chunks are mixed with freshly generated synthetic fragments and
    shuffled exactly as the batch path mixes and shuffles them, then sent one
    request at a time. The answer is newline-delimited JSON: a `phase` and
    `synthetic` event per generation step, one `submitted` event carrying the
    whole submission in provider order, one `result` event per provider call as
    it completes, and a final `done` event carrying the complete run.

    Everything that can fail before the first byte is checked here rather than
    in the stream, so an unknown document, an already-detected one and a
    missing API key stay ordinary HTTP errors with a status code.
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

    return StreamingResponse(
        _detect_events(context.config, store, client, generator, chunks),
        media_type=NDJSON_MEDIA_TYPE,
    )


def _detect_events(
    config: AppConfig,
    store: ChunkStore,
    client: Any,
    generator: Any,
    chunks: list[dict],
) -> Iterator[str]:
    """Wrap the run so a mid-stream failure still reaches the reader.

    The status line is long gone by then, so the only way to report a break is
    a last event saying so.
    """
    try:
        yield from _run_detection_stream(config, store, client, generator, chunks)
    except Exception as exc:  # noqa: BLE001 - the stream must not die silently
        logger.exception("Detection stream failed")
        yield _ndjson({"type": EVENT_ERROR, "detail": _readable(exc)})


def _run_detection_stream(
    config: AppConfig,
    store: ChunkStore,
    client: Any,
    generator: Any,
    chunks: list[dict],
) -> Iterator[str]:
    """Generate, submit and collect, emitting one event per finished step."""
    yield _ndjson(_phase_event(PHASE_PLANNING, f"{len(chunks)} real fragments"))

    counts = _CountingProbe()
    plan_synthetic_fragments(config, lambda: counts, len(chunks))

    synthetic_config = config.synthetic
    if (
        synthetic_config.honeytokens_enabled
        or synthetic_config.chaff_enabled
        or synthetic_config.canaries_enabled
    ):
        llm = synthetic_config.llm
        detail = (
            f"generating via {llm.model} at {llm.base_url}"
            if llm.enabled
            else PHASE_SYNTHETICS_TEMPLATE_DETAIL
        )
        yield _ndjson(_phase_event(PHASE_SYNTHETICS, detail))

    fragments: list[Any] = []
    # One fragment per call rather than one call per kind: the point of the
    # stream is that a slow local model is visible while it works.
    for kind, make, total in (
        (KIND_HONEYTOKEN, generator.make_honeytokens, counts.honeytokens),
        (KIND_CHAFF, generator.make_chaff, counts.chaff),
    ):
        for index in range(1, total + 1):
            made = make(1)
            if not made:
                # The mechanism answered with nothing (it is switched off);
                # there is no fragment to report.
                break
            fragments.extend(made)
            yield _ndjson(
                {
                    "type": EVENT_SYNTHETIC,
                    "kind": kind,
                    "index": index,
                    "total": total,
                    "fragment_id": made[0].fragment_id,
                }
            )

    if synthetic_config.canaries_enabled:
        yield _ndjson(_phase_event(PHASE_CANARIES, PHASE_CANARIES_DETAIL))
    canaries = plan_synthetic_fragments(
        config, lambda: _CanaryProbe(generator), len(chunks)
    )
    for index, canary in enumerate(canaries, start=1):
        yield _ndjson(
            {
                "type": EVENT_SYNTHETIC,
                "kind": KIND_CANARY,
                "index": index,
                "total": len(canaries),
                "fragment_id": canary.fragment_id,
            }
        )
    fragments.extend(canaries)

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

    template = _payload_template(config)
    by_kind = Counter(fragment.kind for fragment in fragments)
    composition = {KIND_REAL: len(chunks)}
    composition.update({kind: by_kind[kind] for kind in SYNTHETIC_KINDS})

    # Emitted after the shuffle and the marking, before the first provider
    # call: this is exactly what is about to leave the machine, in order.
    yield _ndjson(
        {
            "type": EVENT_SUBMITTED,
            "batch_id": batch_id,
            "composition": composition,
            "payload_template": template,
            "requests": planned,
        }
    )

    planted_by_id = {
        fragment.fragment_id: fragment.planted
        for fragment in fragments
        if fragment.kind == KIND_HONEYTOKEN
    }

    results: list[dict] = []
    entities_stored = 0
    honeytokens_found = 0
    # A worker does one thing: send a fragment and time the answer. It touches
    # no store and no shared mutable state, so every SQLite write - and every
    # decision about where a result belongs - stays on this thread by design:
    # the store's connection belongs to the thread that opened it, and the
    # routing rules are easier to trust in one sequential place. The Anthropic
    # client is shared: the SDK's client is safe to send requests from several
    # threads.
    workers = max(1, config.anthropic.detect_concurrency)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            pool.submit(_run_detection, client, template, item["text"]): item
            for item in planned
        }
        # Completion order, not submission order: a result is reported the
        # moment it lands, which is the whole point of running several at once.
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            outcome = future.result()
            result = {
                "custom_id": item["custom_id"],
                "kind": item["kind"],
                "status": outcome.status,
                "entities": _entity_payloads(outcome.entities),
                "raw_text": outcome.raw_text,
                "latency_ms": outcome.latency_ms,
                "detail": outcome.detail,
            }
            results.append(result)
            yield _ndjson(
                {"type": EVENT_RESULT, "index": index, "total": len(planned), **result}
            )

            if outcome.status != STATUS_OK:
                continue
            if item["kind"] == KIND_REAL:
                store.add_entities(item["custom_id"], outcome.entities)
                entities_stored += len(outcome.entities)
            elif item["kind"] == KIND_HONEYTOKEN:
                store.record_honeytoken_result(
                    item["custom_id"], batch_id, outcome.entities
                )
                found_names = {name for name, _ in outcome.entities}
                honeytokens_found += sum(
                    1
                    for name, _ in planted_by_id[item["custom_id"]]
                    if name in found_names
                )
            # Chaff exists to dilute and a canary to be answered about later;
            # what the model reported about either of them is of no interest.
    finally:
        # A reader that walks away must not keep the run going: whatever is
        # still queued is dropped rather than waited for.
        pool.shutdown(wait=False, cancel_futures=True)

    store.set_batch_status(batch_id, BATCH_STATUS_SYNC_COMPLETED)
    logger.info("Sync batch %s finished: %s", batch_id, composition)

    yield _ndjson(
        {
            "type": EVENT_DONE,
            "batch_id": batch_id,
            "composition": composition,
            "payload_template": template,
            "requests": planned,
            "results": results,
            "honeytoken_recall": _honeytoken_recall(planted_by_id, honeytokens_found),
            "entities_stored": entities_stored,
        }
    )


def _phase_event(phase: str, detail: str) -> dict:
    """Build a progress event announcing which step is now running."""
    return {"type": EVENT_PHASE, "phase": phase, "detail": detail}


def _ndjson(event: dict) -> str:
    """Render one event as a line of newline-delimited JSON."""
    return json.dumps(event, ensure_ascii=False) + "\n"


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


@app.get("/api/llm-models")
def llm_models(context: RequestContext = Depends(context_dependency)) -> dict:
    """Offer the catalogued local models, saying which are actually installed.

    The catalog is config data (measured figures, not promises); availability
    is asked of the local server right here, because a model listed but not
    pulled is the difference between a choice and a broken run. An unreachable
    endpoint is a normal answer - no model is available - rather than an error:
    the deterministic templates remain a valid choice either way.
    """
    llm = context.config.synthetic.llm
    available, reachable = _local_models(llm.base_url)
    return {
        "catalog": [
            {
                "model": entry.model,
                "size": entry.size,
                "seconds_per_fragment": entry.seconds_per_fragment,
                "first_try_validity": entry.first_try_validity,
                "note": entry.note,
                "available": _serves_model(entry.model, available),
            }
            for entry in llm.catalog
        ],
        "catalog_note": llm.catalog_note,
        "available": available,
        "llm_reachable": reachable,
    }


def _local_models(base_url: str) -> tuple[list[str], bool]:
    """Ask the local endpoint what it serves; ([], False) when it does not answer."""
    url = f"{base_url.rstrip('/')}{MODELS_PATH}"
    try:
        client = get_http_client()
        try:
            response = client.get(url)
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - unreachable is a normal outcome
        logger.info("Local LLM unreachable at %s: %s", url, exc)
        return [], False

    if getattr(response, "status_code", None) != 200:
        logger.info("Local LLM at %s answered HTTP %s", url, getattr(response, "status_code", None))
        return [], False

    try:
        return [item["id"] for item in response.json()["data"]], True
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Local LLM at %s returned an unexpected payload: %s", url, exc)
        return [], False


@app.post("/api/verify")
def verify(context: RequestContext = Depends(context_dependency)) -> dict:
    """Check every external dependency the pipeline needs before a run.

    Each check is independent, timed and never raises: a broken one is a
    reported failure, not a 500, because the whole point is to tell the user
    which piece of their environment is missing.
    """
    checks = [
        _check_anthropic(context),
        _check_local_llm(context.config),
        _check_conversion(context.config),
        _check_database(context.config),
    ]
    all_ok = all(check["ok"] for check in checks)
    logger.info(
        "Preflight: %s",
        ", ".join(f"{check['name']}={'ok' if check['ok'] else 'failed'}" for check in checks),
    )
    return {"checks": checks, "all_ok": all_ok}


def _check(name: str, ok: bool, detail: str, started: float) -> dict:
    """Render one finished check."""
    return {
        "name": name,
        "label": CHECK_LABELS[name],
        "ok": ok,
        "detail": detail,
        "latency_ms": _elapsed_ms(started),
    }


def _check_anthropic(context: RequestContext) -> dict:
    """Confirm that the stored key can retrieve the configured model."""
    started = time.perf_counter()
    model = context.config.anthropic.model
    if not context.api_key:
        return _check(CHECK_ANTHROPIC, False, VERIFY_NO_KEY_DETAIL, started)

    try:
        client = get_anthropic_client(context.api_key)
        retrieved = client.models.retrieve(model)
    except Exception as exc:  # noqa: BLE001 - a failed check is the answer
        logger.warning("Preflight: Anthropic check failed: %s", exc)
        return _check(CHECK_ANTHROPIC, False, _readable(exc), started)

    model_id = getattr(retrieved, "id", None) or model
    return _check(CHECK_ANTHROPIC, True, f"model {model_id} confirmed", started)


def _check_local_llm(config: AppConfig) -> dict:
    """Confirm the local endpoint answers and knows the configured model.

    The endpoint is optional: with the local LLM switched off the generator
    phrases its fragments from deterministic templates, so nothing is probed
    and the check passes - a component deliberately not in use must not report
    the environment as broken.
    """
    started = time.perf_counter()
    llm = config.synthetic.llm
    if not llm.enabled:
        return _check(CHECK_LOCAL_LLM, True, VERIFY_LLM_DISABLED_DETAIL, started)

    base_url = llm.base_url.rstrip("/")
    hint = f"start it: ollama serve && ollama pull {llm.model}"

    try:
        client = get_http_client()
        try:
            response = client.get(f"{base_url}{MODELS_PATH}")
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - unreachable is a normal outcome
        logger.warning("Preflight: local LLM unreachable at %s: %s", base_url, exc)
        return _check(
            CHECK_LOCAL_LLM, False, f"{base_url} unreachable ({exc}); {hint}", started
        )

    status_code = getattr(response, "status_code", None)
    if status_code != 200:
        return _check(
            CHECK_LOCAL_LLM,
            False,
            f"{base_url}{MODELS_PATH} answered HTTP {status_code}; {hint}",
            started,
        )

    try:
        model_ids = [item["id"] for item in response.json()["data"]]
    except (ValueError, KeyError, TypeError) as exc:
        return _check(
            CHECK_LOCAL_LLM,
            False,
            f"{base_url}{MODELS_PATH} returned an unexpected payload ({exc})",
            started,
        )

    if _serves_model(llm.model, model_ids):
        return _check(
            CHECK_LOCAL_LLM, True, f"{base_url} serves {llm.model}", started
        )
    available = ", ".join(model_ids) if model_ids else "none"
    return _check(
        CHECK_LOCAL_LLM,
        False,
        f"{base_url} does not serve {llm.model}; available: {available}",
        started,
    )


def _serves_model(model: str, model_ids: list[str]) -> bool:
    """Whether `model` is among `model_ids`.

    Ollama reports a pulled model under its full tag, so a configured
    "llama3.2" is served by "llama3.2:latest"; that one implicit tag is
    accepted, anything else has to match exactly.
    """
    return model in model_ids or f"{model}{OLLAMA_DEFAULT_TAG}" in model_ids


def _check_conversion(config: AppConfig) -> dict:
    """Convert a tiny sample through the conversion path uploads actually take.

    With a conversion service configured the sample goes through its /convert
    endpoint - the same route every upload takes - so a green check proves the
    path that is really in use. Without one the app is in text-only mode
    (Markdown and plain text passthrough); that is a deliberate state, not a
    broken environment, so the check passes and says what it means.
    """
    started = time.perf_counter()
    service_url = config.conversion.service_url
    if not service_url:
        return _check(
            CHECK_CONVERSION,
            True,
            (
                "no conversion service configured - only Markdown and plain "
                "text uploads are accepted; set the Conversion service URL in "
                "Settings to convert PDF, DOCX or HTML"
            ),
            started,
        )

    hint = (
        "; start the conversion service, or clear the Conversion service URL "
        "in Settings to run in text-only mode"
    )
    try:
        markdown = _to_markdown(
            VERIFY_SAMPLE_NAME, VERIFY_SAMPLE_HTML, config.conversion
        )
    except HTTPException as exc:
        logger.warning("Preflight: conversion check failed: %s", exc.detail)
        return _check(CHECK_CONVERSION, False, f"{exc.detail}{hint}", started)
    except Exception as exc:  # noqa: BLE001 - a failed check is the answer
        logger.warning("Preflight: conversion check failed: %s", exc)
        return _check(CHECK_CONVERSION, False, f"{_readable(exc)}{hint}", started)

    if not markdown.strip():
        return _check(
            CHECK_CONVERSION,
            False,
            f"conversion service at {service_url} returned no markdown{hint}",
            started,
        )
    return _check(
        CHECK_CONVERSION,
        True,
        f"conversion service at {service_url}: converted a sample document "
        f"to {len(markdown)} characters of markdown",
        started,
    )


def _check_database(config: AppConfig) -> dict:
    """Open the store to prove the database file exists and is writable.

    Opening it is the write: `ChunkStore` runs its schema script and commits
    before returning, so a missing directory or a read-only file fails here.
    """
    started = time.perf_counter()
    path = config.database.path
    try:
        store = get_store(config)
        try:
            documents = len(store.list_documents())
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 - a failed check is the answer
        logger.warning("Preflight: database check failed: %s", exc)
        return _check(CHECK_DATABASE, False, _readable(exc), started)

    return _check(
        CHECK_DATABASE, True, f"{path} is writable ({documents} documents)", started
    )


def _readable(exc: Exception) -> str:
    """Render an exception for a user: its message, or its type when silent."""
    message = str(exc).strip()
    return message or exc.__class__.__name__


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
