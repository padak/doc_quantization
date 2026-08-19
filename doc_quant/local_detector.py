"""Name detection against a local OpenAI-compatible LLM (Ollama, LM Studio).

The parity core of the local backend: both transports - the webapp's
streaming endpoint and the CLI's `detect` command - build their requests and
read the answers exclusively through this module, so the two can never drift
apart (the same rule the Anthropic paths follow with `doc_quant.detector`).

Local mode sends nothing off the machine, so the decontextualization
apparatus is deliberately absent here: no synthetic fragments, no mixing, no
shuffle. Chunks are still the detection unit - small local models recall
names better on short fragments - and results land in the same entities
table the remote paths fill.

The constrained-output shape (`response_format.type == "json_schema"` with a
nested `json_schema` object) is what Ollama's and LM Studio's
OpenAI-compatible endpoints document; llama.cpp documents a flat variant and
is out of scope. A model that ignores the constraint is caught by the same
schema validation the remote paths apply, plus the verbatim guard below.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass

import httpx

from doc_quant.config import AppConfig
from doc_quant.detector import (
    DETECTION_SYSTEM_PROMPT,
    ENTITY_SCHEMA,
    parse_entities_payload,
)
from doc_quant.local_llm import LocalLLMClient, LocalLLMError

logger = logging.getLogger(__name__)

# Structural constants, not tunables: detection wants the most deterministic
# answer the server can give, and one retry is for a sloppy sample, not for a
# dead server (a transport error is not retried at all).
DETECTION_TEMPERATURE = 0.0
LOCAL_DETECTION_ATTEMPTS = 2
RESPONSE_SCHEMA_NAME = "entities"

# Cheap availability check; the endpoint every OpenAI-compatible server has.
PROBE_PATH = "/models"
PROBE_TIMEOUT_SECONDS = 5.0

# A local run is still recorded as a batch so chunk submission state, the
# stored-run view and redaction share one ledger with the remote runs. The
# prefix tells the runs apart; the statuses reuse the webapp's sync lifecycle.
LOCAL_BATCH_PREFIX = "local-"
LOCAL_BATCH_ID_CHARS = 12
LOCAL_BATCH_STATUS_RUNNING = "sync"
LOCAL_BATCH_STATUS_COMPLETED = "sync-completed"

STATUS_OK = "ok"
STATUS_ERROR = "error"

# Keys of the payload template that are reported to the operator but must not
# reach the wire: base_url says where requests go, system becomes a message.
_REPORTING_ONLY_KEYS = ("base_url", "system")


@dataclass(frozen=True)
class LocalDetectionOutcome:
    """What one local detection call produced."""

    status: str
    entities: list
    raw_text: str | None
    latency_ms: int
    detail: str | None
    dropped: int


def get_local_client(
    config: AppConfig, transport: httpx.BaseTransport | None = None
) -> LocalLLMClient:
    """Build the client for the configured local detection endpoint."""
    local = config.detection.local
    return LocalLLMClient(
        base_url=local.base_url,
        model=local.model,
        temperature=DETECTION_TEMPERATURE,
        timeout_seconds=local.timeout_seconds,
        transport=transport,
    )


def build_local_payload_template(config: AppConfig) -> dict:
    """Build the per-request payload minus the fragment text.

    This dict is both what is sent (through `build_local_request`) and what
    the webapp reports as `payload_template`, so the two can never drift
    apart. The same system prompt and schema as the remote paths: only the
    transport differs, never the question.
    """
    local = config.detection.local
    return {
        "base_url": local.base_url,
        "model": local.model,
        "temperature": DETECTION_TEMPERATURE,
        "stream": False,
        "system": DETECTION_SYSTEM_PROMPT,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": RESPONSE_SCHEMA_NAME,
                "strict": True,
                "schema": ENTITY_SCHEMA,
            },
        },
    }


def build_local_request(template: dict, text: str) -> dict:
    """Turn the template into the wire payload for one fragment."""
    payload = {
        key: value for key, value in template.items() if key not in _REPORTING_ONLY_KEYS
    }
    payload["messages"] = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": text},
    ]
    return payload


def detect_local(
    client: LocalLLMClient, template: dict, text: str
) -> LocalDetectionOutcome:
    """Send one fragment to the local model and classify the answer.

    A transport error is a per-fragment outcome rather than a failure of the
    run - the remaining fragments still go out, matching the remote sync
    path. An answer that fails JSON or schema validation is retried once: a
    local model may produce one sloppy sample, but a server that answers
    garbage twice is an errored fragment, not a stalled pipeline.
    """
    started = time.perf_counter()
    raw_text: str | None = None
    detail = "unknown"
    for attempt in range(1, LOCAL_DETECTION_ATTEMPTS + 1):
        try:
            raw_text = client.chat_completion(build_local_request(template, text))
        except LocalLLMError as exc:
            return LocalDetectionOutcome(
                STATUS_ERROR, [], raw_text, _elapsed_ms(started), str(exc), 0
            )
        try:
            entities = parse_entities_payload(json.loads(raw_text))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            detail = str(exc)
            logger.warning(
                "Local detection output invalid on attempt %d/%d: %s",
                attempt,
                LOCAL_DETECTION_ATTEMPTS,
                exc,
            )
            continue
        kept, dropped = _filter_verbatim(entities, text)
        return LocalDetectionOutcome(
            STATUS_OK, kept, raw_text, _elapsed_ms(started), None, dropped
        )

    return LocalDetectionOutcome(
        STATUS_ERROR, [], raw_text, _elapsed_ms(started), detail, 0
    )


def _filter_verbatim(entities: list, text: str) -> tuple[list, int]:
    """Keep only entities that appear verbatim in the fragment.

    Anthropic structured outputs enforce the exact-substring contract hard; a
    small local model can hallucinate an entity that appears nowhere in the
    fragment, and a hallucinated string must not pollute the entities table
    even though the redactor would never match it.
    """
    kept = [
        (entity_text, entity_type)
        for entity_text, entity_type in entities
        if entity_text in text
    ]
    dropped = len(entities) - len(kept)
    if dropped:
        logger.warning("Dropped %d hallucinated entities from a local answer", dropped)
    return kept, dropped


def probe_local_server(
    config: AppConfig, transport: httpx.BaseTransport | None = None
) -> None:
    """Fail fast when the local detection server is unreachable.

    Called by both transports before any store write, so a dead server leaves
    the document exactly as it was - the same rule the remote paths apply to
    a missing API key.

    Raises:
        LocalLLMError: with an actionable message naming the endpoint.
    """
    local = config.detection.local
    url = f"{local.base_url.rstrip('/')}{PROBE_PATH}"
    try:
        with httpx.Client(
            transport=transport, timeout=PROBE_TIMEOUT_SECONDS
        ) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise _unreachable(local.base_url, local.model, str(exc)) from exc
    if response.status_code != 200:
        raise _unreachable(local.base_url, local.model, f"HTTP {response.status_code}")


def _unreachable(base_url: str, model: str, detail: str) -> LocalLLMError:
    """Build an error that says what to do about it, not just what broke."""
    return LocalLLMError(
        f"Local detection server unreachable at {base_url}: {detail}. "
        f"Start one, e.g. Ollama (`ollama serve`, `ollama pull {model}`), "
        "or change detection.local.base_url in config/config.json."
    )


def new_local_batch_id() -> str:
    """Mint the id a local run is recorded under."""
    return f"{LOCAL_BATCH_PREFIX}{uuid.uuid4().hex[:LOCAL_BATCH_ID_CHARS]}"


def _elapsed_ms(started: float) -> int:
    """Milliseconds since `started`, as measured by a monotonic clock."""
    return int((time.perf_counter() - started) * 1000)
