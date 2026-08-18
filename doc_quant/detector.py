"""Name detection over decontextualized chunks via the Anthropic Batches API.

Every chunk is submitted as an isolated fragment under its random chunk id:
no document identity, no ordering, no neighbour identity. The request list is
shuffled before submission so that even the position of a request inside the
batch carries no information about the document it came from. The only context
a chunk gets is a few margin tokens of adjacent text, which is needed so that a
name split across a chunk boundary can still be recognised.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from doc_quant.config import AppConfig, require_api_key

logger = logging.getLogger(__name__)

DETECTION_SYSTEM_PROMPT = (
    "You receive a short fragment of text with no surrounding context. "
    "Identify every person name and every company or organization name that "
    "appears verbatim in the fragment. Report each one exactly as it appears "
    "in the fragment, as an exact substring. Partial names at the fragment "
    "edges count too. If there are none, return an empty list."
)

ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ["person", "company"]},
                },
                "required": ["text", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

VALID_ENTITY_TYPES = frozenset({"person", "company"})

BATCH_STATUS_FETCHED = "fetched"


class Detector:
    """Submits chunks for detection and folds the results back into the store."""

    def __init__(self, config: AppConfig, store: Any, chunker: Any, client: Any = None) -> None:
        self._config = config
        self._store = store
        self._chunker = chunker
        self._client = client

    def _get_client(self) -> Any:
        """Return the API client, creating a real one only when first needed.

        Creation is deferred so that offline commands never require a key, and
        so tests can inject a stand-in client.
        """
        if self._client is None:
            require_api_key()
            self._client = anthropic.Anthropic()
        return self._client

    def _build_request(self, chunk: dict, window_text: str) -> Request:
        return Request(
            custom_id=chunk["chunk_id"],
            params=MessageCreateParamsNonStreaming(
                model=self._config.anthropic.model,
                max_tokens=self._config.anthropic.max_tokens,
                system=DETECTION_SYSTEM_PROMPT,
                output_config={
                    "effort": self._config.anthropic.effort,
                    "format": {"type": "json_schema", "schema": ENTITY_SCHEMA},
                },
                messages=[{"role": "user", "content": window_text}],
            ),
        )

    def submit(self) -> str | None:
        """Send every not-yet-submitted chunk as one shuffled batch.

        Returns the batch id, or None when there is nothing to submit.
        """
        chunks = self._store.get_unsubmitted_chunks()
        if not chunks:
            logger.info("No unsubmitted chunks found")
            return None

        # Fetch each document's ordered texts once; the window is built from
        # them but only the window itself leaves this process.
        document_texts: dict[str, list[str]] = {}
        requests: list[Request] = []
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in document_texts:
                document_texts[doc_id] = self._store.get_document_chunk_texts(doc_id)
            window_text = self._chunker.window(
                document_texts[doc_id],
                chunk["seq"],
                self._config.chunking.detection_margin_tokens,
            )
            requests.append(self._build_request(chunk, window_text))

        # Order must not correlate with document identity or chunk sequence.
        random.shuffle(requests)

        client = self._get_client()
        batch = client.messages.batches.create(requests=requests)
        self._store.record_batch(batch.id, batch.processing_status)
        self._store.mark_chunks_submitted([chunk["chunk_id"] for chunk in chunks], batch.id)
        logger.info("Submitted %d chunks as batch %s", len(requests), batch.id)
        return batch.id

    def check(self, batch_id: str) -> str:
        """Refresh and return the processing status of a batch."""
        client = self._get_client()
        batch = client.messages.batches.retrieve(batch_id)
        self._store.set_batch_status(batch_id, batch.processing_status)
        return batch.processing_status

    def fetch(self, batch_id: str) -> dict:
        """Collect batch results and store the detected entities per chunk."""
        client = self._get_client()
        counts = {"succeeded": 0, "errored": 0, "refused": 0, "entities": 0}

        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            result_type = result.result.type

            if result_type != "succeeded":
                counts["errored"] += 1
                logger.warning("Chunk %s result was %s", custom_id, result_type)
                continue

            message = result.result.message
            if getattr(message, "stop_reason", None) == "refusal":
                # A safety classifier declined; the output need not match the
                # schema, so there is nothing to parse.
                counts["refused"] += 1
                logger.warning("Chunk %s was refused by the model", custom_id)
                continue

            try:
                entities = _parse_entities(message)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                counts["errored"] += 1
                logger.warning("Chunk %s returned unparsable output: %s", custom_id, exc)
                continue

            self._store.add_entities(custom_id, entities)
            counts["succeeded"] += 1
            counts["entities"] += len(entities)

        self._store.set_batch_status(batch_id, BATCH_STATUS_FETCHED)
        logger.info("Fetched batch %s: %s", batch_id, counts)
        return counts


def _parse_entities(message: Any) -> list[tuple[str, str]]:
    """Extract (text, type) pairs from a successful detection message.

    Raises ValueError, KeyError or TypeError when the payload does not match
    the requested schema; callers count that as an errored chunk.
    """
    text_block = None
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text_block = block
            break
    if text_block is None:
        raise ValueError("no text content block in message")

    payload = json.loads(text_block.text)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object, got {type(payload).__name__}")

    raw_entities = payload["entities"]
    if not isinstance(raw_entities, list):
        raise TypeError(f"expected an entities list, got {type(raw_entities).__name__}")

    entities: list[tuple[str, str]] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            raise TypeError(f"expected an entity object, got {type(item).__name__}")
        entity_text = item["text"]
        entity_type = item["type"]
        if not isinstance(entity_text, str) or not isinstance(entity_type, str):
            raise TypeError("entity text and type must be strings")
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"unknown entity type {entity_type!r}")
        entities.append((entity_text, entity_type))
    return entities
