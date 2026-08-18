"""Name detection over decontextualized chunks via the Anthropic Batches API.

Every chunk is submitted as an isolated fragment under its random chunk id:
no document identity, no ordering, no neighbour identity. The request list is
shuffled before submission so that even the position of a request inside the
batch carries no information about the document it came from. The only context
a chunk gets is a few margin tokens of adjacent text, which is needed so that a
name split across a chunk boundary can still be recognised.

Outbound batches are additionally mixed with synthetic fragments (see
`doc_quant.synthetic`):

* honeytokens carry known planted names, so the returned detections measure
  recall on fragments whose ground truth we own;
* chaff dilutes the batch, so the share of real customer text in what the
  provider receives drops;
* canaries carry an invented fact bound to an invented name, so a later plain
  question to a model can reveal that the fragment was trained on.

Synthetic requests are built through the exact same code path as real ones -
same model, system prompt, schema and effort - so nothing but the store knows
which id is real. Their results never reach the entity tables.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from doc_quant.config import AppConfig, require_api_key
from doc_quant.synthetic import KIND_CANARY, KIND_CHAFF, KIND_HONEYTOKEN, SyntheticGenerator

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

    def __init__(
        self,
        config: AppConfig,
        store: Any,
        chunker: Any,
        client: Any = None,
        generator: Any = None,
    ) -> None:
        self._config = config
        self._store = store
        self._chunker = chunker
        self._client = client
        self._generator = generator

    def _get_client(self) -> Any:
        """Return the API client, creating a real one only when first needed.

        Creation is deferred so that offline commands never require a key, and
        so tests can inject a stand-in client.
        """
        if self._client is None:
            require_api_key()
            self._client = anthropic.Anthropic()
        return self._client

    def _get_generator(self) -> Any:
        """Return the synthetic fragment generator, building it on first use.

        Construction is deferred because it opens a connection to the local
        model server; a submission with every synthetic feature switched off
        must not need one.
        """
        if self._generator is None:
            self._generator = SyntheticGenerator(self._config, self._store)
        return self._generator

    def _build_request(self, custom_id: str, text: str) -> Request:
        """Build one detection request.

        Real chunks and synthetic fragments both go through here: the request
        shape must not betray which of the two an id stands for.
        """
        return Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=self._config.anthropic.model,
                max_tokens=self._config.anthropic.max_tokens,
                system=DETECTION_SYSTEM_PROMPT,
                output_config={
                    "effort": self._config.anthropic.effort,
                    "format": {"type": "json_schema", "schema": ENTITY_SCHEMA},
                },
                messages=[{"role": "user", "content": text}],
            ),
        )

    def _make_synthetic_fragments(self, real_count: int) -> list[Any]:
        """Generate the synthetic fragments that ride along with `real_count` chunks.

        Every fragment returned here is already registered in the store by the
        generator, so a later result can be recognised as synthetic.
        """
        synthetic_config = self._config.synthetic
        fragments: list[Any] = []

        if synthetic_config.honeytokens_enabled:
            # At least one honeytoken per batch, otherwise small batches would
            # never be measured at all.
            count = max(1, round(real_count * synthetic_config.honeytoken_rate))
            fragments.extend(self._get_generator().make_honeytokens(count))

        if synthetic_config.chaff_enabled:
            count = round(real_count * synthetic_config.chaff_ratio)
            if count > 0:
                fragments.extend(self._get_generator().make_chaff(count))

        if synthetic_config.canaries_enabled:
            # The canary set is persistent; each batch carries a random sample
            # of it so that no single batch exposes the whole set.
            canaries = self._get_generator().ensure_canaries()
            sample_size = min(synthetic_config.canaries_per_batch, len(canaries))
            if sample_size > 0:
                fragments.extend(random.sample(canaries, sample_size))

        return fragments

    def submit(self) -> str | None:
        """Send every not-yet-submitted chunk, mixed with synthetic fragments.

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
            requests.append(self._build_request(chunk["chunk_id"], window_text))

        synthetic_fragments = self._make_synthetic_fragments(len(chunks))
        for fragment in synthetic_fragments:
            requests.append(self._build_request(fragment.fragment_id, fragment.text))

        # Order must not correlate with document identity, chunk sequence, or
        # with a request being real rather than synthetic.
        random.shuffle(requests)

        client = self._get_client()
        batch = client.messages.batches.create(requests=requests)
        self._store.record_batch(batch.id, batch.processing_status)
        self._store.mark_chunks_submitted([chunk["chunk_id"] for chunk in chunks], batch.id)
        if synthetic_fragments:
            self._store.mark_synthetic_submitted(
                [fragment.fragment_id for fragment in synthetic_fragments], batch.id
            )

        by_kind = Counter(fragment.kind for fragment in synthetic_fragments)
        logger.info(
            "Submitted batch %s: %d requests (real=%d honeytokens=%d chaff=%d canaries=%d)",
            batch.id,
            len(requests),
            len(chunks),
            by_kind[KIND_HONEYTOKEN],
            by_kind[KIND_CHAFF],
            by_kind[KIND_CANARY],
        )
        return batch.id

    def check(self, batch_id: str) -> str:
        """Refresh and return the processing status of a batch."""
        client = self._get_client()
        batch = client.messages.batches.retrieve(batch_id)
        self._store.set_batch_status(batch_id, batch.processing_status)
        return batch.processing_status

    def fetch(self, batch_id: str) -> dict:
        """Collect batch results and store the detected entities per chunk.

        Results belonging to synthetic fragments are diverted before anything
        is written: honeytokens are scored for recall, chaff and canary output
        is dropped. The `succeeded` and `entities` counts therefore stay a
        measure of real chunks only.
        """
        client = self._get_client()
        counts = {
            "succeeded": 0,
            "errored": 0,
            "refused": 0,
            "entities": 0,
            "honeytokens_scored": 0,
            "synthetic_discarded": 0,
        }

        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id

            # Ask the store first: an id is real only when it is not synthetic.
            synthetic = self._store.get_synthetic_fragment(custom_id)
            if synthetic is not None:
                self._handle_synthetic_result(batch_id, custom_id, synthetic, result, counts)
                continue

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

    def _handle_synthetic_result(
        self,
        batch_id: str,
        custom_id: str,
        synthetic: dict,
        result: Any,
        counts: dict,
    ) -> None:
        """Score a honeytoken result, or drop a chaff / canary result.

        Neither path may call `add_entities`: a synthetic fragment belongs to
        no document, and its detections must never reach a redaction.
        """
        kind = synthetic["kind"]

        if kind != KIND_HONEYTOKEN:
            # Chaff exists to dilute, a canary to be answered about later; what
            # the model reported about either of them is of no interest.
            counts["synthetic_discarded"] += 1
            logger.debug("Discarded %s result for %s", kind, custom_id)
            return

        if result.result.type != "succeeded":
            counts["errored"] += 1
            logger.warning("Honeytoken %s result was %s", custom_id, result.result.type)
            return

        message = result.result.message
        if getattr(message, "stop_reason", None) == "refusal":
            counts["refused"] += 1
            logger.warning("Honeytoken %s was refused by the model", custom_id)
            return

        try:
            found = _parse_entities(message)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            counts["errored"] += 1
            logger.warning("Honeytoken %s returned unparsable output: %s", custom_id, exc)
            return

        self._store.record_honeytoken_result(custom_id, batch_id, found)
        counts["honeytokens_scored"] += 1


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
