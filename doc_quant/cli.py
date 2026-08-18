"""Command line interface for the doc_quant pipeline.

Usage: python -m doc_quant.cli <command> [options]

The commands follow the lifecycle of a document: ingest it into chunks, submit
the chunks for detection, fetch the results, then write out redacted copies.
The `chunk get` / `chunk set` pair exists so a single fragment can be handed to
an outside processor and the transformed text put back under the same id.

Two further commands cover the synthetic fragments that ride along with every
batch: `synthetic-report` reads back what was measured (offline), and
`canary-probe` asks a model about the invented canary people to see whether a
fragment we sent has since been trained on.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import anthropic

from doc_quant.chunker import Chunker
from doc_quant.config import AppConfig, ConfigError, load_config, require_api_key
from doc_quant.detector import KIND_CANARY, Detector
from doc_quant.redactor import redact_text
from doc_quant.store import ChunkStore
from doc_quant.synthetic import CANARY_FACT_MARKER, LocalLLMError

logger = logging.getLogger(__name__)

MARKDOWN_GLOB = "*.md"
REDACTED_SUFFIX = ".redacted.md"

# Entity type used by the detector for people; a canary is probed by its name.
PERSON_ENTITY_TYPE = "person"

CANARY_PROBE_QUESTION = (
    "What can you tell me about {name}? If you have never heard of them, say so plainly."
)

# How much of a probe answer is kept as evidence next to the verdict.
RESPONSE_EXCERPT_CHARS = 200
EXCERPT_REFUSAL = "(refusal)"
EXCERPT_ERROR = "(error)"

# Shortest word that can serve as a canary fact's nonce; see `_fact_nonce`.
MIN_NONCE_WORD_LENGTH = 6

# Runs of letters only: digits and punctuation never make a usable nonce.
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Separator between columns of the synthetic report tables.
COLUMN_GAP = "  "


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m doc_quant.cli",
        description="Chunk Markdown documents, detect names out of context, redact.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log progress details to stderr.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Chunk every Markdown file in a directory.")
    ingest.add_argument("directory", type=Path, help="Directory searched recursively for *.md.")

    subparsers.add_parser("docs", help="List ingested documents.")

    chunk = subparsers.add_parser("chunk", help="Read or replace a single chunk.")
    chunk_subparsers = chunk.add_subparsers(dest="chunk_command", required=True)

    chunk_get = chunk_subparsers.add_parser("get", help="Print a chunk's text to stdout.")
    chunk_get.add_argument("chunk_id")

    chunk_set = chunk_subparsers.add_parser("set", help="Replace a chunk's text.")
    chunk_set.add_argument("chunk_id")
    chunk_set_source = chunk_set.add_mutually_exclusive_group(required=True)
    chunk_set_source.add_argument("--text", help="Replacement text given inline.")
    chunk_set_source.add_argument("--file", type=Path, help="File holding the replacement text.")

    reconstruct = subparsers.add_parser("reconstruct", help="Reassemble a document.")
    reconstruct.add_argument("doc_id")
    reconstruct.add_argument(
        "-o", "--output", type=Path, help="Write to this file instead of stdout."
    )

    subparsers.add_parser("submit", help="Send unsubmitted chunks to the Batches API.")

    status = subparsers.add_parser("status", help="Show batch status.")
    status.add_argument("batch_id", nargs="?", help="Omit to list all known batches.")

    fetch = subparsers.add_parser("fetch", help="Collect the results of a finished batch.")
    fetch.add_argument("batch_id")

    redact = subparsers.add_parser("redact", help="Write redacted copies of every document.")
    redact.add_argument("output_dir", type=Path)

    subparsers.add_parser(
        "synthetic-report",
        help="Show synthetic fragment counts, honeytoken recall and canary probes.",
    )

    canary_probe = subparsers.add_parser(
        "canary-probe",
        help="Ask a model about every canary person and record whether it knows them.",
    )
    canary_probe.add_argument(
        "--model",
        help="Model to probe; defaults to the configured Anthropic model.",
    )

    return parser


def _cmd_ingest(args: argparse.Namespace, store: ChunkStore, chunker: Chunker) -> int:
    directory: Path = args.directory
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 1

    ingested = 0
    for path in sorted(directory.rglob(MARKDOWN_GLOB)):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            logger.info("Skipping empty file %s", path)
            continue
        relative_path = path.relative_to(directory)
        chunk_texts = chunker.chunk(text)
        doc_id = store.add_document(str(relative_path), chunk_texts)
        print(f"{doc_id}  {relative_path}  {len(chunk_texts)} chunks")
        ingested += 1

    if ingested == 0:
        print("No Markdown files ingested.", file=sys.stderr)
    return 0


def _cmd_docs(store: ChunkStore) -> int:
    documents = store.list_documents()
    if not documents:
        print("No documents ingested yet.", file=sys.stderr)
        return 0
    for document in documents:
        print(f"{document['doc_id']}  {document['chunk_count']:>5}  {document['path']}")
    return 0


def _cmd_chunk_get(args: argparse.Namespace, store: ChunkStore) -> int:
    chunk = store.get_chunk(args.chunk_id)
    # Written raw so the fragment can be piped without picking up decoration.
    sys.stdout.write(chunk["text"])
    return 0


def _cmd_chunk_set(args: argparse.Namespace, store: ChunkStore) -> int:
    if args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = args.text
    store.update_chunk_text(args.chunk_id, text)
    print(f"Updated chunk {args.chunk_id} ({len(text)} characters).", file=sys.stderr)
    return 0


def _cmd_reconstruct(args: argparse.Namespace, store: ChunkStore) -> int:
    text = store.reconstruct(args.doc_id)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def _cmd_submit(detector: Detector) -> int:
    batch_id = detector.submit()
    if batch_id is None:
        print("nothing to submit")
        return 0
    print(f"Submitted batch {batch_id}")
    return 0


def _cmd_status(args: argparse.Namespace, store: ChunkStore, detector: Detector) -> int:
    if args.batch_id:
        status = detector.check(args.batch_id)
        print(f"{args.batch_id}  {status}")
        return 0

    batches = store.list_batches()
    if not batches:
        print("No batches submitted yet.", file=sys.stderr)
        return 0
    for batch in batches:
        print(f"{batch['batch_id']}  {batch['created_at']}  {batch['status']}")
    return 0


def _cmd_fetch(args: argparse.Namespace, detector: Detector) -> int:
    counts = detector.fetch(args.batch_id)
    print(
        f"succeeded={counts['succeeded']} errored={counts['errored']} "
        f"refused={counts['refused']} entities={counts['entities']} "
        f"honeytokens_scored={counts['honeytokens_scored']} "
        f"synthetic_discarded={counts['synthetic_discarded']}"
    )
    return 0


def _cmd_redact(args: argparse.Namespace, store: ChunkStore, config) -> int:
    output_dir: Path = args.output_dir
    documents = store.list_documents()
    if not documents:
        print("No documents ingested yet.", file=sys.stderr)
        return 0

    for document in documents:
        doc_id = document["doc_id"]
        text = store.reconstruct(doc_id)
        entities = store.get_document_entities(doc_id)
        redacted = redact_text(
            text,
            entities,
            config.redaction.person,
            config.redaction.company,
        )
        output_path = output_dir / Path(document["path"]).with_suffix(REDACTED_SUFFIX)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(redacted, encoding="utf-8")
        print(output_path)
    return 0


def _format_recall(found_total: int, planted_total: int) -> str:
    """Render a recall figure as a percentage, or 'n/a' when nothing was planted.

    Computed from the totals rather than read from the store's `recall` field
    so that per-batch and aggregate rows come from one formula, and so that a
    batch with nothing planted reads as unmeasured instead of as zero recall.
    """
    if planted_total <= 0:
        return "n/a"
    return f"{found_total / planted_total * 100:.1f}%"


def _single_line(text: str) -> str:
    """Collapse whitespace so an excerpt stays on one output line."""
    return " ".join(text.split())


def _print_table(header: list[str], rows: list[list[str]], right_align_from: int) -> None:
    """Print an indented table whose columns are sized to their contents.

    Columns from `right_align_from` onwards are right aligned, which is where
    the numbers live. Sizing from the data keeps timestamps and model ids from
    running into the next column.
    """
    widths = [
        max(len(header[index]), *(len(row[index]) for row in rows)) if rows else len(header[index])
        for index in range(len(header))
    ]

    def render(cells: list[str]) -> str:
        rendered = [
            cell.rjust(widths[index]) if index >= right_align_from else cell.ljust(widths[index])
            for index, cell in enumerate(cells)
        ]
        return f"  {COLUMN_GAP.join(rendered).rstrip()}"

    print(render(header))
    for row in rows:
        print(render(row))


def _cmd_synthetic_report(store: ChunkStore) -> int:
    """Print what the synthetic fragments have measured so far. Offline only."""
    fragments = store.list_synthetic_fragments()
    print("synthetic fragments")
    if not fragments:
        print("  none generated yet")
    else:
        counts_by_kind: dict[str, int] = {}
        for fragment in fragments:
            kind = fragment["kind"]
            counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
        rows = [[kind, str(counts_by_kind[kind])] for kind in sorted(counts_by_kind)]
        rows.append(["total", str(len(fragments))])
        _print_table(["KIND", "COUNT"], rows, right_align_from=1)

    print()
    print("honeytoken recall by batch")
    stats = store.honeytoken_stats()
    if not stats:
        print("  no honeytoken results yet")
    else:
        scored_sum = 0
        planted_sum = 0
        found_sum = 0
        rows = []
        for stat in stats:
            scored_sum += stat["honeytokens_scored"]
            planted_sum += stat["planted_total"]
            found_sum += stat["found_total"]
            rows.append(
                [
                    stat["batch_id"],
                    str(stat["honeytokens_scored"]),
                    str(stat["planted_total"]),
                    str(stat["found_total"]),
                    _format_recall(stat["found_total"], stat["planted_total"]),
                ]
            )
        # Recomputed from the totals rather than averaged, so that batches with
        # more planted names weigh more.
        rows.append(
            [
                "aggregate",
                str(scored_sum),
                str(planted_sum),
                str(found_sum),
                _format_recall(found_sum, planted_sum),
            ]
        )
        _print_table(["BATCH", "SCORED", "PLANTED", "FOUND", "RECALL"], rows, right_align_from=1)

    print()
    print("canary probes")
    probes = store.list_canary_probes()
    if not probes:
        print("  never probed")
        return 0

    header = ["PROBED AT", "MODEL", "FRAGMENT", "RESULT"]
    rows = [
        [
            probe["probed_at"],
            probe["model"],
            probe["fragment_id"],
            "TRIPPED" if probe["tripped"] else "clean",
        ]
        for probe in probes
    ]
    _print_table(header, rows, right_align_from=len(header))
    return 0


def _first_person_name(planted: list) -> str | None:
    """Return the first planted person name, or None when the fragment has none."""
    for entry in planted:
        text, entity_type = entry[0], entry[1]
        if entity_type == PERSON_ENTITY_TYPE:
            return text
    return None


def _fact_nonce(fact: str, planted: list) -> str | None:
    """Return the distinctive token of a canary fact.

    A canary fact binds an invented person to an invented place. The names are
    the bait and appear in the question we ask, so they can never be evidence;
    the invented place is what only a model that has seen the fragment can
    reproduce.

    Heuristic: drop every word belonging to a planted name, then take the
    longest remaining word of at least MIN_NONCE_WORD_LENGTH characters. Short
    words are skipped because ordinary vocabulary would match by chance.

    Facts written from the generator's template are narrowed to the place
    phrase first (everything past CANARY_FACT_MARKER), because the template's
    own wording - "coordinates", "records" - is long enough to win the
    comparison and would trip on any answer that happened to use it. Returns
    None when nothing distinctive enough to test is left.
    """
    planted_words = {
        word.casefold() for entry in planted for word in WORD_PATTERN.findall(entry[0])
    }

    marker_index = fact.find(CANARY_FACT_MARKER)
    if marker_index >= 0:
        searched = fact[marker_index + len(CANARY_FACT_MARKER) :]
    else:
        searched = fact

    candidates = [
        word
        for word in WORD_PATTERN.findall(searched)
        if len(word) >= MIN_NONCE_WORD_LENGTH and word.casefold() not in planted_words
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _message_text(message: Any) -> str:
    """Join the text blocks of a message into one string."""
    return "\n".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )


def _probe_canary_name(
    config: AppConfig,
    client: Any,
    model: str,
    name: str,
    nonce: str,
) -> tuple[bool, str]:
    """Ask the model about one canary person; return (tripped, excerpt).

    The question is a plain one: no structured output, no hint of what the
    right answer would be. Anything but a clean answer counts as not tripped,
    because absence of evidence is not evidence of training.
    """
    question = CANARY_PROBE_QUESTION.format(name=name)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=config.anthropic.max_tokens,
            output_config={"effort": config.anthropic.effort},
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.APIError as exc:
        logger.warning("Canary probe for %s failed: %s", name, exc)
        return False, EXCERPT_ERROR

    if getattr(message, "stop_reason", None) == "refusal":
        logger.warning("Canary probe for %s was refused by the model", name)
        return False, EXCERPT_REFUSAL

    text = _message_text(message)
    if not text.strip():
        logger.warning("Canary probe for %s returned no text content", name)
        return False, EXCERPT_ERROR

    tripped = nonce.casefold() in text.casefold()
    return tripped, text[:RESPONSE_EXCERPT_CHARS]


def run_canary_probe(
    config: AppConfig,
    store: ChunkStore,
    client: Any,
    model: str | None = None,
) -> list[dict]:
    """Probe every stored canary and record each verdict.

    Returns one result dict per probed canary: fragment_id, model, tripped,
    excerpt, name and nonce. Canaries without a planted person name or without
    a testable fact are skipped with a warning, since their answer could not be
    judged either way.
    """
    model_used = model or config.anthropic.model
    results: list[dict] = []

    for canary in store.list_synthetic_fragments(kind=KIND_CANARY):
        fragment_id = canary["fragment_id"]
        planted = canary["planted"] or []
        fact = canary["fact"]
        name = _first_person_name(planted)
        nonce = _fact_nonce(fact, planted) if fact else None
        if name is None or nonce is None:
            logger.warning(
                "Skipping canary %s: no planted person name or no distinctive fact word",
                fragment_id,
            )
            continue

        tripped, excerpt = _probe_canary_name(config, client, model_used, name, nonce)
        store.record_canary_probe(fragment_id, model_used, tripped, excerpt)
        results.append(
            {
                "fragment_id": fragment_id,
                "model": model_used,
                "tripped": tripped,
                "excerpt": excerpt,
                "name": name,
                "nonce": nonce,
            }
        )
    return results


def _cmd_canary_probe(args: argparse.Namespace, config: AppConfig, store: ChunkStore) -> int:
    require_api_key()
    client = anthropic.Anthropic()
    results = run_canary_probe(config, store, client, model=args.model)
    if not results:
        print("No canaries to probe.", file=sys.stderr)
        return 0

    for result in results:
        verdict = "TRIPPED" if result["tripped"] else "clean"
        print(
            f"{result['fragment_id']}  {result['model']}  {verdict}  "
            f"{_single_line(result['excerpt'])}"
        )
    tripped = sum(1 for result in results if result["tripped"])
    print(f"{tripped}/{len(results)} canaries tripped")
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    config = load_config()
    store = ChunkStore(config.database.path)
    try:
        chunker = Chunker(
            config.chunking.encoding,
            config.chunking.chunk_size_tokens,
            config.chunking.name_run_max_extension_tokens,
        )
        detector = Detector(config, store, chunker)

        if args.command == "ingest":
            return _cmd_ingest(args, store, chunker)
        if args.command == "docs":
            return _cmd_docs(store)
        if args.command == "chunk":
            if args.chunk_command == "get":
                return _cmd_chunk_get(args, store)
            return _cmd_chunk_set(args, store)
        if args.command == "reconstruct":
            return _cmd_reconstruct(args, store)
        if args.command == "submit":
            return _cmd_submit(detector)
        if args.command == "status":
            return _cmd_status(args, store, detector)
        if args.command == "fetch":
            return _cmd_fetch(args, detector)
        if args.command == "redact":
            return _cmd_redact(args, store, config)
        if args.command == "synthetic-report":
            return _cmd_synthetic_report(store)
        if args.command == "canary-probe":
            return _cmd_canary_probe(args, config, store)

        raise ValueError(f"Unhandled command: {args.command}")
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return _dispatch(args)
    except (ConfigError, LocalLLMError) as exc:
        # Both carry a message meant for the operator: a missing setting, or
        # how to bring the local model server up.
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
