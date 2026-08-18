"""Command line interface for the doc_quant pipeline.

Usage: python -m doc_quant.cli <command> [options]

The commands follow the lifecycle of a document: ingest it into chunks, submit
the chunks for detection, fetch the results, then write out redacted copies.
The `chunk get` / `chunk set` pair exists so a single fragment can be handed to
an outside processor and the transformed text put back under the same id.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from doc_quant.chunker import Chunker
from doc_quant.config import ConfigError, load_config
from doc_quant.detector import Detector
from doc_quant.redactor import redact_text
from doc_quant.store import ChunkStore

logger = logging.getLogger(__name__)

MARKDOWN_GLOB = "*.md"
REDACTED_SUFFIX = ".redacted.md"


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
        f"refused={counts['refused']} entities={counts['entities']}"
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


def _dispatch(args: argparse.Namespace) -> int:
    config = load_config()
    store = ChunkStore(config.database.path)
    try:
        chunker = Chunker(config.chunking.encoding, config.chunking.chunk_size_tokens)
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
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
