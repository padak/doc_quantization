# doc_quantization

Decontextualization pipeline for anonymizing Markdown documents. Documents are
split into small fixed-size token chunks that are processed out of context —
by an LLM or by a human — so that no single party ever sees the full document,
while the original can always be reassembled byte-exactly.

## How it works

```
ingest                     submit / fetch                    redact
Markdown ── tokenize ──> SQLite KV store ── shuffled ──> Anthropic Batch API
             22-token      random chunk IDs   fragments     (name detection)
             chunks        order kept local        |
                                |                  v
                          reconstruct <──── detected entities
                          (byte-exact)             |
                                └──> **PERSON** / **COMPANY** redaction
```

1. **Ingest** — each Markdown file is tokenized (tiktoken) and split into
   consecutive chunks of 22 tokens (configurable). Chunks are stored in a
   SQLite key-value store under random UUIDs. The ordering lives only in a
   local column; a chunk ID leaks neither document identity nor position.
2. **Detect** — chunks are sent shuffled and context-free to the Anthropic
   Message Batches API. Each request carries a single fragment (plus a few
   margin tokens from its neighbors so names split across a chunk boundary
   are still found). Structured outputs enforce an exact-substring entity
   schema: `{"entities": [{"text": ..., "type": "person" | "company"}]}`.
3. **Redact** — documents are reassembled byte-exactly from their chunks and
   every detected name is replaced: persons with `**PERSON**`, companies with
   `**COMPANY**`.

Chunking is lossless by construction: token slices are cut via `decode_bytes`
and the boundary is extended whenever a cut would split a multi-byte UTF-8
character, so `"".join(chunks) == original` holds for any input (verified for
Czech diacritics, emoji and CJK).

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` in your environment for the `submit`, `status` and
`fetch` commands. Offline commands work without it.

## Usage

```bash
# 1. Ingest a directory of Markdown files
python -m doc_quant.cli ingest ./documents

# 2. List ingested documents
python -m doc_quant.cli docs

# 3. Submit unprocessed chunks (shuffled) to the Batch API
python -m doc_quant.cli submit

# 4. Poll the batch until it has ended, then collect results
python -m doc_quant.cli status <batch_id>
python -m doc_quant.cli fetch <batch_id>

# 5. Write redacted documents
python -m doc_quant.cli redact ./redacted
```

### Manual fragment workflow

Any chunk can be handed to a person (for example a translator) who sees only
the isolated fragment, never the document:

```bash
python -m doc_quant.cli chunk get <chunk_id> > fragment.txt
# ... the fragment is transformed externally ...
python -m doc_quant.cli chunk set <chunk_id> --file fragment.txt
python -m doc_quant.cli reconstruct <doc_id> -o document.md
```

## Configuration

All tunables live in `config/config.json`:

| Key | Default | Meaning |
| --- | --- | --- |
| `chunking.chunk_size_tokens` | `22` | Tokens per stored chunk |
| `chunking.encoding` | `cl100k_base` | tiktoken encoding |
| `chunking.detection_margin_tokens` | `8` | Neighbor context sent with each detection request (never stored) |
| `anthropic.model` | `claude-opus-5` | Detection model |
| `anthropic.effort` | `low` | Reasoning effort for detection requests |
| `anthropic.max_tokens` | `1024` | Max output tokens per request |
| `redaction.person` | `**PERSON**` | Placeholder for person names |
| `redaction.company` | `**COMPANY**` | Placeholder for company names |

## Privacy model, honestly

Decontextualization reduces context exposure; it does not eliminate it:

- The provider still reads the names themselves — that is the essence of the
  detection task. What it never receives is the full document, its identity,
  or the chunk ordering.
- All chunks are submitted from one account in one batch, so "these fragments
  belong together" is observable; "in what order and forming what" is not.
- Each detection request exposes at most `chunk_size + 2 * margin` tokens of
  contiguous text (38 tokens with the defaults).

If you need stronger guarantees, run a local NER model instead — nothing
leaves the machine and this pipeline is unnecessary.

## Known limitations

- Replacement is case-sensitive and verbatim: inflected name forms must be
  returned by the model as separate entities, otherwise they remain.
- When different chunks classify the same string as both `person` and
  `company`, `person` wins (safer over-redaction).
- Re-ingesting the same directory creates new documents; there is no
  deduplication.

## Development

```bash
.venv/bin/pytest -q
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
