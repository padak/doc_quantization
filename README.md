# doc_quantization

Decontextualization pipeline for anonymizing Markdown documents. Documents are
split into small fixed-size token chunks that are processed out of context —
by an LLM or by a human — so that no single party ever sees the full document,
while the original can always be reassembled byte-exactly. Outbound traffic is
additionally salted with synthetic fragments (honeytokens, chaff, canaries)
that only the local registry can subtract.

## How it works

```mermaid
flowchart LR
    docs["Markdown documents"] -- "ingest" --> store[("SQLite key-value store<br/>22-token chunks, random IDs<br/>ordering kept local")]
    store -- "real fragments" --> batch["Shuffled batch"]
    synth["Synthetic generator<br/>(local LLM + name factory)"] -- "honeytokens · chaff · canaries" --> batch
    batch -- "submit" --> api["Anthropic Batch API"]
    api -- "fetch" --> filter{"Registry<br/>filter"}
    filter -- "real results" --> entities["Detected entities"]
    filter -- "honeytokens" --> recall["Recall stats"]
    filter -- "chaff + canaries" --> discard["Discarded"]
    store -- "reconstruct (byte-exact)" --> redact["Redacted documents"]
    entities --> redact
```

1. **Ingest** — each Markdown file is tokenized (tiktoken) and split into
   consecutive chunks of 22 tokens (configurable). Chunks are stored in a
   SQLite key-value store under random UUIDs. The ordering lives only in a
   local column; a chunk ID leaks neither document identity nor position.
2. **Detect** — chunks are sent shuffled and context-free to the Anthropic
   Message Batches API, mixed with synthetic fragments that are
   indistinguishable from real ones by request shape. Structured outputs
   enforce an exact-substring entity schema:
   `{"entities": [{"text": ..., "type": "person" | "company"}]}`.
3. **Redact** — documents are reassembled byte-exactly from their chunks and
   every detected name is replaced: persons with `**PERSON**`, companies with
   `**COMPANY**`.

Chunking is lossless by construction: token slices are cut via `decode_bytes`
and the boundary is extended whenever a cut would split a multi-byte UTF-8
character, so `"".join(chunks) == original` holds for any input (verified for
Czech diacritics, emoji and CJK).

## Canaries and chaff

Every outbound batch carries three kinds of synthetic fragments. The provider
cannot tell them from real ones; the local registry subtracts them at fetch
time, and a synthetic result can never enter the entities table or the
redacted output (enforced by tests).

| Mechanism | Purpose | Volume |
| --- | --- | --- |
| Honeytokens | Fragments with known fake names; the share the model finds is a live recall measurement per batch and model | ~2% of requests |
| Chaff | Business-prose decoys diluting and poisoning the outbound corpus; a retained copy is worthless without the registry | 1:1 with real fragments |
| Canaries | ~50 globally unique fabricated facts continuously seeded into traffic; later probes test models for knowledge of them | ~5 per batch |

```mermaid
flowchart LR
    reg[("Canary registry<br/>~50 unique fake facts")] -- "seeded into every batch" --> traffic["Outbound traffic"]
    reg -- "canary-probe" --> ask["Ask the model about each fact"]
    ask -- "model knows the fact" --> tripped["TRIPPED<br/>evidence of training misuse"]
    ask -- "model has never heard of it" --> clean["Clean"]
```

Design invariant: fake names and canary facts come from a deterministic
seeded factory and are registered locally **before** anything is sent. The
local LLM only wraps given names in natural prose; its output is validated
verbatim (3 retries, then a deterministic template fallback), so a
hallucination can never poison the registry or the tripwire.

## Installation

Requires **Python 3.9 or newer** (verified on 3.9, 3.11, 3.13 and 3.14).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q          # 234 tests should pass
```

On macOS, `python3` may point at the system interpreter shipped with Xcode,
which comes with a very old pip. If installation fails with
`Could not find a version that satisfies the requirement ...`, upgrade pip
inside the virtual environment and retry:

```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` in your environment for the `submit`, `status`,
`fetch` and `canary-probe` commands. Offline commands work without it.

Generating synthetic fragments requires a running local LLM server with an
OpenAI-compatible endpoint — for example [Ollama](https://ollama.com)
(`ollama serve`, `ollama pull qwen2.5:7b`), LM Studio, or a llama.cpp server.
Point `synthetic.llm.base_url` in `config/config.json` at it. Without one,
commands fail fast with an actionable message.

## Usage

```bash
# 1. Ingest a directory of Markdown files
python -m doc_quant.cli ingest ./documents

# 2. List ingested documents
python -m doc_quant.cli docs

# 3. Submit unprocessed chunks + synthetics (shuffled) to the Batch API
python -m doc_quant.cli submit

# 4. Poll the batch until it has ended, then collect results
python -m doc_quant.cli status <batch_id>
python -m doc_quant.cli fetch <batch_id>

# 5. Write redacted documents
python -m doc_quant.cli redact ./redacted

# Synthetic fragments: recall report and training-misuse probe
python -m doc_quant.cli synthetic-report
python -m doc_quant.cli canary-probe [--model MODEL]
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
| `chunking.name_run_max_extension_tokens` | `12` | Max extra tokens a cut may move so it never splits a capitalized name run |
| `conversion.service_url` | `""` (empty) | Optional external conversion service; empty means the built-in markitdown converter is used |
| `anthropic.model` | `claude-opus-5` | Detection model |
| `anthropic.effort` | `low` | Reasoning effort for detection requests |
| `anthropic.max_tokens` | `1024` | Max output tokens per request |
| `anthropic.detect_concurrency` | `6` | Detection requests kept in flight by the web app's synchronous path |
| `redaction.person` | `**PERSON**` | Placeholder for person names |
| `redaction.company` | `**COMPANY**` | Placeholder for company names |
| `redaction.email` | `**EMAIL**` | Placeholder for email addresses, replaced whole and without the detector |
| `redaction.url` | `**URL**` | Placeholder for URLs, replaced whole and without the detector |
| `synthetic.honeytokens_enabled` | `true` | Mix honeytokens into batches |
| `synthetic.chaff_enabled` | `true` | Mix chaff into batches |
| `synthetic.canaries_enabled` | `true` | Seed canaries into batches |
| `synthetic.chaff_ratio` | `1.0` | Chaff fragments per real fragment |
| `synthetic.honeytoken_rate` | `0.02` | Honeytokens per real fragment |
| `synthetic.canary_set_size` | `50` | Size of the persisted canary set |
| `synthetic.canaries_per_batch` | `5` | Canaries re-seeded into each batch |
| `synthetic.seed` | `20260818` | Seed for the deterministic name factory |
| `synthetic.llm.base_url` | `http://localhost:11434/v1` | OpenAI-compatible local LLM endpoint |
| `synthetic.llm.model` | `qwen2.5:7b` | Local model for synthetic prose |
| `synthetic.llm.catalog` | 3 entries | Local models the Settings view offers, with measured size, seconds per fragment and first-try validity |
| `synthetic.llm.catalog_note` | see config | Where those figures were measured |

## Privacy model, honestly

This pipeline is built to survive an assume-breach argument. The question it
answers is not "do we trust the provider?" but "what would the provider hold
if it ignored every contract and kept everything we ever sent?".

### Worst case: the provider retains all traffic

Even then, it holds:

- **Fragments, not documents.** Isolated snippets of roughly `chunk_size`
  tokens of contiguous text (~22 with the defaults), stripped of filenames,
  titles, authorship and any document identity.
- **A shuffled pool under meaningless IDs.** Chunk ordering lives only in the
  local database and is never transmitted; the random UUIDs encode nothing.
- **A corpus that is half fake.** Chaff is mixed 1:1 and is indistinguishable
  from real fragments by request shape. Only the local registry can subtract
  it — a retained copy is polluted beyond trustworthy use.
- **Fifty tripwires.** Globally unique fabricated facts are seeded into the
  traffic continuously. `canary-probe` later asks models about them; a model
  that "knows" a canary is evidence of training misuse. The no-training
  clause stops being a promise and becomes a testable claim.
- **An audit trail working for you.** Honeytokens turn every batch into a
  measurement: a recall figure per batch and model that you can put in front
  of a regulator or an auditor instead of an assurance.

### What it does not hide

- **Unknown names must travel** — finding them is the job. What never has to
  travel are names you already know (roadmap: a local-first detection pass so
  known entities are pre-redacted before anything leaves the machine).
- **Co-membership.** All fragments arrive from one account, so "these belong
  to the same client" is observable; what they form is not.
- **Reassembly is a statistical fight, not a lookup.** Outbound fragments are
  mutually disjoint — chunk cuts are name-aware, so no overlap seams exist to
  chain them. What remains for an adversary is linguistic stitching of
  22-token crumbs across a shuffled, half-fake pool. Residual risk: names
  that defy capitalization conventions, or capitalized runs longer than the
  extension cap, can still straddle a cut.
- **This is not GDPR anonymization.** Fragments containing real names remain
  personal data. The pipeline delivers data minimization and unlinkability on
  top of your DPA — it complements the contract, it does not replace it.

In one sentence: even in the scenario your lawyers fear most, the adversary
ends up with a shuffled, half-fabricated pile of disjoint 22-token crumbs,
no way to tell real from fake, and fifty landmines that convert cheating
into evidence.

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
