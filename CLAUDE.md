# CLAUDE.md — project memory for doc_quantization

Decontextualization pipeline for anonymizing documents: chunk under random
IDs, mix with synthetic decoys, detect names via LLM, reassemble redacted.
The threat model is **unlinkability**, not secrecy: it is fine for a provider
to see a name alone or content alone — never the join of identity and
content, and never enough to reassemble a document. The product's real
audience is the customer's lawyers: the design must survive an assume-breach
argument ("what would the provider hold if it ignored every contract?").

Companion repo: [doc_converter](https://github.com/padak/doc_converter)
(AGPL-3.0, PyMuPDF) — document conversion behind a generic HTTP contract.

## Architecture

| Piece | Role |
| --- | --- |
| `doc_quant/chunker.py` | tiktoken chunking (~22 tokens), lossless, name-aware cuts |
| `doc_quant/store.py` | SQLite KV: documents, chunks, batches, entities, synthetic registry, probes |
| `doc_quant/synthetic.py` | honeytokens / chaff / canaries; name factory + prose via `local_llm` |
| `doc_quant/local_llm.py` | `LocalLLMClient` for OpenAI-compatible servers (Ollama, LM Studio); raw `chat_completion`. Lived in `synthetic.py`, still re-exported there |
| `doc_quant/detector.py` | Anthropic **Message Batches** path (CLI); shared `plan_synthetic_fragments`, `parse_entities`, `DETECTION_SYSTEM_PROMPT`, `ENTITY_SCHEMA` |
| `doc_quant/local_detector.py` | local detection parity core: same prompt + schema as remote, OpenAI-style `response_format` json_schema, verbatim guard, `/models` probe before any store write, `local-` batch ids with `sync` / `sync-completed` |
| `doc_quant/redactor.py` | deterministic replacement: URLs → emails → entity alternation |
| `doc_quant/cli.py` | ingest / chunk get+set / reconstruct / submit / status / fetch / detect / redact / synthetic-report / canary-probe |
| `webapp/server.py` + `webapp/static/` | FastAPI observability console (port 8801); **synchronous** per-fragment detection branching on `detection.provider`, payloads identical to the CLI path of the same provider |
| `webapp/settings.py` | overrides on top of config, persisted in gitignored `data/settings.json` |

## Invariants — do not break these

1. **Lossless chunking.** `"".join(chunks) == original` for any input. Cuts
   go through `decode_bytes` with UTF-8 boundary extension. Token *display*
   uses byte-accumulating segments (`token_display_segments`) — per-token
   decode corrupts multi-byte chars ("Š" → ��).
2. **No overlaps between outbound fragments.** Detection unit == stored
   chunk. Cuts are name-aware (extend up to `name_run_max_extension_tokens`
   so a capitalized name run is never split). Detection margins/`window()`
   existed once and were **removed deliberately** — overlapping fragments are
   exact-match seams an adversary can chain documents back together with.
   Do not reintroduce.
3. **Ordering never leaves the machine.** `seq` is local; chunk/fragment IDs
   are random `uuid4().hex`; submission order is shuffled.
4. **Synthetic integrity.** Fake names and canary facts come from the
   deterministic seeded factory and are registered in the store **before**
   anything is sent. The local LLM only wraps given names in prose; output is
   validated verbatim (3 retries → deterministic template fallback). An LLM
   hallucination must never mint a tracked name (it could be a real person).
   Synthetic results must never reach the entities table or redacted output —
   tests enforce this; keep them.
5. **Payload parity between transports, per provider.** Whatever the backend,
   the webapp and the CLI must send byte-identical request payloads. For
   `anthropic`: the webapp's sync detection and the CLI batch path share
   `detector`'s helpers (same system prompt, schema, effort, mixing math).
   For `local`: the webapp's local branch and the CLI `detect` build every
   request and read every answer exclusively through `local_detector` (same
   `DETECTION_SYSTEM_PROMPT` and `ENTITY_SCHEMA` as remote — only the
   transport differs, never the question). Do not fork either.
6. **Deterministic redaction order.** URLs first, then emails, then a single
   longest-first entity alternation with word-boundary lookarounds; `person`
   beats `company` for the same string; entity texts equal to a placeholder's
   inner word are skipped. Emails/URLs never need the LLM.
7. **SQLite thread affinity.** In parallel flows (detect stream, canary
   probe) workers do API calls only; every store write happens on the
   generator/calling thread via `as_completed`. The webapp's per-request
   store is the one sanctioned exception: FastAPI hands a request's
   dependency, endpoint and cleanup to different thread-pool workers
   *sequentially*, so it opens with `allow_cross_thread=True`; everywhere
   else the guard stays on (default) to keep catching real races.
8. **Config discipline.** All tunables in `config/config.json`, fail-fast
   loaders (`_require`, `_require_bool` — a JSON `"false"` string must not
   pass as a bool). User overrides live in `data/settings.json` with
   *presence* semantics (an explicitly stored empty string is a choice, an
   absent key falls back to config). API key: env / `.env` / settings only;
   never echoed (masked `sk-a...wxyz`), never in git.
9. **No converter in the core — and no AGPL, ever.** Markitdown was removed
   on purpose. `.md`/`.txt` pass through; everything else requires the
   conversion service (`conversion.service_url`; empty = text-only mode with
   a clear 422). PyMuPDF is AGPL and lives in doc_converter behind an
   arms-length HTTP boundary precisely so this repo stays clean Apache-2.0.
   The service contract (`POST /convert`, `GET /health`) is deliberately
   generic and replaceable.
10. **Local mode is all-local, and deliberately bare.** With
   `detection.provider = "local"` nothing leaves the machine, so the
   decontextualization apparatus has nothing to do: mixing, honeytokens,
   chaff, canaries and the shuffle are switched off for that run and
   `honeytoken_recall` is reported as `null`, never as a zero. Do not "restore
   parity" by re-enabling them. Chunks stay the detection unit anyway — small
   local models recall names better on short fragments. Local answers pass a
   verbatim guard (an entity that is not a substring of its own fragment is
   dropped and counted) because only Anthropic's structured outputs enforce
   the exact-substring contract hard. The CLI dispatcher refuses the other
   backend's commands: `submit`/`fetch`/`status <id>` under `local`, `detect`
   under `anthropic`; bare `status` only lists stored batches and stays
   offline-usable under both.

## Anthropic API specifics

- Model `claude-opus-5`, `output_config={"effort": ..., "format": json_schema}`,
  detection effort `low` (NER on ~22 tokens needs no deep thinking).
- Structured outputs enforce `{"entities": [{"text", "type": person|company}]}` —
  exact substrings, never model-reported coordinates (LLMs miscount offsets).
- `stop_reason == "refusal"` is counted and skipped, never raised.
- The `fallbacks` parameter is **rejected on the Batches API** — don't add it.
- Message Batches are unavailable on Bedrock/Vertex/Foundry; the hyperscaler
  story (customer's own cloud, multi-model within one platform) is roadmap.

## Dev workflow

- venv at `.venv`; `.venv/bin/pytest -q` (503 tests, all offline — fakes for
  Anthropic, local LLM, conversion service); `node --check webapp/static/app.js`.
- `requirements.txt` holds **direct deps with lower bounds only** — never
  `pip freeze` (breaks Python 3.9 colleagues); verified 3.9–3.14.
- Run: `.venv/bin/uvicorn webapp.server:app --port 8801` (launch.json name
  `webapp`); doc_converter on 8802; Ollama on 11434.
- Local LLM: default `qwen2.5:7b` (best prose, 100% first-try validity);
  `llama3.2:1b` measured ~2× faster at 94% validity; template mode
  (`llm_enabled=false`) is instant and needs no server. Catalog with measured
  stats lives in config (`synthetic.llm.catalog`) — data, not code.
  `detection.local` is a *separate* endpoint/model from `synthetic.llm` on
  purpose: switching the detection backend must not silently change which
  model writes chaff. Its model field is offered from the server's own
  `/models` list (`GET /api/detection/local-models`), not from a catalog.
- Git: PR flow for everything (branch → PR → merge), branches deleted after
  merge. The maintainer merges fast — push *all* companion commits (README,
  docs) before announcing a PR as ready, or they get orphaned on a merged
  branch (happened twice). Keep the README test count current.

## Data sensitivity

- `data/` (gitignored) holds the maintainer's **real sensitive documents**
  (contracts, tax emails). Never commit, never quote in PRs, and **never
  screenshot the UI against the real database** — for README/demo material,
  move `data/chunks.db` aside, use a fictional generated document, restore
  after. Fictional demo content: "Project Meridian" style memos.
- Restarting the server kills in-flight sync detections and strands their
  batch in status `sync` with chunks marked submitted. Recovery:
  `UPDATE chunks SET batch_id=NULL WHERE batch_id='<id>'; UPDATE batches SET
  status='aborted' ...` — then re-run detection.

## Known gaps / roadmap (discussed, not built)

- **Local-first funnel**: gazetteer + local NER ensemble (GLiNER-PII, spaCy
  trf, open-weights LLM) over whole documents; pre-redact known names; send
  only candidate-centered residual spans to the API. Kills the "known names
  travel" gap. Distinct from the shipped `detection.provider = local`, which
  is the all-or-nothing end of the same axis (everything local, nothing sent);
  the funnel is the hybrid in between.
- Bare identifiers in prose (case numbers like "3979-4561-9198") are not
  redacted — neither URL, email, nor name.
- Batch-API mode in the webapp UI (submit/poll/fetch with the same
  observability views).
- Worst-case one-pager generator for lawyers (numbers from honeytoken recall,
  dilution ratio, canary probes).
- Broken PDF text layers (glued words) need OCR — no extractor fixes them.
- Rename proposal pending: **ContextShred** (PyPI + GitHub free as of
  2026-08; "the shredder only you can reassemble").
