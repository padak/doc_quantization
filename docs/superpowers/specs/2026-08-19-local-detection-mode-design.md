# Local detection mode — design

Date: 2026-08-19
Status: approved (chat), implementation pending

## Motivation

Today every detection request leaves the machine: chunks go to the Anthropic
API (Batches via CLI, synchronous via webapp), mixed with synthetic decoys so
the provider never holds the join of identity and content. A fully local mode
removes the remote provider entirely: detection runs on the local LLM already
integrated for synthetic prose (Ollama or any OpenAI-compatible server). With
nothing leaving the machine, the entire decontextualization apparatus —
mixing, shuffling, honeytokens, chaff, canaries — becomes unnecessary for
that run.

## Decisions (made with the maintainer)

1. **Granularity: per-chunk.** Local detection runs over the existing ~22
   token chunks, exactly like the remote paths. Small local models have
   better recall on short fragments, and the whole downstream pipeline
   (store, entities table, redactor, observability views) is reused
   unchanged. Local calls are free, so their count does not matter.
2. **Synthetics: all off in local mode.** No honeytokens, no chaff, no
   canaries. The local path never touches `SyntheticGenerator` or the
   synthetic registry, so invariant 4 (synthetic results never reach the
   entities table) holds vacuously.
3. **Surfaces: webapp and CLI.** A `detection.provider` switch selects the
   backend for both. The CLI gains a synchronous `detect` command; the
   Batches commands are guarded (see below).

## Configuration

New required section in `config/config.json`:

```json
"detection": {
  "provider": "anthropic",
  "local": {
    "base_url": "http://localhost:11434/v1",
    "model": "qwen2.5:7b",
    "timeout_seconds": 120,
    "concurrency": 2
  }
}
```

- `provider` must be `"anthropic"` or `"local"`; anything else fails fast at
  load time (`ConfigError`).
- `detection.local` is its own endpoint config, deliberately not shared with
  `synthetic.llm`: the detection model and the prose model may differ. It has
  no `enabled` flag — selecting `provider: "local"` is the switch.
- New dataclasses in `doc_quant/config.py`: `LocalDetectionConfig`
  (base_url, model, timeout_seconds, concurrency) and `DetectionConfig`
  (provider, local), wired into `AppConfig` with fail-fast loaders.
- Detection sampling temperature is a structural constant (0.0) in code, not
  a tunable: detection must be as deterministic as the server allows, and a
  knob whose only correct value is 0 is not a knob.

### Settings overrides (webapp)

New override keys in `webapp/settings.py`, same presence/empty semantics as
existing string keys:

- `detection_provider` — validated against `{"anthropic", "local"}` on both
  read and write (a stored invalid value must never be discovered later);
- `detection_local_base_url`, `detection_local_model` — free-text strings,
  empty clears the override.

`effective_config` patches these onto `config.detection`. The settings UI
(webapp settings panel) gets a provider selector and the two text fields.

## Shared local-LLM client refactor

`LocalLLMClient` and `LocalLLMError` move from `doc_quant/synthetic.py` to a
new `doc_quant/local_llm.py`. `synthetic.py` imports them from there and
re-exports the names so existing imports and tests keep working. The client
gains a low-level `chat_completion(payload: dict) -> str` method that POSTs
the given JSON body to `/chat/completions` and returns the assistant message
content; the existing `generate(prompt, seed)` becomes a thin wrapper that
builds its payload and delegates. Error semantics (`LocalLLMError` with
actionable message) are unchanged and shared.

## New module: `doc_quant/local_detector.py`

The parity core for the local backend — both transports (webapp, CLI) build
requests and read answers exclusively through this module (invariant 5,
per-provider).

- `build_local_payload_template(config) -> dict`: the per-request payload
  minus the fragment text —

  ```python
  {
      "model": config.detection.local.model,
      "temperature": 0.0,
      "stream": False,
      "messages_system": DETECTION_SYSTEM_PROMPT,   # same prompt as remote
      "response_format": {
          "type": "json_schema",
          "json_schema": {
              "name": "entities",
              "strict": True,
              "schema": ENTITY_SCHEMA,               # same schema as remote
          },
      },
  }
  ```

  (Exact dict shape decided in implementation; the invariant is: same system
  prompt, same schema, one builder used by both transports, and the template
  is both what is sent and what the webapp reports, so they cannot drift.)

  The nested `json_schema` shape is what Ollama's and LM Studio's
  OpenAI-compatible endpoints document; llama.cpp documents a flat
  `response_format.schema` and is out of scope for now (validation + retry
  still catches malformed output, but the constrained decoding may not
  engage there).

- `detect_local(client, template, text) -> DetectionOutcome`-style function:
  sends one fragment, measures latency, classifies the answer as
  ok/error. Retries once (`LOCAL_DETECTION_ATTEMPTS = 2`, structural
  constant) when the answer fails JSON or schema validation; a transport
  error (`LocalLLMError`) is a per-fragment error outcome, not a run
  failure.

- **Entity parsing reuses the remote rules.** `detector.parse_entities` is
  refactored: the dict-level validation moves into a shared
  `parse_entities_payload(payload: dict) -> list[tuple[str, str]]`;
  `parse_entities(message)` extracts the text block and delegates. The local
  path parses `json.loads(content)` through the same
  `parse_entities_payload`.

- **Verbatim guard (local-only).** After parsing, every entity whose `text`
  is not a verbatim substring of the fragment is dropped (and counted in a
  `dropped` figure surfaced in results/logs). Anthropic structured outputs
  enforce the schema hard; a small local model can hallucinate an entity
  that appears nowhere in the fragment, and a hallucinated string must not
  pollute the entities table even if the redactor would never match it.

- **Availability probe.** `probe_local_server(config)` issues a cheap
  `GET {base_url}/models` and raises `LocalLLMError` with the standard
  actionable message when the server is unreachable. Both transports call it
  before any store write, mirroring the "missing API key leaves the document
  untouched" rule.

## Webapp changes

`POST /api/detect` resolves the effective provider and branches:

- **anthropic** — existing flow, untouched.
- **local** — no API key requirement, no `SyntheticGenerator`, no synthetic
  phases, no shuffle (nothing leaves the machine; local `seq` order is fine
  and more watchable). The stream emits the same event vocabulary:
  - `phase` (planning),
  - `submitted` with `composition: {"real": N}`, the local
    `payload_template` (includes `base_url` and `model` so the UI shows
    where requests go), and the requests list (all `kind: "real"`),
  - one `result` per chunk as it completes (same fields; `detail` may carry
    the dropped-entity count),
  - `done` with `honeytoken_recall: null`, `entities_stored`.
- Batch bookkeeping: id `local-<12 hex>`, statuses `sync` /
  `sync-completed` reused, chunks marked submitted under it — so the
  stored-run view, redaction flow and the CLAUDE.md recovery note all keep
  working. The stored-run reconstruction must render local batches
  (implementation must check any `sync-` prefix assumptions).
- Concurrency: `ThreadPoolExecutor(max_workers=config.detection.local.concurrency)`;
  workers do HTTP only, every store write stays on the generator thread via
  `as_completed` (invariant 7).
- Frontend (`webapp/static/app.js`, `index.html`): settings panel gains the
  provider selector + local endpoint fields; the detect view renders the
  local payload template and the all-real composition without special
  casing beyond what the events already carry.

## CLI changes

- New `detect` command (synchronous): loads config, requires
  `detection.provider == "local"`, probes the server, takes every
  not-yet-submitted chunk (same selection as `submit`), runs them through
  `local_detector` with the configured concurrency, writes entities on the
  calling thread, records a `local-` batch, prints a summary
  (chunks, ok, errored, entities, dropped).
- **Guards.** With `provider: "local"`, `submit`/`status`/`fetch` refuse to
  run with a clear error ("detection.provider is 'local'; use `detect`, or
  switch the provider back to 'anthropic'") — a machine configured local
  must not send data out by habit. Symmetrically, `detect` refuses under
  `provider: "anthropic"`.

## Error handling

- Local server unreachable: webapp answers an ordinary HTTP error (503)
  before the first byte and before any store write; CLI exits with the
  actionable `LocalLLMError` message. The document stays untouched.
- Mid-run transport errors: per-fragment `error` outcomes; the run
  continues (partial results are observable, matching the remote sync
  path).
- Unparsable/invalid JSON after retry: per-fragment `error` outcome.

## Testing (all offline)

Fake local server via `httpx.MockTransport`, as the synthetic tests already
do. Coverage:

- config: `detection` section required, provider validated, local fields
  fail fast; settings: new keys validated on read and write, effective
  config patching.
- payload parity: the webapp local path and the CLI local path produce
  byte-identical request payloads for the same fragment.
- parsing: `parse_entities_payload` shared behaviour; verbatim guard drops
  hallucinated entities and counts them; retry on invalid JSON, error after
  the last attempt.
- isolation: a local-mode run constructs no Anthropic client and no
  `SyntheticGenerator`; no synthetic rows are written; entities land only
  for real chunks.
- webapp: event sequence for a local run (planning → submitted → results →
  done), 503 on unreachable server with no store writes, stored-run view
  renders a `local-` batch.
- CLI: `detect` end-to-end against the fake transport; all four guard
  combinations.

## Documentation

- README: new mode description, updated test count.
- CLAUDE.md: architecture table row for `local_llm.py`/`local_detector.py`,
  invariant 5 restated as per-provider parity, note that local mode disables
  synthetics by design.

## Out of scope

- Whole-document or windowed local detection (revisit if per-chunk recall
  disappoints).
- llama.cpp `response_format` variant.
- Local mode for the Batches-style deferred workflow (meaningless locally).
- The gazetteer / local NER ensemble funnel from the roadmap (this feature
  is a step toward it, not it).
