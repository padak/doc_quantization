# Local Detection Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `detection.provider` switch that routes name detection to a local OpenAI-compatible LLM (Ollama) instead of the Anthropic API, with synthetics/mixing/shuffle disabled in local mode, in both the webapp and the CLI.

**Architecture:** A new `doc_quant/local_detector.py` is the parity core for the local backend (payload builder + answer parsing + verbatim guard); `LocalLLMClient` is extracted to `doc_quant/local_llm.py` and gains a raw `chat_completion` method; the webapp's `/api/detect` and a new CLI `detect` command both consume only `local_detector`. The Anthropic paths stay untouched.

**Tech Stack:** Python (3.9-compatible style of the repo), FastAPI, httpx (+ `httpx.MockTransport` in tests), pytest, vanilla JS frontend.

**Spec:** `docs/superpowers/specs/2026-08-19-local-detection-mode-design.md`

## Global Constraints

- All tunables live in `config/config.json`; loaders fail fast (`_require`, `_require_bool`). No hardcoded tunables; structural constants (retry counts, temperature 0.0) are module constants with a comment, following `doc_quant/synthetic.py` precedent.
- `requirements.txt` must not change (no new dependencies) and must never be `pip freeze`d.
- All tests offline: no network, fake local server via `httpx.MockTransport`, fake Anthropic clients as existing tests do.
- Every store write in parallel flows happens on the generator/calling thread via `as_completed` (invariant 7).
- Local detection must never construct an Anthropic client, a `SyntheticGenerator`, or write synthetic rows.
- Payload parity per provider: webapp-local and CLI-local build requests exclusively through `doc_quant.local_detector`.
- Same detection contract as remote: `DETECTION_SYSTEM_PROMPT`, `ENTITY_SCHEMA`, entity types `person|company`, exact substrings.
- Files, comments, commit messages in English. No "Co-Authored-By" lines in commits.
- Run tests with `.venv/bin/pytest -q` (subset: `.venv/bin/pytest tests/<file> -q`). Frontend syntax check: `node --check webapp/static/app.js`.
- Match the repo's docstring-heavy, "why not what" comment style.

## Execution waves (for parallel dispatch)

- Wave 1 (parallel): Task 1 (config), Task 2 (local_llm extraction), Task 3 (parse refactor)
- Wave 2 (parallel): Task 4 (settings), Task 5 (local_detector)
- Wave 3 (parallel): Task 6 (CLI), Task 7 (webapp backend)
- Wave 4: Task 8 (frontend)
- Wave 5: Task 9 (docs + full verification)

---

### Task 1: Detection config section

**Files:**
- Modify: `config/config.json`
- Modify: `doc_quant/config.py`
- Test: `tests/test_config.py` (follow the existing test style in that file; create alongside existing config tests if the file has another name — search `grep -rl "load_config" tests/`)

**Interfaces:**
- Consumes: existing `_require`, `ConfigError`, `AppConfig`.
- Produces (later tasks rely on these exact names):
  - `doc_quant.config.DETECTION_PROVIDER_ANTHROPIC = "anthropic"`
  - `doc_quant.config.DETECTION_PROVIDER_LOCAL = "local"`
  - `doc_quant.config.VALID_DETECTION_PROVIDERS: frozenset[str]`
  - `LocalDetectionConfig(base_url: str, model: str, timeout_seconds: float, concurrency: int)` (frozen dataclass)
  - `DetectionConfig(provider: str, local: LocalDetectionConfig)` (frozen dataclass)
  - `AppConfig.detection: DetectionConfig`

- [ ] **Step 1: Write failing tests**

Add to the existing config test file:

```python
def test_detection_section_is_required(tmp_path):
    # Copy the shipped config, drop "detection", expect ConfigError.
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.pop("detection")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="detection"):
        load_config(path)


def test_detection_provider_is_validated(tmp_path):
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["detection"]["provider"] = "remote"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="detection.provider"):
        load_config(path)


def test_detection_local_fields_load():
    config = load_config()
    assert config.detection.provider == "anthropic"
    assert config.detection.local.base_url.startswith("http")
    assert config.detection.local.model
    assert config.detection.local.timeout_seconds > 0
    assert config.detection.local.concurrency >= 1
```

- [ ] **Step 2: Run tests, verify they fail** (`AttributeError: detection` / KeyError)

- [ ] **Step 3: Implement**

In `config/config.json`, add after the `"anthropic"` section:

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

In `doc_quant/config.py`:

```python
DETECTION_PROVIDER_ANTHROPIC = "anthropic"
DETECTION_PROVIDER_LOCAL = "local"
VALID_DETECTION_PROVIDERS = frozenset(
    {DETECTION_PROVIDER_ANTHROPIC, DETECTION_PROVIDER_LOCAL}
)


@dataclass(frozen=True)
class LocalDetectionConfig:
    """Local OpenAI-compatible endpoint used when detection.provider is "local".

    Deliberately separate from synthetic.llm: the detection model and the
    prose model may differ, and switching the provider must not silently
    repoint the synthetic pipeline.
    """

    base_url: str
    model: str
    timeout_seconds: float
    concurrency: int


@dataclass(frozen=True)
class DetectionConfig:
    """Which backend detection requests go to."""

    provider: str
    local: LocalDetectionConfig


def _load_detection(raw: dict) -> DetectionConfig:
    """Build the detection section, failing fast on any missing key.

    The provider is validated here rather than where it is first used, so a
    typo cannot select the remote backend by accident at detection time.
    """
    provider = _require(raw, "provider", "detection")
    if provider not in VALID_DETECTION_PROVIDERS:
        raise ConfigError(
            f"Config key detection.provider must be one of "
            f"{sorted(VALID_DETECTION_PROVIDERS)}, got {provider!r}"
        )
    raw_local = _require(raw, "local", "detection")
    if not isinstance(raw_local, dict):
        raise ConfigError(
            f"Config key detection.local must be an object, got {type(raw_local).__name__}"
        )
    local = LocalDetectionConfig(
        base_url=_require(raw_local, "base_url", "detection.local"),
        model=_require(raw_local, "model", "detection.local"),
        timeout_seconds=float(_require(raw_local, "timeout_seconds", "detection.local")),
        concurrency=int(_require(raw_local, "concurrency", "detection.local")),
    )
    return DetectionConfig(provider=provider, local=local)
```

Add `detection: DetectionConfig` to `AppConfig`, `"detection"` to the required-sections loop in `load_config`, and `detection=_load_detection(raw["detection"])` to the `AppConfig(...)` construction.

- [ ] **Step 4: Run the config tests, then the whole suite** — some fixtures may build configs by hand; fix any that now miss the section.

- [ ] **Step 5: Commit** — `feat: add detection provider config section`

---

### Task 2: Extract LocalLLMClient into doc_quant/local_llm.py

**Files:**
- Create: `doc_quant/local_llm.py`
- Modify: `doc_quant/synthetic.py`
- Test: existing synthetic tests must keep passing unchanged; add `tests/test_local_llm.py`

**Interfaces:**
- Produces:
  - `doc_quant.local_llm.LocalLLMError` (moved, unchanged)
  - `doc_quant.local_llm.LocalLLMClient` (moved) with a NEW method
    `chat_completion(self, payload: dict) -> str` — POSTs `payload` verbatim
    to `{base_url}/chat/completions`, returns
    `body["choices"][0]["message"]["content"]`, raises `LocalLLMError` on
    connect/timeout/non-200/unusable payload (same messages as today).
  - `LocalLLMClient.generate(prompt, seed)` — unchanged behaviour, now a thin
    wrapper building its payload and delegating to `chat_completion`.
- `doc_quant.synthetic` re-exports `LocalLLMClient` and `LocalLLMError` (import at top, keep names in the module namespace) so `doc_quant/cli.py` and existing tests keep importing from `doc_quant.synthetic`.

- [ ] **Step 1: Write failing tests** (`tests/test_local_llm.py`)

```python
import httpx
import json
import pytest

from doc_quant.local_llm import LocalLLMClient, LocalLLMError


def _client(handler, timeout=5.0):
    return LocalLLMClient(
        base_url="http://fake-llm/v1",
        model="test-model",
        temperature=0.0,
        timeout_seconds=timeout,
        transport=httpx.MockTransport(handler),
    )


def test_chat_completion_posts_payload_verbatim_and_returns_content():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
        )

    payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    assert _client(handler).chat_completion(payload) == "hello"
    assert seen["body"] == payload
    assert seen["url"] == "http://fake-llm/v1/chat/completions"


def test_chat_completion_raises_on_http_error():
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(LocalLLMError, match="HTTP 500"):
        _client(handler).chat_completion({"model": "m", "messages": []})


def test_generate_still_works_through_chat_completion():
    def handler(request):
        body = json.loads(request.content)
        assert body["seed"] == 7
        assert body["stream"] is False
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "prose"}}]}
        )

    assert _client(handler).generate("write", seed=7) == "prose"


def test_synthetic_reexports_stay_importable():
    from doc_quant.synthetic import LocalLLMClient as A, LocalLLMError as B
    assert A is LocalLLMClient
    assert B is LocalLLMError
```

- [ ] **Step 2: Run, verify failure** (`ModuleNotFoundError: doc_quant.local_llm`)

- [ ] **Step 3: Implement**

Create `doc_quant/local_llm.py`: move `LocalLLMError` and `LocalLLMClient` verbatim from `synthetic.py` (module docstring: minimal client for any OpenAI-compatible local server). Restructure the client so the HTTP mechanics live in `chat_completion`:

```python
def chat_completion(self, payload: dict) -> str:
    """POST `payload` to /chat/completions and return the assistant content.

    Raises:
        LocalLLMError: when the server is unreachable, times out, answers
            with a non-200 status, or returns an unusable payload.
    """
    url = f"{self._base_url}/chat/completions"
    try:
        with httpx.Client(
            transport=self._transport, timeout=self._timeout_seconds
        ) as client:
            response = client.post(url, json=payload)
    except httpx.ConnectError as exc:
        raise self._unusable(f"connection refused ({exc})") from exc
    except httpx.TimeoutException as exc:
        raise self._unusable(
            f"no answer within {self._timeout_seconds}s ({exc})"
        ) from exc

    if response.status_code != 200:
        raise self._unusable(f"HTTP {response.status_code}")

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise self._unusable(f"unexpected response payload ({exc})") from exc

    if not isinstance(content, str):
        raise self._unusable(f"expected string content, got {type(content).__name__}")
    return content


def generate(self, prompt: str, seed: int) -> str:
    """Return the assistant message for `prompt`, sampled with `seed`."""
    return self.chat_completion(
        {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "seed": seed,
            "stream": False,
        }
    )
```

In `synthetic.py`: delete the moved code, add `from doc_quant.local_llm import LocalLLMClient, LocalLLMError` (top of imports), keep the module docstring sentence about the local model. Do NOT change any other synthetic behaviour.

- [ ] **Step 4: Run `tests/test_local_llm.py` and the synthetic test files** — all pass, no synthetic test edited.

- [ ] **Step 5: Commit** — `refactor: extract LocalLLMClient into doc_quant/local_llm with raw chat_completion`

---

### Task 3: Shared entity-payload parser in detector.py

**Files:**
- Modify: `doc_quant/detector.py:342-379` (`parse_entities`)
- Test: the existing detector tests keep passing; add payload-level tests next to them (find with `grep -rl "parse_entities" tests/`)

**Interfaces:**
- Produces: `doc_quant.detector.parse_entities_payload(payload: Any) -> list[tuple[str, str]]` — the dict-level validation currently inside `parse_entities` (type checks, `VALID_ENTITY_TYPES`, raises `ValueError`/`KeyError`/`TypeError`).
- `parse_entities(message)` keeps its exact signature and behaviour: extract the text block, `json.loads`, delegate to `parse_entities_payload`.

- [ ] **Step 1: Write failing tests**

```python
def test_parse_entities_payload_accepts_valid_dict():
    payload = {"entities": [{"text": "Jan Novak", "type": "person"}]}
    assert parse_entities_payload(payload) == [("Jan Novak", "person")]


def test_parse_entities_payload_rejects_non_dict():
    with pytest.raises(TypeError):
        parse_entities_payload(["not", "a", "dict"])


def test_parse_entities_payload_rejects_unknown_type():
    with pytest.raises(ValueError):
        parse_entities_payload({"entities": [{"text": "x", "type": "place"}]})
```

- [ ] **Step 2: Run, verify failure** (ImportError)

- [ ] **Step 3: Implement** — split `parse_entities` exactly at the `payload = json.loads(...)` seam:

```python
def parse_entities_payload(payload: Any) -> list[tuple[str, str]]:
    """Extract (text, type) pairs from a decoded detection answer.

    Shared by every transport: the Anthropic paths hand over the parsed JSON
    of a message's text block, the local path the parsed JSON of a chat
    completion. One set of rules, so no two paths can disagree about what a
    provider returned.

    Raises ValueError, KeyError or TypeError when the payload does not match
    the requested schema; callers count that as an errored chunk.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object, got {type(payload).__name__}")
    # ... (move the existing body from parse_entities verbatim)


def parse_entities(message: Any) -> list[tuple[str, str]]:
    """Extract (text, type) pairs from a successful detection message."""
    text_block = None
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text_block = block
            break
    if text_block is None:
        raise ValueError("no text content block in message")
    return parse_entities_payload(json.loads(text_block.text))
```

- [ ] **Step 4: Run detector tests + new tests** — pass.

- [ ] **Step 5: Commit** — `refactor: split parse_entities into shared payload-level parser`

---

### Task 4: Settings overrides for the detection provider

**Files:**
- Modify: `webapp/settings.py`
- Test: the existing settings test file (find with `grep -rl "save_overrides" tests/`)

**Interfaces:**
- Consumes: Task 1's `DetectionConfig`, `VALID_DETECTION_PROVIDERS`.
- Produces: override keys `"detection_provider"`, `"detection_local_base_url"`, `"detection_local_model"` accepted by `load_overrides`/`save_overrides`; `effective_config` patches `config.detection` accordingly.

- [ ] **Step 1: Write failing tests**

```python
def test_detection_provider_override_is_validated(tmp_path):
    path = tmp_path / "settings.json"
    with pytest.raises(ConfigError, match="detection_provider"):
        save_overrides(path, {"detection_provider": "remote"})


def test_detection_provider_override_applies(tmp_path):
    path = tmp_path / "settings.json"
    save_overrides(path, {"detection_provider": "local"})
    overrides = load_overrides(path)
    config = effective_config(load_config(), overrides)
    assert config.detection.provider == "local"


def test_detection_local_endpoint_overrides_apply(tmp_path):
    path = tmp_path / "settings.json"
    save_overrides(
        path,
        {
            "detection_local_base_url": "http://elsewhere:9999/v1",
            "detection_local_model": "llama3.2:1b",
        },
    )
    config = effective_config(load_config(), load_overrides(path))
    assert config.detection.local.base_url == "http://elsewhere:9999/v1"
    assert config.detection.local.model == "llama3.2:1b"


def test_invalid_stored_provider_fails_on_read(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"detection_provider": "remote"}', encoding="utf-8")
    with pytest.raises(ConfigError, match="detection_provider"):
        load_overrides(path)
```

- [ ] **Step 2: Run, verify failures** (unknown settings key)

- [ ] **Step 3: Implement**

In `webapp/settings.py`:

- Import `VALID_DETECTION_PROVIDERS` from `doc_quant.config` and `replace` is already imported.
- Add an enum-validated key mechanism mirroring `NUMERIC_SETTINGS_SPECS` (validated on BOTH read and write):

```python
# String keys whose value must come from a fixed set. Validated on both read
# and write, so a stored invalid value can never be discovered only later.
ENUM_SETTINGS_SPECS: dict[str, frozenset] = {
    "detection_provider": VALID_DETECTION_PROVIDERS,
}
ENUM_SETTINGS_KEYS: tuple[str, ...] = tuple(ENUM_SETTINGS_SPECS)


def _validate_enum(key: str, value: object) -> str:
    allowed = ENUM_SETTINGS_SPECS[key]
    if not isinstance(value, str):
        raise ConfigError(
            f"Settings key {key} must be a string, got {type(value).__name__}"
        )
    if value not in allowed:
        raise ConfigError(
            f"Settings key {key} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value
```

- Add `"detection_local_base_url"`, `"detection_local_model"` as ordinary free-text keys and `*ENUM_SETTINGS_KEYS` to `SETTINGS_KEYS`.
- In `load_overrides`: before the generic string branch, handle `key in ENUM_SETTINGS_KEYS` with `_validate_enum` (skip empty string as "cleared" like other free-text keys — an empty enum value is dropped, not validated).
- In `save_overrides`: in the merge loop, after the numeric branch add `elif key in ENUM_SETTINGS_KEYS: merged[key] = _validate_enum(key, value)` (the `value is None or value == ""` clearing branch above already handles clears; enum keys are clearable-empty).
- In `effective_config`, add:

```python
detection_config = replace(
    config.detection,
    provider=_text(overrides, "detection_provider") or config.detection.provider,
    local=replace(
        config.detection.local,
        base_url=_text(overrides, "detection_local_base_url")
        or config.detection.local.base_url,
        model=_text(overrides, "detection_local_model")
        or config.detection.local.model,
    ),
)
```

and `detection=detection_config` in the final `replace(config, ...)`.

- [ ] **Step 4: Run settings tests** — pass.

- [ ] **Step 5: Commit** — `feat: settings overrides for detection provider and local endpoint`

---

### Task 5: doc_quant/local_detector.py

**Files:**
- Create: `doc_quant/local_detector.py`
- Test: `tests/test_local_detector.py`

**Interfaces:**
- Consumes: Task 1 (`AppConfig.detection`, provider constants), Task 2 (`LocalLLMClient.chat_completion`, `LocalLLMError`), Task 3 (`parse_entities_payload`), `detector.DETECTION_SYSTEM_PROMPT`, `detector.ENTITY_SCHEMA`.
- Produces (Tasks 6 and 7 rely on these exact names):

```python
LOCAL_BATCH_PREFIX = "local-"
LOCAL_BATCH_ID_CHARS = 12
LOCAL_BATCH_STATUS_RUNNING = "sync"           # same lifecycle as webapp sync runs
LOCAL_BATCH_STATUS_COMPLETED = "sync-completed"
STATUS_OK = "ok"
STATUS_ERROR = "error"

@dataclass(frozen=True)
class LocalDetectionOutcome:
    status: str                      # STATUS_OK | STATUS_ERROR
    entities: list                   # list[tuple[str, str]], verbatim-filtered
    raw_text: str | None             # last assistant content, for observability
    latency_ms: int
    detail: str | None               # error detail, or None
    dropped: int                     # entities removed by the verbatim guard

def get_local_client(config: AppConfig, transport=None) -> LocalLLMClient
def build_local_payload_template(config: AppConfig) -> dict
def build_local_request(template: dict, text: str) -> dict
def detect_local(client: LocalLLMClient, template: dict, text: str) -> LocalDetectionOutcome
def probe_local_server(config: AppConfig, transport=None) -> None   # raises LocalLLMError
def new_local_batch_id() -> str
```

- [ ] **Step 1: Write failing tests** (`tests/test_local_detector.py`)

```python
import json

import httpx
import pytest

from doc_quant.config import load_config
from doc_quant.detector import DETECTION_SYSTEM_PROMPT, ENTITY_SCHEMA
from doc_quant.local_detector import (
    LOCAL_BATCH_PREFIX,
    STATUS_ERROR,
    STATUS_OK,
    build_local_payload_template,
    build_local_request,
    detect_local,
    get_local_client,
    new_local_batch_id,
    probe_local_server,
)
from doc_quant.local_llm import LocalLLMError


def _answer(entities):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps({"entities": entities})}}]},
    )


def _client(handler):
    return get_local_client(load_config(), transport=httpx.MockTransport(handler))


def test_template_carries_shared_prompt_schema_and_endpoint():
    template = build_local_payload_template(load_config())
    assert template["system"] == DETECTION_SYSTEM_PROMPT
    assert template["response_format"]["json_schema"]["schema"] == ENTITY_SCHEMA
    assert template["temperature"] == 0.0
    assert "base_url" in template  # reported, stripped from the wire payload


def test_request_strips_reporting_keys_and_builds_messages():
    template = build_local_payload_template(load_config())
    payload = build_local_request(template, "Some fragment")
    assert "base_url" not in payload
    assert "system" not in payload
    assert payload["messages"] == [
        {"role": "system", "content": DETECTION_SYSTEM_PROMPT},
        {"role": "user", "content": "Some fragment"},
    ]


def test_detect_local_parses_entities():
    def handler(request):
        return _answer([{"text": "Jan Novak", "type": "person"}])

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "Jan Novak signed.")
    assert outcome.status == STATUS_OK
    assert outcome.entities == [("Jan Novak", "person")]
    assert outcome.dropped == 0


def test_verbatim_guard_drops_hallucinated_entities():
    def handler(request):
        return _answer(
            [
                {"text": "Jan Novak", "type": "person"},
                {"text": "Elvira Ghost", "type": "person"},
            ]
        )

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "Jan Novak signed.")
    assert outcome.status == STATUS_OK
    assert outcome.entities == [("Jan Novak", "person")]
    assert outcome.dropped == 1


def test_detect_local_retries_invalid_json_then_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "not json"}}]}
        )

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "text")
    assert outcome.status == STATUS_ERROR
    assert calls["n"] == 2  # LOCAL_DETECTION_ATTEMPTS


def test_detect_local_retry_recovers():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "not json"}}]}
            )
        return _answer([])

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "text")
    assert outcome.status == STATUS_OK
    assert calls["n"] == 2


def test_transport_error_is_an_error_outcome_not_an_exception():
    def handler(request):
        raise httpx.ConnectError("refused")

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "text")
    assert outcome.status == STATUS_ERROR
    assert "unreachable" in outcome.detail


def test_probe_raises_actionable_error_when_server_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LocalLLMError, match="unreachable"):
        probe_local_server(load_config(), transport=httpx.MockTransport(handler))


def test_probe_passes_when_models_endpoint_answers():
    def handler(request):
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": []})

    probe_local_server(load_config(), transport=httpx.MockTransport(handler))


def test_batch_id_shape():
    batch_id = new_local_batch_id()
    assert batch_id.startswith(LOCAL_BATCH_PREFIX)
    assert len(batch_id) == len(LOCAL_BATCH_PREFIX) + 12
```

- [ ] **Step 2: Run, verify failure** (ModuleNotFoundError)

- [ ] **Step 3: Implement** `doc_quant/local_detector.py`:

```python
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
from doc_quant.detector import DETECTION_SYSTEM_PROMPT, ENTITY_SCHEMA, parse_entities_payload
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


def get_local_client(config: AppConfig, transport: httpx.BaseTransport | None = None) -> LocalLLMClient:
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
    payload = {key: value for key, value in template.items() if key not in _REPORTING_ONLY_KEYS}
    payload["messages"] = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": text},
    ]
    return payload


def detect_local(client: LocalLLMClient, template: dict, text: str) -> LocalDetectionOutcome:
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
    kept = [(entity_text, entity_type) for entity_text, entity_type in entities if entity_text in text]
    dropped = len(entities) - len(kept)
    if dropped:
        logger.warning("Dropped %d hallucinated entities from a local answer", dropped)
    return kept, dropped


def probe_local_server(config: AppConfig, transport: httpx.BaseTransport | None = None) -> None:
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
        with httpx.Client(transport=transport, timeout=PROBE_TIMEOUT_SECONDS) as client:
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
```

- [ ] **Step 4: Run `tests/test_local_detector.py`** — pass.

- [ ] **Step 5: Commit** — `feat: add local detection parity core (doc_quant/local_detector)`

---

### Task 6: CLI `detect` command and provider guards

**Files:**
- Modify: `doc_quant/cli.py` (parser at `_build_parser`, dispatch at `_dispatch`)
- Test: `tests/test_cli_detect.py` (new; follow the style of existing CLI tests — find them with `grep -rl "doc_quant.cli" tests/`)

**Interfaces:**
- Consumes: Task 5's `probe_local_server`, `get_local_client`, `build_local_payload_template`, `detect_local`, `new_local_batch_id`, `LOCAL_BATCH_STATUS_RUNNING`, `LOCAL_BATCH_STATUS_COMPLETED`, `STATUS_OK`; Task 1's provider constants; `store.get_unsubmitted_chunks()`, `store.record_batch`, `store.mark_chunks_submitted`, `store.set_batch_status`, `store.add_entities`.
- Produces: `python -m doc_quant.cli detect` (synchronous local detection); guard errors.

**Guard semantics (small spec refinement):** `submit` and `fetch` refuse whenever `detection.provider == "local"`; `status <batch_id>` refuses too (it calls the remote API); bare `status` still lists batches — it is offline and local batches should be listable. `detect` refuses when the provider is `"anthropic"`.

- [ ] **Step 1: Write failing tests**

Use a tmp config/store fixture in the style of the existing CLI tests (monkeypatch `load_config` to return a config whose `database.path` is a tmp file and whose `detection.provider` is set per test via `dataclasses.replace`). Test through `cli.main([...])` with `capsys`.

```python
def test_detect_refuses_under_anthropic_provider(anthropic_cli_env, capsys):
    assert cli.main(["detect"]) == 1
    assert "detection.provider" in capsys.readouterr().err


def test_submit_refuses_under_local_provider(local_cli_env, capsys):
    assert cli.main(["submit"]) == 1
    assert "detection.provider is 'local'" in capsys.readouterr().err


def test_fetch_refuses_under_local_provider(local_cli_env, capsys):
    assert cli.main(["fetch", "batch_x"]) == 1
    assert "detection.provider is 'local'" in capsys.readouterr().err


def test_status_with_batch_id_refuses_under_local_provider(local_cli_env, capsys):
    assert cli.main(["status", "batch_x"]) == 1
    assert "detection.provider is 'local'" in capsys.readouterr().err


def test_bare_status_still_lists_under_local_provider(local_cli_env, capsys):
    assert cli.main(["status"]) == 0


def test_detect_runs_locally_and_stores_entities(local_cli_env, capsys):
    # local_cli_env ingests one small document with a known name and wires
    # get_local_client/probe_local_server to a MockTransport answering
    # {"entities": [{"text": "<name>", "type": "person"}]} for every request.
    assert cli.main(["detect"]) == 0
    out = capsys.readouterr().out
    assert "entities=" in out
    # entities landed, chunks marked submitted under a local- batch
    store = local_cli_env.open_store()
    docs = store.list_documents()
    assert store.get_document_entities(docs[0]["doc_id"])
    chunks = store.get_document_chunks(docs[0]["doc_id"])
    assert all(c["batch_id"].startswith("local-") for c in chunks)


def test_detect_with_nothing_to_do(local_cli_env_empty, capsys):
    assert cli.main(["detect"]) == 0
    assert "nothing to detect" in capsys.readouterr().out
```

Mock the transport by monkeypatching `doc_quant.cli.get_local_client` and `doc_quant.cli.probe_local_server`-visible names (import them into `cli.py` as module attributes so tests can patch `cli.get_local_client`).

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Implement**

In `_build_parser`, after the `submit` parser:

```python
subparsers.add_parser(
    "detect",
    help="Detect names locally over unsubmitted chunks (detection.provider=local).",
)
```

New command implementation:

```python
def _cmd_detect(config: AppConfig, store: ChunkStore) -> int:
    """Run local detection over every not-yet-submitted chunk, synchronously.

    Local mode sends nothing off the machine, so there is no mixing, no
    shuffle and no synthetic traffic: the chunks go to the configured local
    endpoint as-is and the entities land in the same table the remote paths
    fill. Workers only send HTTP requests; every store write happens on this
    thread via `as_completed`, because the SQLite connection belongs to the
    thread that opened it.
    """
    probe_local_server(config)

    chunks = store.get_unsubmitted_chunks()
    if not chunks:
        print("nothing to detect")
        return 0

    client = get_local_client(config)
    template = build_local_payload_template(config)

    batch_id = new_local_batch_id()
    store.record_batch(batch_id, LOCAL_BATCH_STATUS_RUNNING)
    store.mark_chunks_submitted([chunk["chunk_id"] for chunk in chunks], batch_id)

    ok = errored = entities_stored = dropped = 0
    workers = max(1, config.detection.local.concurrency)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            pool.submit(detect_local, client, template, chunk["text"]): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            outcome = future.result()
            if outcome.status == LOCAL_STATUS_OK:
                store.add_entities(chunk["chunk_id"], outcome.entities)
                ok += 1
                entities_stored += len(outcome.entities)
                dropped += outcome.dropped
            else:
                errored += 1
                logger.warning("Chunk %s errored: %s", chunk["chunk_id"], outcome.detail)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    store.set_batch_status(batch_id, LOCAL_BATCH_STATUS_COMPLETED)
    print(
        f"Detected batch {batch_id}: chunks={len(chunks)} ok={ok} "
        f"errored={errored} entities={entities_stored} dropped={dropped}"
    )
    return 0
```

Imports (import the functions as names on the `cli` module so tests can monkeypatch them):

```python
from doc_quant.config import DETECTION_PROVIDER_ANTHROPIC, DETECTION_PROVIDER_LOCAL
from doc_quant.local_detector import (
    LOCAL_BATCH_STATUS_COMPLETED,
    LOCAL_BATCH_STATUS_RUNNING,
    STATUS_OK as LOCAL_STATUS_OK,
    build_local_payload_template,
    detect_local,
    get_local_client,
    new_local_batch_id,
    probe_local_server,
)
```

Guards, in `_dispatch` before the corresponding command branches:

```python
def _require_provider(config: AppConfig, wanted: str, command: str) -> None:
    """Refuse a command whose backend the config has switched away from.

    A machine configured local must not send data out by habit, and `detect`
    on an anthropic-configured machine would silently bypass the mixing
    pipeline - both are configuration mistakes worth a hard stop.
    """
    actual = config.detection.provider
    if actual == wanted:
        return
    if actual == DETECTION_PROVIDER_LOCAL:
        raise ConfigError(
            f"detection.provider is 'local'; `{command}` talks to the Anthropic "
            "API. Use `detect`, or switch detection.provider back to 'anthropic'."
        )
    raise ConfigError(
        f"detection.provider is 'anthropic'; `detect` runs locally only. "
        "Use `submit`/`status`/`fetch`, or switch detection.provider to 'local'."
    )
```

Dispatch wiring:

```python
if args.command == "submit":
    _require_provider(config, DETECTION_PROVIDER_ANTHROPIC, "submit")
    return _cmd_submit(detector)
if args.command == "status":
    if args.batch_id:
        _require_provider(config, DETECTION_PROVIDER_ANTHROPIC, "status")
    return _cmd_status(args, store, detector)
if args.command == "fetch":
    _require_provider(config, DETECTION_PROVIDER_ANTHROPIC, "fetch")
    return _cmd_fetch(args, detector)
if args.command == "detect":
    _require_provider(config, DETECTION_PROVIDER_LOCAL, "detect")
    return _cmd_detect(config, store)
```

(`ConfigError` is already caught in `main` and printed to stderr with exit 1; `LocalLLMError` likewise.)

- [ ] **Step 4: Run `tests/test_cli_detect.py` and existing CLI tests** — pass.

- [ ] **Step 5: Commit** — `feat: CLI detect command with provider guards`

---

### Task 7: Webapp local detection branch + settings API

**Files:**
- Modify: `webapp/server.py` (imports; `SettingsUpdate`; `_settings_payload`; `/api/detect`; new `_run_local_detection_stream`; `LocalLLMError` exception handler)
- Test: extend the webapp test file(s) (find with `grep -rl "api/detect" tests/`)

**Interfaces:**
- Consumes: Task 4's settings keys, Task 5's whole surface, Task 1's provider constants.
- Produces:
  - `GET/PUT /api/settings` carry `detection_provider`, `detection_local_base_url`, `detection_local_model` (GET returns effective values; `pipeline_defaults` untouched — these are not pipeline-parameter fields).
  - `POST /api/detect` on a local-provider config streams:
    `phase(planning)` → `submitted` (`composition: {"real": N}`, local `payload_template`, requests all `kind:"real"` in seq order, unshuffled) → one `result` per chunk (fields as the remote path, plus `dropped`) → `done` (`honeytoken_recall: null`, `entities_stored`, same `payload_template`/`requests`).
  - HTTP 503 with the actionable message when the local server is down, before any store write.
  - Module-level factory `get_local_detection_client(config) -> LocalLLMClient` (monkeypatch seam for tests, mirroring `get_anthropic_client`), delegating to `local_detector.get_local_client`. Also import `probe_local_server` as a module attribute so tests can patch `server.probe_local_server`.

- [ ] **Step 1: Write failing tests**

Follow the existing webapp-test fixture pattern (TestClient + monkeypatched factories + tmp store). Key tests:

```python
def test_settings_carry_detection_fields(client):
    payload = client.get("/api/settings").json()
    assert payload["detection_provider"] == "anthropic"
    assert payload["detection_local_base_url"]
    assert payload["detection_local_model"]


def test_settings_accept_provider_update(client):
    answer = client.put("/api/settings", json={"detection_provider": "local"})
    assert answer.status_code == 200
    assert answer.json()["detection_provider"] == "local"


def test_settings_reject_unknown_provider(client):
    answer = client.put("/api/settings", json={"detection_provider": "remote"})
    assert answer.status_code == 400


def test_local_detect_streams_and_stores(local_mode_client, ingested_doc):
    # local_mode_client: provider=local settings override; fake transport
    # returning one person entity that appears in the document text.
    with local_mode_client.stream(
        "POST", "/api/detect", json={"doc_id": ingested_doc}
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]
    types = [event["type"] for event in events]
    assert types[0] == "phase"
    assert "synthetic" not in types
    submitted = next(e for e in events if e["type"] == "submitted")
    assert submitted["composition"] == {"real": len(submitted["requests"])}
    assert all(item["kind"] == "real" for item in submitted["requests"])
    assert submitted["batch_id"].startswith("local-")
    assert submitted["payload_template"]["base_url"]
    done = events[-1]
    assert done["type"] == "done"
    assert done["honeytoken_recall"] is None
    assert done["entities_stored"] >= 1


def test_local_detect_needs_no_api_key(local_mode_client_without_key, ingested_doc):
    # get_anthropic_client is monkeypatched to raise if called.
    with local_mode_client_without_key.stream(
        "POST", "/api/detect", json={"doc_id": ingested_doc}
    ) as response:
        assert response.status_code == 200


def test_local_detect_makes_no_synthetics(local_mode_client, ingested_doc, opened_store):
    # SyntheticGenerator factory monkeypatched to raise if constructed.
    ...  # run detect, then:
    assert opened_store.list_synthetic_fragments() == []


def test_local_detect_503_when_server_down(local_mode_client_down, ingested_doc, opened_store):
    answer = local_mode_client_down.post("/api/detect", json={"doc_id": ingested_doc})
    assert answer.status_code == 503
    chunks = opened_store.get_document_chunks(ingested_doc)
    assert all(chunk["batch_id"] is None for chunk in chunks)


def test_stored_run_view_renders_local_batch(local_mode_client, ingested_doc):
    # after a local detect run, the document detail endpoint reports the run
    answer = local_mode_client.get(f"/api/documents/{ingested_doc}")
    assert answer.status_code == 200
    assert answer.json()["has_detection"] is True
```

Payload-parity test (place in `tests/test_local_detector.py` or the webapp file):

```python
def test_webapp_and_cli_local_payloads_are_identical():
    config = load_config()
    template = build_local_payload_template(config)
    # Both transports call build_local_request(template, text); assert the
    # webapp reports exactly this template and sends exactly this payload.
    assert build_local_request(template, "abc") == build_local_request(template, "abc")
```

(The real parity guarantee is structural — one builder — so the test asserts the webapp passes `template` from `build_local_payload_template` into both the `submitted` event and the workers; assert via the `submitted` event's `payload_template` equaling `build_local_payload_template(effective_config)`.)

- [ ] **Step 2: Run, verify failures**

- [ ] **Step 3: Implement**

Imports in `server.py`:

```python
from doc_quant.config import DETECTION_PROVIDER_LOCAL
from doc_quant.local_detector import (
    LOCAL_BATCH_STATUS_COMPLETED,
    LOCAL_BATCH_STATUS_RUNNING,
    build_local_payload_template,
    detect_local,
    get_local_client,
    new_local_batch_id,
    probe_local_server,
)
from doc_quant.local_llm import LocalLLMError
```

Constant: `HTTP_SERVICE_UNAVAILABLE = 503`.

Exception handler (module level, next to the ConfigError handler):

```python
@app.exception_handler(LocalLLMError)
async def local_llm_error_handler(request: Request, exc: LocalLLMError) -> JSONResponse:
    """Report an unreachable local model server as a 503 carrying its message."""
    logger.warning("Local LLM error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=HTTP_SERVICE_UNAVAILABLE, content={"detail": str(exc)})
```

Factory:

```python
def get_local_detection_client(config: AppConfig) -> Any:
    """Build the client for the local detection endpoint."""
    return get_local_client(config)
```

`/api/detect` branch — after the existing `_require_document`/no-unsubmitted checks, before the Anthropic client is built:

```python
if context.config.detection.provider == DETECTION_PROVIDER_LOCAL:
    # Before anything is marked as submitted: a dead local server must
    # leave the document exactly as it was.
    probe_local_server(context.config)
    local_client = get_local_detection_client(context.config)
    return StreamingResponse(
        _local_detect_events(context.config, store, local_client, chunks),
        media_type=NDJSON_MEDIA_TYPE,
    )
```

`_local_detect_events` mirrors `_detect_events` (same try/except yielding `EVENT_ERROR`). `_run_local_detection_stream`:

```python
def _run_local_detection_stream(
    config: AppConfig,
    store: ChunkStore,
    client: Any,
    chunks: list[dict],
) -> Iterator[str]:
    """Submit chunks to the local model, emitting one event per finished step.

    No synthetics, no shuffle: nothing leaves the machine, so there is no
    provider to hide the document's shape from, and seq order is the more
    watchable one. The event vocabulary is the remote sync path's, so the
    frontend renders both runs with one code path.
    """
    yield _ndjson(_phase_event(PHASE_PLANNING, f"{len(chunks)} real fragments, local model"))

    planned = [
        {
            "custom_id": chunk["chunk_id"],
            "kind": KIND_REAL,
            "seq": chunk["seq"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    batch_id = new_local_batch_id()
    store.record_batch(batch_id, LOCAL_BATCH_STATUS_RUNNING)
    store.mark_chunks_submitted([chunk["chunk_id"] for chunk in chunks], batch_id)

    template = build_local_payload_template(config)
    yield _ndjson(
        {
            "type": EVENT_SUBMITTED,
            "batch_id": batch_id,
            "composition": {KIND_REAL: len(chunks)},
            "payload_template": template,
            "requests": planned,
        }
    )

    results: list[dict] = []
    entities_stored = 0
    workers = max(1, config.detection.local.concurrency)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            pool.submit(detect_local, client, template, item["text"]): item
            for item in planned
        }
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
                "dropped": outcome.dropped,
            }
            results.append(result)
            yield _ndjson(
                {"type": EVENT_RESULT, "index": index, "total": len(planned), **result}
            )
            if outcome.status == STATUS_OK:
                store.add_entities(item["custom_id"], outcome.entities)
                entities_stored += len(outcome.entities)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    store.set_batch_status(batch_id, LOCAL_BATCH_STATUS_COMPLETED)
    logger.info("Local batch %s finished: %d chunks", batch_id, len(chunks))

    yield _ndjson(
        {
            "type": EVENT_DONE,
            "batch_id": batch_id,
            "composition": {KIND_REAL: len(chunks)},
            "payload_template": template,
            "requests": planned,
            "results": results,
            "honeytoken_recall": None,
            "entities_stored": entities_stored,
        }
    )
```

(NOTE: `local_detector.STATUS_OK` equals the webapp's `STATUS_OK` string `"ok"` — assert this once with a module-level `assert` or reuse the webapp constant in the comparison.)

Settings: add the three fields to `SettingsUpdate` (all `str | None = None`) and to `_settings_payload`:

```python
"detection_provider": config.detection.provider,
"detection_local_base_url": config.detection.local.base_url,
"detection_local_model": config.detection.local.model,
```

Stored-run view: read the reconstruction helper around `webapp/server.py:576-720` and verify nothing filters on the `sync-` prefix; if anything does, treat `local-` batches identically. The test `test_stored_run_view_renders_local_batch` pins this.

- [ ] **Step 4: Run the webapp tests** — pass.

- [ ] **Step 5: Commit** — `feat: local detection branch in webapp detect endpoint and settings API`

---

### Task 8: Frontend — provider settings and local run rendering

**Files:**
- Modify: `webapp/static/app.js`, `webapp/static/index.html`
- Test: `node --check webapp/static/app.js`; manual verification via the browser preview

**Interfaces:**
- Consumes: Task 7's settings payload fields and the local detect event stream.

- [ ] **Step 1: Read the settings panel and detect-view code in `app.js`/`index.html`** to match existing patterns (how `llm_base_url`/`llm_model`/`model` fields are bound and saved).

- [ ] **Step 2: Implement settings UI**

- Add a "Detection" group to the settings panel: a `<select>` with options `anthropic` / `local` bound to `detection_provider`, and two text inputs bound to `detection_local_base_url` and `detection_local_model`, following exactly the binding/save pattern of the existing `llm_*` fields (placeholders from GET payload, PUT on save, empty string clears).
- Show a short hint under the provider select: local mode sends nothing to Anthropic and disables synthetic mixing for detection runs.

- [ ] **Step 3: Implement detect-view tolerance**

- Verify the detect view renders a run whose `composition` lacks synthetic kinds, whose `honeytoken_recall` is `null`, and whose `payload_template` has local keys (`base_url`, `response_format`) instead of Anthropic keys (`system`/`output_config`); the template is rendered generically (key/value or JSON dump) — adjust only if the current code hardcodes Anthropic keys.
- If a result row carries `dropped > 0`, show it (e.g. in the detail column: `dropped N hallucinated`).

- [ ] **Step 4: Verify**

- `node --check webapp/static/app.js`
- Start the dev server (launch.json name `webapp`, port 8801) against a THROWAWAY database (`data/chunks.db` must not be touched — point `database.path` at a scratch copy or move the real one aside and restore). Ingest a fictional "Project Meridian" memo, switch provider to local in Settings, run detection with Ollama up if available; otherwise verify the 503 message renders cleanly. Verify the anthropic mode settings round-trip unchanged.

- [ ] **Step 5: Commit** — `feat: detection provider settings and local run rendering in webapp UI`

---

### Task 9: Documentation and full verification

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Run the whole suite** — `.venv/bin/pytest -q`; note the new test count.

- [ ] **Step 2: README** — document the local detection mode: the `detection.provider` switch, the CLI `detect` command, the guard behaviour, what local mode deliberately drops (mixing/synthetics) and why (nothing leaves the machine), Ollama/LM Studio compatibility note. Update the test count.

- [ ] **Step 3: CLAUDE.md** — add `local_llm.py` + `local_detector.py` rows to the architecture table; restate invariant 5 as per-provider parity (webapp-local and CLI-local share `local_detector`); note local mode disables synthetics by design and that `submit`/`fetch`/`status <id>` are provider-guarded.

- [ ] **Step 4: Commit** — `docs: document local detection mode`

---

## Self-review notes

- Spec coverage: config (T1), settings (T4), client extraction (T2), parser share (T3), parity core + verbatim guard + probe (T5), CLI + guards (T6), webapp branch + 503 + stored-run view (T7), frontend (T8), docs (T9). The spec's "status refuses" is refined in T6 (bare `status` stays offline-listable) — flagged there explicitly.
- Type consistency: `LocalDetectionOutcome` fields match what T6/T7 read (`status/entities/raw_text/latency_ms/detail/dropped`); provider constants come from `doc_quant.config` everywhere; batch statuses come from `local_detector` in both transports.
