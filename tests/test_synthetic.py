"""Tests for doc_quant.synthetic.

Everything runs offline: the local LLM is either a fake object or an httpx
MockTransport, so no test ever opens a socket.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from doc_quant.config import (
    DEFAULT_CONFIG_PATH,
    DETECTION_PROVIDER_ANTHROPIC,
    AnthropicConfig,
    AppConfig,
    ChunkingConfig,
    ConfigError,
    ConversionConfig,
    DatabaseConfig,
    DetectionConfig,
    LLMCatalogEntry,
    LocalDetectionConfig,
    RedactionConfig,
    SyntheticConfig,
    SyntheticLLMConfig,
    load_config,
)
from doc_quant.store import ChunkStore
from doc_quant.synthetic import (
    CANARY_FACT_MARKER,
    LLM_ATTEMPTS,
    MAX_FRAGMENT_CHARS,
    FakeNameFactory,
    LocalLLMClient,
    LocalLLMError,
    SyntheticFragment,
    SyntheticGenerator,
)

BASE_URL = "http://localhost:11434/v1"
MODEL = "test-model"
SEED = 20260818

STUBBORN_OUTPUT = "The parties agreed to revisit the matter at a later date."

# The offered-model catalog the settings view renders; irrelevant to generation
# itself, so one entry is enough here.
CATALOG = (
    LLMCatalogEntry(
        model=MODEL,
        size="1.0 GB",
        seconds_per_fragment=0.9,
        first_try_validity=0.81,
        note="test entry",
    ),
)
CATALOG_NOTE = "Measured on a test machine."


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def make_config(tmp_path: Path, **synthetic_overrides) -> AppConfig:
    """Build a complete AppConfig with an easily overridden synthetic part."""
    synthetic = SyntheticConfig(
        honeytokens_enabled=True,
        chaff_enabled=True,
        canaries_enabled=True,
        chaff_ratio=1.0,
        honeytoken_rate=0.02,
        canary_set_size=3,
        canaries_per_batch=2,
        seed=SEED,
        llm=SyntheticLLMConfig(
            enabled=True,
            base_url=BASE_URL,
            model=MODEL,
            temperature=0.8,
            timeout_seconds=5.0,
            catalog=CATALOG,
            catalog_note=CATALOG_NOTE,
        ),
    )
    return AppConfig(
        chunking=ChunkingConfig(
            chunk_size_tokens=22,
            encoding="cl100k_base",
            name_run_max_extension_tokens=12,
        ),
        database=DatabaseConfig(path=tmp_path / "chunks.db"),
        conversion=ConversionConfig(service_url=""),
        anthropic=AnthropicConfig(
            model="claude-opus-5", effort="low", max_tokens=1024, detect_concurrency=6
        ),
        # The synthetic pipeline never consults this section, but AppConfig is
        # whole-or-nothing: leaving it out would only prove the dataclass
        # requires it.
        detection=DetectionConfig(
            provider=DETECTION_PROVIDER_ANTHROPIC,
            local=LocalDetectionConfig(
                base_url=BASE_URL,
                model=MODEL,
                timeout_seconds=5.0,
                concurrency=2,
            ),
        ),
        redaction=RedactionConfig(
            person="**PERSON**",
            company="**COMPANY**",
            email="**EMAIL**",
            url="**URL**",
        ),
        synthetic=replace(synthetic, **synthetic_overrides),
    )


class FakeLLM:
    """Stand-in for LocalLLMClient; records calls and never opens a socket."""

    def __init__(self, respond: Callable[[str, int], str]) -> None:
        self._respond = respond
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, seed: int) -> str:
        self.calls.append((prompt, seed))
        return self._respond(prompt, seed)

    @property
    def prompts(self) -> list[str]:
        return [prompt for prompt, _ in self.calls]

    @property
    def seeds(self) -> list[int]:
        return [seed for _, seed in self.calls]


def echo(prompt: str, seed: int) -> str:
    """Cooperative answer.

    Echoing the prompt necessarily reproduces every requested name verbatim,
    because the prompt lists them - so this passes validation without the fake
    having to know what the factory minted.
    """
    return f"Internal note. {prompt}"


def stubborn(prompt: str, seed: int) -> str:
    """Sloppy answer that never contains the requested names."""
    return STUBBORN_OUTPUT


def blank(prompt: str, seed: int) -> str:
    """Whitespace-only answer."""
    return "   \n  "


def overlong(prompt: str, seed: int) -> str:
    """Correct content, far beyond the accepted fragment length."""
    return f"{prompt} " + "padding " * MAX_FRAGMENT_CHARS


@pytest.fixture
def store(tmp_path: Path):
    chunk_store = ChunkStore(tmp_path / "chunks.db")
    yield chunk_store
    chunk_store.close()


# ----------------------------------------------------------------------
# config wiring
# ----------------------------------------------------------------------


def test_load_config_reads_the_synthetic_section() -> None:
    synthetic = load_config().synthetic

    assert synthetic.honeytokens_enabled is True
    assert synthetic.chaff_enabled is True
    assert synthetic.canaries_enabled is True
    assert synthetic.chaff_ratio == pytest.approx(1.0)
    assert synthetic.honeytoken_rate == pytest.approx(0.02)
    assert synthetic.canary_set_size == 50
    assert synthetic.canaries_per_batch == 5
    assert synthetic.seed == 20260818
    assert synthetic.llm.base_url == "http://localhost:11434/v1"
    assert synthetic.llm.model
    assert synthetic.llm.temperature == pytest.approx(0.8)
    assert synthetic.llm.timeout_seconds == pytest.approx(120)


def test_missing_synthetic_section_fails_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "chunking": {
                    "chunk_size_tokens": 22,
                    "encoding": "cl100k_base",
                    "name_run_max_extension_tokens": 12,
                },
                "database": {"path": "data/chunks.db"},
                "conversion": {"service_url": ""},
                "anthropic": {
                    "model": "claude-opus-5",
                    "effort": "low",
                    "max_tokens": 1024,
                },
                "redaction": {
                    "person": "**PERSON**",
                    "company": "**COMPANY**",
                    "email": "**EMAIL**",
                    "url": "**URL**",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="synthetic"):
        load_config(config_path)


def test_missing_synthetic_key_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["synthetic"]["seed"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="synthetic.seed"):
        load_config(config_path)


def test_missing_llm_key_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["synthetic"]["llm"]["base_url"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="synthetic.llm.base_url"):
        load_config(config_path)


def test_conversion_defaults_to_the_built_in_converter() -> None:
    # An empty service_url means the optional external service is not in use.
    assert load_config(DEFAULT_CONFIG_PATH).conversion.service_url == ""


def test_conversion_service_url_is_read_and_trimmed(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["conversion"]["service_url"] = "  http://converter:9000  "
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_config(config_path).conversion.service_url == "http://converter:9000"


def test_missing_conversion_section_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["conversion"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="conversion"):
        load_config(config_path)


def test_missing_conversion_service_url_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["conversion"]["service_url"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="conversion.service_url"):
        load_config(config_path)


def test_non_string_conversion_service_url_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["conversion"]["service_url"] = 9000
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a string"):
        load_config(config_path)


def test_llm_catalog_is_loaded_with_every_measured_field(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    catalog = config.synthetic.llm.catalog
    assert len(catalog) >= 1
    assert config.synthetic.llm.catalog_note
    for entry in catalog:
        assert entry.model and entry.size and entry.note
        assert entry.seconds_per_fragment > 0
        assert 0.0 <= entry.first_try_validity <= 1.0
    # The configured default model is one a user can also pick from the list.
    assert config.synthetic.llm.model in [entry.model for entry in catalog]


def test_catalog_entry_missing_a_field_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["synthetic"]["llm"]["catalog"][1]["first_try_validity"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"synthetic.llm.catalog\[1\].first_try_validity"):
        load_config(config_path)


def test_catalog_that_is_not_a_list_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["synthetic"]["llm"]["catalog"] = {"model": "llama3.2:1b"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a list"):
        load_config(config_path)


def test_missing_catalog_note_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["synthetic"]["llm"]["catalog_note"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="synthetic.llm.catalog_note"):
        load_config(config_path)


def test_non_boolean_switch_fails_fast(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["synthetic"]["chaff_enabled"] = "false"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a boolean"):
        load_config(config_path)


# ----------------------------------------------------------------------
# FakeNameFactory
# ----------------------------------------------------------------------


def _draw_sequence(factory: FakeNameFactory, rounds: int = 20) -> list[str]:
    drawn = []
    for _ in range(rounds):
        drawn.extend([factory.person(), factory.company(), factory.unique_place()])
    return drawn


def test_factory_is_deterministic_for_a_given_seed() -> None:
    assert _draw_sequence(FakeNameFactory(SEED)) == _draw_sequence(
        FakeNameFactory(SEED)
    )


def test_factory_differs_between_seeds() -> None:
    assert _draw_sequence(FakeNameFactory(SEED)) != _draw_sequence(
        FakeNameFactory(SEED + 1)
    )


def test_factory_never_issues_the_same_name_twice() -> None:
    factory = FakeNameFactory(SEED)
    drawn = _draw_sequence(factory, rounds=200)

    assert len(drawn) == len(set(drawn))


def test_factory_names_look_plausible() -> None:
    factory = FakeNameFactory(SEED)

    person = factory.person()
    company = factory.company()
    place = factory.unique_place()

    for name in (person, company, place):
        assert len(name.split(" ")) == 2
        assert name == name.strip()
        assert name[0].isupper()
        assert all(part[0].isupper() for part in name.split(" "))


# ----------------------------------------------------------------------
# LocalLLMClient
# ----------------------------------------------------------------------


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> LocalLLMClient:
    return LocalLLMClient(
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.8,
        timeout_seconds=5.0,
        transport=httpx.MockTransport(handler),
    )


def test_generate_posts_the_openai_payload_and_returns_the_content() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "prose"}}]},
        )

    assert _client(handler).generate("write something", 42) == "prose"
    assert seen["url"] == f"{BASE_URL}/chat/completions"
    assert seen["body"] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "write something"}],
        "temperature": 0.8,
        "seed": 42,
        "stream": False,
    }


def test_base_url_trailing_slash_does_not_double_up() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = LocalLLMClient(
        base_url=f"{BASE_URL}/",
        model=MODEL,
        temperature=0.8,
        timeout_seconds=5.0,
        transport=httpx.MockTransport(handler),
    )
    client.generate("prompt", 1)

    assert seen["url"] == f"{BASE_URL}/chat/completions"


def _assert_actionable(message: str) -> None:
    """The error has to tell the operator what to do, not just what broke."""
    assert "Local LLM server unreachable" in message
    assert BASE_URL in message
    assert "ollama serve" in message
    assert f"ollama pull {MODEL}" in message
    assert "LM Studio" in message
    assert "synthetic.llm.base_url" in message
    assert "config/config.json" in message


def test_connect_error_becomes_an_actionable_local_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LocalLLMError) as excinfo:
        _client(handler).generate("prompt", 1)

    _assert_actionable(str(excinfo.value))


def test_timeout_becomes_an_actionable_local_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(LocalLLMError) as excinfo:
        _client(handler).generate("prompt", 1)

    _assert_actionable(str(excinfo.value))
    assert isinstance(excinfo.value.__cause__, httpx.TimeoutException)


def test_non_200_becomes_an_actionable_local_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(LocalLLMError) as excinfo:
        _client(handler).generate("prompt", 1)

    _assert_actionable(str(excinfo.value))
    assert "HTTP 404" in str(excinfo.value)


def test_unusable_payload_becomes_a_local_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(LocalLLMError):
        _client(handler).generate("prompt", 1)


def test_non_string_content_becomes_a_local_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": 7}}]})

    with pytest.raises(LocalLLMError):
        _client(handler).generate("prompt", 1)


# ----------------------------------------------------------------------
# honeytokens
# ----------------------------------------------------------------------


def test_honeytokens_plant_their_names_verbatim(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    fragments = generator.make_honeytokens(4)

    assert len(fragments) == 4
    for fragment in fragments:
        assert isinstance(fragment, SyntheticFragment)
        assert fragment.kind == "honeytoken"
        assert fragment.fact is None
        assert uuid.UUID(hex=fragment.fragment_id).version == 4
        persons = [name for name, kind in fragment.planted if kind == "person"]
        companies = [name for name, kind in fragment.planted if kind == "company"]
        assert 1 <= len(persons) <= 2
        assert 0 <= len(companies) <= 1
        for name, _ in fragment.planted:
            assert name in fragment.text


def test_honeytoken_names_are_unique_across_fragments(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    fragments = generator.make_honeytokens(25)

    names = [name for fragment in fragments for name, _ in fragment.planted]
    assert len(names) == len(set(names))


def test_honeytokens_are_registered_before_being_returned(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    fragments = generator.make_honeytokens(3)

    stored = store.list_synthetic_fragments(kind="honeytoken")
    assert len(stored) == 3
    by_id = {row["fragment_id"]: row for row in stored}
    for fragment in fragments:
        row = by_id[fragment.fragment_id]
        assert row["text"] == fragment.text
        assert row["planted"] == fragment.planted
        assert row["fact"] is None
        assert row["batch_id"] is None


def test_prompt_asks_for_the_names_the_factory_minted(
    tmp_path: Path, store: ChunkStore
) -> None:
    llm = FakeLLM(echo)
    generator = SyntheticGenerator(make_config(tmp_path), store, llm)

    fragment = generator.make_honeytokens(1)[0]

    assert len(llm.calls) == 1
    for name, _ in fragment.planted:
        assert name in llm.prompts[0]


def test_zero_or_negative_count_generates_nothing(
    tmp_path: Path, store: ChunkStore
) -> None:
    llm = FakeLLM(echo)
    generator = SyntheticGenerator(make_config(tmp_path), store, llm)

    assert generator.make_honeytokens(0) == []
    assert generator.make_chaff(-3) == []
    assert llm.calls == []
    assert store.list_synthetic_fragments() == []


# ----------------------------------------------------------------------
# validation, retries and fallback
# ----------------------------------------------------------------------


def test_sloppy_model_is_retried_then_replaced_by_the_template(
    tmp_path: Path, store: ChunkStore, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLM(stubborn)
    generator = SyntheticGenerator(make_config(tmp_path), store, llm)

    with caplog.at_level(logging.WARNING, logger="doc_quant.synthetic"):
        fragment = generator.make_honeytokens(1)[0]

    assert len(llm.calls) == LLM_ATTEMPTS
    assert len(set(llm.seeds)) == LLM_ATTEMPTS
    assert fragment.text != STUBBORN_OUTPUT
    for name, _ in fragment.planted:
        assert name in fragment.text
    assert 0 < len(fragment.text) <= MAX_FRAGMENT_CHARS
    assert "deterministic template" in caplog.text


def test_fallback_text_is_persisted_not_the_rejected_output(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(stubborn))

    fragment = generator.make_honeytokens(1)[0]

    assert store.get_synthetic_fragment(fragment.fragment_id)["text"] == fragment.text


def test_retry_stops_as_soon_as_the_model_gets_it_right(
    tmp_path: Path, store: ChunkStore
) -> None:
    attempts = {"count": 0}

    def flaky(prompt: str, seed: int) -> str:
        attempts["count"] += 1
        return stubborn(prompt, seed) if attempts["count"] == 1 else echo(prompt, seed)

    llm = FakeLLM(flaky)
    generator = SyntheticGenerator(make_config(tmp_path), store, llm)

    fragment = generator.make_honeytokens(1)[0]

    assert len(llm.calls) == 2
    assert fragment.text.startswith("Internal note.")


def test_blank_output_is_rejected(tmp_path: Path, store: ChunkStore) -> None:
    llm = FakeLLM(blank)
    generator = SyntheticGenerator(make_config(tmp_path), store, llm)

    fragment = generator.make_honeytokens(1)[0]

    assert len(llm.calls) == LLM_ATTEMPTS
    assert fragment.text.strip()


def test_overlong_output_is_rejected(tmp_path: Path, store: ChunkStore) -> None:
    llm = FakeLLM(overlong)
    generator = SyntheticGenerator(make_config(tmp_path), store, llm)

    fragment = generator.make_honeytokens(1)[0]

    assert len(llm.calls) == LLM_ATTEMPTS
    assert len(fragment.text) <= MAX_FRAGMENT_CHARS


def test_llm_seeds_are_derived_from_the_configured_seed(
    tmp_path: Path, store: ChunkStore
) -> None:
    llm = FakeLLM(echo)
    generator = SyntheticGenerator(make_config(tmp_path), store, llm)

    generator.make_honeytokens(3)

    assert llm.seeds == [SEED + 1, SEED + 2, SEED + 3]


def test_generation_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    def run(name: str) -> list[SyntheticFragment]:
        store = ChunkStore(tmp_path / f"{name}.db")
        try:
            generator = SyntheticGenerator(
                make_config(tmp_path), store, FakeLLM(echo)
            )
            return generator.make_honeytokens(5)
        finally:
            store.close()

    first = run("first")
    second = run("second")

    assert [f.planted for f in first] == [f.planted for f in second]
    assert [f.text for f in first] == [f.text for f in second]


def test_unreachable_server_is_not_swallowed(tmp_path: Path, store: ChunkStore) -> None:
    """A dead endpoint is an operator problem, not a sloppy answer."""

    def unreachable(prompt: str, seed: int) -> str:
        raise LocalLLMError("Local LLM server unreachable at " + BASE_URL)

    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(unreachable))

    with pytest.raises(LocalLLMError):
        generator.make_honeytokens(1)


def test_the_llm_client_is_built_lazily(tmp_path: Path, store: ChunkStore) -> None:
    """Nothing is contacted while there is nothing to generate."""
    config = make_config(tmp_path, llm=SyntheticLLMConfig(
        enabled=True,
        base_url="http://127.0.0.1:1/v1",
        model=MODEL,
        temperature=0.8,
        timeout_seconds=0.1,
        catalog=CATALOG,
        catalog_note=CATALOG_NOTE,
    ))
    generator = SyntheticGenerator(config, store)

    assert generator.make_honeytokens(0) == []
    assert generator.make_chaff(0) == []


class _ExplodingLLM:
    """Fails the test if the generator touches the LLM at all."""

    def generate(self, prompt: str, seed: int) -> str:
        raise AssertionError("LLM must not be called when synthetic.llm.enabled is false")


def test_disabled_llm_uses_templates_directly(tmp_path: Path, store: ChunkStore) -> None:
    """With synthetic.llm.enabled=false, templates are used with no LLM contact."""
    config = make_config(tmp_path, llm=SyntheticLLMConfig(
        enabled=False,
        base_url="http://127.0.0.1:1/v1",
        model=MODEL,
        temperature=0.8,
        timeout_seconds=0.1,
        catalog=CATALOG,
        catalog_note=CATALOG_NOTE,
    ))
    generator = SyntheticGenerator(config, store, llm=_ExplodingLLM())

    honeytokens = generator.make_honeytokens(2)
    chaff = generator.make_chaff(2)
    canaries = generator.ensure_canaries()

    for fragment in honeytokens + chaff:
        for name, _kind in fragment.planted:
            assert name in fragment.text
    for canary in canaries:
        assert canary.fact is not None
        assert canary.fact in canary.text
    assert store.get_synthetic_fragment(honeytokens[0].fragment_id) is not None


# ----------------------------------------------------------------------
# chaff
# ----------------------------------------------------------------------


def test_chaff_is_registered_with_its_planted_names(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    fragments = generator.make_chaff(12)

    assert len(fragments) == 12
    stored = {
        row["fragment_id"]: row for row in store.list_synthetic_fragments(kind="chaff")
    }
    assert len(stored) == 12
    for fragment in fragments:
        assert fragment.kind == "chaff"
        assert fragment.fact is None
        assert 0 <= len(fragment.planted) <= 2
        for name, entity_type in fragment.planted:
            assert entity_type in {"person", "company"}
            assert name in fragment.text
        assert stored[fragment.fragment_id]["planted"] == fragment.planted


def test_chaff_covers_both_the_named_and_the_nameless_case(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    fragments = generator.make_chaff(40)

    sizes = {len(fragment.planted) for fragment in fragments}
    assert 0 in sizes
    assert sizes - {0, 1, 2} == set()


def test_chaff_names_do_not_collide_with_honeytoken_names(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    honeytokens = generator.make_honeytokens(10)
    chaff = generator.make_chaff(10)

    honey_names = {name for f in honeytokens for name, _ in f.planted}
    chaff_names = {name for f in chaff for name, _ in f.planted}
    assert honey_names.isdisjoint(chaff_names)


# ----------------------------------------------------------------------
# canaries
# ----------------------------------------------------------------------


def _place_of(fragment: SyntheticFragment) -> str:
    """Extract the nonce a probe would later string-match on."""
    assert fragment.fact is not None
    return fragment.fact.split(CANARY_FACT_MARKER)[1].rstrip(".")


def test_ensure_canaries_creates_the_configured_set(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    canaries = generator.ensure_canaries()

    assert len(canaries) == 3
    for fragment in canaries:
        assert fragment.kind == "canary"
        assert fragment.fact is not None
        assert CANARY_FACT_MARKER in fragment.fact
        assert fragment.fact.endswith(".")
        assert fragment.fact in fragment.text
        person = fragment.planted[0][0]
        assert fragment.planted == [(person, "person")]
        assert fragment.fact.startswith(person)


def test_canary_place_nonces_are_unique(tmp_path: Path, store: ChunkStore) -> None:
    generator = SyntheticGenerator(
        make_config(tmp_path, canary_set_size=15), store, FakeLLM(echo)
    )

    canaries = generator.ensure_canaries()

    places = [_place_of(fragment) for fragment in canaries]
    assert len(places) == len(set(places)) == 15


def test_ensure_canaries_is_idempotent(tmp_path: Path, store: ChunkStore) -> None:
    config = make_config(tmp_path)
    first_llm = FakeLLM(echo)
    first = SyntheticGenerator(config, store, first_llm).ensure_canaries()

    second_llm = FakeLLM(echo)
    second = SyntheticGenerator(config, store, second_llm).ensure_canaries()

    assert second_llm.calls == []
    assert len(store.list_synthetic_fragments(kind="canary")) == 3
    assert [f.fragment_id for f in second] == [f.fragment_id for f in first]
    assert [f.fact for f in second] == [f.fact for f in first]
    assert [f.text for f in second] == [f.text for f in first]
    assert [f.planted for f in second] == [f.planted for f in first]


def test_ensure_canaries_only_creates_the_shortfall(
    tmp_path: Path, store: ChunkStore
) -> None:
    config = make_config(tmp_path)
    SyntheticGenerator(config, store, FakeLLM(echo)).ensure_canaries()

    topped_up_config = replace(
        config, synthetic=replace(config.synthetic, canary_set_size=5)
    )
    llm = FakeLLM(echo)
    canaries = SyntheticGenerator(topped_up_config, store, llm).ensure_canaries()

    assert len(llm.calls) == 2
    assert len(canaries) == 5
    assert len({fragment.fragment_id for fragment in canaries}) == 5


def test_ensure_canaries_does_not_shrink_an_oversized_set(
    tmp_path: Path, store: ChunkStore
) -> None:
    config = make_config(tmp_path, canary_set_size=4)
    SyntheticGenerator(config, store, FakeLLM(echo)).ensure_canaries()

    smaller = replace(config, synthetic=replace(config.synthetic, canary_set_size=2))
    canaries = SyntheticGenerator(smaller, store, FakeLLM(echo)).ensure_canaries()

    assert len(canaries) == 4


def test_canary_fact_is_a_template_and_never_llm_output(
    tmp_path: Path, store: ChunkStore
) -> None:
    """A hallucinated fact would break the tripwire, so the fact is ours."""
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(stubborn))

    canaries = generator.ensure_canaries()

    for fragment in canaries:
        assert STUBBORN_OUTPUT not in fragment.text
        assert CANARY_FACT_MARKER in fragment.fact
        assert fragment.text == fragment.fact


def test_canaries_are_persisted_with_their_fact(
    tmp_path: Path, store: ChunkStore
) -> None:
    generator = SyntheticGenerator(make_config(tmp_path), store, FakeLLM(echo))

    canaries = generator.ensure_canaries()

    for fragment in canaries:
        row = store.get_synthetic_fragment(fragment.fragment_id)
        assert row["kind"] == "canary"
        assert row["fact"] == fragment.fact
        assert row["text"] == fragment.text
        assert row["planted"] == fragment.planted


# ----------------------------------------------------------------------
# switches
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "call"),
    [
        ("honeytokens_enabled", lambda gen: gen.make_honeytokens(3)),
        ("chaff_enabled", lambda gen: gen.make_chaff(3)),
        ("canaries_enabled", lambda gen: gen.ensure_canaries()),
    ],
)
def test_disabled_mechanism_generates_nothing(
    tmp_path: Path, store: ChunkStore, flag: str, call: Callable
) -> None:
    llm = FakeLLM(echo)
    generator = SyntheticGenerator(make_config(tmp_path, **{flag: False}), store, llm)

    assert call(generator) == []
    assert llm.calls == []
    assert store.list_synthetic_fragments() == []
