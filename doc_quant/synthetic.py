"""Synthetic fragments mixed into outbound batches: honeytokens, chaff, canaries.

Three mechanisms share one pipeline:

* **honeytokens** carry fake names we already know, so the share of them the
  provider reports back measures detection recall on real chunks;
* **chaff** dilutes the batch with fabricated business prose, so what is
  learned from the stream as a whole is mostly noise;
* **canaries** plant a globally unique fact, so a later probe of a model that
  repeats the fact is evidence the fragment was trained on.

**Design invariant: uniqueness and bookkeeping NEVER depend on the LLM.**
Fake names and unique facts come from a deterministic local factory
(`FakeNameFactory`) and are recorded in the local registry verbatim before
anything is sent anywhere; the LLM only wraps given names in natural prose.
This is because an LLM could hallucinate a real person's name, which would
poison recall measurement and break the tripwire: a "planted" name that
belongs to a real person makes a hit ambiguous, and a canary fact that is not
actually unique makes a trip meaningless. So the LLM output is never parsed to
learn what was planted - it is only validated against what the factory already
decided, and a fragment whose text fails validation falls back to a
deterministic template rather than blocking the pipeline.

The prose comes from a LOCAL model reached over an OpenAI-compatible endpoint
(Ollama, LM Studio, llama.cpp server), configured under `synthetic.llm`.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from doc_quant.config import AppConfig

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from doc_quant.store import ChunkStore

logger = logging.getLogger(__name__)

# Structural constants of the generation protocol. These are not user-facing
# tunables (those live in config/config.json under "synthetic"); they define
# what counts as a usable fragment and how patient we are with a sloppy local
# model before falling back to a deterministic template.
LLM_ATTEMPTS = 3
MAX_FRAGMENT_CHARS = 600
HONEYTOKEN_MIN_PERSONS = 1
HONEYTOKEN_MAX_PERSONS = 2
HONEYTOKEN_MIN_COMPANIES = 0
HONEYTOKEN_MAX_COMPANIES = 1
CHAFF_MIN_NAMES = 0
CHAFF_MAX_NAMES = 2
CHAFF_MIN_WORDS = 30
CHAFF_MAX_WORDS = 50
CHAFF_PERSON_SHARE = 0.5
NAME_DRAW_ATTEMPTS = 64

KIND_HONEYTOKEN = "honeytoken"
KIND_CHAFF = "chaff"
KIND_CANARY = "canary"

TYPE_PERSON = "person"
TYPE_COMPANY = "company"

# The canary fact is a template and never LLM output: probes later look for the
# place nonce by exact string match, so the sentence must be known verbatim.
CANARY_FACT_TEMPLATE = "{person} coordinates the records office in {place}."
CANARY_FACT_MARKER = " coordinates the records office in "


@dataclass(frozen=True)
class SyntheticFragment:
    """One fabricated fragment, ready to be mixed into an outbound batch."""

    fragment_id: str
    kind: str
    text: str
    planted: list[tuple[str, str]]
    fact: str | None


class LocalLLMError(Exception):
    """Raised when the local OpenAI-compatible endpoint cannot be used."""


class LocalLLMClient:
    """Minimal client for an OpenAI-compatible `/chat/completions` endpoint.

    Deliberately tiny: only the one call this module needs, so that any server
    speaking the protocol (Ollama, LM Studio, llama.cpp) works without a
    vendor SDK.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: float,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure the endpoint.

        Args:
            base_url: OpenAI-compatible root, e.g. http://localhost:11434/v1.
            model: model name as the local server knows it.
            temperature: sampling temperature for the prose.
            timeout_seconds: per-request timeout.
            transport: optional httpx transport override; tests use it to stay
                offline, production leaves it at None.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def generate(self, prompt: str, seed: int) -> str:
        """Return the assistant message for `prompt`, sampled with `seed`.

        Raises:
            LocalLLMError: when the server is unreachable, times out, answers
                with a non-200 status, or returns an unusable payload.
        """
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "seed": seed,
            "stream": False,
        }

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
            raise self._unusable(
                f"expected string content, got {type(content).__name__}"
            )
        return content

    def _unusable(self, detail: str) -> LocalLLMError:
        """Build an error that says what to do about it, not just what broke."""
        return LocalLLMError(
            f"Local LLM server unreachable at {self._base_url}: {detail}. "
            f"Start one, e.g. Ollama (`ollama serve`, `ollama pull {self._model}`) "
            "or LM Studio, or change synthetic.llm.base_url in config/config.json."
        )


# Syllable parts. Assembled names read like plausible European surnames and
# company names while being practically collision-free with real entities,
# which is what keeps a honeytoken hit unambiguous.
_FIRST_HEADS = (
    "Torv", "Brann", "Kel", "Marv", "Ostr", "Hald", "Rhen", "Sild",
    "Vald", "Yrsa", "Ebb", "Norv", "Grel", "Ilm", "Tarv", "Wend",
)
_FIRST_TAILS = (
    "ald", "ik", "ora", "und", "isse", "ar", "eldt", "ina",
    "olf", "yn", "ret", "usz", "ovic", "elm", "ida", "orn",
)
_LAST_HEADS = (
    "Grims", "Vorn", "Quell", "Hask", "Aldr", "Ladd", "Brem", "Cast",
    "Dun", "Ferr", "Holm", "Ives", "Kirn", "Loft", "Merr", "Ost",
)
_LAST_TAILS = (
    "bury", "beck", "thorpe", "vane", "wold", "ridge", "stead", "mere",
    "holt", "combe", "shaw", "gard", "ling", "worth", "stone", "hall",
)
_COMPANY_HEADS = (
    "Quell", "Vorn", "Mard", "Alsp", "Threx", "Bram", "Ceral", "Dorn",
    "Ellim", "Fask", "Gravn", "Hulst", "Ivast", "Jorm", "Kelv", "Lorn",
)
_COMPANY_TAILS = (
    "andic", "beck", "ovia", "arde", "isen", "oria", "unt", "esco",
    "avia", "ondo", "iska", "erra", "ulon", "ithe", "arno", "edge",
)
_COMPANY_SUFFIXES = (
    "Systems", "Analytics", "Holdings", "Logistics", "Partners", "Industries",
    "Consulting", "Instruments", "Technologies", "Trading", "Services", "Works",
)
_PLACE_HEADS = (
    "Krelling", "Ashen", "Bram", "Corvet", "Dunmar", "Ellisk", "Farrow", "Gild",
    "Harrow", "Ilder", "Jarns", "Kelder", "Lomand", "Merrow", "Nessel", "Orval",
)
_PLACE_TAILS = (
    "hausen", "wick", "moor", "field", "haven", "bridge", "cross", "gate",
    "mouth", "hollow", "vale", "burn", "ford", "cliff", "marsh", "reach",
)
_PLACE_FEATURES = (
    "Flats", "Hollow", "Reach", "Basin", "Downs", "Bend",
    "Crossing", "Common", "Rise", "Weir", "Landing", "Verge",
)
# Appended only if the plain combinations are exhausted; keeps names
# pronounceable while guaranteeing the uniqueness contract.
_EXTENSION_SYLLABLES = ("son", "sen", "ov", "ing", "by", "dal")


class FakeNameFactory:
    """Deterministic source of fake persons, companies and place nonces.

    The same seed always yields the same sequence, and one instance never
    hands out the same name twice, so a fragment's planted names are known
    exactly and cannot silently collide with each other.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._issued: set[str] = set()

    def person(self) -> str:
        """Return a fresh fake person name, e.g. "Torvald Grimsbury"."""
        return self._unique(
            lambda: (
                f"{self._rng.choice(_FIRST_HEADS)}{self._rng.choice(_FIRST_TAILS)} "
                f"{self._rng.choice(_LAST_HEADS)}{self._rng.choice(_LAST_TAILS)}"
            )
        )

    def company(self) -> str:
        """Return a fresh fake company name, e.g. "Quellandic Systems"."""
        return self._unique(
            lambda: (
                f"{self._rng.choice(_COMPANY_HEADS)}{self._rng.choice(_COMPANY_TAILS)}"
                f" {self._rng.choice(_COMPANY_SUFFIXES)}"
            )
        )

    def unique_place(self) -> str:
        """Return a fresh invented toponym, e.g. "Krellinghausen Flats".

        Used as the canary nonce: probes match on this string, so it has to be
        something no real document would contain.
        """
        return self._unique(
            lambda: (
                f"{self._rng.choice(_PLACE_HEADS)}{self._rng.choice(_PLACE_TAILS)}"
                f" {self._rng.choice(_PLACE_FEATURES)}"
            )
        )

    def _unique(self, build: Callable[[], str]) -> str:
        """Draw from `build` until the result has not been issued before."""
        for _ in range(NAME_DRAW_ATTEMPTS):
            candidate = build()
            if candidate not in self._issued:
                self._issued.add(candidate)
                return candidate

        # The plain combinations look exhausted; grow the name by a syllable
        # at a time. Each pass makes it longer, so it must eventually exceed
        # everything already issued and terminate.
        candidate = build()
        while candidate in self._issued:
            candidate = f"{candidate}{self._rng.choice(_EXTENSION_SYLLABLES)}"
        self._issued.add(candidate)
        logger.debug("Extended a fake name to keep it unique: %s", candidate)
        return candidate


def build_honeytoken_prompt(names: list[str]) -> str:
    """Prompt asking for short business prose around exactly `names`."""
    return (
        "Write 1 to 3 sentences of realistic English business-document prose "
        "(meeting notes, contract, invoice or report style) that mention "
        f"exactly these names verbatim: {_join_names(names)}. "
        "Use no other proper names. Plain text only, no markdown, no lists."
    )


def build_chaff_prompt(names: list[str]) -> str:
    """Prompt asking for a generic filler fragment of business prose."""
    base = (
        f"Write a single fragment of {CHAFF_MIN_WORDS} to {CHAFF_MAX_WORDS} words "
        "of generic English business-document prose (status report, meeting "
        "notes or invoice style). Plain text only, no markdown, no lists."
    )
    if not names:
        return f"{base} Use no proper names at all."
    return f"{base} Mention exactly these names verbatim: {_join_names(names)}."


def build_canary_prompt(fact: str) -> str:
    """Prompt asking for one wrapping sentence around an untouched fact."""
    return (
        "Write one sentence of neutral English business-document prose and put "
        "it before or after the following sentence, which you must reproduce "
        f"exactly, character for character, without any change: {fact} "
        "Use no other proper names. Plain text only, no markdown, no lists."
    )


class SyntheticGenerator:
    """Builds and registers the synthetic fragments for outbound batches."""

    def __init__(
        self,
        config: AppConfig,
        store: "ChunkStore",
        llm: LocalLLMClient | None = None,
    ) -> None:
        """Wire the generator to its config, store and prose source.

        Args:
            config: the loaded application config.
            store: the registry synthetic fragments are recorded in.
            llm: prose source; when None a real client is built from the
                config on first use, so offline commands never touch it.
        """
        self._config = config
        self._store = store
        self._llm = llm
        # Two independent deterministic streams from the one configured seed:
        # one decides the shape of a fragment, one mints the names.
        self._rng = random.Random(config.synthetic.seed)
        self._factory = FakeNameFactory(config.synthetic.seed)
        self._seed_counter = 0

    # ------------------------------------------------------------------
    # fragment kinds
    # ------------------------------------------------------------------

    def make_honeytokens(self, count: int) -> list[SyntheticFragment]:
        """Create `count` honeytokens and register them before returning."""
        if not self._config.synthetic.honeytokens_enabled:
            logger.info("Honeytokens are disabled; generating none")
            return []

        fragments = self._make_named_fragments(
            KIND_HONEYTOKEN,
            count,
            self._draw_honeytoken_names,
            build_honeytoken_prompt,
        )
        logger.info("Generated %d honeytokens", len(fragments))
        return fragments

    def make_chaff(self, count: int) -> list[SyntheticFragment]:
        """Create `count` chaff fragments and register them before returning."""
        if not self._config.synthetic.chaff_enabled:
            logger.info("Chaff is disabled; generating none")
            return []

        fragments = self._make_named_fragments(
            KIND_CHAFF,
            count,
            self._draw_chaff_names,
            build_chaff_prompt,
        )
        logger.info("Generated %d chaff fragments", len(fragments))
        return fragments

    def ensure_canaries(self) -> list[SyntheticFragment]:
        """Top the canary set up to its configured size and return all of it.

        Idempotent: only the shortfall is created, and the answer is read back
        from the store, so repeated calls neither duplicate canaries nor lose
        the ones planted in earlier runs.
        """
        if not self._config.synthetic.canaries_enabled:
            logger.info("Canaries are disabled; generating none")
            return []

        existing = self._store.list_synthetic_fragments(kind=KIND_CANARY)
        shortfall = self._config.synthetic.canary_set_size - len(existing)
        if shortfall > 0:
            self._register([self._make_canary() for _ in range(shortfall)])
            logger.info("Added %d canaries to the set", shortfall)
        else:
            logger.debug("Canary set already complete (%d fragments)", len(existing))

        return [
            _fragment_from_row(row)
            for row in self._store.list_synthetic_fragments(kind=KIND_CANARY)
        ]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _make_named_fragments(
        self,
        kind: str,
        count: int,
        draw: Callable[[], list[tuple[str, str]]],
        build_prompt: Callable[[list[str]], str],
    ) -> list[SyntheticFragment]:
        """Build `count` fragments of one kind and register them.

        Honeytokens and chaff differ only in how many names they carry and how
        the prose is asked for; everything else - minting the names locally,
        validating that they survived into the text, and recording them before
        the fragments are handed back - has to stay identical.
        """
        fragments: list[SyntheticFragment] = []
        for _ in range(max(count, 0)):
            planted = draw()
            names = [name for name, _ in planted]
            text = self._write_text(
                build_prompt(names), names, _template_fragment(planted)
            )
            fragments.append(
                SyntheticFragment(
                    fragment_id=uuid.uuid4().hex,
                    kind=kind,
                    text=text,
                    planted=planted,
                    fact=None,
                )
            )

        self._register(fragments)
        return fragments

    def _make_canary(self) -> SyntheticFragment:
        """Mint one canary: a fake person tied to a unique invented place."""
        person = self._factory.person()
        place = self._factory.unique_place()
        fact = CANARY_FACT_TEMPLATE.format(person=person, place=place)
        # The fact is the payload; the LLM may only add prose around it, and
        # if it cannot manage that, the bare fact is a perfectly good fragment.
        text = self._write_text(build_canary_prompt(fact), [fact], fact)
        return SyntheticFragment(
            fragment_id=uuid.uuid4().hex,
            kind=KIND_CANARY,
            text=text,
            planted=[(person, TYPE_PERSON)],
            fact=fact,
        )

    def _draw_honeytoken_names(self) -> list[tuple[str, str]]:
        """Draw 1-2 fake persons and 0-1 fake companies for one honeytoken."""
        person_count = self._rng.randint(HONEYTOKEN_MIN_PERSONS, HONEYTOKEN_MAX_PERSONS)
        company_count = self._rng.randint(
            HONEYTOKEN_MIN_COMPANIES, HONEYTOKEN_MAX_COMPANIES
        )
        planted = [
            (self._factory.person(), TYPE_PERSON) for _ in range(person_count)
        ]
        planted.extend(
            (self._factory.company(), TYPE_COMPANY) for _ in range(company_count)
        )
        return planted

    def _draw_chaff_names(self) -> list[tuple[str, str]]:
        """Draw 0-2 fake names of mixed kinds for one chaff fragment."""
        planted: list[tuple[str, str]] = []
        for _ in range(self._rng.randint(CHAFF_MIN_NAMES, CHAFF_MAX_NAMES)):
            if self._rng.random() < CHAFF_PERSON_SHARE:
                planted.append((self._factory.person(), TYPE_PERSON))
            else:
                planted.append((self._factory.company(), TYPE_COMPANY))
        return planted

    def _write_text(self, prompt: str, required: list[str], fallback: str) -> str:
        """Ask the LLM for prose, validate it, and fall back on a template.

        The LLM is retried with a fresh seed while its answer fails validation.
        A model too sloppy to follow the instruction must not stall the
        pipeline, so the deterministic `fallback` is used after the last try.

        Raises:
            LocalLLMError: when the endpoint itself is unusable; that is an
                operator problem worth surfacing, not a sloppy answer.
        """
        llm = self._get_llm()
        for attempt in range(1, LLM_ATTEMPTS + 1):
            candidate = llm.generate(prompt, self._next_seed()).strip()
            problem = _validation_problem(candidate, required)
            if problem is None:
                return candidate
            logger.warning(
                "Local LLM output rejected on attempt %d/%d: %s",
                attempt,
                LLM_ATTEMPTS,
                problem,
            )

        logger.warning(
            "Local LLM failed %d times; using the deterministic template instead",
            LLM_ATTEMPTS,
        )
        return fallback

    def _get_llm(self) -> LocalLLMClient:
        """Return the prose source, building a real client on first use."""
        if self._llm is None:
            llm_config = self._config.synthetic.llm
            self._llm = LocalLLMClient(
                base_url=llm_config.base_url,
                model=llm_config.model,
                temperature=llm_config.temperature,
                timeout_seconds=llm_config.timeout_seconds,
            )
            logger.debug("Built local LLM client for %s", llm_config.base_url)
        return self._llm

    def _next_seed(self) -> int:
        """Return the next deterministic sampling seed for an LLM call."""
        self._seed_counter += 1
        return self._config.synthetic.seed + self._seed_counter

    def _register(self, fragments: list[SyntheticFragment]) -> None:
        """Record fragments locally; nothing may be submitted before this."""
        if not fragments:
            return
        self._store.add_synthetic_fragments(
            [
                {
                    "fragment_id": fragment.fragment_id,
                    "kind": fragment.kind,
                    "text": fragment.text,
                    "planted": fragment.planted,
                    "fact": fragment.fact,
                }
                for fragment in fragments
            ]
        )


def _validation_problem(text: str, required: list[str]) -> str | None:
    """Return why `text` is unusable, or None when it passes.

    Validation is deliberately about the planted strings only: the model is
    trusted to write prose, never to decide what was planted.
    """
    if not text:
        return "empty output"
    if len(text) > MAX_FRAGMENT_CHARS:
        return f"output too long ({len(text)} > {MAX_FRAGMENT_CHARS} chars)"
    missing = [item for item in required if item not in text]
    if missing:
        return f"missing verbatim: {', '.join(missing)}"
    return None


def _template_fragment(planted: list[tuple[str, str]]) -> str:
    """Compose a deterministic fragment embedding every planted name."""
    persons = [name for name, entity_type in planted if entity_type == TYPE_PERSON]
    companies = [name for name, entity_type in planted if entity_type == TYPE_COMPANY]

    if persons and companies:
        return (
            f"Meeting notes: {_join_names(persons)} signed the revised service "
            f"agreement with {_join_names(companies)} before the agreed "
            "delivery date."
        )
    if persons:
        return (
            f"Meeting notes: {_join_names(persons)} approved the revised "
            "delivery schedule and closed the outstanding action items."
        )
    if companies:
        return (
            f"Meeting notes: the revised service agreement with "
            f"{_join_names(companies)} was approved without further amendments."
        )
    return (
        "Meeting notes: the revised delivery schedule was approved without "
        "further amendments."
    )


def _join_names(names: list[str]) -> str:
    """Join names for prose: "A", "A and B", "A, B and C"."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _fragment_from_row(row: dict) -> SyntheticFragment:
    """Rebuild a SyntheticFragment from a store row."""
    return SyntheticFragment(
        fragment_id=row["fragment_id"],
        kind=row["kind"],
        text=row["text"],
        planted=[(name, entity_type) for name, entity_type in row["planted"]],
        fact=row["fact"],
    )
