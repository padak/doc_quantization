"""Tests for doc_quant.chunker.

The critical property is lossless reconstruction: joining the chunks of any
input must reproduce that input byte for byte, including Czech diacritics,
emoji and CJK text whose UTF-8 encoding spans several bytes per character.

The second property is that a cut never falls inside a run of capitalized
words, so that a name always lies wholly inside one chunk and detection needs
no context from the neighbouring fragments. The texts below are engineered so
that the *tentative* cut - the plain `CHUNK_SIZE`-token boundary - lands in a
specific spot; `tentative_cut` asserts that engineering still holds, so a
future tokenizer change turns into a clear failure rather than a test that
silently stops testing anything.
"""

from __future__ import annotations

import pytest
import tiktoken

from doc_quant.chunker import Chunker

ENCODING = "cl100k_base"
CHUNK_SIZE = 22
MAX_EXTENSION = 12

ENGLISH_MARKDOWN = (
    "# Release notes\n\n"
    "The parser now handles nested lists correctly.\n\n"
    "- Fixed a crash in the tokenizer\n"
    "- Added support for footnotes\n"
    "- Improved error messages for malformed tables\n\n"
    "See the [documentation](https://example.com/docs) for details.\n"
)

CZECH_TEXT = (
    "Příliš žluťoučký kůň úpěl ďábelské ódy. Řekl: „Ať žije Škoda Auto!\" "
    "Ředitelka Věra Nováková podepsala smlouvu s firmou Čerpadla Plzeň s.r.o. "
    "v úterý ráno, přestože účetní oddělení mělo výhrady."
)

EMOJI_CJK_TEXT = (
    "Launch party 🎉🚀 with the team 👨‍👩‍👧‍👦 in Tokyo.\n"
    "日本語のテキストと中文文本、그리고 한국어 텍스트가 섞여 있습니다。\n"
    "Flags: 🇨🇿 🇯🇵 🇰🇷 and math symbols ∑ ∫ √ π.\n"
)

SHORT_TEXT = "Hi."

SAMPLE_TEXTS = {
    "english_markdown": ENGLISH_MARKDOWN,
    "czech_diacritics": CZECH_TEXT,
    "emoji_and_cjk": EMOJI_CJK_TEXT,
    "shorter_than_chunk": SHORT_TEXT,
    "empty": "",
}


@pytest.fixture(scope="module")
def chunker() -> Chunker:
    return Chunker(
        encoding_name=ENCODING,
        chunk_size=CHUNK_SIZE,
        name_run_max_extension_tokens=MAX_EXTENSION,
    )


def token_counts(texts: list[str]) -> list[int]:
    encoding = tiktoken.get_encoding(ENCODING)
    return [len(encoding.encode(text, disallowed_special=())) for text in texts]


def tentative_cut(text: str) -> tuple[str, str]:
    """Return the text before and after the plain `CHUNK_SIZE`-token boundary."""
    encoding = tiktoken.get_encoding(ENCODING)
    tokens = encoding.encode(text, disallowed_special=())
    assert len(tokens) > CHUNK_SIZE, "sample must be longer than one chunk"
    return encoding.decode(tokens[:CHUNK_SIZE]), encoding.decode(tokens[CHUNK_SIZE:])


def chunk_containing(chunks: list[str], needle: str) -> str:
    matches = [text for text in chunks if needle in text]
    assert len(matches) == 1, f"expected {needle!r} in exactly one chunk, got {len(matches)}"
    return matches[0]


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_size", [0, -1])
def test_chunk_size_must_be_positive(bad_size: int) -> None:
    with pytest.raises(ValueError):
        Chunker(encoding_name=ENCODING, chunk_size=bad_size)


@pytest.mark.parametrize("bad_extension", [-1, -12])
def test_name_run_extension_must_not_be_negative(bad_extension: int) -> None:
    with pytest.raises(ValueError):
        Chunker(
            encoding_name=ENCODING,
            chunk_size=CHUNK_SIZE,
            name_run_max_extension_tokens=bad_extension,
        )


# ----------------------------------------------------------------------
# lossless reconstruction
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SAMPLE_TEXTS))
def test_chunk_roundtrip_is_lossless(chunker: Chunker, name: str) -> None:
    text = SAMPLE_TEXTS[name]
    assert "".join(chunker.chunk(text)) == text


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 22, 1000])
@pytest.mark.parametrize("name", sorted(SAMPLE_TEXTS))
def test_roundtrip_is_lossless_for_any_chunk_size(name: str, chunk_size: int) -> None:
    text = SAMPLE_TEXTS[name]
    local_chunker = Chunker(encoding_name=ENCODING, chunk_size=chunk_size)
    assert "".join(local_chunker.chunk(text)) == text


def test_roundtrip_preserves_mixed_document(chunker: Chunker) -> None:
    text = f"{ENGLISH_MARKDOWN}\n{CZECH_TEXT}\n{EMOJI_CJK_TEXT}"
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_empty_text_yields_no_chunks(chunker: Chunker) -> None:
    assert chunker.chunk("") == []


def test_text_shorter_than_chunk_size_yields_single_chunk(chunker: Chunker) -> None:
    chunks = chunker.chunk(SHORT_TEXT)
    assert chunks == [SHORT_TEXT]


# ----------------------------------------------------------------------
# chunk sizes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", [ENGLISH_MARKDOWN, CZECH_TEXT, EMOJI_CJK_TEXT], ids=["en", "cz", "emoji"]
)
def test_chunk_token_counts_match_chunk_size(chunker: Chunker, text: str) -> None:
    """Every chunk but the last carries `CHUNK_SIZE` tokens.

    Boundaries that would split a multi-byte character are pushed forward by a
    token or two, and re-encoding an isolated chunk can merge differently than
    the original slice, so a small tolerance is allowed.
    """
    tolerance = 4
    chunks = chunker.chunk(text)
    assert len(chunks) > 1

    encoding = tiktoken.get_encoding(ENCODING)
    for position, chunk_text in enumerate(chunks[:-1]):
        token_count = len(encoding.encode(chunk_text, disallowed_special=()))
        assert CHUNK_SIZE - tolerance <= token_count <= CHUNK_SIZE + tolerance, (
            f"chunk {position} has {token_count} tokens"
        )

    last_count = len(encoding.encode(chunks[-1], disallowed_special=()))
    assert 1 <= last_count <= CHUNK_SIZE + tolerance


def test_chunk_count_scales_with_chunk_size() -> None:
    text = f"{ENGLISH_MARKDOWN}{CZECH_TEXT}"
    small = Chunker(encoding_name=ENCODING, chunk_size=10).chunk(text)
    large = Chunker(encoding_name=ENCODING, chunk_size=50).chunk(text)
    assert len(small) > len(large)
    assert "".join(small) == "".join(large) == text



# ----------------------------------------------------------------------
# name-aware cuts
# ----------------------------------------------------------------------

# The tentative cut falls between the two words of "Margaret Wetherby".
NAME_STRADDLING_CUT = (
    "Notes: the quarterly report was reviewed again and again by the auditors "
    "before it was filed away our auditor Margaret Wetherby signed the amended "
    "agreement without delay."
)

# The tentative cut falls inside "McKinsey", between "McKin" and "sey".
WORD_STRADDLING_CUT = (
    "Notes: the quarterly report was reviewed again and again by the auditors "
    "before it was filed the consultants at McKinsey delivered the amended "
    "agreement without delay."
)

# The tentative cut falls before the connector: "... the Bank | of America ...".
CONNECTOR_RUN_CUT_BEFORE_CONNECTOR = (
    "Notes: the quarterly report was reviewed again and again by the auditors "
    "before it was filed away in the Bank of America underwrote the amended "
    "agreement without any delay."
)

# The tentative cut falls after the connector: "... the Bank of | America ...".
CONNECTOR_RUN_CUT_AFTER_CONNECTOR = (
    "Notes: the quarterly report was reviewed again and again by the auditors "
    "before it was filed away the Bank of America underwrote the amended "
    "agreement without any delay."
)

# The tentative cut falls exactly on the sentence boundary after "Peter.".
SENTENCE_BOUNDARY_CUT = (
    "Notes: the quarterly report was reviewed again and again by the auditors "
    "before the whole team finally met Peter. The auditors signed the amended "
    "agreement."
)

# A Title Case run far longer than the extension budget can carry.
TITLE_RUN_WORDS = [
    "Grand", "Northern", "Valley", "Harbour", "Trust", "Mutual", "Holdings",
    "Regional", "Advisory", "Council", "Annual", "General", "Meeting",
    "Minutes", "Volume", "Seven", "Winter", "Session", "Special", "Edition",
    "Appendix", "Charter", "Review", "Board", "Summary", "Preface", "Errata",
    "Index", "Notice", "Ledger",
]
LONG_TITLE_RUN = (
    "Notes: the auditors filed a document titled "
    + " ".join(TITLE_RUN_WORDS)
    + " and then everyone went home."
)

NAME_AWARE_SAMPLES = [
    NAME_STRADDLING_CUT,
    WORD_STRADDLING_CUT,
    CONNECTOR_RUN_CUT_BEFORE_CONNECTOR,
    CONNECTOR_RUN_CUT_AFTER_CONNECTOR,
    SENTENCE_BOUNDARY_CUT,
    LONG_TITLE_RUN,
]


@pytest.mark.parametrize("text", NAME_AWARE_SAMPLES, ids=range(len(NAME_AWARE_SAMPLES)))
def test_name_aware_cuts_stay_lossless(chunker: Chunker, text: str) -> None:
    """Moving a cut may never cost a byte."""
    assert "".join(chunker.chunk(text)) == text


def test_name_split_by_the_default_cut_ends_up_in_one_chunk(chunker: Chunker) -> None:
    before, after = tentative_cut(NAME_STRADDLING_CUT)
    assert before.endswith("Margaret") and after.startswith(" Wetherby")

    chunks = chunker.chunk(NAME_STRADDLING_CUT)

    assert "Margaret Wetherby" in chunk_containing(chunks, "Margaret")
    assert "".join(chunks) == NAME_STRADDLING_CUT


def test_cut_inside_a_capitalized_word_is_extended_past_it(chunker: Chunker) -> None:
    before, after = tentative_cut(WORD_STRADDLING_CUT)
    assert before.endswith("McKin") and after.startswith("sey")

    chunks = chunker.chunk(WORD_STRADDLING_CUT)

    assert "McKinsey" in chunk_containing(chunks, "McKin")
    assert not any(text.endswith("McKin") for text in chunks)
    assert "".join(chunks) == WORD_STRADDLING_CUT


@pytest.mark.parametrize(
    ("text", "expected_before", "expected_after"),
    [
        (CONNECTOR_RUN_CUT_BEFORE_CONNECTOR, "Bank", " of"),
        (CONNECTOR_RUN_CUT_AFTER_CONNECTOR, "Bank of", " America"),
    ],
    ids=["cut_before_connector", "cut_after_connector"],
)
def test_connector_run_is_never_cut(
    chunker: Chunker, text: str, expected_before: str, expected_after: str
) -> None:
    """"Bank of America" is one run: the connector may not become a seam."""
    before, after = tentative_cut(text)
    assert before.endswith(expected_before) and after.startswith(expected_after)

    chunks = chunker.chunk(text)

    assert "Bank of America" in chunk_containing(chunks, "Bank")
    assert "".join(chunks) == text


def test_sentence_boundary_cut_is_left_alone(chunker: Chunker) -> None:
    """"...met Peter. The auditors..." is two runs, so the seam is legal."""
    before, after = tentative_cut(SENTENCE_BOUNDARY_CUT)
    assert before.endswith("Peter.") and after.startswith(" The")

    chunks = chunker.chunk(SENTENCE_BOUNDARY_CUT)

    # No extension happened: the first chunk is exactly the tentative cut.
    assert chunks[0] == before
    assert token_counts(chunks[:-1]) == [CHUNK_SIZE] * (len(chunks) - 1)
    assert "".join(chunks) == SENTENCE_BOUNDARY_CUT


def test_extension_stops_at_the_configured_cap(chunker: Chunker) -> None:
    """A run longer than the budget is cut through rather than chased forever."""
    chunks = chunker.chunk(LONG_TITLE_RUN)
    counts = token_counts(chunks)

    assert max(counts) <= CHUNK_SIZE + MAX_EXTENSION
    # The budget really was spent: this run is long enough to exhaust it.
    assert counts[0] == CHUNK_SIZE + MAX_EXTENSION
    assert "".join(chunks) == LONG_TITLE_RUN


def test_zero_extension_budget_keeps_the_plain_cut() -> None:
    plain = Chunker(
        encoding_name=ENCODING, chunk_size=CHUNK_SIZE, name_run_max_extension_tokens=0
    )
    before, _ = tentative_cut(NAME_STRADDLING_CUT)

    chunks = plain.chunk(NAME_STRADDLING_CUT)

    assert chunks[0] == before
    assert "".join(chunks) == NAME_STRADDLING_CUT


@pytest.mark.parametrize(
    "text",
    NAME_AWARE_SAMPLES + [ENGLISH_MARKDOWN, CZECH_TEXT, EMOJI_CJK_TEXT],
    ids=range(len(NAME_AWARE_SAMPLES) + 3),
)
def test_chunking_is_deterministic(text: str) -> None:
    """Same input, same cuts - the adjustment introduces no state."""
    first = Chunker(
        encoding_name=ENCODING,
        chunk_size=CHUNK_SIZE,
        name_run_max_extension_tokens=MAX_EXTENSION,
    ).chunk(text)
    second = Chunker(
        encoding_name=ENCODING,
        chunk_size=CHUNK_SIZE,
        name_run_max_extension_tokens=MAX_EXTENSION,
    ).chunk(text)
    assert first == second == Chunker(ENCODING, CHUNK_SIZE, MAX_EXTENSION).chunk(text)
