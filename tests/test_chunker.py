"""Tests for doc_quant.chunker.

The critical property is lossless reconstruction: joining the chunks of any
input must reproduce that input byte for byte, including Czech diacritics,
emoji and CJK text whose UTF-8 encoding spans several bytes per character.
"""

from __future__ import annotations

import pytest
import tiktoken

from doc_quant.chunker import Chunker

ENCODING = "cl100k_base"
CHUNK_SIZE = 22
MARGIN = 8

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
    return Chunker(encoding_name=ENCODING, chunk_size=CHUNK_SIZE)


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_size", [0, -1])
def test_chunk_size_must_be_positive(bad_size: int) -> None:
    with pytest.raises(ValueError):
        Chunker(encoding_name=ENCODING, chunk_size=bad_size)


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
# detection windows
# ----------------------------------------------------------------------


def test_window_of_middle_chunk_includes_neighbour_fragments(
    chunker: Chunker,
) -> None:
    chunks = chunker.chunk(f"{ENGLISH_MARKDOWN}{CZECH_TEXT}{EMOJI_CJK_TEXT}")
    assert len(chunks) >= 3

    index = len(chunks) // 2
    chunk_text = chunks[index]
    window = chunker.window(chunks, index, MARGIN)

    assert chunk_text in window
    assert window != chunk_text

    prefix, _, suffix = window.partition(chunk_text)
    assert prefix, "expected trailing context from the previous chunk"
    assert suffix, "expected leading context from the next chunk"
    assert chunks[index - 1].endswith(prefix)
    assert chunks[index + 1].startswith(suffix)


def test_window_of_first_chunk_has_no_left_margin(chunker: Chunker) -> None:
    chunks = chunker.chunk(f"{ENGLISH_MARKDOWN}{CZECH_TEXT}")
    window = chunker.window(chunks, 0, MARGIN)

    assert window.startswith(chunks[0])
    assert len(window) > len(chunks[0])
    assert chunks[1].startswith(window[len(chunks[0]) :])


def test_window_of_last_chunk_has_no_right_margin(chunker: Chunker) -> None:
    chunks = chunker.chunk(f"{ENGLISH_MARKDOWN}{CZECH_TEXT}")
    last = len(chunks) - 1
    window = chunker.window(chunks, last, MARGIN)

    assert window.endswith(chunks[last])
    assert len(window) > len(chunks[last])
    assert chunks[last - 1].endswith(window[: -len(chunks[last])])


def test_window_with_zero_margin_returns_chunk_unchanged(chunker: Chunker) -> None:
    chunks = chunker.chunk(f"{ENGLISH_MARKDOWN}{CZECH_TEXT}")
    for index, chunk_text in enumerate(chunks):
        assert chunker.window(chunks, index, 0) == chunk_text


def test_window_of_single_chunk_document_returns_chunk(chunker: Chunker) -> None:
    chunks = chunker.chunk(SHORT_TEXT)
    assert chunker.window(chunks, 0, MARGIN) == SHORT_TEXT


def test_window_keeps_unicode_intact_across_margins(chunker: Chunker) -> None:
    """Margins may drop bytes at their outer edge but never emit broken text."""
    chunks = chunker.chunk(EMOJI_CJK_TEXT + CZECH_TEXT)
    for index in range(len(chunks)):
        window = chunker.window(chunks, index, MARGIN)
        # A str that round-trips through UTF-8 contains no lone surrogates or
        # replacement damage introduced by the margin trimming.
        assert window.encode("utf-8").decode("utf-8") == window
        assert "�" not in window
        assert chunks[index] in window


@pytest.mark.parametrize("bad_index", [-1, 5, 99])
def test_window_raises_index_error_for_out_of_range_index(
    chunker: Chunker, bad_index: int
) -> None:
    chunks = chunker.chunk(SHORT_TEXT)
    assert len(chunks) == 1
    with pytest.raises(IndexError):
        chunker.window(chunks, bad_index, MARGIN)


def test_window_on_empty_chunk_list_raises_index_error(chunker: Chunker) -> None:
    with pytest.raises(IndexError):
        chunker.window([], 0, MARGIN)
