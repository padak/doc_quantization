"""Token-based chunking of Markdown documents.

The chunker slices a document into consecutive fixed-size token chunks so the
pieces can be shipped out of context (see `doc_quant.store`) and the original
document reassembled later. Reconstruction must be byte-exact, therefore:

    "".join(Chunker(...).chunk(text)) == text

for any input. Decoding a token slice through `decode_bytes` and enforcing a
valid UTF-8 boundary is what guarantees that property: a naive
`encoding.decode(tokens[a:b])` silently replaces a truncated multi-byte
character with U+FFFD and the document can no longer be reassembled.

A chunk is also the detection unit: exactly the stored text is what leaves this
process (see `doc_quant.detector`), with no context borrowed from a neighbour.
That is what keeps the outbound fragments disjoint - overlapping seams would
let whoever receives them re-stitch the document by matching shared text. For
detection to still work without such context, the cuts themselves are made
name-aware: a boundary that would fall inside a run of capitalized words is
pushed forward until the run is over, so every name lies wholly inside one
chunk. The push is bounded (`name_run_max_extension_tokens`); on the rare
document where the bound is hit the cut lands mid-run anyway, which is a
documented residual risk - such a name may then go undetected, but the
lossless invariant is never at stake, since this logic only moves cut points.

Tunables (encoding name, chunk size, extension budget) are never hardcoded
here; they are passed in by the caller from the application config.
"""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)

# How much decoded text on each side of a candidate cut is inspected when
# deciding whether the cut lands inside a name run. Bounded so that chunking
# stays linear in the document length; a name longer than this many tokens per
# side is beyond what the extension budget could rescue anyway.
NAME_RUN_ANALYSIS_TOKENS = 15

# Lowercase words that may sit *inside* a name run when the word after them is
# capitalized again: "Bank of America", "Johnson & Johnson", "Ludwig van
# Beethoven". Documents are English, so the list stays short and literal.
NAME_RUN_CONNECTORS = frozenset(
    {"of", "and", "de", "van", "von", "der", "da", "di", "la", "le", "al", "bin", "ter", "&"}
)

# Punctuation that ends a name run when it is attached to the preceding word,
# so that "...met Peter. The committee..." may be cut between the sentences.
SENTENCE_PUNCTUATION = frozenset(".,!?:;")

# Decoration that may wrap a word without changing whether it is capitalized or
# whether it carries terminating punctuation: quotes, brackets, Markdown marks.
LEADING_DECORATION = "\"'“‘([{*_#>~"
TRAILING_DECORATION = "\"'”’)]}*_~"


class Chunker:
    """Split text into token chunks whose cuts never fall inside a name run."""

    def __init__(
        self,
        encoding_name: str,
        chunk_size: int,
        name_run_max_extension_tokens: int = 12,
    ) -> None:
        """Create a chunker.

        Args:
            encoding_name: tiktoken encoding name, e.g. "cl100k_base".
            chunk_size: target number of tokens per chunk (must be >= 1).
            name_run_max_extension_tokens: how many tokens beyond the first
                UTF-8-valid cut a boundary may be pushed to avoid splitting a
                capitalized name run; 0 disables the name-aware adjustment.

        Raises:
            ValueError: if `chunk_size` is smaller than 1, or if
                `name_run_max_extension_tokens` is negative.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if name_run_max_extension_tokens < 0:
            raise ValueError(
                "name_run_max_extension_tokens must be >= 0, "
                f"got {name_run_max_extension_tokens}"
            )
        self.encoding_name = encoding_name
        self.chunk_size = chunk_size
        self.name_run_max_extension_tokens = name_run_max_extension_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    def _encode(self, text: str) -> list[int]:
        """Encode text, treating special-token markup as ordinary text.

        `disallowed_special=()` keeps the encoder from raising on documents
        that literally contain strings such as "<|endoftext|>", which would
        otherwise make chunking fail on perfectly valid Markdown.
        """
        return self._encoding.encode(text, disallowed_special=())

    def token_strings(self, text: str) -> list[str]:
        """Decode every token of `text` on its own, for display purposes.

        The list length is the token count, and for text whose characters each
        live inside a single token the concatenation is `text` again. A
        character split across two tokens cannot be shown that way, so the
        pieces are decoded with `errors="replace"`: this output is only ever
        shown to a reader, never stored and never sent anywhere, so a U+FFFD in
        it costs nothing. Chunk text itself is produced by `chunk`, which is
        lossless.
        """
        return [
            self._encoding.decode_single_token_bytes(token).decode(
                "utf-8", errors="replace"
            )
            for token in self._encode(text)
        ]

    def token_display_segments(self, text: str) -> list[str]:
        """Split `text` into displayable pieces that follow the token boundaries.

        Like `token_strings`, but lossless: bytes are accumulated token by token
        and a segment is emitted as soon as what has accumulated is valid UTF-8.
        A token whose bytes end mid-character therefore merges with the tokens
        that complete it, so a name such as "Šimeček" is shown as written
        instead of as the U+FFFD pairs an individually decoded token yields.

        Guarantees, for any `text`:

            "".join(token_display_segments(text)) == text

        and no segment carries a U+FFFD that `text` did not already carry. Most
        segments are exactly one token; only the ones spanning a split
        character are longer, which is why this is a display aid and not a
        token count - use `token_strings` when the number of tokens is what
        matters.
        """
        segments: list[str] = []
        pending = b""
        for token in self._encode(text):
            pending += self._encoding.decode_single_token_bytes(token)
            try:
                segments.append(pending.decode("utf-8"))
            except UnicodeDecodeError:
                # The character is not complete yet; the next token carries the
                # rest of its bytes.
                continue
            pending = b""

        if pending:
            # Unreachable for text that round-trips through the encoder: its
            # last token cannot end mid-character. Kept lossy-but-visible
            # rather than silently dropping bytes.
            logger.warning(
                "Trailing %d undecodable byte(s) while building display segments",
                len(pending),
            )
            segments.append(pending.decode("utf-8", errors="replace"))
        return segments

    def chunk(self, text: str) -> list[str]:
        """Split `text` into consecutive chunks of roughly `chunk_size` tokens.

        The concatenation of the returned chunks always equals `text`. Two
        things may push a boundary past the nominal token count: a cut that
        would land inside a multi-byte UTF-8 character, and a cut that would
        split a run of capitalized words (see `_splits_name_run`). The latter
        is bounded by `name_run_max_extension_tokens`; when that budget runs
        out the last UTF-8-valid cut is used even though it splits a run.

        Args:
            text: document text; an empty string yields an empty list.

        Returns:
            List of chunk texts in document order.
        """
        if not text:
            return []

        tokens = self._encode(text)
        total = len(tokens)
        chunks: list[str] = []
        start = 0

        while start < total:
            end, chunk_text = self._decode_slice(
                tokens, start, min(start + self.chunk_size, total), total
            )
            # The budget is counted from the first UTF-8-valid cut, so a
            # boundary already pushed by a multi-byte character does not eat
            # into the allowance for name runs.
            max_end = min(end + self.name_run_max_extension_tokens, total)

            while end < total and self._splits_name_run(tokens, end):
                candidate_end, candidate_text = self._decode_slice(
                    tokens, start, end + 1, total
                )
                if candidate_end > max_end:
                    logger.debug(
                        "Name-run extension budget of %d tokens exhausted at "
                        "token %d; cutting inside the run",
                        self.name_run_max_extension_tokens,
                        end,
                    )
                    break
                end, chunk_text = candidate_end, candidate_text
                logger.debug("Moved chunk boundary to token %d to keep a name run whole", end)

            chunks.append(chunk_text)
            start = end

        return chunks

    def _decode_slice(
        self, tokens: list[int], start: int, end: int, total: int
    ) -> tuple[int, str]:
        """Decode `tokens[start:end]`, pushing `end` to a valid UTF-8 boundary.

        Returns the (possibly advanced) end index together with the decoded
        text. This is the mechanism the lossless invariant rests on: the text
        returned here is always exactly what `tokens[start:end]` encodes.

        Raises:
            ValueError: if even the whole remaining tail fails to decode, which
                cannot happen for text that round-trips through the encoder.
        """
        while True:
            raw = self._encoding.decode_bytes(tokens[start:end])
            try:
                return end, raw.decode("utf-8")
            except UnicodeDecodeError:
                if end >= total:
                    # Unreachable for text that round-trips through the
                    # encoder: the tail of a valid UTF-8 document starting
                    # on a character boundary is itself valid UTF-8.
                    raise ValueError(
                        "Cannot decode trailing token slice as UTF-8; "
                        f"encoding {self.encoding_name!r} produced a "
                        "non-recoverable boundary"
                    )
                end += 1
                logger.debug(
                    "Extended chunk boundary to token %d to land on a "
                    "valid UTF-8 character boundary",
                    end,
                )

    def _splits_name_run(self, tokens: list[int], cut_index: int) -> bool:
        """Report whether cutting at `cut_index` would break a name-like run.

        A name-like run is a maximal sequence of whitespace-separated words
        each starting with an uppercase letter; a single connector word (see
        `NAME_RUN_CONNECTORS`) may sit inside the run when the word following
        it is capitalized again. A run ends at sentence punctuation attached to
        a word, at a newline, or at a word that is neither capitalized nor a
        connector.

        The cut splits a run when either
        (a) it falls inside a word whose first letter is uppercase
            ("McKin|sey"), or
        (b) it falls at whitespace between a word that belongs to a run and a
            word that continues it ("Margaret |Wetherby").

        Only a bounded window around the cut is examined, and it is decoded
        with `errors="replace"` because a window edge may land mid-character.
        That is safe here and here only: this text is used for the decision
        alone and never becomes chunk text.
        """
        if cut_index <= 0 or cut_index >= len(tokens):
            return False

        left = self._encoding.decode_bytes(
            tokens[max(0, cut_index - NAME_RUN_ANALYSIS_TOKENS) : cut_index]
        ).decode("utf-8", errors="replace")
        right = self._encoding.decode_bytes(
            tokens[cut_index : cut_index + NAME_RUN_ANALYSIS_TOKENS]
        ).decode("utf-8", errors="replace")

        if not left or not right:
            return False

        # (a) mid-word: letters on both sides of the seam.
        if left[-1].isalpha() and right[0].isalpha():
            return _fragment_starts_uppercase(left)

        # (b) between words: the seam must be whitespace without a newline.
        left_body = left.rstrip()
        right_body = right.lstrip()
        gap = left[len(left_body) :] + right[: len(right) - len(right_body)]
        if not gap or "\n" in gap:
            return False

        before = left_body.split()
        after = right_body.split()
        if not before or not after:
            return False

        return _ends_inside_run(before) and _continues_run(after)


def _fragment_starts_uppercase(left: str) -> str:
    """Return whether the word fragment ending `left` starts with an uppercase letter."""
    index = len(left)
    while index > 0 and left[index - 1].isalpha():
        index -= 1
    fragment = left[index:]
    return bool(fragment) and fragment[0].isupper()


def _core(word: str) -> str:
    """Strip wrapping decoration so a word can be judged on its letters."""
    return word.lstrip(LEADING_DECORATION).rstrip(TRAILING_DECORATION)


def _is_capitalized(word: str) -> bool:
    core = _core(word)
    return bool(core) and core[0].isupper()


def _is_connector(word: str) -> bool:
    return _core(word).casefold() in NAME_RUN_CONNECTORS


def _is_terminated(word: str) -> bool:
    """Whether sentence punctuation is attached to the end of `word`."""
    core = _core(word)
    return bool(core) and core[-1] in SENTENCE_PUNCTUATION


def _ends_inside_run(before: list[str]) -> bool:
    """Whether the last word of `before` is part of an unterminated name run.

    A trailing connector counts when the word before it is itself capitalized,
    so that a cut in "Bank of| America" is recognised as splitting the run just
    like a cut in "Bank |of America" is.
    """
    last = before[-1]
    if _is_terminated(last):
        return False
    if _is_capitalized(last):
        return True
    return (
        _is_connector(last)
        and len(before) >= 2
        and _is_capitalized(before[-2])
        and not _is_terminated(before[-2])
    )


def _continues_run(after: list[str]) -> bool:
    """Whether the first word of `after` carries a name run onwards."""
    first = after[0]
    if _is_capitalized(first):
        return True
    return _is_connector(first) and len(after) >= 2 and _is_capitalized(after[1])
