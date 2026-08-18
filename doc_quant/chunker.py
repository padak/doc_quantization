"""Token-based chunking of Markdown documents.

The chunker slices a document into consecutive fixed-size token chunks so the
pieces can be shipped out of context (see `doc_quant.store`) and the original
document reassembled later. Reconstruction must be byte-exact, therefore:

    "".join(Chunker(...).chunk(text)) == text

for any input. Decoding a token slice through `decode_bytes` and enforcing a
valid UTF-8 boundary is what guarantees that property: a naive
`encoding.decode(tokens[a:b])` silently replaces a truncated multi-byte
character with U+FFFD and the document can no longer be reassembled.

Tunables (encoding name, chunk size, detection margin) are never hardcoded
here; they are passed in by the caller from the application config.
"""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)


class Chunker:
    """Split text into token chunks and build detection windows around them."""

    def __init__(self, encoding_name: str, chunk_size: int) -> None:
        """Create a chunker.

        Args:
            encoding_name: tiktoken encoding name, e.g. "cl100k_base".
            chunk_size: target number of tokens per chunk (must be >= 1).

        Raises:
            ValueError: if `chunk_size` is smaller than 1.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.encoding_name = encoding_name
        self.chunk_size = chunk_size
        self._encoding = tiktoken.get_encoding(encoding_name)

    def _encode(self, text: str) -> list[int]:
        """Encode text, treating special-token markup as ordinary text.

        `disallowed_special=()` keeps the encoder from raising on documents
        that literally contain strings such as "<|endoftext|>", which would
        otherwise make chunking fail on perfectly valid Markdown.
        """
        return self._encoding.encode(text, disallowed_special=())

    def chunk(self, text: str) -> list[str]:
        """Split `text` into consecutive chunks of roughly `chunk_size` tokens.

        The concatenation of the returned chunks always equals `text`. A chunk
        boundary that would land inside a multi-byte UTF-8 character is pushed
        forward one token at a time until the accumulated bytes decode cleanly,
        so individual chunks may occasionally exceed `chunk_size` by a token or
        two. The following chunk then starts after the adjusted boundary.

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
            end = min(start + self.chunk_size, total)
            while True:
                raw = self._encoding.decode_bytes(tokens[start:end])
                try:
                    chunks.append(raw.decode("utf-8"))
                    break
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
            start = end

        return chunks

    def window(self, chunk_texts: list[str], index: int, margin_tokens: int) -> str:
        """Return chunk `index` padded with context from its neighbours.

        Up to `margin_tokens` trailing tokens of the previous chunk and up to
        `margin_tokens` leading tokens of the next chunk are prepended and
        appended. Windows are used for name detection only and are never fed
        back into reconstruction, so a margin whose outer edge cuts a
        multi-byte character simply drops the offending bytes at that edge.

        Args:
            chunk_texts: all chunks of a document, in order.
            index: position of the chunk to build a window for.
            margin_tokens: context budget per side; 0 (or less) returns the
                chunk text unchanged.

        Returns:
            The window text.

        Raises:
            IndexError: if `index` is outside the range of `chunk_texts`.
        """
        if index < 0 or index >= len(chunk_texts):
            raise IndexError(
                f"chunk index {index} out of range for {len(chunk_texts)} chunks"
            )

        chunk_text = chunk_texts[index]
        if margin_tokens <= 0:
            return chunk_text

        prefix = ""
        if index > 0:
            previous_tokens = self._encode(chunk_texts[index - 1])
            prefix = self._decode_margin(
                previous_tokens[-margin_tokens:], drop_from_start=True
            )

        suffix = ""
        if index < len(chunk_texts) - 1:
            next_tokens = self._encode(chunk_texts[index + 1])
            suffix = self._decode_margin(
                next_tokens[:margin_tokens], drop_from_start=False
            )

        return f"{prefix}{chunk_text}{suffix}"

    def _decode_margin(self, tokens: list[int], drop_from_start: bool) -> str:
        """Decode a margin token slice, trimming bytes at the cut edge.

        Args:
            tokens: the margin tokens.
            drop_from_start: True for a trailing margin taken from the previous
                chunk (the cut is at the beginning of the byte string), False
                for a leading margin taken from the next chunk.

        Returns:
            The decoded margin text; empty if nothing survives trimming.
        """
        if not tokens:
            return ""

        raw = self._encoding.decode_bytes(tokens)
        while raw:
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                raw = raw[1:] if drop_from_start else raw[:-1]
                logger.debug("Trimmed one byte from a window margin edge")
        return ""
