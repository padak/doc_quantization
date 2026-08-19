"""Placeholder substitution for detected person and company names.

The redactor receives the entities discovered for a whole document (collected
from many independently processed chunks) and rewrites the reassembled text.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

PERSON = "person"
COMPANY = "company"
EMAIL = "email"
URL = "url"

# Pragmatic rather than RFC-complete: it matches what documents actually
# contain. An email address is fully determined by its own shape, so it needs
# no detector - and because the whole address is replaced as one unit, the
# local part (usually a person's name) can never survive next to a redacted
# domain, which is how "pavel.dolezal@**COMPANY**" used to leak.
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# A URL identifies as much as a name does: the host names the company and the
# path often carries a case or account identifier, as in
# "resolve.picrights.com/Home/Settlement/3979-4561-9198". Both the scheme form
# and the bare "www." form are matched, up to the first whitespace or bracket.
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"'\)\]]+")

# Sentence punctuation that a URL at the end of a sentence would otherwise
# swallow: "see https://example.com/x." must not redact the full stop.
URL_TRAILING_PUNCTUATION = ".,;:!?"


def _trim_url(candidate: str) -> str:
    """Drop sentence punctuation a match picked up at its end."""
    return candidate.rstrip(URL_TRAILING_PUNCTUATION)


def _unique(values: list[str]) -> list[str]:
    """Deduplicate while keeping the order of first appearance."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def find_emails(text: str) -> list[str]:
    """Return the unique email addresses in `text`, first appearance first."""
    return _unique([match.group(0) for match in EMAIL_PATTERN.finditer(text)])


def find_urls(text: str) -> list[str]:
    """Return the unique URLs in `text`, first appearance first.

    Trailing sentence punctuation is trimmed, so the same address written
    mid-sentence and at the end of one is reported once.
    """
    return _unique(
        [
            trimmed
            for trimmed in (_trim_url(match.group(0)) for match in URL_PATTERN.finditer(text))
            if trimmed
        ]
    )


def _placeholder_words(*placeholders: str) -> set[str]:
    """The bare words inside placeholders, e.g. "EMAIL" from "**EMAIL**".

    An entity reported as one of these would match inside an already inserted
    placeholder - the punctuation around it is not a word character, so the
    word-boundary lookarounds do not stop it - and rewriting a placeholder into
    another placeholder helps nobody.
    """
    words: set[str] = set()
    for placeholder in placeholders:
        if placeholder:
            words.update(re.findall(r"\w+", placeholder))
    return words


def _resolve_placeholders(
    entities: list[tuple[str, str]],
    person_placeholder: str,
    company_placeholder: str,
    reserved: set[str],
) -> dict[str, str]:
    """Map each unique entity string to the placeholder that replaces it.

    The same string may be reported as a person in one chunk and as a company
    in another; "person" wins because leaking a personal name is the more
    sensitive failure.
    """
    resolved: dict[str, str] = {}
    for entity_text, entity_type in entities:
        if not entity_text or not entity_text.strip():
            logger.debug("Skipping empty entity of type %s", entity_type)
            continue
        if entity_type not in (PERSON, COMPANY):
            logger.warning(
                "Skipping entity %r with unknown type %r", entity_text, entity_type
            )
            continue
        if entity_text in reserved:
            logger.warning(
                "Skipping entity %r: it is a placeholder word", entity_text
            )
            continue
        placeholder = person_placeholder if entity_type == PERSON else company_placeholder
        if resolved.get(entity_text) == person_placeholder:
            # "person" already claimed this string and always wins.
            continue
        resolved[entity_text] = placeholder
    return resolved


def redact_text(
    text: str,
    entities: list[tuple[str, str]],
    person_placeholder: str,
    company_placeholder: str,
    *,
    email_placeholder: str | None = None,
    url_placeholder: str | None = None,
) -> str:
    """Replace every occurrence of each entity with its placeholder.

    Matching is case-sensitive and verbatim, bounded by word-boundary
    lookarounds so that "Ann" does not match inside "Announcement".

    All entities are compiled into a single alternation, longest first, and
    substituted in one pass. This is what keeps a shorter entity from matching
    inside a placeholder that a longer entity just produced: the replacement
    text is never rescanned.

    The two shape-detected kinds go first, before any entity is looked at:
    nothing about them depends on what a detector reported. URLs precede
    emails, because a URL may carry an "@" in its userinfo or path and the
    email pass would otherwise eat a piece of it and leave the rest behind.
    """
    reserved = _placeholder_words(
        person_placeholder,
        company_placeholder,
        email_placeholder or "",
        url_placeholder or "",
    )
    if url_placeholder:
        text = URL_PATTERN.sub(
            lambda match: url_placeholder + match.group(0)[len(_trim_url(match.group(0))):],
            text,
        )
    if email_placeholder:
        text = EMAIL_PATTERN.sub(email_placeholder, text)

    placeholders = _resolve_placeholders(
        entities, person_placeholder, company_placeholder, reserved
    )
    if not placeholders:
        return text

    # Longest first so "Jan Novak" is preferred over the "Jan" prefix.
    ordered = sorted(placeholders, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(entity) for entity in ordered) + r")(?!\w)"
    )
    return pattern.sub(lambda match: placeholders[match.group(0)], text)
