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


def _resolve_placeholders(
    entities: list[tuple[str, str]],
    person_placeholder: str,
    company_placeholder: str,
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
) -> str:
    """Replace every occurrence of each entity with its placeholder.

    Matching is case-sensitive and verbatim, bounded by word-boundary
    lookarounds so that "Ann" does not match inside "Announcement".

    All entities are compiled into a single alternation, longest first, and
    substituted in one pass. This is what keeps a shorter entity from matching
    inside a placeholder that a longer entity just produced: the replacement
    text is never rescanned.
    """
    placeholders = _resolve_placeholders(entities, person_placeholder, company_placeholder)
    if not placeholders:
        return text

    # Longest first so "Jan Novak" is preferred over the "Jan" prefix.
    ordered = sorted(placeholders, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(entity) for entity in ordered) + r")(?!\w)"
    )
    return pattern.sub(lambda match: placeholders[match.group(0)], text)
