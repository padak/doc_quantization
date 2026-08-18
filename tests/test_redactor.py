"""Tests for placeholder substitution."""

from __future__ import annotations

from doc_quant.redactor import redact_text

PERSON_PLACEHOLDER = "**PERSON**"
COMPANY_PLACEHOLDER = "**COMPANY**"


def redact(text: str, entities: list[tuple[str, str]]) -> str:
    return redact_text(text, entities, PERSON_PLACEHOLDER, COMPANY_PLACEHOLDER)


def test_replaces_person_and_company():
    text = "Jan Novak works at Keboola on Tuesdays."
    result = redact(text, [("Jan Novak", "person"), ("Keboola", "company")])
    assert result == "**PERSON** works at **COMPANY** on Tuesdays."


def test_no_entities_returns_text_unchanged():
    text = "Nothing to hide here."
    assert redact(text, []) == text


def test_longest_entity_wins_over_its_own_prefix():
    # "Jan" is a prefix of "Jan Novak"; the full name must be matched first.
    text = "Jan Novak called. Jan called again."
    result = redact(text, [("Jan", "person"), ("Jan Novak", "person")])
    assert result == "**PERSON** called. **PERSON** called again."
    assert "Novak" not in result


def test_longest_match_wins_across_types():
    text = "Acme Corporation invoiced us."
    result = redact(
        text,
        [("Acme", "company"), ("Acme Corporation", "company")],
    )
    assert result == "**COMPANY** invoiced us."


def test_person_wins_when_same_string_has_both_types():
    text = "Ford signed the contract."
    person_first = redact(text, [("Ford", "person"), ("Ford", "company")])
    company_first = redact(text, [("Ford", "company"), ("Ford", "person")])
    assert person_first == "**PERSON** signed the contract."
    assert company_first == "**PERSON** signed the contract."


def test_word_boundaries_prevent_matching_inside_longer_words():
    text = "Announcement: Ann and Anna arrived. Anncorp is unrelated."
    result = redact(text, [("Ann", "person")])
    assert result == "Announcement: **PERSON** and Anna arrived. Anncorp is unrelated."


def test_no_recursive_replacement_into_generated_placeholders():
    # "PERSON" is reported as a company name. The placeholder produced for
    # "Jan" contains that substring, and must not be rewritten again.
    text = "Jan sent the report."
    result = redact(text, [("Jan", "person"), ("PERSON", "company")])
    assert result == "**PERSON** sent the report."
    assert "COMPANY" not in result


def test_replaces_all_occurrences():
    text = "Keboola, Keboola and Keboola again; Keboola."
    result = redact(text, [("Keboola", "company")])
    assert result == "**COMPANY**, **COMPANY** and **COMPANY** again; **COMPANY**."


def test_duplicate_entities_are_deduplicated():
    text = "Keboola shipped it."
    result = redact(
        text,
        [("Keboola", "company"), ("Keboola", "company"), ("Keboola", "company")],
    )
    assert result == "**COMPANY** shipped it."


def test_czech_diacritics_are_matched_exactly():
    text = "Tomáš Marný a Žofie Křížová dorazili do Řeporyj."
    result = redact(
        text,
        [("Tomáš Marný", "person"), ("Žofie Křížová", "person")],
    )
    assert result == "**PERSON** a **PERSON** dorazili do Řeporyj."


def test_czech_diacritics_respect_word_boundaries():
    # An accented letter is a word character, so the declined form must survive.
    text = "Tomáš mluvil s Tomášem."
    result = redact(text, [("Tomáš", "person")])
    assert result == "**PERSON** mluvil s Tomášem."


def test_case_sensitive_matching():
    text = "Keboola and keboola are different strings."
    result = redact(text, [("Keboola", "company")])
    assert result == "**COMPANY** and keboola are different strings."


def test_empty_and_whitespace_entities_are_skipped():
    text = "Jan  Novak stayed."
    result = redact(text, [("", "person"), ("   ", "company"), ("Jan", "person")])
    assert result == "**PERSON**  Novak stayed."


def test_unknown_entity_type_is_ignored():
    text = "Prague is a city."
    assert redact(text, [("Prague", "location")]) == text


def test_regex_metacharacters_in_entities_are_literal():
    text = "C++ Solutions s.r.o. and C.. Solutions differ."
    result = redact(text, [("C++ Solutions s.r.o.", "company")])
    assert result == "**COMPANY** and C.. Solutions differ."
