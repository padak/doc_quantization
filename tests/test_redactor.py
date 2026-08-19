"""Tests for placeholder substitution."""

from __future__ import annotations

from doc_quant.redactor import find_emails, find_urls, redact_text

PERSON_PLACEHOLDER = "**PERSON**"
COMPANY_PLACEHOLDER = "**COMPANY**"
EMAIL_PLACEHOLDER = "**EMAIL**"
URL_PLACEHOLDER = "**URL**"


def redact(text: str, entities: list[tuple[str, str]]) -> str:
    return redact_text(text, entities, PERSON_PLACEHOLDER, COMPANY_PLACEHOLDER)


def redact_with_emails(text: str, entities: list[tuple[str, str]]) -> str:
    return redact_text(
        text,
        entities,
        PERSON_PLACEHOLDER,
        COMPANY_PLACEHOLDER,
        email_placeholder=EMAIL_PLACEHOLDER,
    )


def redact_all(text: str, entities: list[tuple[str, str]]) -> str:
    """Everything on: URLs, emails and entities, as the app runs it."""
    return redact_text(
        text,
        entities,
        PERSON_PLACEHOLDER,
        COMPANY_PLACEHOLDER,
        email_placeholder=EMAIL_PLACEHOLDER,
        url_placeholder=URL_PLACEHOLDER,
    )


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


# ---------------------------------------------------------------------------
# emails
# ---------------------------------------------------------------------------


def test_find_emails_returns_unique_addresses_in_order_of_appearance():
    text = "a@b.com wrote, then c@d.org, then a@b.com again."
    assert find_emails(text) == ["a@b.com", "c@d.org"]


def test_find_emails_on_text_without_any():
    assert find_emails("No addresses here (at all).") == []


def test_email_is_replaced_as_a_whole_including_dots_plus_and_subdomains():
    text = (
        "Write to pavel.dolezal+invoices@mail.corp.keboola.com or to "
        "padak@keboola.com before Friday."
    )
    result = redact_with_emails(text, [])
    assert result == "Write to **EMAIL** or to **EMAIL** before Friday."
    assert "pavel.dolezal" not in result
    assert "keboola.com" not in result


def test_email_inside_angle_brackets_is_replaced():
    text = "Petr Simecek <petr@keboola.com> signed it."
    result = redact_with_emails(text, [("Petr Simecek", "person")])
    assert result == "**PERSON** <**EMAIL**> signed it."


def test_entity_name_that_is_also_an_email_local_part():
    # The standalone occurrence is redacted as a person; the one inside the
    # address is gone with the whole address, so no half-redacted mix survives.
    text = "Novak approved it. Write to Novak@keboola.com for the details."
    result = redact_with_emails(
        text, [("Novak", "person"), ("keboola", "company")]
    )
    assert result == "**PERSON** approved it. Write to **EMAIL** for the details."
    assert "Novak@" not in result
    assert "@**COMPANY**" not in result


def test_company_domain_no_longer_leaks_the_local_part():
    # The regression this whole mechanism exists for.
    text = "Contact pavel.dolezal@keboola.com about it."
    result = redact_with_emails(text, [("keboola.com", "company")])
    assert result == "Contact **EMAIL** about it."
    assert "pavel.dolezal" not in result


def test_entity_equal_to_a_placeholder_word_is_skipped():
    # "EMAIL" as an entity would otherwise match inside "**EMAIL**".
    text = "Ask EMAIL Systems, or write to a@b.com."
    result = redact_with_emails(text, [("EMAIL", "company")])
    assert result == "Ask EMAIL Systems, or write to **EMAIL**."


def test_without_email_placeholder_addresses_are_left_alone():
    text = "Write to padak@keboola.com about Keboola."
    result = redact(text, [("Keboola", "company")])
    assert result == "Write to padak@keboola.com about **COMPANY**."


def test_without_email_placeholder_entity_matching_inside_an_address_is_unchanged():
    # Today's behaviour, kept verbatim: the domain word is still substituted.
    text = "Write to padak@keboola.com."
    result = redact(text, [("keboola", "company")])
    assert result == "Write to padak@**COMPANY**.com."


def test_emails_are_replaced_when_there_are_no_entities_at_all():
    text = "Just a@b.com here."
    assert redact_with_emails(text, []) == "Just **EMAIL** here."


# ---------------------------------------------------------------------------
# urls
# ---------------------------------------------------------------------------


def test_find_urls_returns_unique_addresses_in_order_of_appearance():
    text = "See https://a.com/x and www.b.org, then https://a.com/x again."
    assert find_urls(text) == ["https://a.com/x", "www.b.org"]


def test_url_with_path_and_query_is_replaced_wholly():
    # The leak this exists for: the host names the company and the path
    # carries a case identifier.
    text = "Open https://resolve.picrights.com/Home/Settlement/3979-4561-9198?ref=7 now."
    result = redact_all(text, [])
    assert result == "Open **URL** now."
    assert "picrights" not in result
    assert "3979-4561-9198" not in result


def test_bare_www_urls_are_replaced():
    text = "Try www.keboola.com/pricing for the details."
    assert redact_all(text, []) == "Try **URL** for the details."


def test_trailing_sentence_punctuation_is_not_swallowed():
    text = "Read https://example.com/a. Then https://example.com/b, or https://example.com/c!"
    result = redact_all(text, [])
    assert result == "Read **URL**. Then **URL**, or **URL**!"


def test_a_url_in_brackets_keeps_its_bracket():
    text = "The report (https://example.com/report) arrived."
    assert redact_all(text, []) == "The report (**URL**) arrived."


def test_url_containing_an_email_like_part_is_replaced_once():
    # URL-first: the email pass must not eat the userinfo and leave the rest.
    text = "Login at https://user@mail.example.com/inbox?to=jan@example.com today."
    result = redact_all(text, [])
    assert result == "Login at **URL** today."
    assert "**EMAIL**" not in result


def test_url_email_and_person_all_get_their_own_placeholder():
    text = (
        "Jan Novak from Keboola wrote to padak@keboola.com and linked "
        "https://keboola.com/blog/post-1."
    )
    result = redact_all(text, [("Jan Novak", "person"), ("Keboola", "company")])
    assert result == (
        "**PERSON** from **COMPANY** wrote to **EMAIL** and linked **URL**."
    )


def test_entity_equal_to_the_url_placeholder_word_is_skipped():
    text = "URL Systems maintains https://example.com/x."
    result = redact_all(text, [("URL", "company")])
    assert result == "URL Systems maintains **URL**."


def test_without_url_placeholder_urls_are_left_alone():
    text = "See https://example.com/pricing about Keboola."
    result = redact(text, [("Keboola", "company")])
    assert result == "See https://example.com/pricing about **COMPANY**."


def test_without_url_placeholder_an_entity_inside_a_url_is_still_substituted():
    # Today's behaviour, kept verbatim: only the host word is rewritten, and
    # the rest of the address survives - which is the leak the placeholder fixes.
    text = "See https://keboola.com/pricing."
    result = redact(text, [("keboola", "company")])
    assert result == "See https://**COMPANY**.com/pricing."


def test_url_redaction_without_any_entities_or_emails():
    text = "Only https://example.com here."
    result = redact_text(
        text, [], PERSON_PLACEHOLDER, COMPANY_PLACEHOLDER, url_placeholder=URL_PLACEHOLDER
    )
    assert result == "Only **URL** here."
