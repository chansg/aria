"""
tests.test_web_search
---------------------
Web search query-cleaning regression tests.
"""

from __future__ import annotations

from core.web_search import clean_query


def test_clean_query_strips_correction_prefix_for_finance_news() -> None:
    out = clean_query("No, how did Ryan Cohen interview go today?")

    assert out == "how did ryan cohen interview go today"


def test_clean_query_strips_chained_assistant_and_correction_prefixes() -> None:
    out = clean_query("Aria, actually, I meant look up Ryan Cohen interview today")

    assert out == "ryan cohen interview today"


def test_clean_query_keeps_plain_negative_query_content() -> None:
    out = clean_query("no news about GameStop today")

    assert out == "news about gamestop today"
