"""
Aria Web Search Module
======================
General-purpose web search for Aria using DuckDuckGo.

Scrapes the DuckDuckGo HTML endpoint for any natural language query
and extracts the top result snippets as clean plain text.

Results are cached in data/web_cache.json with a 15-minute TTL.

The old weather-only handler is retained but deprecated — weather
queries now go through the general search_web() path.
"""

import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from config import DATA_DIR

CACHE_FILE = os.path.join(DATA_DIR, "web_cache.json")
CACHE_TTL_MINUTES = 15
MAX_SNIPPETS = 3


# ── Query cleaning ───────────────────────────────────────────────────

# Name variants and prefixes to strip from queries before web search
_ARIA_PREFIXES = {
    "hey aria", "hello aria", "hi aria", "ok aria", "okay aria",
    "aria", "arya", "area", "ariya",
}

_CONVERSATIONAL_PREFIXES = (
    "please ", "can you ", "could you ", "would you ",
    "tell me about ", "tell me ", "show me ",
    "what is the ", "what is ", "what's the ", "what's ",
    "search for ", "look up ", "find out ",
)


def clean_query(raw: str) -> str:
    """Strip Aria name variants and conversational filler from a search query.

    Produces a clean, search-engine-friendly query string from raw
    transcribed speech. Applied before every DuckDuckGo scrape.

    Args:
        raw: Raw transcribed text from the user, e.g.
             'Hello Aria, what is the weather in Slough today?'

    Returns:
        Cleaned query string, e.g. 'weather in Slough today'

    Examples:
        >>> clean_query('Hello Aria, what is the weather in Slough?')
        'weather in Slough'
        >>> clean_query('Aria search for latest AI news')
        'latest AI news'
    """
    text = raw.lower().strip().rstrip("?.,!")

    # Strip Aria name variants from start
    for prefix in sorted(_ARIA_PREFIXES, key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix):].strip().lstrip(",").strip()
            break

    # Strip conversational prefixes — loop to catch chained filler
    # e.g. "can you look up" → strip "can you " → strip "look up "
    changed = True
    while changed:
        changed = False
        for prefix in _CONVERSATIONAL_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
                break

    result = text.strip().rstrip("?.,!")
    print(f"[WebSearch] Cleaned query: '{raw[:50]}' -> '{result}'")
    return result


# ── Public entry point ────────────────────────────────────────────────

def search_web(query: str) -> str:
    """Search the web for a query using DuckDuckGo and return plain text results.

    Scrapes the DuckDuckGo HTML endpoint and extracts the top 3 result
    snippets as clean plain text. Results are cached for 15 minutes.

    The raw query is cleaned first — Aria name variants and conversational
    filler are stripped to produce a search-engine-friendly string.

    Args:
        query: Natural language search query from the user.

    Returns:
        Plain text string containing the top search result snippets,
        or an error message string if scraping fails.
    """
    query = clean_query(query)  # Strip name and conversational filler
    cache_key = _cache_key(query)
    cache = _load_cache()

    # Return cached result if still valid
    if cache_key in cache and _is_cache_valid(cache[cache_key]):
        print(f"[WebSearch] Returning cached result for: {query[:50]}")
        return cache[cache_key]["result"]

    print(f"[WebSearch] Scraping DuckDuckGo for: {query[:50]}...")

    # Try Playwright first, fall back to httpx
    try:
        result = _scrape_ddg_playwright(query)
    except Exception as e:
        print(f"[WebSearch] Playwright failed ({e}), falling back to httpx...")
        try:
            result = _scrape_ddg_httpx(query)
        except Exception as e2:
            print(f"[WebSearch] httpx fallback also failed: {e2}")
            return f"Sorry Chan, I couldn't find results for that query right now."

    if not result or not result.strip():
        return "I searched but didn't find any relevant results, Chan."

    # Cache the result
    cache[cache_key] = {
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }
    _save_cache(cache)

    return result


# ── DuckDuckGo scrapers ──────────────────────────────────────────────

def _scrape_ddg_playwright(query: str) -> str:
    """Scrape DuckDuckGo HTML endpoint using Playwright headless browser.

    Args:
        query: The search query string.

    Returns:
        Plain text snippets from the top results.

    Raises:
        Exception: If Playwright fails or no results are found.
    """
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=15000)
        html = page.content()
        browser.close()

    return _extract_snippets(html)


def _scrape_ddg_httpx(query: str) -> str:
    """Fallback DuckDuckGo scrape using httpx (no browser).

    Args:
        query: The search query string.

    Returns:
        Plain text snippets from the top results.

    Raises:
        Exception: If the HTTP request fails.
    """
    import httpx

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    response = httpx.get(
        url,
        timeout=10.0,
        headers={"User-Agent": "Aria/1.0"},
        follow_redirects=True,
    )
    response.raise_for_status()
    return _extract_snippets(response.text)


def _extract_snippets(html: str) -> str:
    """Extract the top result snippets from DuckDuckGo HTML response.

    Args:
        html: Raw HTML string from DuckDuckGo.

    Returns:
        Clean plain text with the top snippets, one per line.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    snippets = []

    # DuckDuckGo HTML results are in <a class="result__snippet"> elements
    for result in soup.select(".result__snippet")[:MAX_SNIPPETS]:
        text = result.get_text(strip=True)
        if text:
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)
            snippets.append(text)

    if not snippets:
        # Try fallback selectors
        for result in soup.select(".result__body")[:MAX_SNIPPETS]:
            text = result.get_text(strip=True)
            if text:
                text = re.sub(r"\s+", " ", text)
                snippets.append(text)

    return "\n".join(snippets) if snippets else ""


# ── Cache helpers ─────────────────────────────────────────────────────

def _cache_key(query: str) -> str:
    """Generate a stable cache key from a query string.

    Args:
        query: The search query.

    Returns:
        A short hash string suitable as a dict key.
    """
    normalised = query.lower().strip()
    return hashlib.md5(normalised.encode()).hexdigest()[:12]


def _load_cache() -> dict:
    """Load the web cache from disk.

    Returns:
        The cached data dict, or empty dict if missing/corrupt.
    """
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    """Save the web cache to disk.

    Args:
        cache: The cache dict to persist.
    """
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _is_cache_valid(entry: dict) -> bool:
    """Check if a cache entry is still within the TTL window.

    Args:
        entry: Cache entry dict with a 'timestamp' key.

    Returns:
        True if the entry is still valid, False if expired.
    """
    try:
        cached_at = datetime.fromisoformat(entry["timestamp"])
        return datetime.now() - cached_at < timedelta(minutes=CACHE_TTL_MINUTES)
    except Exception:
        return False

