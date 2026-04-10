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


# ── Public entry point ────────────────────────────────────────────────

def search_web(query: str) -> str:
    """Search the web for a query using DuckDuckGo and return plain text results.

    Scrapes the DuckDuckGo HTML endpoint and extracts the top 3 result
    snippets as clean plain text. Results are cached for 15 minutes.

    Args:
        query: Natural language search query from the user.

    Returns:
        Plain text string containing the top search result snippets,
        or an error message string if scraping fails.
    """
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

