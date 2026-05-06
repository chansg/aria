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
from core.logger import get_logger

log = get_logger(__name__)

CACHE_FILE = os.path.join(DATA_DIR, "web_cache.json")
CACHE_TTL_MINUTES = 30          # Bumped from 15 — covers weather caching too
MAX_SNIPPETS = 3

# Weather query keywords — trigger the wttr.in path before DuckDuckGo
_WEATHER_KEYWORDS = (
    "weather", "temperature", "forecast", "raining",
    "sunny", "cloudy", "cold", "warm", "hot", "degrees",
    "snow", "umbrella",
)


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

_CORRECTION_PREFIX_PATTERNS = (
    r"^(?:no|nope|nah)[,\s]+",
    r"^(?:actually|sorry|correction)[,\s]+",
    r"^(?:i meant|i mean|what i meant was|what i mean is)[,\s]+",
)

# Common Whisper mishear corrections for command words.
# Multi-word patterns ensure we only correct in command contexts —
# bare "such" stays untouched ("such a good day" should not be modified),
# but "such the", "such for" etc. are clearly mistranscribed "search".
_MISHEAR_CORRECTIONS = {
    "such the":   "search the",
    "such for":   "search for",
    "such about": "search about",
    "such up":    "look up",
    "surge the":  "search the",
    "surge for":  "search for",
    "searcher":   "search",
    "surging":    "searching",
}


def clean_query(raw: str) -> str:
    """Strip Aria name variants, conversational filler, and correct mishears.

    Produces a clean, search-engine-friendly query string from raw
    transcribed speech. Applied before every DuckDuckGo scrape and every
    wttr.in lookup.

    Args:
        raw: Raw transcribed text from the user, e.g.
             'Hello Aria, what is the weather in Slough today?'

    Returns:
        Cleaned, corrected query string ready for web search.

    Examples:
        >>> clean_query('Hello Aria, what is the weather in Slough?')
        'weather in Slough'
        >>> clean_query('Aria such the latest news in AI')
        'search the latest news in ai'
    """
    text = raw.lower().strip().rstrip("?.,!")

    # Strip Aria name variants, corrections, and conversational prefixes.
    # Loop to catch chained filler, e.g. "Aria, no, can you look up...".
    changed = True
    while changed:
        changed = False
        for prefix in sorted(_ARIA_PREFIXES, key=len, reverse=True):
            if text.startswith(prefix):
                text = text[len(prefix):].strip().lstrip(",").strip()
                changed = True
                break
        if changed:
            continue
        for pattern in _CORRECTION_PREFIX_PATTERNS:
            new_text = re.sub(pattern, "", text).strip()
            if new_text != text:
                text = new_text
                changed = True
                break
        if changed:
            continue
        for prefix in _CONVERSATIONAL_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
                break

    # Apply Whisper mishear corrections (multi-word patterns to avoid false positives)
    for wrong, correct in _MISHEAR_CORRECTIONS.items():
        if wrong in text:
            text = text.replace(wrong, correct)
            log.info("Mishear corrected: %r -> %r", wrong, correct)

    result = text.strip().rstrip("?.,!")
    log.info("Cleaned query: %r -> %r", raw[:50], result)
    return result


def _snippet_count(result: str) -> int:
    """Count non-empty web context lines for diagnostics."""
    return sum(1 for line in result.splitlines() if line.strip())


def _log_web_context(query: str, result: str, *, source: str) -> None:
    """Log web context size without dumping snippet content."""
    log.info(
        "Web context ready: source=%s query=%r chars=%d snippets=%d",
        source,
        query[:80],
        len(result),
        _snippet_count(result),
    )


# ── Weather: wttr.in handler ──────────────────────────────────────────

def _fetch_wttr(location: str) -> str:
    """Fetch weather data from wttr.in for a given location.

    Uses wttr.in's JSON API which returns structured weather data
    including current conditions and a 3-day forecast. More reliable
    than scraping DuckDuckGo for weather queries — Gemini gets concrete
    numbers to reason over instead of HTML snippets.

    Args:
        location: City or location name extracted from user query.

    Returns:
        Formatted plain text weather summary, or empty string if fetch fails.
    """
    import httpx
    try:
        url = f"https://wttr.in/{location}?format=j1"
        resp = httpx.get(url, timeout=8.0, headers={"User-Agent": "Aria/1.0"})
        resp.raise_for_status()
        data = resp.json()

        current   = data["current_condition"][0]
        temp_c    = current["temp_C"]
        feels_c   = current["FeelsLikeC"]
        desc      = current["weatherDesc"][0]["value"]
        humidity  = current["humidity"]
        wind_kmph = current["windspeedKmph"]

        today    = data["weather"][0]
        tomorrow = data["weather"][1]

        today_max     = today["maxtempC"]
        today_min     = today["mintempC"]
        tomorrow_max  = tomorrow["maxtempC"]
        tomorrow_min  = tomorrow["mintempC"]
        tomorrow_desc = tomorrow["hourly"][4]["weatherDesc"][0]["value"]

        summary = (
            f"Current weather in {location}: {desc}, {temp_c}°C "
            f"(feels like {feels_c}°C). Humidity {humidity}%, "
            f"wind {wind_kmph} km/h. "
            f"Today: high {today_max}°C, low {today_min}°C. "
            f"Tomorrow: {tomorrow_desc}, high {tomorrow_max}°C, "
            f"low {tomorrow_min}°C."
        )

        log.info("wttr.in fetch successful for: %s", location)
        return summary

    except Exception as e:
        log.warning("wttr.in fetch failed (%s) — falling back to DuckDuckGo.", e)
        return ""


def _extract_location(query: str) -> str:
    """Extract a location name from a weather query.

    Looks for patterns like 'in Slough', 'in London tomorrow',
    'for Birmingham'. Falls back to 'London' if nothing matches.

    Args:
        query: Cleaned user query string.

    Returns:
        Extracted location name string.
    """
    # Match "in <Location>" or "for <Location>", stopping at time qualifiers
    match = re.search(
        r'\b(?:in|for)\s+([A-Za-z][A-Za-z\s]+?)(?:\s+today|\s+tomorrow|\s+this\s+week|\s+now|$)',
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().title()

    # Fall back: standalone capitalised non-stopword
    stopwords = {"aria", "what", "tell", "search", "find", "weather",
                 "the", "is", "in", "for", "today", "tomorrow"}
    for word in query.split():
        if word and word[0].isupper() and word.lower() not in stopwords:
            return word

    return "London"  # default fallback


# ── Public entry point ────────────────────────────────────────────────

def search_web(query: str) -> str:
    """Search the web for a query and return plain text results.

    For weather queries, fetches from wttr.in directly for structured data.
    For all other queries, scrapes the DuckDuckGo HTML endpoint and extracts
    the top 3 result snippets. Results are cached for CACHE_TTL_MINUTES.

    The raw query is cleaned first — Aria name variants, conversational
    filler, and known Whisper mishears are normalised before lookup.

    Args:
        query: Natural language search query from the user.

    Returns:
        Plain text string for Gemini to reason over,
        or a graceful error string if every backend fails.
    """
    query = clean_query(query)  # Strip name, filler, mishears
    cache_key = _cache_key(query)
    cache = _load_cache()

    # Return cached result if still valid
    if cache_key in cache and _is_cache_valid(cache[cache_key]):
        log.info("Returning cached result for: %s", query[:50])
        result = cache[cache_key]["result"]
        _log_web_context(query, result, source="cache")
        return result

    # Weather queries → wttr.in for structured data
    is_weather = any(kw in query.lower() for kw in _WEATHER_KEYWORDS)
    result = ""
    if is_weather:
        location = _extract_location(query)
        result = _fetch_wttr(location)

    # Fall back to DuckDuckGo if wttr.in failed or query is non-weather
    if not result:
        log.info("Scraping DuckDuckGo for: %s...", query[:50])
        try:
            result = _scrape_ddg_playwright(query)
        except Exception as e:
            log.warning("Playwright failed (%s), falling back to httpx...", e)
            try:
                result = _scrape_ddg_httpx(query)
            except Exception as e2:
                log.error("httpx fallback also failed: %s", e2)
                return "Sorry Chan, I couldn't find results for that query right now."

    if not result or not result.strip():
        return "I searched but didn't find any relevant results, Chan."

    _log_web_context(query, result, source="fetch")

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

