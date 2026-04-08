"""
Aria Web Search Module
======================
Web scraping module for Aria. Provides live data access via
Playwright headless browser. Built with a scalable dispatch
architecture — add new handlers by registering them in HANDLERS.

Current handlers:
    - weather: Scrapes wttr.in for Birmingham weather

Future handlers (not yet implemented):
    - news
    - job_listings
    - general_search
"""

import json
import os
from datetime import datetime, timedelta
from config import DATA_DIR

CACHE_FILE = os.path.join(DATA_DIR, "web_cache.json")
CACHE_TTL_MINUTES = 30


# ── Handler registry ────────────────────────────────────────────────
# To add a new handler:
#   1. Write a _handle_<name>(**kwargs) function below
#   2. Add an entry here: "name": _handle_<name>
# The dispatch function search_web() does the rest.

HANDLERS: dict = {}


def _register(name: str):
    """Decorator to register a handler function in HANDLERS.

    Args:
        name: The query type this handler responds to.
    """
    def decorator(fn):
        HANDLERS[name] = fn
        return fn
    return decorator


def search_web(query_type: str, **kwargs) -> str:
    """Route a query to the appropriate web handler.

    Args:
        query_type: The type of query (e.g. 'weather', 'news').
        **kwargs: Additional arguments passed to the handler.

    Returns:
        A plain-text string result, or an error message if scraping fails.
    """
    handler = HANDLERS.get(query_type)
    if not handler:
        return f"No web handler registered for: {query_type}"

    return handler(**kwargs)


# ── Cache helpers ───────────────────────────────────────────────────

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


# ── Weather handler ─────────────────────────────────────────────────

@_register("weather")
def _handle_weather(location: str = "Birmingham") -> str:
    """Scrape current weather from wttr.in for a given location.

    Uses a 30-minute cache to avoid repeated scraping.
    Tries Playwright first, falls back to httpx if unavailable.

    Args:
        location: The city name to fetch weather for.

    Returns:
        A plain-text weather summary string.
    """
    cache_key = f"weather_{location.lower()}"
    cache = _load_cache()

    # Return cached result if still valid
    if cache_key in cache and _is_cache_valid(cache[cache_key]):
        print(f"[Aria] Weather: returning cached result for {location}.")
        return cache[cache_key]["result"]

    print(f"[Aria] Weather: scraping wttr.in for {location}...")

    # Try Playwright first, fall back to httpx
    try:
        result = _scrape_wttr_playwright(location)
    except Exception as e:
        print(f"[Aria] Playwright failed ({e}), falling back to httpx...")
        try:
            result = _scrape_wttr_httpx(location)
        except Exception as e2:
            print(f"[Aria] httpx fallback also failed: {e2}")
            return f"Sorry Chan, I couldn't fetch the weather for {location} right now."

    # Cache the result
    cache[cache_key] = {
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }
    _save_cache(cache)

    return result


def _scrape_wttr_playwright(location: str) -> str:
    """Scrape weather using a Playwright headless browser with JSON API.

    Uses wttr.in's JSON endpoint and extracts a human-readable summary.

    Args:
        location: City name to query.

    Returns:
        A plain-text weather summary.

    Raises:
        Exception: If Playwright is not installed or scraping fails.
    """
    import json as _json
    from playwright.sync_api import sync_playwright

    url = f"https://wttr.in/{location}?format=j1"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=10000)
        content = page.inner_text("body").strip()
        browser.close()

    if not content:
        raise ValueError(f"Empty response from wttr.in for {location}")

    data = _json.loads(content)
    current = data["current_condition"][0]
    temp_c = current["temp_C"]
    desc = current["weatherDesc"][0]["value"]
    humidity = current["humidity"]
    wind_mph = current["windspeedMiles"]
    return f"{location}: {desc}, {temp_c}°C, humidity {humidity}%, wind {wind_mph}mph"


def _scrape_wttr_httpx(location: str) -> str:
    """Fallback weather scrape using httpx (no browser needed for wttr.in format=3).

    Args:
        location: City name to query.

    Returns:
        A plain-text weather summary.

    Raises:
        Exception: If the HTTP request fails.
    """
    import httpx

    url = f"https://wttr.in/{location}?format=3"
    response = httpx.get(url, timeout=8.0, headers={"User-Agent": "Aria/1.0"})
    response.raise_for_status()
    return response.text.strip()
