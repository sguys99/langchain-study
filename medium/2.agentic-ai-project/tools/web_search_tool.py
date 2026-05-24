# tools/web_search_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# Web Search Tool: uses Serper API to fetch live web results.
#
# Serper returns structured results (title, link, snippet) — much
# cleaner than raw HTML scraping for LLM consumption.
#
# NOTE: This tool only retrieves raw search results. Answer synthesis
# is handled by the central synthesis node.
# ─────────────────────────────────────────────────────────────────────────────

import time
import requests
from config import SERPER_API_KEY, WEB_SEARCH_TOP_K, WEB_SEARCH_TIMEOUT
from observability import trace

SERPER_URL = "https://google.serper.dev/search"
_RETRY_STATUSES = {429, 502, 503, 504}


def _build_query(user_question: str, conversation_history: list[dict]) -> str:
    """Prefix the most recent prior user turn for multi-turn context."""
    if conversation_history:
        for turn in reversed(conversation_history):
            prior = turn.get("content")
            if turn.get("role") == "user" and prior and prior != user_question:
                return f"{prior}\n{user_question}"
    return user_question


def _parse_organic(data: dict) -> list[dict]:
    return [
        {
            "title":   item.get("title"),
            "url":     item.get("link"),
            "content": item.get("snippet"),
        }
        for item in data.get("organic", [])
    ]


@trace(span_type="TOOL", attributes={
    "api.provider": "serper.dev",
    "api.endpoint": "google.serper.dev/search",
})
def run_web_search_tool(user_question: str, conversation_history: list[dict]) -> dict:
    """
    Main entry point for the Web Search tool.

    Returns a dict:
      {
        "results": list[dict],   # [{title, url, content}, ...]
        "query":   str,          # the query actually sent to Serper
        "error":   str | None,   # failure reason, if any
      }
    """
    query = _build_query(user_question, conversation_history)

    if not SERPER_API_KEY:
        return {
            "results": [],
            "query":   query,
            "error":   "SERPER_API_KEY not configured",
        }

    headers = {
        "X-API-KEY":    SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": WEB_SEARCH_TOP_K}

    last_error: str | None = None
    for attempt in range(2):
        try:
            response = requests.post(
                SERPER_URL,
                json=payload,
                headers=headers,
                timeout=WEB_SEARCH_TIMEOUT,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))
            continue

        if response.status_code == 200:
            return {
                "results": _parse_organic(response.json()),
                "query":   query,
                "error":   None,
            }

        last_error = f"Serper HTTP {response.status_code}: {response.text[:200]}"
        if response.status_code not in _RETRY_STATUSES:
            break
        time.sleep(0.5 * (attempt + 1))

    return {
        "results": [],
        "query":   query,
        "error":   last_error,
    }
