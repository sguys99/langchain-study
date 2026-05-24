# agent/state.py
# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state definition.
# All fields flow through the graph; nodes read what they need and return
# only the keys they update.
# ─────────────────────────────────────────────────────────────────────────────

from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_message: str
    conversation_history: list[dict]   # [{"role": "user"|"assistant", "content": str}]

    # ── Routing ───────────────────────────────────────────────────────────────
    route:        str                  # "sql" | "rag" | "web_search"
    route_reason: str                  # explanation from the router LLM

    # ── Tool outputs (only one will be populated per turn) ───────────────────
    sql_result:        Optional[dict]  # {sql, rows, table_md, error}
    rag_result:        Optional[dict]  # {answer, chunks, sources}
    web_search_result: Optional[dict]  # {answer, results, query}

    # ── Final synthesised answer ──────────────────────────────────────────────
    final_answer: str

    # ── Turn metadata ─────────────────────────────────────────────────────────
    turn_number:  int