# agent/state.py
# ─────────────────────────────────────────────────────────────────────────────
# LangGraph 상태 정의.
# 모든 필드는 그래프를 통해 흐르며, 각 노드는 필요한 키만 읽고
# 업데이트할 키만 반환합니다.
# ─────────────────────────────────────────────────────────────────────────────

from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    # ── 입력 ──────────────────────────────────────────────────────────────────
    user_message: str
    conversation_history: list[dict]   # [{"role": "user"|"assistant", "content": str}]

    # ── 라우팅 ────────────────────────────────────────────────────────────────
    route:        str                  # "sql" | "mcp_sql" | "rag" | "web_search"
    route_reason: str                  # 라우터 LLM의 판단 근거

    # ── 도구 출력 (턴 당 하나의 슬롯만 채워짐) ────────────────────────────────
    sql_result:        Optional[dict]  # {sql, rows, table_md, error}
    mcp_sql_result:    Optional[dict]  # {sql, rows, table_md, error, mcp_tool_calls}
    rag_result:        Optional[dict]  # {answer, chunks, sources}
    web_search_result: Optional[dict]  # {answer, results, query}

    # ── 최종 합성 답변 ────────────────────────────────────────────────────────
    final_answer: str

    # ── 턴 메타데이터 ─────────────────────────────────────────────────────────
    turn_number:  int
