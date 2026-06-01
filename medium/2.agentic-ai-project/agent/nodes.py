# agent/nodes.py
# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node functions.  Each node receives the full AgentState dict and
# returns a partial dict with only the keys it updates.
# ─────────────────────────────────────────────────────────────────────────────

import mlflow
from anthropic import Anthropic
from agent.router import route_question
from tools.sql_tool        import run_sql_tool
from tools.mcp_sql_tool    import run_mcp_sql_tool
from tools.rag_tool        import run_rag_tool
from tools.web_search_tool import run_web_search_tool
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, ANTHROPIC_API_KEY
from observability import trace, set_attrs  # Import observability utilities

client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Router
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: dict) -> dict:
    """
    Classify the user message and set state["route"].
    """
    routing = route_question(
        user_message=state["user_message"],
        conversation_history=state["conversation_history"],
    )
    print(f"\n[router] → {routing.route.upper()}  |  {routing.reason}")
    return {
        "route":        routing.route,
        "route_reason": routing.reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2a — SQL Tool Node
# ─────────────────────────────────────────────────────────────────────────────

def sql_node(state: dict) -> dict:
    """
    Generate & execute a SQL query, store raw result in state.
    """
    print("[sql_node] Generating and executing SQL …")
    result = run_sql_tool(
        user_question=state["user_message"],
        conversation_history=state["conversation_history"],
    )
    print(f"[sql_node] SQL: {result['sql']}")
    if result["error"]:
        print(f"[sql_node] ERROR: {result['error']}")
    else:
        print(f"[sql_node] {len(result['rows'])} rows returned.")
    return {"sql_result": result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2a-mcp — MCP SQL Tool Node
# ─────────────────────────────────────────────────────────────────────────────

def mcp_sql_node(state: dict) -> dict:
    """
    Generate & execute a SQL query via the MCP SQLite server,
    store raw result in state.
    """
    print("[mcp_sql_node] Generating and executing SQL via MCP …")
    result = run_mcp_sql_tool(
        user_question=state["user_message"],
        conversation_history=state["conversation_history"],
    )
    print(f"[mcp_sql_node] SQL: {result['sql']}")
    if result["error"]:
        print(f"[mcp_sql_node] ERROR: {result['error']}")
    else:
        n_calls = len(result.get("mcp_tool_calls", []))
        print(
            f"[mcp_sql_node] {len(result['rows'])} rows via "
            f"{n_calls} MCP tool calls."
        )
    return {"mcp_sql_result": result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2b — RAG Tool Node
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: dict) -> dict:
    """
    Retrieve relevant PDF chunks and synthesise an answer.
    """
    print("[rag_node] Retrieving document chunks …")
    result = run_rag_tool(
        user_question=state["user_message"],
        conversation_history=state["conversation_history"],
    )
    print(f"[rag_node] {len(result['chunks'])} chunks retrieved from: {result['sources']}")
    return {"rag_result": result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2c — Web Search Tool Node
# ─────────────────────────────────────────────────────────────────────────────

def web_search_node(state: dict) -> dict:
    """
    Run Tavily search and synthesise an answer.
    """
    print("[web_node] Searching the web …")
    result = run_web_search_tool(
        user_question=state["user_message"],
        conversation_history=state["conversation_history"],
    )
    print(f"[web_node] {len(result['results'])} results for query: '{result['query']}'")
    return {"web_search_result": result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Answer Synthesiser
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHESISE_SYSTEM = """\
당신은 친절하고 정확한 이커머스 분석 어시스턴트입니다.
사용자에게 명확하고 잘 정리된 최종 답변을 전달하는 것이 당신의 역할입니다.

[작성 지침]
- 간결하되 빠짐없이 작성하세요.
- SQL 결과를 활용할 때는 핵심 수치를 먼저 요약하고, 이어서 설명을 덧붙이세요.
- 문서(RAG)에서 가져온 내용이라면 출처 문서명을 함께 표기하세요.
- 웹 검색 결과를 활용한다면 참고한 URL을 반드시 함께 노출하세요.
- 가독성이 좋아진다면 마크다운(불릿, 굵게, 표)을 적극 사용하세요.
- 항상 정중하고 전문적인 어조를 유지하세요.
"""


def _call_synthesiser(user_content: str, history: list[dict]) -> str:
    """Anthropic Claude로 최종 답변을 합성한다."""
    messages = list(history[-4:])
    messages.append({"role": "user", "content": user_content})
    response = client.messages.create(
        model=LLM_MODEL,
        system=_SYNTHESISE_SYSTEM,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.content[0].text.strip()


@trace(span_type="CHAIN", model=LLM_MODEL)
def synthesise_node(state: dict) -> dict:
    """
    Combine tool output into a polished final answer using Claude.
    - For SQL: synthesize narrative from raw rows
    - For RAG: synthesize answer from retrieved document chunks
    - For Web Search: synthesize answer from web search results
    """
    route = state.get("route", "")

    # ── SQL / MCP SQL: tool returns raw rows — synthesise a narrative ───────
    if route in ("sql", "mcp_sql"):
        result_key = "sql_result" if route == "sql" else "mcp_sql_result"
        via        = "직접 연결" if route == "sql" else "MCP"
        sql_result = state.get(result_key, {}) or {}
        sql_query  = sql_result.get("sql", "")
        table_md   = sql_result.get("table_md", "_결과 없음_")
        error      = sql_result.get("error")

        if error:
            content = (
                f"SQL 쿼리 실행 중 오류가 발생했습니다 (via {via}):\n\n"
                f"```sql\n{sql_query}\n```\n\n"
                f"오류: {error}\n\n"
                "질문을 다시 표현해 주세요."
            )
        else:
            content = (
                f"실행된 SQL 쿼리 (via {via}):\n```sql\n{sql_query}\n```\n\n"
                f"결과:\n{table_md}"
            )

        user_content = (
            f"원래 질문: {state['user_message']}\n\n"
            f"쿼리 결과:\n{content}\n\n"
            "위 결과를 비즈니스 관점에서 명확하고 이해하기 쉽게 요약해 주세요."
        )

        # Capture prompts as span inputs
        span = mlflow.get_current_active_span()
        if span:
            span.set_inputs({
                "system_prompt": _SYNTHESISE_SYSTEM,
                "user_prompt": user_content,
                "synthesis_type": route,
                "user_message": state['user_message']
            })

        answer = _call_synthesiser(user_content, state["conversation_history"])

    # ── RAG: synthesize answer from retrieved document chunks ───────────────
    elif route == "rag":
        rag_result = state.get("rag_result", {})
        chunks     = rag_result.get("chunks", [])
        sources    = rag_result.get("sources", [])

        if not chunks:
            answer = "불러온 문서에서 관련 정보를 찾지 못했습니다."
        else:
            # Build context block from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(
                    f"[발췌 {i} — {chunk['source']} (점수: {chunk['score']:.3f})]:\n"
                    f"{chunk['text']}"
                )
            context = "\n\n---\n\n".join(context_parts)

            user_content = (
                f"문서 발췌:\n\n{context}\n\n"
                f"질문: {state['user_message']}"
            )

            # Capture prompts as span inputs
            span = mlflow.get_current_active_span()
            if span:
                span.set_inputs({
                    "system_prompt": _SYNTHESISE_SYSTEM,
                    "user_prompt": user_content,
                    "synthesis_type": "rag",
                    "retrieval_sources": ", ".join(sources) if sources else "none",
                    "user_message": state['user_message']
                })

            answer = _call_synthesiser(user_content, state["conversation_history"])

            if sources:
                answer += f"\n\n_출처: {', '.join(sources)}_"

    # ── Web Search: synthesize answer from search results ───────────────────
    elif route == "web_search":
        web_result = state.get("web_search_result", {})
        results    = web_result.get("results", [])
        query      = web_result.get("query", "")

        if not results:
            answer = "웹 검색 결과가 없습니다."
        else:
            # Format results for context
            result_blocks = []
            for i, r in enumerate(results, 1):
                block = (
                    f"[검색 결과 {i}]\n"
                    f"제목: {r.get('title', 'N/A')}\n"
                    f"URL:  {r.get('url', 'N/A')}\n"
                    f"요약: {r.get('content', '')}"
                )
                result_blocks.append(block)
            formatted_results = "\n\n".join(result_blocks)

            user_content = (
                f"'{query}'에 대한 웹 검색 결과:\n\n{formatted_results}\n\n"
                f"원래 질문: {state['user_message']}\n\n"
                "위 결과를 바탕으로 명확하고 정확한 답변을 작성하세요. "
                "사용자가 검증할 수 있도록 출처 URL을 반드시 함께 명시해 주세요."
            )

            # Capture prompts as span inputs
            span = mlflow.get_current_active_span()
            if span:
                span.set_inputs({
                    "system_prompt": _SYNTHESISE_SYSTEM,
                    "user_prompt": user_content,
                    "synthesis_type": "web_search",
                    "web_search_query": query,
                    "user_message": state['user_message']
                })

            answer = _call_synthesiser(user_content, state["conversation_history"])

    else:
        answer = "질문에 답변할 방법을 결정하지 못했습니다."

    print(f"[synthesise_node] Answer ready ({len(answer)} chars).")
    return {"final_answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — History Updater
# ─────────────────────────────────────────────────────────────────────────────

def update_history_node(state: dict) -> dict:
    """
    Append the current turn to conversation_history.
    (Turn number is already incremented in session.ask())
    """
    history = list(state["conversation_history"])
    history.append({"role": "user",      "content": state["user_message"]})
    history.append({"role": "assistant", "content": state["final_answer"]})
    return {
        "conversation_history": history,
        "turn_number": state.get("turn_number", 0),
    }
