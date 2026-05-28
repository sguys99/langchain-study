# tools/mcp_sql_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# MCP SQL Tool: Claude tool-use 루프로 MCP SQLite 서버를 호출한다.
#
# 흐름:
#   1. MultiServerMCPClient 로 tools/mcp_servers/sqlite_server.py (stdio) 연결
#   2. 노출 도구(list_tables / describe_table / sample_table / execute_sql) 를
#      Anthropic tool-use 스키마로 변환해 Claude 에 바인딩
#   3. stop_reason == "tool_use" 인 동안 멀티턴 루프를 돌며 MCP 도구를 호출
#   4. execute_sql 결과에서 sql/rows/error 를 캡처해 직접 SQL 도구와 동일한
#      반환 형태({sql, rows, table_md, error}) + mcp_tool_calls 로 돌려준다
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json
import sys
from pathlib import Path

from anthropic import Anthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, ANTHROPIC_API_KEY
from observability import trace
from tools.sql_tool import _rows_to_markdown

client = Anthropic(api_key=ANTHROPIC_API_KEY)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = PROJECT_ROOT / "tools" / "mcp_servers" / "sqlite_server.py"

MAX_TOOL_TURNS = 8

MCP_SYSTEM_PROMPT = """\
You are an expert e-commerce SQL analyst with access to an MCP SQLite server.

Available tools:
  - list_tables()                 : discover available tables
  - describe_table(table_name)    : inspect column schema
  - sample_table(table_name, n)   : peek a few rows to understand data shape
  - execute_sql(query)            : run a READ-ONLY SELECT

Workflow:
  1. (Optional) Use list_tables / describe_table / sample_table to ground your
     query in the actual schema.  Skip these if the question is unambiguous.
  2. Compose a single SELECT and call execute_sql once.
  3. Stop calling tools as soon as you have the answer.

Rules:
  - Read-only SELECT only.  Never attempt INSERT/UPDATE/DELETE/DDL.
  - Always add an explicit LIMIT (e.g. LIMIT 50) unless the user asked for all rows.
  - Use table aliases and explicit column names; avoid SELECT *.
  - For date filtering: strftime('%Y-%m', order_purchase_timestamp) = '2018-01'.
"""


def _unwrap_tool_result(raw):
    """
    MCP 도구 결과를 파이썬 객체로 풀어낸다.

    langchain_mcp_adapters 가 도구 호출 결과를 LangChain content-block 리스트
    ( [{"type": "text", "text": "<json>"}, ...] ) 형태로 돌려주는 경우가 있어,
    text 블록을 합쳐 JSON 으로 파싱한다.  실패하면 원문 문자열을 그대로 반환.
    """
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        text = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return raw


def _to_anthropic_tools(mcp_tools) -> list[dict]:
    """LangChain MCP Tool 리스트를 Anthropic tool-use 스펙으로 변환."""
    spec = []
    for t in mcp_tools:
        schema = t.args_schema if isinstance(t.args_schema, dict) else {
            "type": "object", "properties": {},
        }
        spec.append({
            "name":         t.name,
            "description":  t.description or "",
            "input_schema": schema,
        })
    return spec


async def _run_agentic_loop(user_question: str, history: list[dict]) -> dict:
    """Claude tool-use 루프로 MCP 서버를 구동하고 결과를 모은다."""
    mcp_client = MultiServerMCPClient(connections={
        "sqlite": {
            "transport": "stdio",
            "command":   sys.executable,
            "args":      [str(SERVER_SCRIPT)],
        },
    })

    mcp_tools       = await mcp_client.get_tools()
    tool_map        = {t.name: t for t in mcp_tools}
    anthropic_tools = _to_anthropic_tools(mcp_tools)

    messages: list[dict] = list(history[-6:])
    messages.append({"role": "user", "content": user_question})

    mcp_tool_calls: list[dict] = []
    final_sql:   str            = ""
    final_rows:  list[dict]     = []
    final_error: str | None     = None

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model       = LLM_MODEL,
            system      = MCP_SYSTEM_PROMPT,
            messages    = messages,
            tools       = anthropic_tools,
            temperature = LLM_TEMPERATURE,
            max_tokens  = LLM_MAX_TOKENS,
        )

        if response.stop_reason != "tool_use":
            break

        # assistant 의 tool_use 블록을 보존
        messages.append({
            "role":    "assistant",
            "content": [b.model_dump() for b in response.content],
        })

        tool_result_blocks = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool = tool_map.get(block.name)
            if tool is None:
                payload = {"error": f"unknown tool: {block.name}"}
            else:
                raw     = await tool.ainvoke(block.input)
                payload = _unwrap_tool_result(raw)

            mcp_tool_calls.append({
                "tool":   block.name,
                "input":  block.input,
                "output": payload,
            })

            if block.name == "execute_sql" and isinstance(payload, dict):
                final_sql   = payload.get("sql") or block.input.get("query", "")
                final_rows  = payload.get("rows", [])
                final_error = payload.get("error")

            tool_result_blocks.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     json.dumps(payload, default=str, ensure_ascii=False),
            })

        messages.append({"role": "user", "content": tool_result_blocks})

    table_md = (
        f"_SQL error: {final_error}_" if final_error
        else _rows_to_markdown(final_rows)
    )

    return {
        "sql":            final_sql,
        "rows":           final_rows,
        "table_md":       table_md,
        "error":          final_error,
        "mcp_tool_calls": mcp_tool_calls,
    }


@trace(span_type="TOOL", model=LLM_MODEL, attributes={"db.type": "sqlite_mcp"})
def run_mcp_sql_tool(user_question: str, conversation_history: list[dict]) -> dict:
    """
    MCP 기반 SQL 도구 진입점.

    Returns dict:
      {
        "sql":            str,          # execute_sql 이 마지막으로 실행한 SQL
        "rows":           list[dict],   # execute_sql 결과 행
        "table_md":       str,          # 마크다운 테이블 또는 에러 메시지
        "error":          str | None,
        "mcp_tool_calls": list[dict],   # [{tool, input, output}, ...] — 호출 trail
      }
    """
    return asyncio.run(_run_agentic_loop(user_question, conversation_history))
