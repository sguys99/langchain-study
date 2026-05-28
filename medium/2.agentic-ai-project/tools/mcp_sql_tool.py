# tools/mcp_sql_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# MCP-based SQL Tool
#
# 기존 sql_tool.py 가 sqlite3 로 DB 에 직접 연결하는 반면, 이 도구는
# tools/mcp_servers/sqlite_server.py 를 별도 STDIO subprocess 로 띄우고
# FastMCP Client 로 호출한다.
#
# Flow:
#   1) 첫 호출 시 백그라운드 이벤트 루프 + MCP Client 부팅 (subprocess 1회)
#   2) list_tables → describe_table × N → sample_table × N 로 스키마 introspection
#      (모듈 레벨 캐시; 프로세스 수명 동안 1회만 fetch)
#   3) Claude 로 자연어 → SQL 생성 (동적 스키마를 system prompt 에 주입)
#   4) MCP execute_sql 도구 호출 → 결과를 dict 로 반환
#
# 반환 dict 는 sql_tool.run_sql_tool 와 키 호환 (sql/rows/table_md/error)
# 이므로 agent/synthesise_node 의 SQL 분기를 그대로 재사용한다.
# 추가로 "mcp_tool_calls" 키에 호출된 MCP 도구 이름 목록을 담는다.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import atexit
import threading
from concurrent.futures import Future
from typing import Any

from anthropic import Anthropic
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from config import (
    ANTHROPIC_API_KEY,
    BASE_DIR,
    DB_PATH,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from observability import trace
from tools.sql_tool import _rows_to_markdown  # 기존 유틸 재사용

# ── MCP 서버 스크립트 경로 ──────────────────────────────────────────────────
_SERVER_SCRIPT = f"{BASE_DIR}/tools/mcp_servers/sqlite_server.py"

# ── Anthropic 클라이언트 ────────────────────────────────────────────────────
_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── 백그라운드 이벤트 루프 + persistent MCP Client ──────────────────────────
# 동기 코드(에이전트)에서 async MCP Client 를 호출해야 하므로, 별도 스레드에서
# 영구 이벤트 루프를 돌리고 asyncio.run_coroutine_threadsafe 로 작업을 제출한다.
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

_client: Client | None = None
_client_ready = False
_schema_cache: dict | None = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """백그라운드 이벤트 루프를 lazy 하게 시작하고 반환한다."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None or not _loop.is_running():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop.run_forever,
                name="mcp-sql-loop",
                daemon=True,
            )
            _loop_thread.start()
    return _loop


def _run_coro(coro, timeout: float = 30.0):
    """동기 컨텍스트에서 백그라운드 루프에 코루틴을 제출하고 결과를 기다린다."""
    loop = _ensure_loop()
    fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=timeout)


async def _ensure_client() -> Client:
    """MCP Client (와 subprocess) 를 1회만 부팅한다."""
    global _client, _client_ready
    if _client is None or not _client_ready:
        transport = PythonStdioTransport(script_path=_SERVER_SCRIPT)
        _client = Client(transport)
        await _client.__aenter__()
        _client_ready = True
    return _client


async def _call_tool_async(name: str, args: dict[str, Any] | None = None):
    """MCP 도구를 호출하고 CallToolResult 의 data 를 반환한다."""
    client = await _ensure_client()
    result = await client.call_tool(name, args or {})
    # FastMCP 3.x: structured 반환은 result.data 로 노출 (structured_content 가 dict).
    # 일부 도구가 list 를 반환할 때는 result.data 가 그대로 list.
    return result.data


def _call_tool(name: str, args: dict[str, Any] | None = None):
    return _run_coro(_call_tool_async(name, args))


def _shutdown():
    """프로세스 종료 시 client / loop 를 정리한다."""
    global _client, _client_ready, _loop
    if _client is not None and _loop is not None and _loop.is_running():
        try:
            fut = asyncio.run_coroutine_threadsafe(
                _client.__aexit__(None, None, None), _loop
            )
            fut.result(timeout=5)
        except Exception:
            pass
        _client = None
        _client_ready = False
    if _loop is not None and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)


atexit.register(_shutdown)


# ─────────────────────────────────────────────────────────────────────────────
# 스키마 introspection (MCP 도구 list_tables / describe_table / sample_table)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_schema() -> dict:
    """MCP 도구로 전체 스키마를 fetch 한다. 첫 호출만 비용 발생."""
    schema: dict = {}
    tables: list[str] = _call_tool("list_tables")
    for tbl in tables:
        desc = _call_tool("describe_table", {"table_name": tbl})
        sample = _call_tool("sample_table", {"table_name": tbl, "n": 3})
        schema[tbl] = {
            "columns": desc.get("columns", []),
            "sample":  sample.get("rows", []),
        }
    return schema


def _get_schema() -> dict:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = _fetch_schema()
    return _schema_cache


def _build_dynamic_system_prompt() -> str:
    """MCP 로 조회한 실시간 스키마를 system prompt 로 변환한다."""
    schema = _get_schema()
    lines: list[str] = [
        "You are an expert SQLite query writer for an e-commerce analytics database.",
        "The following schema was introspected at runtime via MCP tools.",
        "",
    ]
    for table, info in schema.items():
        lines.append(f"TABLE: {table}")
        for col in info["columns"]:
            pk = "  PK" if col.get("pk") else ""
            null = "" if col.get("nullable") else "  NOT NULL"
            lines.append(f"  {col['name']:30} {col['type']}{pk}{null}")
        if info["sample"]:
            lines.append(f"  -- sample row: {info['sample'][0]}")
        lines.append("")
    lines.extend([
        "RULES:",
        "  - Output ONLY the raw SQLite SQL query, no markdown, no explanation.",
        "  - Use table aliases for readability.",
        "  - For date filtering use: strftime('%Y-%m', order_purchase_timestamp) = '2018-01'",
        "  - Always add LIMIT 50 unless the user asks for all rows.",
        "  - Never use SELECT *; always name columns explicitly.",
    ])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SQL 생성
# ─────────────────────────────────────────────────────────────────────────────
def _generate_sql(user_question: str, conversation_history: list[dict]) -> str:
    """Claude 로 자연어 질문을 SQLite SQL 로 번역한다."""
    system_prompt = _build_dynamic_system_prompt()
    messages = list(conversation_history[-6:])
    messages.append({"role": "user", "content": user_question})

    response = _anthropic.messages.create(
        model=LLM_MODEL,
        system=system_prompt,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=512,
    )
    return response.content[0].text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────────────────────────────────────
@trace(span_type="TOOL", model=LLM_MODEL, attributes={
    "db.type": "sqlite",
    "db.path": DB_PATH,
    "mcp.transport": "stdio",
    "mcp.server": "sqlite-ecommerce",
})
def run_mcp_sql_tool(user_question: str, conversation_history: list[dict]) -> dict:
    """
    MCP 기반 SQL 도구의 진입점.

    Returns dict:
      {
        "sql":             str,
        "rows":            list[dict],
        "table_md":        str,
        "error":           str | None,
        "mcp_tool_calls":  list[str],   # 이번 호출에서 사용한 MCP 도구 이름
      }
    """
    tool_calls: list[str] = []

    # 1) 스키마 캐시 (첫 호출만 list_tables + describe_table*N + sample_table*N)
    if _schema_cache is None:
        tool_calls.extend(["list_tables", "describe_table", "sample_table"])
    _get_schema()

    # 2) Claude 로 SQL 생성
    sql = _generate_sql(user_question, conversation_history)

    # 3) MCP execute_sql 도구 호출
    tool_calls.append("execute_sql")
    exec_result = _call_tool("execute_sql", {"query": sql})

    rows = exec_result.get("rows", [])
    error = exec_result.get("error")
    actual_sql = exec_result.get("sql", sql)

    return {
        "sql":            actual_sql,
        "rows":           rows,
        "table_md":       _rows_to_markdown(rows) if not error else f"_SQL error: {error}_",
        "error":          error,
        "mcp_tool_calls": tool_calls,
    }
