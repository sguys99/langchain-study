# tools/mcp_servers/sqlite_server.py
# ─────────────────────────────────────────────────────────────────────────────
# FastMCP 기반 SQLite MCP 서버
#
# 이커머스 SQLite DB(data/ecommerce.db) 를 MCP(Model Context Protocol) 로
# 외부에 노출한다. STDIO transport 로 동작하며, 자식 프로세스로 실행되어
# 부모(에이전트) 프로세스와 stdin/stdout JSON-RPC 로 통신한다.
#
# 노출 도구:
#   - execute_sql(query)            : read-only SQL 실행
#   - list_tables()                 : 테이블 목록 조회
#   - describe_table(table_name)    : 컬럼 스키마 조회
#   - sample_table(table_name, n)   : 샘플 행 조회
#
# 안전 장치:
#   - sqlite3 connection 에 mode=ro URI 강제 (OS 수준 read-only)
#   - execute_sql 은 DDL/DML 키워드를 정규식으로 1차 차단
#   - describe_table / sample_table 은 list_tables 결과와 화이트리스트 매칭
#
# 주의:
#   - stdout 은 MCP 프로토콜 전용이므로 print() 사용 금지. 로그는 stderr 로.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import re
import sqlite3
import sys
from pathlib import Path

# subprocess 로 실행되므로 sys.path 에 프로젝트 루트를 명시적으로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP  # noqa: E402
from config import DB_PATH  # noqa: E402

# ── 로깅: stderr 로만 출력 (stdout 오염 금지) ────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[mcp-sqlite] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

mcp = FastMCP("sqlite-ecommerce")

# DDL/DML 키워드 차단 정규식 (mode=ro 로 이미 차단되지만 명시적 422 응답 위함)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|PRAGMA)\b",
    re.IGNORECASE,
)


def _ro_conn() -> sqlite3.Connection:
    """Read-only 모드로 SQLite 연결을 연다."""
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _list_table_names() -> list[str]:
    """sqlite_master 에서 사용자 테이블 이름만 반환."""
    with _ro_conn() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# 도구 1: execute_sql
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def execute_sql(query: str) -> dict:
    """
    Run a READ-ONLY SQLite query against the e-commerce database and return rows.

    Use this for SELECT queries (aggregations, joins, filters, etc).
    Forbidden statements (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/REPLACE/
    TRUNCATE/ATTACH/PRAGMA) are rejected before execution.

    If no LIMIT clause is present, `LIMIT 500` is appended automatically.

    Returns dict with keys:
        rows      (list[dict]) - result rows
        row_count (int)        - number of rows returned
        sql       (str)        - the SQL actually executed (after auto-LIMIT)
        error     (str | None) - error message if execution failed
    """
    if _FORBIDDEN.search(query):
        log.warning("execute_sql rejected (forbidden keyword): %s", query[:120])
        return {
            "rows": [],
            "row_count": 0,
            "sql": query,
            "error": "forbidden statement: only SELECT is allowed",
        }

    sql = query.strip().rstrip(";")
    if "limit" not in sql.lower():
        sql = f"{sql} LIMIT 500"

    try:
        with _ro_conn() as conn:
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
        return {"rows": rows, "row_count": len(rows), "sql": sql, "error": None}
    except sqlite3.Error as exc:
        log.warning("execute_sql failed: %s | sql=%s", exc, sql[:200])
        return {"rows": [], "row_count": 0, "sql": sql, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 도구 2: list_tables
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def list_tables() -> list[str]:
    """
    List all user-defined tables in the e-commerce database.

    Call this first to discover available data before using describe_table or
    composing an execute_sql query.
    """
    return _list_table_names()


# ─────────────────────────────────────────────────────────────────────────────
# 도구 3: describe_table
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def describe_table(table_name: str) -> dict:
    """
    Return column schema for the given table.

    The table_name must be one of the names returned by list_tables();
    unknown names are rejected to prevent SQL identifier injection.

    Returns dict:
        table   (str)
        columns (list[dict]) - each {name, type, nullable, pk}
        error   (str | None)
    """
    valid = set(_list_table_names())
    if table_name not in valid:
        return {
            "table": table_name,
            "columns": [],
            "error": f"unknown table: {table_name!r}. Use list_tables() first.",
        }

    with _ro_conn() as conn:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()

    columns = [
        {
            "name":     r["name"],
            "type":     r["type"],
            "nullable": not bool(r["notnull"]),
            "pk":       bool(r["pk"]),
        }
        for r in rows
    ]
    return {"table": table_name, "columns": columns, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# 도구 4: sample_table
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def sample_table(table_name: str, n: int = 5) -> dict:
    """
    Return up to n sample rows from the given table (default 5, max 50).

    Useful for understanding the data shape before composing a query.
    The table_name must be one of the names returned by list_tables().

    Returns dict:
        table     (str)
        rows      (list[dict])
        row_count (int)
        error     (str | None)
    """
    valid = set(_list_table_names())
    if table_name not in valid:
        return {
            "table": table_name,
            "rows": [],
            "row_count": 0,
            "error": f"unknown table: {table_name!r}. Use list_tables() first.",
        }

    n = max(1, min(int(n), 50))
    with _ro_conn() as conn:
        cur = conn.execute(f"SELECT * FROM {table_name} LIMIT ?", (n,))
        rows = [dict(r) for r in cur.fetchall()]

    return {"table": table_name, "rows": rows, "row_count": len(rows), "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# 엔트리 포인트 (STDIO transport)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("starting sqlite-ecommerce MCP server (db=%s)", DB_PATH)
    mcp.run(transport="stdio")
