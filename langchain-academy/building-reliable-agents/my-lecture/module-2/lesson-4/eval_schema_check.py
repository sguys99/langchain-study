"""
평가자(Evaluator): 쿼리 전 스키마 확인 검사 (Schema-Before-Query Check)

에이전트가 query_database를 사용할 때, 데이터 쿼리를 실행하기 전에
먼저 데이터베이스 스키마를 확인(PRAGMA table_info 또는 sqlite_master를 통해)하는지
검사한다. 이를 통해 에이전트가 컬럼명을 무작정 추측하지 않도록 보장한다.
"""
import re


# 스키마 조회(스키마 확인) 쿼리로 간주할 정규식 패턴 목록
SCHEMA_PATTERNS = [
    r"PRAGMA\s+table_info",
    r"SELECT\s+.*FROM\s+sqlite_master",
    r"PRAGMA\s+database_list",
    r"\.schema",
]


def _is_schema_query(sql: str) -> bool:
    """해당 SQL이 스키마 확인용 쿼리이면 True를 반환한다."""
    for pattern in SCHEMA_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE):
            return True
    return False


def _extract_tool_calls(run) -> list[dict]:
    """run의 출력 메시지에서 도구 호출(tool call) 정보를 추출한다."""
    # run 객체가 outputs 속성을 가지면 그것을 사용하고, 아니면 dict로 접근한다
    run_outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    messages = run_outputs.get("messages", [])

    tool_calls = []
    for msg in messages:
        if isinstance(msg, dict):
            # 각 메시지의 tool_calls에서 함수 이름과 인자를 추출한다
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                tool_calls.append({
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                })
    return tool_calls


def schema_before_query(run, example) -> dict:
    """에이전트가 데이터 쿼리 전에 DB 스키마를 확인하면 1점, 아니면 0점을 부여한다.

    에이전트가 query_database를 한 번도 호출하지 않으면 1점을 부여한다(해당 없음).
    """
    tool_calls = _extract_tool_calls(run)

    # query_database 호출만 따로 모은다
    db_calls = [tc for tc in tool_calls if tc["name"] == "query_database"]

    # 데이터베이스 호출이 없음 — 확인할 대상이 없음
    if not db_calls:
        return {"score": 1, "comment": "No query_database calls — schema check not applicable"}

    # 첫 번째 데이터 쿼리(스키마 쿼리가 아닌) 이전에 스키마 쿼리가 나타나는지 확인한다
    seen_schema_check = False
    for tc in db_calls:
        sql = tc.get("arguments", "")
        if _is_schema_query(sql):
            seen_schema_check = True
        else:
            # 첫 번째 실제 데이터 쿼리 — 그 이전에 스키마 확인이 있었는가?
            if not seen_schema_check:
                return {
                    "score": 0,
                    "comment": f"Agent queried data without checking schema first. First query: {sql[:200]}",
                }
            break  # 첫 데이터 쿼리 이전에 스키마를 확인함 — 통과

    if seen_schema_check:
        return {"score": 1, "comment": "Agent checked schema before querying data"}

    return {"score": 1, "comment": "All query_database calls were schema inspections"}
