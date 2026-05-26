# agent/router.py
# ─────────────────────────────────────────────────────────────────────────────
# Router: Claude(Anthropic)로 사용자 메시지를 세 경로 중 하나로 분류합니다.
#
# Routes:
#   sql        → 로컬 SQLite 이커머스 데이터베이스 질의
#   rag        → 업로드된 PDF 문서 검색
#   web_search → Serper를 통한 실시간 웹 검색
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
from anthropic import Anthropic
from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    ANTHROPIC_API_KEY,
    ROUTES,
)
from observability import trace

client = Anthropic(api_key=ANTHROPIC_API_KEY)

ROUTER_SYSTEM = """\
당신은 이커머스 분석 어시스턴트의 라우팅 에이전트입니다.
사용자 메시지와 대화 이력을 보고, 어떤 도구를 사용할지 결정해야 합니다.

[사용 가능한 도구]
  sql        — 데이터베이스에 저장된 정보를 묻는 경우 사용합니다.
               예) 주문, 주문 항목, 고객, 결제, 상품, 매출, 집계/카운트,
                   추세 분석, 특정 주문/고객 조회 등 정형 이커머스 데이터.

  rag        — 업로드된 문서(PDF)에서 정보를 찾아야 하는 경우 사용합니다.
               예) 정책, 매뉴얼, 문서, FAQ 등. 업로드된 PDF가 없어도
                   문서 기반 질문이면 이 경로를 선택합니다.

  web_search — 다음과 같은 경우 사용합니다.
               • 최신 뉴스, 시사, 실시간 정보를 묻는 질문
               • 이커머스 산업 동향, 벤치마크 등 일반 지식
               • DB와 PDF 어디에서도 답할 수 없는 질문
               • "X란 무엇인가?", "X는 어떻게 동작하는가?" 같은 일반 상식

[출력 형식]
반드시 아래 JSON 형식만 출력하세요 (다른 텍스트 금지):
{
  "route": "<sql|rag|web_search>",
  "reason": "<한 문장으로 선택 이유 설명>"
}
"""


@trace(span_type="PARSER", model=LLM_MODEL)
def route_question(user_message: str, conversation_history: list[dict]) -> dict:
    """
    user_message를 ROUTES 중 하나로 분류합니다.

    반환값:
      {
        "route":  str,   # "sql", "rag", "web_search" 중 하나
        "reason": str,   # 라우팅 근거
      }
    """
    # 최근 4개 메시지(2턴)만 컨텍스트로 포함
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in conversation_history[-4:]
    ]
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=LLM_MODEL,
        system=ROUTER_SYSTEM,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=128,
    )

    raw_text = response.content[0].text.strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    raw = match.group(0) if match else raw_text

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # 파싱 실패 시 원문 텍스트에서 라우트 키워드 추출
        lowered = raw.lower()
        for route in ROUTES:
            if route in lowered or route.replace("_", " ") in lowered:
                return {"route": route, "reason": "extracted from malformed JSON"}
        return {"route": "web_search", "reason": "could not parse router output"}

    route = parsed.get("route", "web_search").lower().strip()
    if route not in ROUTES:
        route = "web_search"

    return {
        "route":  route,
        "reason": parsed.get("reason", ""),
    }
