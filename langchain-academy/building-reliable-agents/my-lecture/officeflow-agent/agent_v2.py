import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import voyageai
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from langsmith import traceable, uuid7
from langsmith.wrappers import wrap_anthropic

# .env 파일에서 환경 변수(API 키 등)를 로드
load_dotenv()

# ===== 클라이언트 초기화 =====
# Anthropic: LLM 호출 (Claude 모델 - 대화 및 도구 호출)
#   - wrap_anthropic으로 감싸면 LangSmith가 모든 messages.create 호출을 자동 추적
# Voyage AI: 임베딩 생성 (knowledge base 의미 검색용)
#   - 임베딩 호출은 @traceable 데코레이터로 감싼 도구 함수 내부에서 추적됨
anthropic_client = wrap_anthropic(
    AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
)
voyage_client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))

# ===== 설정 =====
MODEL = "claude-sonnet-4-6"          # 채팅용 Claude 모델 (속도/비용/성능 균형)
EMBEDDING_MODEL = "voyage-4-lite"    # 임베딩 모델 (200M 무료 토큰 제공)
thread_id = str(uuid7())             # LangSmith 추적용 스레드 ID (UUIDv7)

# ===== 대화 히스토리 저장소 =====
# 메모리에 저장하는 단순 구조 → 프로세스가 재시작되면 모두 사라짐.
# 실제 서비스에서는 Redis, PostgreSQL 등 영속성 있는 저장소를 사용해야 함.
thread_store: dict[str, list] = {}

# ===== Knowledge Base 저장소 =====
# 앱 시작 시 한 번만 로드하여 메모리에 보관 (글로벌 변수)
# - knowledge_base_docs: (chunk 이름, 본문) 튜플 리스트
# - knowledge_base_embeddings: 각 chunk에 대응되는 임베딩 벡터(부동소수점 리스트)
knowledge_base_docs: List[Tuple[str, str]] = []
knowledge_base_embeddings: List[List[float]] = []

system_prompt = """당신은 북미 전역의 중소기업을 대상으로 종이 및 사무용품을 유통하는 OfficeFlow Supply Co.의 고객지원 전문가 Emma입니다.

당신의 역할:
당신은 고객 경험(Customer Experience) 팀의 일원으로, OfficeFlow에서 3년째 근무하고 있습니다. 도움이 되고, 효율적이며, 고객의 문제를 진심으로 해결하려는 자세로 잘 알려져 있습니다. 당신의 매니저는 모든 상호작용이 신뢰와 충성도를 쌓을 수 있는 기회라고 강조합니다.

도와드릴 수 있는 업무:
✓ 제품 정보 - 사무용품, 종이 제품, 필기구, 정리 도구, 책상 액세서리 카탈로그에 대한 문의 답변
✓ 재고 및 가용성 - 현재 재고 수준 확인 및 고객이 필요한 제품을 찾도록 지원
✓ 제품 추천 - 고객의 요구, 사용 패턴, 예산에 따른 제품 제안
✓ 일반 문의 - 회사, 제품군, 서비스에 대한 질문 응대

직접 처리할 수 없는 업무:
✗ 주문 접수 - 제품 정보는 안내할 수 있지만, 실제 주문은 웹 포털을 이용하거나 영업팀(sales@officeflow.com)에 문의해야 합니다
✗ 주문 상태 및 배송 추적 - 고객에게 계정 포털을 확인하거나 fulfillment@officeflow.com으로 문의하도록 안내하세요
✗ 반품 및 환불 - 반품 부서(returns@officeflow.com)의 승인이 필요합니다
✗ 계정 변경 - 결제, 결제 수단, 계정 설정은 accounts@officeflow.com을 통해 처리됩니다
✗ 기술 지원 - 웹사이트 문제는 support@officeflow.com으로 안내하세요

커뮤니케이션 스타일:
- 따뜻하고 전문적이되, 기계적이거나 지나치게 격식 있는 표현은 피하세요
- 자연스러운 표현 사용 - "지원해 드리겠습니다" 대신 "기꺼이 도와드릴게요"와 같이 표현
- 고객이 불편함을 느낄 때 공감을 표현하세요
- 정보는 구체적이고 정확하게 제공하세요
- 모르는 것이 있다면 솔직하게 말하고, 적절한 담당자나 자원을 안내하세요
- 고객이 이름을 알려주면 이름을 사용하세요
- 응답은 간결하면서도 충분한 내용을 담아야 합니다

중요 - 데이터베이스를 먼저 확인하세요:
고객이 제품이나 재고에 대해 질문할 때는, 추가 질문을 하기 전에 반드시 데이터베이스를 먼저 확인하세요. 추가 정보를 묻기보다는 확인한 내용을 바탕으로 유용한 정보를 제공하세요. 예를 들어, 고객이 "종이 있나요?"라고 물으면 "어떤 종류의 종이를 찾으시나요?"라고 묻지 말고, 재고에 있는 종이 제품을 확인하여 어떤 것들이 있는지 알려주세요.

상호작용 가이드라인:
1. 항상 고객을 따뜻하게 맞이하고 질문을 인지했음을 표현하세요
2. 가용 정보를 확인한 후 정말 필요한 경우에만 추가 질문을 하세요
3. 제품과 재고에 대해 완전하고 정확한 정보를 제공하세요
4. 제품을 추천할 때는 왜 적합한지 이유를 설명하세요
5. 대화를 마칠 때 추가로 필요한 것이 있는지 확인하세요
6. 직접 도울 수 없는 경우, 고객이 필요로 하는 구체적인 연락처나 자원을 안내하세요
7. 절대로 정보를 지어내지 마세요 - 확실하지 않다면 솔직히 말하고 잘 아는 담당자를 연결해 드리겠다고 제안하세요

사용 가능한 도구:
당신은 고객 지원을 위해 두 가지 강력한 도구를 사용할 수 있습니다:

1. query_database - 제품 관련 질문에 사용:
   - 제품 가용성 및 재고 수준
   - 제품 가격 및 가격 정보
   - 제품 세부 정보 및 사양
   - 재고에서 특정 품목 검색

2. search_knowledge_base - 회사 정책 및 정보에 사용:
   - 반품 및 환불 정책
   - 배송 정보
   - 주문 절차 및 결제 방법
   - 매장 위치 및 연락처 정보
   - 회사 배경 및 일반 정보
   - 영업 시간 및 휴무일

고객의 질문 내용에 따라 적절한 도구를 선택하세요. 특정 제품에 대한 질문이라면 데이터베이스를, 정책·절차·회사 정보에 대한 질문이라면 지식 베이스를 사용하세요.

예시 상호작용:

고객: "복사 용지 있나요?"
당신: "네, 있습니다! 다양한 종류의 복사 용지를 취급하고 있어요. 표준 8.5x11인치 레터 사이즈를 찾으시나요, 아니면 특정 무게나 마감재가 필요하신가요? 어떤 재고가 있는지 확인해 드릴 수 있어요."

고객: "주문을 반품하고 싶어요"
당신: "반품을 진행하시려는 거군요, 이해했습니다. 제가 직접 반품을 처리할 수는 없지만, 반품 부서에서 기꺼이 도와드릴 거예요. returns@officeflow.com 또는 1-800-OFFICE-1 내선 3번으로 연락하시면 됩니다. 보통 영업일 기준 4시간 내에 답변을 드려요. 다른 도움이 필요한 것이 있으실까요?"

고객: "서류 서명에 가장 좋은 펜이 뭔가요?"
당신: "서류 서명에는 시간이 지나도 잉크가 바래지 않는 보존성 있는 잉크 펜을 추천드려요. 해당 용도에 적합한 제품이 어떤 것이 있는지 확인해 드릴게요."

기억하세요: 당신은 OfficeFlow의 우수한 고객 서비스에 대한 약속을 대표합니다. 모든 상호작용에서 도움이 되고, 정직하며, 인간적으로 응대하세요."""

@traceable(name="query_database", run_type="tool")
def query_database(query: str, db_path: str) -> str:
    """
    SQLite 인벤토리 DB에 SQL 쿼리를 실행하고 결과를 문자열로 반환.
    LLM이 query_database 도구를 호출할 때 실제로 실행되는 함수.
    오류 발생 시에도 LLM이 다시 시도하거나 사용자에게 안내할 수 있도록
    예외를 던지지 않고 에러 문자열을 반환.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error: {str(e)}"

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """
    긴 문서를 검색하기 좋은 작은 단위(chunk)로 분할.
    - chunk_size: 한 chunk의 문자 수
    - overlap: 인접한 chunk 사이에서 겹치는 문자 수 (문맥 단절 방지용)
    문장 경계는 고려하지 않는 단순 슬라이딩 방식 (학습용 단순 구현).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # 공백만 있는 chunk는 제외
            chunks.append(chunk)
        start = end - overlap  # overlap만큼 뒤로 돌아가 다음 chunk 시작점 설정
    return chunks

async def load_knowledge_base(kb_dir: str = "./knowledge_base") -> None:
    """
    knowledge base 문서를 로드하고 임베딩을 준비.
    캐시(JSON)가 있으면 그대로 사용하고, 없으면 Voyage AI로 새로 생성.
    임베딩 생성에는 비용/시간이 들기 때문에 캐싱이 중요.
    """
    global knowledge_base_docs, knowledge_base_embeddings

    kb_path = Path(kb_dir) / "documents"
    # Voyage 임베딩 전용 캐시 (OpenAI 임베딩과 차원이 달라 파일을 분리)
    cache_path = Path(kb_dir) / "embeddings" / "embeddings_voyage.json"

    # 1) 캐시가 있으면 그대로 로드 후 종료 (가장 빠른 경로)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        # JSON에는 list로 저장되므로 다시 tuple로 변환
        knowledge_base_docs = [tuple(doc) for doc in cache_data["docs"]]
        knowledge_base_embeddings = cache_data["embeddings"]
        print(f"Knowledge base loaded from cache: {len(knowledge_base_docs)} chunks")
        return

    # 2) 캐시가 없으면 마크다운 문서들을 읽어 chunk 단위로 분할
    if not kb_path.exists():
        print(f"Warning: Knowledge base directory '{kb_dir}' not found")
        return

    chunks = []
    for file_path in kb_path.glob("*.md"):
        if file_path.name == "CHUNKING_NOTES.md":  # 메타 문서는 인덱싱 제외
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            file_chunks = chunk_text(content)
            for i, chunk in enumerate(file_chunks):
                # chunk 식별자에 원본 파일명과 인덱스를 기록 → 검색 결과 출처 표시에 사용
                chunks.append((f"{file_path.name}:chunk_{i}", chunk))

    if not chunks:
        print(f"Warning: No documents found in '{kb_dir}'")
        return

    knowledge_base_docs = chunks

    # 3) 각 chunk에 대해 임베딩 생성
    # Voyage는 input_type을 명시하면 정확도가 향상됨
    #   - "document": 인덱싱(저장)할 때 사용
    #   - "query":    검색할 때 사용 (search_knowledge_base 참고)
    print(f"Generating embeddings for {len(chunks)} chunks using Voyage AI...")
    embeddings = []
    for chunk_name, content in chunks:
        response = await voyage_client.embed(
            [content],
            model=EMBEDDING_MODEL,
            input_type="document",
        )
        embeddings.append(response.embeddings[0])

    knowledge_base_embeddings = embeddings

    # 4) 다음 실행 때는 재생성하지 않도록 캐시 저장
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({"docs": chunks, "embeddings": embeddings}, f)
    print(f"Knowledge base loaded: {len(chunks)} chunks indexed")

@traceable(name="search_knowledge_base", run_type="tool")
async def search_knowledge_base(query: str, top_k: int = 2) -> str:
    """
    의미 기반 유사도(코사인 유사도)로 knowledge base에서 관련 chunk를 검색.
    LLM이 search_knowledge_base 도구를 호출할 때 실제로 실행되는 함수.
    """
    if not knowledge_base_docs or not knowledge_base_embeddings:
        return "Error: Knowledge base not loaded"

    # 1) 사용자 질의를 임베딩 벡터로 변환
    # 검색 시에는 input_type="query"를 권장 (저장 시 "document"와 구분)
    response = await voyage_client.embed(
        [query],
        model=EMBEDDING_MODEL,
        input_type="query",
    )
    query_embedding = response.embeddings[0]

    # 2) 모든 문서 임베딩과 코사인 유사도 계산
    # cosine_sim = (A · B) / (|A| × |B|)
    # 값이 1에 가까울수록 의미가 비슷하다고 판단
    similarities = []
    for i, doc_embedding in enumerate(knowledge_base_embeddings):
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((i, similarity))

    # 3) 유사도 내림차순 정렬 후 상위 top_k개 선택
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_k]

    # 4) LLM이 읽기 좋은 텍스트 형태로 포맷
    results = []
    for idx, score in top_results:
        filename, content = knowledge_base_docs[idx]
        results.append(f"=== {filename} (relevance: {score:.3f}) ===\n{content}\n")

    return "\n".join(results)

# ===== Anthropic 도구(Tool) 스키마 =====
# OpenAI와 다른 점:
#   - 최상위에 {"type": "function", "function": {...}} 래퍼가 없음
#   - 입력 스키마 키가 "parameters"가 아니라 "input_schema"
# description은 LLM이 "언제 이 도구를 써야 하는지" 판단할 때 사용 → 명확하게 작성.
#
# v2 변경점:
#   - description에 스키마(테이블/컬럼) 탐색 절차를 명시
#   - LLM이 임의로 컬럼명을 추측하지 않고, 실제 DB 구조를 먼저 파악하도록 유도
QUERY_DATABASE_TOOL = {
    "name": "query_database",
    "description": """고객을 위해 재고 정보(제품, 수량, 가격 등)를 조회하기 위한 SQL 쿼리를 실행합니다.

당신은 DB 스키마를 모릅니다. 항상 다음 절차로 먼저 스키마를 탐색하세요:
1. 'SELECT name FROM sqlite_master WHERE type="table"' 쿼리로 사용 가능한 테이블 목록을 확인합니다.
2. 'PRAGMA table_info(table_name)'로 각 테이블의 컬럼 정보를 확인합니다.
3. 스키마를 충분히 이해한 뒤에만 실제 검색 쿼리를 작성하세요.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "인벤토리 데이터베이스에 실행할 SQL 쿼리"
            }
        },
        "required": ["query"]
    }
}

SEARCH_KNOWLEDGE_BASE_TOOL = {
    "name": "search_knowledge_base",
    "description": "회사 지식 베이스에서 정책, 절차, 회사 정보, 배송, 반품, 주문, 연락처, 매장 위치, 영업 시간 등에 대한 정보를 검색합니다. 제품과 무관한 질문에 사용하세요.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "회사 정책이나 정보에 대한 자연어 질문 또는 검색어"
            }
        },
        "required": ["query"]
    }
}

def get_thread_history(thread_id: str) -> list:
    """thread_id에 해당하는 이전 대화 메시지 목록을 반환."""
    return thread_store.get(thread_id, [])

def save_thread_history(thread_id: str, messages: list):
    """대화 히스토리를 저장 (system 프롬프트는 제외, user/assistant 메시지만)."""
    thread_store[thread_id] = messages

@traceable(name="Emma", metadata={"thread_id": thread_id})
async def chat(question: str) -> dict:
    """
    사용자 질문을 받아 도구 호출을 포함한 에이전트 루프를 돌고 최종 응답을 반환.

    Anthropic 에이전트 루프(agentic loop) 흐름:
      1) 사용자 질문 + 이전 대화 + 도구 목록을 보내고 응답 받기
      2) stop_reason == "tool_use"이면 → 도구 실행 → 결과를 user 메시지로 다시 전달
      3) 더 이상 도구 호출이 없을 때까지 2번 반복
      4) 최종 텍스트 응답을 추출해 반환

    반환값은 {"messages": ..., "output": ...} 형식으로,
    LangSmith 추적 시 전체 메시지 흐름과 최종 응답을 함께 확인할 수 있음.
    """
    db_path = str(Path(__file__).parent / 'inventory' / 'inventory.db')
    tools = [QUERY_DATABASE_TOOL, SEARCH_KNOWLEDGE_BASE_TOOL]

    # 이전 대화 히스토리 로드
    history_messages = get_thread_history(thread_id)

    # Anthropic API의 messages 배열 구성
    # 주의: system은 messages에 넣지 않고 messages.create의 system 파라미터로 따로 전달.
    messages = history_messages + [
        {"role": "user", "content": question}
    ]

    # 첫 번째 API 호출
    response = await anthropic_client.messages.create(
        model=MODEL,
        max_tokens=16000,        # 응답 최대 토큰 수 (도구 호출 블록 + 텍스트 합)
        system=system_prompt,    # 별도 파라미터로 전달 (messages에는 포함하지 않음)
        tools=tools,
        messages=messages,
    )

    # ===== 에이전트 루프 =====
    # response.stop_reason 주요 값:
    #   - "end_turn"   : 모델이 응답을 완성함 → 루프 종료
    #   - "tool_use"   : 모델이 도구 호출을 요청함 → 실행 후 결과 전달 필요
    #   - "max_tokens" : 토큰 한도 초과
    while response.stop_reason == "tool_use":
        # 어시스턴트 응답 전체(content blocks)를 그대로 히스토리에 추가.
        # content 안에는 text 블록과 tool_use 블록이 함께 들어 있을 수 있음.
        messages.append({"role": "assistant", "content": response.content})

        # content 안의 모든 tool_use 블록을 실행 (한 번에 여러 도구 호출 가능)
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue  # text 등 다른 블록은 건너뜀

            # 도구 이름에 따라 실제 함수로 라우팅
            if block.name == "query_database":
                result = query_database(
                    query=block.input.get("query"),
                    db_path=db_path,
                )
            elif block.name == "search_knowledge_base":
                result = await search_knowledge_base(
                    query=block.input.get("query"),
                )
            else:
                result = f"Error: Unknown tool {block.name}"

            # tool_result는 tool_use_id로 어떤 호출의 결과인지 명시해야 함
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        # 모든 도구 결과는 하나의 user 메시지(content 배열)에 모아 전달
        messages.append({"role": "user", "content": tool_results})

        # 도구 결과를 포함해 다시 호출 → 모델이 최종 답변하거나 추가 도구 호출
        response = await anthropic_client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

    # ===== 최종 응답 추출 =====
    # content는 여러 블록(text/tool_use 등)으로 구성되므로 text 블록만 골라 합침
    final_content = "".join(
        block.text for block in response.content if block.type == "text"
    )

    # 마지막 어시스턴트 응답도 히스토리에 저장 → 다음 turn에서 문맥으로 활용
    messages.append({"role": "assistant", "content": response.content})
    save_thread_history(thread_id, messages)

    return {"messages": messages, "output": final_content}

async def main():
    """대화형 CLI 진입점."""
    print("Office Supplies Support Chat (Anthropic + Voyage AI)")
    print("=" * 50)
    print(f"Model: {MODEL}")
    print(f"Thread ID: {thread_id}")
    print()

    # 시작 시 knowledge base 로드 (캐시 사용 또는 새로 생성)
    await load_knowledge_base()
    print()
    print("Type 'quit' or 'exit' to end the conversation\n")

    # 사용자 입력 루프
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for chatting! Goodbye!")
            break

        if not user_input:  # 빈 입력은 무시
            continue

        result = await chat(user_input)
        response = result["output"]
        print(f"\nAgent: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
