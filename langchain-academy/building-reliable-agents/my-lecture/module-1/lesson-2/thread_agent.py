# Thread(대화 세션) 단위로 대화 이력을 유지하고, LangSmith에 thread_id 메타데이터로 묶어 추적하는 예제
# 동일한 thread_id를 가진 호출들은 LangSmith UI에서 하나의 스레드로 묶여 보임

from anthropic import Anthropic
from dotenv import load_dotenv
from langsmith import traceable, uuid7  # uuid7: 시간 정렬 가능한 UUID (스레드 ID 생성용)
from langsmith.wrappers import wrap_anthropic  # Anthropic 클라이언트를 LangSmith 추적용으로 감싸는 래퍼


load_dotenv()

# Anthropic 클라이언트를 LangSmith 래퍼로 감싸기
client = wrap_anthropic(Anthropic())

# 스레드 식별자 (대화 세션 ID). uuid7로 생성하면 시간순 정렬이 가능
THREAD_ID = str(uuid7())

MODEL = "claude-sonnet-4-6"  # 사용할 Claude 모델
MAX_TOKENS = 1024            # 응답 최대 토큰 수

# 대화 이력 저장소 (데모용 인메모리 dict — 실제 서비스에서는 DB 사용 필요)
thread_store: dict[str, list] = {}

def get_thread_history(thread_id: str) -> list:
    """주어진 thread_id의 이전 대화 메시지 리스트를 반환 (없으면 빈 리스트)."""
    return thread_store.get(thread_id, [])

def save_thread_history(thread_id: str, messages: list):
    """주어진 thread_id에 전체 대화 메시지 리스트를 저장(덮어쓰기)."""
    thread_store[thread_id] = messages

# @traceable 데코레이터로 함수 호출을 LangSmith에 기록
# - name: LangSmith UI에 표시될 실행 이름
# - metadata.thread_id: 같은 thread_id를 가진 실행끼리 하나의 스레드로 묶임
@traceable(name="Name Agent", metadata={"thread_id": THREAD_ID})
def chat_pipeline(messages: list):
    # 기존 대화 이력 자동 조회 (스레드 컨텍스트 유지의 핵심)
    history_messages = get_thread_history(THREAD_ID)

    # 이전 이력 + 이번 턴의 새 메시지를 합쳐 모델에 전달할 전체 메시지 구성
    all_messages = history_messages + messages

    # 모델 호출 (전체 대화 이력을 함께 전달해야 멀티턴 컨텍스트가 유지됨)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=all_messages,
    )

    # 응답 content 블록 중 text 타입 블록에서 답변 텍스트 추출 (없으면 빈 문자열)
    response_text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )

    # 응답을 이력에 추가해 다음 턴에서도 같은 컨텍스트를 이어갈 수 있게 저장
    full_conversation = all_messages + [{"role": "assistant", "content": response_text}]
    save_thread_history(THREAD_ID, full_conversation)

    return {
        "messages": full_conversation
    }

if __name__ == "__main__":
    # 첫 번째 메시지: 사용자가 자기 이름을 알려줌
    messages = [{"content": "안녕, 내이름은 홍길동이야", "role": "user"}]
    result = chat_pipeline(messages)
    print(result["messages"][-1])

    # 두 번째 메시지: 같은 THREAD_ID이므로 이전 대화를 기억하고 이름을 답할 수 있어야 함
    messages = [{"content": "내 이름이 뭐야?", "role": "user"}]
    result = chat_pipeline(messages)
    print(result["messages"][-1])
