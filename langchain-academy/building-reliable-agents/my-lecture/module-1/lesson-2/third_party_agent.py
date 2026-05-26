# LangChain 외부의 서드파티(Anthropic) SDK를 LangSmith로 수동 추적하는 예제
# 3단계(① 클라이언트 래핑, ② 도구에 traceable, ③ 체인 함수에 traceable)로 추적을 구성한다

from anthropic import Anthropic
from langsmith.wrappers import wrap_anthropic  # Anthropic 클라이언트를 LangSmith 추적용으로 감싸는 래퍼

from langsmith import traceable  # 일반 함수도 LangSmith에서 추적할 수 있게 해주는 데코레이터
from dotenv import load_dotenv

# .env 파일에서 ANTHROPIC_API_KEY, LANGSMITH_API_KEY 등 환경변수 로드
load_dotenv()

# 1. 모델 클라이언트를 LangSmith 래퍼로 감싸기
client = wrap_anthropic(Anthropic())

# 2. 도구 함수에 @traceable 데코레이터를 붙여 도구 실행 자체도 추적 대상에 포함
@traceable(run_type="tool")  # run_type을 "tool"로 지정해 LangSmith UI에서 도구 노드로 표시
def weather_retriever():
    """현재 날씨 정보를 반환하는 더미 도구."""
    return "It is sunny today"

# Anthropic tool 스키마 정의 (OpenAI와 형식이 다름: 최상위에 name/description/input_schema 위치)
WEATHER_TOOL = {
    "name": "weather_retriever",
    "description": "Get the current weather conditions",
    "input_schema": {
        "type": "object",
        "properties": {},  # 입력 파라미터가 없는 단순 도구
        "required": []
    }
}

MODEL = "claude-sonnet-4-6"  # 사용할 Claude 모델
MAX_TOKENS = 1024            # 응답 최대 토큰 수

# 3. 전체 에이전트 흐름을 감싸는 함수에도 @traceable 적용
@traceable  # run_type 기본값이 "chain"이라 생략 → LangSmith에서 체인 노드로 묶여 추적됨
def agent(question: str) -> str:

    # 사용자 질문을 초기 메시지로 구성
    messages = [{"role": "user", "content": question}]

    # 1차 호출: 모델이 도구를 사용할지 직접 답할지 결정
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[WEATHER_TOOL],
        messages=messages,
    )

    # 모델이 도구 호출을 요청(stop_reason == "tool_use")한 경우 → 도구 실행 후 결과를 다시 모델에 전달
    if response.stop_reason == "tool_use":
        # assistant 턴(전체 content 블록: text + tool_use)을 메시지 히스토리에 그대로 추가
        # → tool_use_id 연속성을 유지하기 위해 content 블록 통째로 append 해야 함
        messages.append({"role": "assistant", "content": response.content})

        # 모델이 요청한 도구들을 순회하며 실제 함수 실행
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "weather_retriever":
                result = weather_retriever()  # @traceable 덕분에 이 호출도 LangSmith에 기록됨
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 어떤 tool_use에 대한 응답인지 매칭
                    "content": result,
                })

        # Anthropic 규약상 tool_result는 user 메시지의 content로 전달
        messages.append({"role": "user", "content": tool_results})

        # 2차 호출: 도구 결과를 받은 모델이 최종 자연어 답변 생성
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[WEATHER_TOOL],
            messages=messages,
        )

    # 응답 content 블록 중 text 타입 블록에서 최종 답변 텍스트 추출 (없으면 빈 문자열)
    final_text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )
    # 최종 답변을 메시지 히스토리에 추가 (대화 로깅 용도)
    messages.append({"role": "assistant", "content": final_text})
    return {"messages": messages, "output": final_text}

if __name__ == "__main__":
    # 간단한 동작 확인: 날씨 질문 → 도구 호출 → 최종 답변
    result = agent("오늘 날씨 어때?")
    print(result["output"])
