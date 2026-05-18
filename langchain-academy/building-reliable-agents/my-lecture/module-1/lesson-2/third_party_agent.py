# 3단계를 통해 수동으로 어플리케이션 추적하는 방안

from anthropic import Anthropic
from langsmith.wrappers import wrap_anthropic

from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# 1. 모델을 Wrpper로 감싸기
client = wrap_anthropic(Anthropic())

# 2. 도구를 정의할 때 traceable 데코레이터 사용
@traceable(run_type="tool") # un_type을 사용하여 함수 유형을 지정
def weather_retriever():
    """Retrieve current weather information."""
    return "It is sunny today"

# Anthropic tool 스키마 (OpenAI와 다름: 최상위에 name/description/input_schema)
WEATHER_TOOL = {
    "name": "weather_retriever",
    "description": "Get the current weather conditions",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

@traceable # run_type 기본값이 "chain"이기 때문에 명시하지 않은 것
def agent(question: str) -> str:

    messages = [{"role": "user", "content": question}]

    # 1차 호출
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[WEATHER_TOOL],
        messages=messages,
    )

    # tool_use가 있으면 실행 후 결과를 다시 보냄
    if response.stop_reason == "tool_use":
        # assistant turn(전체 content 블록)을 그대로 append
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "weather_retriever":
                result = weather_retriever()
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # tool_result는 user 메시지로 전달
        messages.append({"role": "user", "content": tool_results})

        # 2차 호출
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[WEATHER_TOOL],
            messages=messages,
        )

    # 최종 텍스트 추출
    final_text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )
    messages.append({"role": "assistant", "content": final_text})
    return {"messages": messages, "output": final_text}

if __name__ == "__main__":
    result = agent("오늘 날씨 어때?")
    print(result["output"])
