"""
두 개의 실험을 비교하는 페어와이즈(pairwise) 간결성 평가기.

run_agents.py 실행 후, 출력된 실험 이름을 사용하여 본 스크립트를 실행하세요.

사용법:
    uv run python eval_conciseness_pairwise.py <experiment-a> <experiment-b>
    uv run python eval_conciseness_pairwise.py agent-v4-3e016f9c agent-v5-7d7ee287
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from langsmith import evaluate
from langsmith.wrappers import wrap_anthropic

# .env 파일에서 환경 변수(API 키 등)를 로드
load_dotenv()

# ===== 클라이언트 초기화 =====
# Anthropic: LLM 호출 (Claude 모델 - 평가자 역할)
#   - wrap_anthropic으로 감싸면 LangSmith가 모든 messages.create 호출을 자동 추적
client = wrap_anthropic(
    Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
)

# ===== 설정 =====
# 평가자 모델 (Haiku: 빠르고 비용 효율적인 평가에 적합)
MODEL = "claude-haiku-4-5-20251001"

# ===== 평가 프롬프트 =====
# 두 응답을 비교해 어느 쪽이 더 간결한지 판단하도록 LLM에 지시하는 프롬프트.
# - "간결성"의 정의와 "필수 정보"의 정의를 명확히 제시
# - 단순히 짧다고 좋은 응답이 아니라는 점을 강조 (필수 정보 보존 조건 포함)
# - 출력 형식은 0/1/2 중 하나의 숫자로 제한하여 후처리를 단순화
CONCISENESS_PROMPT = """당신은 같은 고객 질문에 대한 두 개의 응답을 평가하고 있습니다.
어느 응답이 필수 정보를 모두 포함하면서도 더 간결(MORE CONCISE)한지 판단하세요.

**간결성(Conciseness)** 이란 핵심을 바로 전달하고, 불필요한 군더더기를 피하며, 동일한 정보를 반복하지 않는 것을 의미합니다.
**필수 정보(Crucial information)** 에는 직접적인 답변, 필요한 맥락, 요구되는 다음 단계가 포함됩니다.

더 짧은 응답이라도 필수 정보를 누락했다면 자동으로 더 좋은 응답이 아닙니다.

**질문:** {question}

**응답 A:**
{response_a}

**응답 B:**
{response_b}

판정을 단 하나의 숫자로만 출력하세요:
1 = 응답 A가 필수 정보를 유지하면서 더 간결함
2 = 응답 B가 필수 정보를 유지하면서 더 간결함
0 = 두 응답이 대체로 동등함"""


def conciseness_evaluator(inputs: dict, outputs: list[dict]) -> list[int]:
    """두 응답의 간결성을 비교하여 페어와이즈 점수를 반환.

    Args:
        inputs: 평가 대상 질문이 포함된 입력 딕셔너리 (key: "question")
        outputs: 비교 대상 두 실험의 출력 리스트
                 - outputs[0]: 실험 A의 응답
                 - outputs[1]: 실험 B의 응답

    Returns:
        [A_score, B_score] 형식의 리스트
        - [1, 0]: A 승리
        - [0, 1]: B 승리
        - [0, 0]: 무승부
    """
    # Anthropic Messages API 호출
    # - system 메시지로 평가자 역할과 출력 형식을 명확히 지정
    # - max_tokens는 단일 숫자만 받으면 되므로 작게 설정
    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        system="당신은 간결성 평가자입니다. 오직 하나의 숫자(0, 1, 2)만 응답하세요.",
        messages=[
            {
                "role": "user",
                "content": CONCISENESS_PROMPT.format(
                    question=inputs["question"],
                    response_a=outputs[0].get("answer", "N/A"),
                    response_b=outputs[1].get("answer", "N/A"),
                ),
            }
        ],
    )

    # Anthropic 응답의 첫 번째 content 블록에서 텍스트를 추출하여 정수로 변환
    preference = int(response.content[0].text.strip())

    # 페어와이즈 점수 매핑
    if preference == 1:
        return [1, 0]  # A 승리
    elif preference == 2:
        return [0, 1]  # B 승리
    else:
        return [0, 0]  # 무승부


if __name__ == "__main__":
    import sys

    # CLI 인자 검증: 비교할 두 실험의 이름이 반드시 필요
    if len(sys.argv) != 3:
        print("Usage: python eval_conciseness_pairwise.py <experiment-a> <experiment-b>")
        print("Example: python eval_conciseness_pairwise.py agent-v4-3e016f9c agent-v5-7d7ee287")
        sys.exit(1)

    # LangSmith pairwise 평가 실행
    # - 첫 번째 인자: 비교할 두 실험명을 튜플로 전달
    # - evaluators: 페어와이즈 평가 함수 리스트
    # - randomize_order: A/B 순서를 무작위화하여 위치 편향(position bias) 제거
    evaluate(
        (sys.argv[1], sys.argv[2]),
        evaluators=[conciseness_evaluator],
        randomize_order=True,
    )
