"""
두 버전의 Anthropic 기반 에이전트(agent_v4, agent_v5)를 LangSmith 데이터셋으로 평가 실행하는 스크립트.

목적:
    - 동일한 데이터셋(officeflow-dataset)에 대해 두 에이전트를 각각 실행하고
      LangSmith에 실험(Experiment)으로 기록한다.
    - 본 스크립트는 단순히 실험을 "생성"만 한다.
    - 두 실험의 응답을 비교하는 페어와이즈 평가는
      eval_conciseness_pairwise.py 에서 별도로 실행한다.

실행 흐름:
    1) 두 에이전트가 공통으로 사용하는 knowledge base(문서 + 임베딩)를 메모리에 로드
    2) agent_v4 의 chat 함수를 LangSmith aevaluate 로 데이터셋 전 항목에 대해 실행
    3) agent_v5 의 chat 함수도 동일하게 실행
    4) 두 실험의 이름을 출력 → 이후 페어와이즈 평가에 사용

사용법:
    uv run python run_agents.py
"""

import asyncio
import sys
from pathlib import Path

# ===== 모듈 경로 설정 =====
# run_agents.py 는 .../my-lecture/module-2/lesson-6/ 아래에 있고,
# 에이전트 코드는 .../my-lecture/officeflow-agent/ 아래에 있다.
# 따라서 상위 디렉터리 3단계(lesson-6 → module-2 → my-lecture)를 거슬러 올라간 뒤
# "officeflow-agent" 폴더를 sys.path 에 추가해야 agent_v4/agent_v5 를 import 할 수 있다.
agent_dir = Path(__file__).resolve().parent.parent.parent / "officeflow-agent"
sys.path.insert(0, str(agent_dir))

from langsmith import aevaluate
# agent_v4, agent_v5 모두 내부적으로 Anthropic Claude 모델(claude-sonnet-4-6 등)을
# 호출하는 비동기 챗 에이전트이다.
# - chat: 사용자 질문을 입력받아 도구 호출 루프를 돌고 최종 응답을 반환
# - load_knowledge_base: 문서 임베딩을 메모리에 적재 (검색 도구가 사용)
from agent_v4 import chat as chat_v4, load_knowledge_base as load_kb_v4
from agent_v5 import chat as chat_v5, load_knowledge_base as load_kb_v5
from dotenv import load_dotenv

# ANTHROPIC_API_KEY, VOYAGE_API_KEY, LANGSMITH_API_KEY 등을 .env 에서 로드
load_dotenv()

# ===== 설정 =====
# LangSmith 에 등록되어 있는 평가용 데이터셋 이름.
# 데이터셋의 각 example 은 {"question": "..."} 형태의 입력을 가진다고 가정.
DATASET_NAME = "officeflow-dataset"

# knowledge base 문서/임베딩이 저장된 디렉터리 (officeflow-agent/knowledge_base)
KB_DIR = str(agent_dir / "knowledge_base")


async def chat_wrapper_v4(inputs: dict) -> dict:
    """
    LangSmith aevaluate 가 데이터셋의 각 example 마다 호출하는 래퍼 함수 (agent_v4 용).

    aevaluate 는 데이터셋의 example 을 그대로 inputs 로 전달하는데,
    Anthropic 기반 chat 함수는 단일 문자열을 인자로 받기 때문에
    "question" 키를 꺼내 chat 에 넘기는 어댑터가 필요하다.

    Args:
        inputs: 데이터셋 example 의 입력 딕셔너리. {"question": "..."} 형태를 기대.

    Returns:
        {"answer": "..."} 형태의 출력 딕셔너리.
        eval_conciseness_pairwise.py 의 평가자는 outputs[i].get("answer") 로 접근하므로
        키 이름을 반드시 "answer" 로 맞춰야 한다.
    """
    question = inputs.get("question", "")
    # agent_v4.chat 은 {"messages": [...], "output": "최종 응답 텍스트"} 를 반환
    result = await chat_v4(question)
    return {"answer": result["output"]}


async def chat_wrapper_v5(inputs: dict) -> dict:
    """
    agent_v5 용 래퍼. chat_wrapper_v4 와 동일한 인터페이스를 제공한다.
    동일한 데이터셋을 동일한 형태로 평가하기 위해 입력/출력 스키마를 맞춘다.
    """
    question = inputs.get("question", "")
    result = await chat_v5(question)
    return {"answer": result["output"]}


async def main():
    """
    두 Anthropic 에이전트의 LangSmith 실험을 순차적으로 생성하는 메인 진입점.

    동작 순서:
        1. 두 에이전트가 공유하는 knowledge base 를 각각 로드
           (v4/v5 는 임베딩 청크 전략이 다를 수 있어 각자 로드 함수를 호출)
        2. agent_v4 실험 실행 → 실험 이름 자동 생성 (예: "agent-v4-3e016f9c")
        3. agent_v5 실험 실행 → 실험 이름 자동 생성 (예: "agent-v5-7d7ee287")
        4. 두 실험 이름을 출력하여 페어와이즈 평가 단계에서 활용
    """
    # ----- 1) Knowledge Base 로드 -----
    # 임베딩 캐시가 유효하면 캐시에서 즉시 로드, 문서가 변경되었다면 재생성한다.
    # 한 번 메모리에 적재되면 이후 모든 chat 호출에서 공유된다.
    print("지식 베이스(knowledge base)를 로드합니다...")
    await load_kb_v4(KB_DIR)
    await load_kb_v5(KB_DIR)

    # ----- 2) agent_v4 실험 실행 -----
    # aevaluate 는 데이터셋의 모든 example 에 대해 chat_wrapper_v4 를 비동기로 호출하고
    # 각 실행을 LangSmith 에 자동으로 트레이싱한다.
    # experiment_prefix 는 LangSmith 에 표시되는 실험 이름의 접두어로 사용된다.
    print("\nagent_v4 (Anthropic Claude) 실험을 실행합니다...")
    v4_results = await aevaluate(
        chat_wrapper_v4,
        data=DATASET_NAME,
        experiment_prefix="agent-v4",
    )

    # ----- 3) agent_v5 실험 실행 -----
    # 동일한 데이터셋에 대해 다른 버전의 에이전트를 실행하여
    # 같은 질문에 대한 두 응답을 비교 가능한 형태로 확보한다.
    print("\nagent_v5 (Anthropic Claude) 실험을 실행합니다...")
    v5_results = await aevaluate(
        chat_wrapper_v5,
        data=DATASET_NAME,
        experiment_prefix="agent-v5",
    )

    # ----- 4) 결과 출력 -----
    # 출력된 두 실험 이름을 eval_conciseness_pairwise.py 의 CLI 인자로 전달하면
    # 페어와이즈(간결성) 비교 평가를 수행할 수 있다.
    print(f"\nv4 실험 이름: {v4_results.experiment_name}")
    print(f"v5 실험 이름: {v5_results.experiment_name}")
    print("완료! 위 실험 이름을 페어와이즈 평가 스크립트에 인자로 전달하세요.")


if __name__ == "__main__":
    # aevaluate, chat 모두 비동기 함수이므로 asyncio.run 으로 이벤트 루프를 구동한다.
    asyncio.run(main())
