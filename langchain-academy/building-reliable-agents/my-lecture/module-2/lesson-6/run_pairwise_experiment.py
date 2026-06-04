"""
두 에이전트(v4, v5)의 실험 생성부터 페어와이즈 간결성 평가까지 한 번에 실행하는
엔드투엔드(end-to-end) 오케스트레이터 스크립트.

eval_conciseness_pairwise.py와의 차이:
    - eval_conciseness_pairwise.py: 이미 존재하는 두 실험 이름을 인자로 받아 비교만 수행
    - run_pairwise_experiment.py(본 파일): 실험을 직접 생성한 뒤 그 결과를 페어와이즈 평가

실행 흐름:
    1. agent_v4 / agent_v5의 지식 베이스(knowledge base)를 로드
    2. 각 에이전트를 LangSmith 데이터셋으로 실험 실행 (aevaluate)
    3. 새로 만들어진 두 실험을 conciseness_evaluator로 페어와이즈 비교 (evaluate)

사용법:
    uv run python run_pairwise_experiment.py
"""

import asyncio
import sys
from pathlib import Path

# ===== sys.path 조정 =====
# officeflow-agent 디렉토리를 import 경로에 추가
#   - 본 스크립트는 my-lecture/module-2/lesson-6/ 하위에 있음
#   - agent_v4, agent_v5 모듈은 officeflow-agent/ 하위에 위치
#   - 상대 경로 import가 불가능하므로 sys.path에 절대 경로를 직접 삽입
agent_dir = Path(__file__).resolve().parent.parent.parent / "officeflow-agent"
sys.path.insert(0, str(agent_dir))

from langsmith import aevaluate, evaluate
from agent_v4 import chat as chat_v4, load_knowledge_base as load_kb_v4
from agent_v5 import chat as chat_v5, load_knowledge_base as load_kb_v5
from eval_conciseness_pairwise import conciseness_evaluator
from dotenv import load_dotenv

# .env 파일에서 환경 변수(API 키 등)를 로드
load_dotenv()

# ===== 설정 =====
# LangSmith에 미리 등록된 평가용 데이터셋 이름
DATASET_NAME = "officeflow-dataset"
# 에이전트가 참조하는 지식 베이스 디렉토리 경로 (officeflow-agent/knowledge_base)
KB_DIR = str(agent_dir / "knowledge_base")


async def chat_wrapper_v4(inputs: dict) -> dict:
    """LangSmith aevaluate가 요구하는 시그니처에 맞춘 agent_v4 호출 래퍼.

    - inputs: 데이터셋의 example에서 전달되는 입력 (key: "question")
    - 반환값은 페어와이즈 평가자(conciseness_evaluator)가 기대하는 "answer" 키 형태로 정규화
    """
    question = inputs.get("question", "")
    result = await chat_v4(question)
    return {"answer": result["output"]}


async def chat_wrapper_v5(inputs: dict) -> dict:
    """agent_v5용 동일한 형태의 호출 래퍼.

    v4와 동일한 입력/출력 스키마를 유지해야 동일 데이터셋에서 공정 비교가 가능.
    """
    question = inputs.get("question", "")
    result = await chat_v5(question)
    return {"answer": result["output"]}


async def main():
    # ----- 사전 준비: 두 에이전트의 지식 베이스 로드 -----
    # 각 에이전트가 내부적으로 사용하는 KB를 미리 메모리에 적재
    #   - 실험 실행 중 매번 로드되는 것을 막아 속도/안정성 확보
    print("Loading knowledge bases...")
    await load_kb_v4(KB_DIR)
    await load_kb_v5(KB_DIR)

    # ----- Step 1: agent_v4 실험 실행 -----
    # aevaluate는 데이터셋의 모든 example을 비동기로 돌려 LangSmith에 실험으로 기록
    #   - experiment_prefix: LangSmith UI에서 식별하기 쉽도록 접두사 지정
    #   - 결과 객체에서 experiment_name을 꺼내 이후 페어와이즈 평가에 사용
    print("\n" + "="*50)
    print("Running experiment for agent_v4...")
    print("="*50)
    v4_results = await aevaluate(
        chat_wrapper_v4,
        data=DATASET_NAME,
        experiment_prefix="agent-v4",
    )

    # ----- Step 2: agent_v5 실험 실행 -----
    # v4와 동일한 데이터셋/방식으로 실행하여 공정한 비교 조건 보장
    print("\n" + "="*50)
    print("Running experiment for agent_v5...")
    print("="*50)
    v5_results = await aevaluate(
        chat_wrapper_v5,
        data=DATASET_NAME,
        experiment_prefix="agent-v5",
    )

    # 방금 생성된 두 실험의 실제 이름을 추출
    # (experiment_prefix에 LangSmith가 자동으로 해시를 붙여 고유한 이름을 만듦)
    v4_experiment = v4_results.experiment_name
    v5_experiment = v5_results.experiment_name

    print(f"\nv4 experiment: {v4_experiment}")
    print(f"v5 experiment: {v5_experiment}")

    # ----- Step 3: 페어와이즈(pairwise) 간결성 평가 -----
    # eval_conciseness_pairwise.py에서 정의된 평가자를 재사용
    #   - 첫 번째 인자: 비교할 두 실험명 튜플
    #   - randomize_order: A/B 순서를 랜덤화하여 LLM 평가자의 위치 편향(position bias) 제거
    print("\n" + "="*50)
    print("Running pairwise evaluation...")
    print("="*50)
    evaluate(
        (v4_experiment, v5_experiment),
        evaluators=[conciseness_evaluator],
        randomize_order=True,
    )

    # 결과는 LangSmith UI에서 확인 (pairwise score, A/B 승률 등)
    print("\n" + "="*50)
    print("Done! Check LangSmith for results.")
    print("="*50)

if __name__ == "__main__":
    # aevaluate는 비동기 함수이므로 asyncio 이벤트 루프 위에서 실행
    asyncio.run(main())
