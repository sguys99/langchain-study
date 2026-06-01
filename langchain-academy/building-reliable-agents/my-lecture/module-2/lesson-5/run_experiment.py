import asyncio
import sys
from pathlib import Path

# 상위 디렉토리를 import 경로에 추가
# 현재 파일 기준으로 3단계 상위 디렉토리(my-lecture의 부모)를 가져옴
python_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(python_dir))
# officeflow-agent 디렉토리도 import 경로에 추가하여 agent_v4 모듈을 불러올 수 있게 함
sys.path.insert(0, str(python_dir / "officeflow-agent"))

from langsmith import aevaluate  # LangSmith의 비동기 평가 함수
from agent_v4 import chat, load_knowledge_base  # 평가 대상 에이전트의 핵심 함수들
from dotenv import load_dotenv

# .env 파일에서 환경변수(API 키 등)를 로드
load_dotenv()

# LangSmith UI에서 평가자(evaluator)가 바인딩되어 있는 데이터셋 이름
# 이 데이터셋의 각 샘플에 대해 평가가 자동으로 수행됨
dataset_name = "officeflow-dataset"

async def chat_wrapper(inputs: dict) -> dict:
    """데이터셋의 입력 형식을 chat 함수의 시그니처에 맞게 변환하는 래퍼 함수.

    LangSmith 데이터셋의 입력 구조({"question": "..."})를
    chat 함수가 기대하는 형태로 어댑팅하고,
    출력 또한 평가에 사용할 수 있는 dict 구조로 반환한다.
    """
    # 데이터셋의 입력에서 "question" 필드를 추출 (없으면 빈 문자열)
    question = inputs.get("question", "")
    # 에이전트의 chat 함수를 호출하여 응답을 받음
    result = await chat(question)
    # 평가에 필요한 답변(answer)과 전체 메시지 히스토리(messages)를 반환
    return {"answer": result["output"], "messages": result["messages"]}

async def main():
    # 평가 실행 전에 에이전트가 사용할 지식 베이스를 먼저 로드
    kb_path = str(python_dir / "officeflow-agent" / "knowledge_base")
    print(f"Loading knowledge base from {kb_path}...")
    await load_knowledge_base(kb_dir=kb_path)
    print()

    # aevaluate를 호출하면 데이터셋의 각 샘플에 대해 chat_wrapper가 실행되고,
    # 데이터셋에 바인딩된 평가자가 자동으로 채점을 수행함
    results = await aevaluate(
        chat_wrapper,  # 평가 대상이 되는 함수 (각 입력에 대해 호출됨)
        data=dataset_name  # 평가에 사용할 LangSmith 데이터셋 이름
    )
    print(f"Evaluation complete! Results: {results}")
    return results

if __name__ == "__main__":
    # 비동기 main 함수를 실행
    asyncio.run(main())
