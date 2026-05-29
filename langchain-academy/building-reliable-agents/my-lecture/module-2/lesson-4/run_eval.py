"""
officeflow-dataset에 대해 쿼리 전 스키마 확인(schema-before-query) 평가자를 실행한다.

사용법:
    uv run --env-file ../../.env python run_eval.py
"""
import asyncio
import sys
from pathlib import Path

from langsmith import evaluate
from langsmith import uuid7

# 에이전트 디렉터리를 import 경로(path)에 추가한다
agent_dir = str(Path(__file__).resolve().parent.parent.parent / "officeflow-agent")
sys.path.insert(0, agent_dir)

import agent_v5
from agent_v5 import chat, load_knowledge_base
from eval_schema_check import schema_before_query


async def setup():
    """평가를 실행하기 전에 지식 베이스(knowledge base)를 로드한다."""
    kb_dir = str(Path(agent_dir) / "knowledge_base")
    await load_knowledge_base(kb_dir)


def run_agent(inputs: dict) -> dict:
    """매번 새로운 thread_id로 에이전트를 호출한다."""
    # 각 평가 예시마다 thread_id를 새로 생성하여 대화 상태가 섞이지 않도록 한다
    agent_v5.thread_id = str(uuid7())
    return asyncio.run(chat(inputs["question"]))


if __name__ == "__main__":
    # 평가 실행 전 지식 베이스를 먼저 로드한다
    asyncio.run(setup())

    # LangSmith의 evaluate로 데이터셋의 각 예시에 대해 에이전트를 실행하고
    # schema_before_query 평가자로 결과를 채점한다
    results = evaluate(
        run_agent,
        data="officeflow-dataset",
        evaluators=[schema_before_query],
        experiment_prefix="schema-check-v5",
        max_concurrency=1,  # 한 번에 하나씩 실행하여 분당 입력 토큰 한도(rate limit) 초과를 방지한다
    )
