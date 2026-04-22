"""TODO 추적 및 가상 파일 시스템을 지원하는 딥 에이전트용 상태 관리.

이 모듈은 다음을 지원하는 확장된 에이전트 state 구조를 정의합니다:
- TODO 목록을 통한 작업 계획 및 진행 상황 추적
- state에 저장된 가상 파일 시스템을 통한 컨텍스트 오프로딩
- reducer 함수를 활용한 효율적인 state 병합
"""

from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

#from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.agents import AgentState  # updated in 1.0

class Todo(TypedDict):
    """복잡한 워크플로우의 진행 상황을 추적하기 위한 구조화된 작업 항목입니다.

    Attuributes:
        content: 작업에 대한 간결하고 구체적인 설명
        status: 현재 state - pending, in_progress, 또는 completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """두 파일 딕셔너리(dictionary)을 병합하며, 오른쪽 값이 우선 적용됩니다.

    에이전트 state의 files 필드에 대한 리듀서 함수로 사용되며,
    가상 파일 시스템에 대한 incremental 업데이트를 가능하게 합니다.

    Args:
        left: 왼쪽 dictionary (기존 파일)
        right: 오른쪽 dictionary (새 파일 또는 업데이트된 파일)

    Returns:
        왼쪽 값을 오른쪽 값으로 덮어쓴 병합된 사전
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right} # 두 딕셔너리를 하나로 합치되, 키가 겹치면 오른쪽 값을 사용


class DeepAgentState(AgentState):
    """작업 추적 및 가상 파일 시스템을 포함하는 확장된 에이전트 state.

    LangGraph의 AgentState를 상속하며 다음을 추가합니다:
    - todos: 작업 계획 및 진행 상황 추적을 위한 Todo item list
    - files: 파일 이름을 콘텐츠에 매핑하는 dictionary로 저장된 가상 파일 시스템
    """

    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
