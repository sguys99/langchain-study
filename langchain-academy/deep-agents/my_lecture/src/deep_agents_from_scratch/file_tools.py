"""Virtual file system tools for agent state management.

This module provides tools for managing a virtual filesystem stored in agent state,
enabling context offloading and information persistence across agent interactions.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agents_from_scratch.prompts import (
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
)
from deep_agents_from_scratch.state import DeepAgentState


@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""
    return list(state.get("files", {}).keys())


@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """선택 사항인 오프셋 및 제한을 사용하여 가상 파일 시스템에서 파일 내용을 읽습니다.

    Args:
        file_path: 읽을 파일의 경로
        state: 가상 파일 시스템을 포함하는 에이전트 상태(툴 노드에 주입됨)
        offset: 읽기를 시작할 줄 번호(기본값: 0)
        limit: 읽을 최대 줄 수(기본값: 2000)

    Returns:
        줄 번호가 포함된 형식화된 파일 내용, 또는 파일을 찾을 수 없는 경우 오류 메시지
    """
    files = state.get("files", {})
    if file_path not in files:
        return f"Error: File '{file_path}' not found"

    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    return "\n".join(result_lines)


@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """가상 파일 시스템의 파일에 내용을 기록합니다.

    Args:
        file_path: 파일을 생성하거나 업데이트할 경로
        content: 파일에 기록할 내용
        state: 가상 파일 시스템을 포함하는 에이전트 상태(툴 노드에 주입됨)
        tool_call_id: 메시지 응답을 위한 툴 호출 식별자(툴 노드에 주입됨)

    Returns:
        새로운 파일 내용으로 에이전트 상태를 업데이트하는 명령
    """
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )
