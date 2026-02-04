from dotenv import load_dotenv

load_dotenv("../../.env")

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Dict, Any
from requests import get


mcp = FastMCP("mcp_server")

tavily_client = TavilyClient()


# Tool for searching the web
@mcp.tool()
def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information"""

    results = tavily_client.search(query)

    return results


# Resources - provide access to langchain-ai repo files
@mcp.resource("github://langchain-ai/langchain-mcp-adapters/blob/main/README.md")
def github_file():
    """
    Resource for accessing langchain-ai/langchain-mcp-adapters/README.md file

    """
    url = f"https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/blob/main/README.md"
    try:
        resp = get(url)
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"


# Prompt template
@mcp.prompt()
def prompt():
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return """
    당신은 Langchain, LangGraph, LangSmith에 대한 질문에 답변하는 친절한 어시스턴트 입니다.

    질문에 답변하기 위해 다음 tools/resources를 사용할 수 있습니다.
    - search_web: Search the web for information
    - github_file: Access the langchain-ai repo files
    
    사용자가 LangChain, LangGraph or LangSmith와 관련 없는 질문을 하면, "죄송합니다. 저는 랭체인, 랭그레프, 랭스미스에 대해서만 답변할수 있어요." 라고 답변합니다.
    
    당신은 사용자 질문에 답변을 위해 다수의 tool과 resouce를 사용할 수 있습니다.
    
    당신은 사용자의 질문을 보다 잘 이해하기 위해 사용자에게 질문의 명확하게 식별하기 위한 질문을 할수 있습니다.(reasking)
    """

if __name__ == "__main__":
    mcp.run(transport="stdio")