from typing import TypedDict, List
from langgraph.graph import START, StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    
    
def create_workflow(rag):
    
    llm = ChatOpenAI(
        model="gpt-5o-mini",
        temperature=0
    )
    
    workflow = StateGraph(GraphState)
    
    # 리트리버 노드
    async def retrieve_node(state: GraphState):
        docs = rag.retrieve(state["question"])
        return {"docs": docs}
    
    
    # reasoning node
    async def reasoning_node(state: GraphState):
        question = state["question"]
        docs = state.get("docs", [])

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        당신은 기술 지원 담당자입니다..

        질문에 답변하기 위해 아래의 Context만 사용하세요.
        답변에 필요한 정보가 Context에 업다면, 모든다고 답변하세요.

        Context:
        {context}

        Question:
        {question}
        """

        response = await llm.ainvoke(prompt)

        return {"answer": response.content}

    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reason", reasoning_node)

    # Connect nodes
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "reason")
    workflow.add_edge("reason", END)

    return workflow.compile()    