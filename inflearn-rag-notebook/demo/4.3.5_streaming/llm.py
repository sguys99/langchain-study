from typing import Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# 여기를 참고 0.3에서는 안쓰는듯
# https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#adding-chat-history

# get_ai_message에 리턴하는 ai_message에는 입력, 컨텍스트 등 많은 정보를 가지고 있음
# 그래서 답변인 ai_message["answer"] 만 추출하는 것임
# 그런데 이렇게 하면 스트리밍 구현이 어렵다.
# chain에서 출력만 뽑도록 pick을 사용한다.


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



def get_llm(model = "gpt-4o"):
    llm = ChatOpenAI(model = model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()    
    prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 질문만 리턴해주세요
    사전: {dictionary}
    
    질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()    
    return dictionary_chain


def get_retriever():
    embedding = OpenAIEmbeddings(model = "text-embedding-3-large")
    index_name = "tax-markdown-index"
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )    
    retriever = database.as_retriever(search_kwargs = {"k":4})
    return retriever


def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()
 
    contextualize_q_system_prompt = ( # 시스템 프롬프트
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages( # 세로은 프롬프트를 만든다.
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # retriever를 가져온다.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # 메모리를 위해 추가된 부분
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    ).pick("answer")# 출력만 뽑도록
    return conversational_rag_chain

def get_ai_message(user_message):
    dictionary_chain = get_dictionary_chain()
    qa_chain = get_qa_chain()
    tax_chain = {"input": dictionary_chain} | qa_chain 
    ai_message = tax_chain.stream( # invoke 대신 stream으로 바꿔줌, Iterator임
        {
            "question": user_message
        },
        config={
        "configurable": {"session_id": "abc123"}
        })
    
    #return ai_message["answer"]
    return ai_message # pick을 했으므로

# 그리고 chat.py도 바꿔 줘야함