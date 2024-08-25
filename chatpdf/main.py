# 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/
# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
# https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/
# https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/
# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/
# https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/
# https://python.langchain.com/v0.2/docs/tutorials/rag/

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv


def pdf_to_documnet(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


load_dotenv()

st.title("ChatPDF")
st.write("---")

#file upload
uploaded_file = st.file_uploader("PDF 파일을 올려주세요.", type=['pdf'])
st.write("---")

#업로드되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_documnet(uploaded_file)

    # #loader
    # loader = PyPDFLoader("data/unsu2.pdf")
    # pages = loader.load_and_split()

    #split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    #embedding
    embedding_model = OpenAIEmbeddings()

    #load it into Chroma
    db = Chroma.from_documents(texts, embedding_model)

    # 질문
    st.header("PDF에게 질문을 해보세요.")
    question = st.text_input("질문을 입력하세요.")
    
    if st.button("질문하기"):
        with st.spinner("wait for it..."):
            llm = ChatOpenAI(model="gpt-4o")
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(), 
                llm=llm
                )

            # # docs = retriever_from_llm.get_relevant_documents(query=question)
            # docs = retriever_from_llm.invoke(input=question)

            # print(len(docs))
            # print(docs)

            # QA 체인을 생성
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",  # or "map_reduce" or "refine" depending on your use case
                retriever=retriever_from_llm,
            )
            result = qa_chain.invoke({"query": question})
            st.write(result["result"])