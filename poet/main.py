# import streamlit as st
# from langchain_community.llms.ctransformers import CTransformers


# llm = CTransformers(
#     model="llama-2-7b-chat.ggmlv3.q2_K.bin", #chat 모델과 llm 모델이 있음
#     model_type="llama"
# )

# st.title("인공지능 시인")

# content = st.text_input("시의 주제을 제시해 주세요")


# if st.button("시 작성 요청하기"):
#     with st.spinner("시 작성중..."):
#         result = llm.predict(content + "에 대한 시를 써줘")
#         st.write(result)

# # ch2.8
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0
# )

# result = llm.invoke("hi")
# print(result.content)

#-------------------------
# ch3
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_openai import ChatOpenAI

# load_dotenv()

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0
# )

# st.title("인공지능 시인")
# content = st.text_input("시의 주제를 제시해 주세요")

# if st.button("시 작성 요청하기"):
#     with st.spinner("시 작성 중"):
#         result = llm.invoke(content + "에 대한 시를 써줘")
#         st.write(result.content)
#-------------------------
# ch4: ollama serve 명령 후 진행해야 한다.
from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_models import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="llama3.1")
st.title("인공지능 시인")
content = st.text_input("시의 주제를 제시해 주세요")

if st.button("시 작성 요청하기"):
    with st.spinner("시 작성 중"):
        result = llm.invoke(content + "에 대한 시를 써줘")
        st.write(result.content)
