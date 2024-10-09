import streamlit as st
from llm import get_ai_message


st.set_page_config(page_title="소득세 챗봇", page_icon=":robot_face:") 

st.title("🤖 소득세 챗봇")  
st.caption("소득세에 관련된 모든 것을 답변해 드립니다.")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        

        
        
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해 주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    with st.spinner("답변을 생성하는 중입니다."):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            #st.write(ai_message)
            ai_result = st.write_stream(ai_message) # iterator, generator 기반을 출력하기 위해 변경
            # 그리고 변수를 전달해야한다. 완성된 것을 저장해야하기 때문이다.
        st.session_state.message_list.append({"role": "ai", "content": ai_result})
        
        # 문제점: 스트림이 동작하지 않고, 히스토리도 저장되지 않는다. 확인 필요함