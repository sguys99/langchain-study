import streamlit as st
from trading_graph import graph 

# Streamlit 페이지 설정
# 페이지 제목과 아이콘을 설정하여 웹 애플리케이션의 기본 정보를 정의합니다.
st.set_page_config(page_title="LangGraph + Streamlit", page_icon="📈")

# Streamlit UI 구성
# 애플리케이션의 메인 제목과 설명을 표시합니다.
st.title("주식 분석 멀티 에이전트 시스템")
st.markdown("""
이 앱은 멀티 에이전트 시스템을 사용하여 주식을 분석합니다. 
주식 티커와 질문을 입력하시면 종합적인 분석 결과를 제공해드립니다.
""")

# 세션 상태 초기화
# 사용자와 AI 간의 대화 내역을 저장할 리스트를 세션 상태에 초기화합니다.
# Streamlit의 세션 상태는 페이지가 새로고침되어도 데이터가 유지됩니다.
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전 대화 내역 표시
# 저장된 모든 메시지를 순회하며 채팅 형식으로 표시합니다.
# 각 메시지는 발신자(사용자/AI)에 따라 다른 스타일로 표시됩니다.
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 받기
# 채팅 입력 필드를 통해 사용자로부터 주식 티커와 질문을 받습니다.
user_input = st.chat_input("주식 티커와 질문을 입력하세요 (예: 'AAPL에 투자해야 할까요?')")

# 그래프 실행을 위한 기본 설정
# 스레드 ID를 포함한 설정 객체를 생성합니다.
# 이 설정은 그래프 실행 시 필요한 메타데이터를 포함합니다.
config = {
    'configurable': {
        'thread_id': '1234'
    }
}

# 사용자 입력이 있을 경우 처리
if user_input:
    # 사용자 메시지를 대화 내역에 추가
    # 입력된 메시지를 세션 상태의 message_list에 추가합니다.
    st.session_state.message_list.append({"role": "user", "content": user_input})
    
    # 사용자 메시지를 UI에 표시
    # 채팅 메시지 형식으로 사용자의 입력을 화면에 표시합니다.
    st.chat_message("user").write(user_input)
    
    # AI 응답 생성 중 로딩 표시
    # AI가 응답을 생성하는 동안 사용자에게 로딩 스피너를 보여줍니다.
    with st.spinner("응답을 생성하는 중..."):
        # AI 응답을 채팅 메시지로 표시
        with st.chat_message("assistant"):
            try:
                # 그래프 실행 및 AI 응답 생성
                # 사용자 입력을 기반으로 그래프를 실행하여 AI 응답을 생성합니다.
                ai_message = graph.invoke(
                    {"messages": [("user", user_input)]}, config=config
                );
                
                # 생성된 AI 응답 추출
                # 그래프의 상태에서 마지막 메시지(AI 응답)를 추출합니다.
                ai_message = graph.get_state(config).values['messages'][-1].content
                
                # AI 응답을 UI에 표시
                st.write(ai_message)
            
                # AI 응답을 대화 내역에 추가
                # 생성된 AI 응답을 세션 상태의 message_list에 추가합니다.
                st.session_state.message_list.append({"role": "assistant", "content": ai_message})
            
            except Exception as e:
                # 에러 처리
                # API 호출 중 발생한 오류를 사용자에게 표시합니다.
                st.error(f"Error during streaming: {str(e)}")
                # API 속도 제한으로 인한 오류일 수 있음을 사용자에게 알립니다.
                st.info("This may be due to API rate limits. Please wait a minute and try again with a simpler query.")


