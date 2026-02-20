import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Adaptive RAG Assistant")
st.title("Adaptive RAG Support Assistant")

query = st.text_input("Enter your question")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=60
                )
                response.raise_for_status()

            answer = response.json()["response"]

            st.markdown("### Answer:")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")