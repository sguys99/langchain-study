from tavily import TavilyClient, AsyncTavilyClient
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .models import NewsletterThemeOutput
import pathlib

# 프로젝트 루트 디렉토리 찾기
def get_project_root():
    return pathlib.Path(__file__).parent.parent

# .env 파일 로드
env_path = get_project_root() / '.env'
load_dotenv(env_path)

# API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")    
tavily_api_key = os.getenv("TAVILY_API_KEY")

# API 키 확인
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

def search_recent_news(keyword: str) -> list:
    try:
        client = TavilyClient(api_key=tavily_api_key)
        search_result = client.search(query=keyword, max_results=5, topic="news", days=5)
        titles = [result['title'] for result in search_result['results']]
        return titles
    except Exception as e:
        st.error(f"뉴스 검색 중 오류가 발생했습니다: {str(e)}")
        return []

def subtheme_generator():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    structured_llm_newsletter = llm.with_structured_output(NewsletterThemeOutput)

    system = """
    You are an expert helping to create a newsletter. Based on a list of article titles provided, your task is to choose a single, 
    specific newsletter theme framed as a clear, detailed question that grabs the reader's attention. 

    In addition, generate 3 to 5 sub-themes that are highly specific, researchable news items or insights under the main theme. 
    Ensure these sub-themes reflect the latest trends in the field and frame them as compelling news topics.

    The output should be formatted as:
    - Main theme (in question form)
    - 3-5 sub-themes (detailed and focused on emerging trends, technologies, or insights).

    The sub-themes should create a clear direction for the newsletter, avoiding broad, generic topics.
    All your output should be in Korean
    """

    theme_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Article titles: \n\n {recent_news}"),
        ]
    )

    return theme_prompt | structured_llm_newsletter

async def search_news_for_subtheme(subtheme: str):
    try:
        async_tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
        search_params = {
            "query": subtheme, 
            "max_results": 5,  # 결과 수 증가
            "topic": "news", 
            "days": 30,  # 검색 기간 확장
            "include_images": True,
            "include_raw_content": True,
            "search_depth": "advanced"  # 더 깊은 검색 수행
        }
        with st.status(label=f"'{subtheme}'와 관련된 뉴스 검색중...", expanded=True) as status:
            st.markdown(f"'{subtheme}'와 관련된 뉴스를 검색하고 있습니다.")
            response = await async_tavily_client.search(**search_params)
            images = response.get('images', [])
            results = response.get('results', [])
        
            if not results:
                st.warning(f"'{subtheme}'에 대한 검색 결과가 없습니다.")
                return {subtheme: []}
            
            article_info = []
            for i, result in enumerate(results):
                article_info.append({
                    'title': result.get('title', ''),
                    'image_url': images[i] if i < len(images) else '',
                    'raw_content': result.get('raw_content', '')
                })
        
            status.update(
                label=f"'{subtheme}'와 관련된 {len(article_info)}개의 기사를 찾았습니다.",
                state='complete',
                expanded=False
                )
        return {subtheme: article_info}
    except Exception as e:
        st.error(f"서브테마 '{subtheme}' 검색 중 오류가 발생했습니다: {str(e)}")
        return {subtheme: []} 