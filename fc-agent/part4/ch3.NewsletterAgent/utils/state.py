from typing import Dict, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from .models import NewsletterThemeOutput

def merge_dicts(left:Dict, right:Dict) -> Dict:
    return {**left, **right}

class State(TypedDict):
    keyword: str
    article_titles: List[str]
    newsletter_theme: NewsletterThemeOutput
    sub_theme_articles: Dict[str, List[Dict]]
    results: Annotated[Dict[str, str], merge_dicts]
    messages: Annotated[List, add_messages]
    
