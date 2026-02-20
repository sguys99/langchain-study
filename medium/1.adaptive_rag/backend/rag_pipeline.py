from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class AdaptiveRAG:
    
    def __init__(self, vector_db: FAISS):
        self.db = vector_db
        
    def retrieve(self, query: str)-> List[Document]:
        if not query.strip():
            return [] # 질의가 없으면 빈 리스트 리턴
        
        # 질의 길의에 따라 k수 조정
        token_count = len(query.split())
        k = 3 if token_count < 6 else 8
        
        return self.db.similarity_search(query=query, k=k)
    
    
    
def build_vector_store(texts: List[str]) -> FAISS:
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )
    
    docs = []
    for text in texts:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(chunk)
            
    return FAISS.from_texts(texts=docs, embedding=embeddings)