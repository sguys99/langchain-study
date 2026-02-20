import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import AdaptiveRAG, build_vector_store
from graph_workflow import create_workflow

load_dotenv()

# ---------------------------
# Startup Initialization
# ---------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    # startup
    global workflow

    # Sample knowledge base (replace with real docs)
    sample_docs = [
        "LangGraph supports stateful workflows and retry logic.",
        "Adaptive RAG dynamically changes retrieval depth based on query complexity.",
        "FastAPI is a high-performance async Python framework.",
    ]

    vector_db = build_vector_store(sample_docs)
    rag = AdaptiveRAG(vector_db)
    workflow = create_workflow(rag)

    yield


app = FastAPI(title="Adaptive RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str

    
# ---------------------------
# API Endpoint
# ---------------------------

@app.post("/ask")
async def ask(payload: AskRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = await workflow.ainvoke(
            {"question": payload.query}
        )

        return {"response": result["answer"]}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Internal RAG processing error"
        )