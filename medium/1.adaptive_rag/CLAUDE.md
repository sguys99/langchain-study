# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack Adaptive RAG (Retrieval-Augmented Generation) proof-of-concept based on [this Medium article](https://medium.com/@robi.tomar72/build-a-full-poc-using-adaptive-rag-langgraph-fastapi-streamlit-complete-step-by-step-guide-ab8631ae5dcb). The system uses LangGraph to orchestrate a two-node workflow (retrieve → reason) and serves it via FastAPI, with a Streamlit frontend.

## Running the Application

Both servers must run simultaneously in separate terminals.

```bash
# Backend (FastAPI)
uvicorn backend.app:app --reload

# Frontend (Streamlit)
streamlit run frontend/ui.py
```

Requires a `.env` file in the project root with `OPENAI_API_KEY`.

## Architecture

```
User → Streamlit UI (frontend/ui.py)
     → POST /ask → FastAPI (backend/app.py)
     → LangGraph workflow (backend/graph_workflow.py)
     → AdaptiveRAG retriever (backend/rag_pipeline.py)
     → FAISS vector store (in-memory, built on startup)
     → OpenAI LLM (gpt-4o-mini via langchain_openai)
```

**Key design points:**

- **Adaptive retrieval** ([backend/rag_pipeline.py](backend/rag_pipeline.py)): `AdaptiveRAG.retrieve()` adjusts `k` dynamically — queries with fewer than 6 tokens fetch 3 docs, longer queries fetch 8.
- **LangGraph state** ([backend/graph_workflow.py](backend/graph_workflow.py)): `GraphState` typed dict carries `question`, `docs`, and `answer` through the graph. Two async nodes: `retrieve_node` and `reasoning_node`.
- **Startup initialization** ([backend/app.py](backend/app.py)): Vector store and compiled workflow are built during FastAPI lifespan startup. The `sample_docs` list is hardcoded in `app.py` — replace it with real documents to extend the knowledge base.
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (no API key required).
- **LLM prompt**: The system prompt in `reasoning_node` is written in Korean. The model is instructed to answer only from the retrieved context.
- **`backend/config.py`**: Currently empty; intended for centralised config/environment setup.
- **`langchain-openai`** is a required dependency not listed in `requirements.txt` — install it separately if you encounter import errors: `pip install langchain-openai`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install langchain-openai  # not yet in requirements.txt
```
