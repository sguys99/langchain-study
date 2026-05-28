# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the E-Commerce Conversational Agent
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SERPER_API_KEY    = os.getenv("SERPER_API_KEY", "")
VOYAGE_API_KEY   = os.getenv("VOYAGE_API_KEY", "")

# ── Model ─────────────────────────────────────────────────────────────────────
LLM_MODEL       = "claude-sonnet-4-6"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 1024

# ── Embedding model ───────────────────────────────────────────────────────────
# 상용 Voyage AI 임베딩 사용 (기본값: voyage-4-lite, .env 의 VOYAGE_MODEL 로 오버라이드 가능)
EMBEDDING_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4-lite")

# 로컬(오픈소스) 임베딩 — 필요 시 위 EMBEDDING_MODEL 을 주석 처리하고 아래 줄을 사용
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DB_PATH         = os.path.join(BASE_DIR, "data", "ecommerce.db")
RAW_DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")
PDF_DIR         = os.path.join(BASE_DIR, "pdf_docs")

# ── RAG settings ──────────────────────────────────────────────────────────────
RAG_TOP_K       = 4         # chunks retrieved per query
CHUNK_SIZE      = 500       # characters per chunk
CHUNK_OVERLAP   = 80        # character overlap between chunks

# ── Web search settings ──────────────────────────────────────────────────────
WEB_SEARCH_TOP_K   = 5      # results requested from Serper
WEB_SEARCH_TIMEOUT = 10     # seconds

# ── Routing ───────────────────────────────────────────────────────────────────
# The router LLM call picks one of these exact route names
ROUTES = ["sql", "rag", "web_search"]