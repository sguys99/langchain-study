# tools/rag_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# RAG Tool: PDF ingestion → FAISS vector store → chunk retrieval
#
# Embedding model: sentence-transformers/all-MiniLM-L6-v2  (runs locally, free)
# Vector store:    FAISS (CPU)
#
# NOTE: This tool only retrieves relevant chunks. Answer synthesis
# is handled by the central synthesis node.
#
# Usage:
#   # One-time build (or whenever PDFs change):
#   uv run python -m tools.rag_tool            # build if missing
#   uv run python -m tools.rag_tool --force    # force rebuild
#
#   # Query:
#   result = run_rag_tool("What is the return policy?", history)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import pickle
import re
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from observability import trace  # NEW: Import observability


from config import (
    PDF_DIR, FAISS_INDEX_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K,
)

# ── Embedding helper ──────────────────────────────────────────────────────────

_embedder = None  # Global embedder cache

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[rag_tool] Loading embedding model: {EMBEDDING_MODEL} …")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 array of L2-normalised embeddings."""
    model  = _get_embedder()
    vecs   = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")

# ── PDF → chunks ──────────────────────────────────────────────────────────────

def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    pages  = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    Each chunk is a dict: {"text": str, "source": str, "chunk_id": int}
    """
    text   = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start  = 0
    idx    = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source, "chunk_id": idx})
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── Index build & load ────────────────────────────────────────────────────────

_INDEX_FILE = None   # will be set after FAISS_INDEX_DIR is resolved
_META_FILE  = None


def _index_paths():
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    idx_file  = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    meta_file = os.path.join(FAISS_INDEX_DIR, "metadata.pkl")
    return idx_file, meta_file


def build_index(force: bool = False) -> None:
    """
    Ingest all PDFs in PDF_DIR and build a FAISS flat IP index.
    Saves index + metadata to FAISS_INDEX_DIR.
    """
    os.makedirs(PDF_DIR, exist_ok=True)
    idx_file, meta_file = _index_paths()

    if os.path.exists(idx_file) and not force:
        print(f"[rag_tool] FAISS index already exists at {idx_file}. Skipping build.")
        print("           Call build_index(force=True) to rebuild.")
        return

    pdf_files = [
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"[rag_tool] WARNING: No PDFs found in {PDF_DIR}.")
        print("           Add PDF files there and re-run build_index().")
        # Build an empty (but valid) index so the rest of the code doesn't crash
        dim   = _get_embedder().get_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, idx_file)
        with open(meta_file, "wb") as fh:
            pickle.dump([], fh)
        return

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        source = os.path.basename(pdf_path)
        print(f"[rag_tool] Parsing {source} …")
        text   = _extract_text_from_pdf(pdf_path)
        chunks = _chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"           → {len(chunks)} chunks")

    texts  = [c["text"] for c in all_chunks]
    vecs   = embed_texts(texts)

    dim    = vecs.shape[1]
    index  = faiss.IndexFlatIP(dim)   # Inner Product on normalised vecs = cosine sim
    index.add(vecs)

    faiss.write_index(index, idx_file)
    with open(meta_file, "wb") as fh:
        pickle.dump(all_chunks, fh)

    print(f"[rag_tool] ✓ Index built: {len(all_chunks)} chunks, dim={dim}")


def _load_index():
    """Load FAISS index and metadata from disk. Build if missing."""
    idx_file, meta_file = _index_paths()
    if not os.path.exists(idx_file):
        build_index()
    index = faiss.read_index(idx_file)
    with open(meta_file, "rb") as fh:
        metadata = pickle.load(fh)
    return index, metadata


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, k: int = RAG_TOP_K) -> list[dict]:
    """
    Embed query and return top-k most similar chunks.
    Each returned dict: {"text": str, "source": str, "chunk_id": int, "score": float}
    """
    index, metadata = _load_index()

    if index.ntotal == 0:
        return []

    q_vec  = embed_texts([query])
    scores, idxs = index.search(q_vec, min(k, index.ntotal))

    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        chunk = dict(metadata[i])
        chunk["score"] = float(score)
        results.append(chunk)
    return results

@trace(span_type="RETRIEVER", attributes={  # NEW: Add tracing
    "retrieval.model": EMBEDDING_MODEL,
    "retrieval.top_k": RAG_TOP_K,
})
def run_rag_tool(user_question: str, conversation_history: list[dict]) -> dict:
    """
    Main entry point for the RAG tool.

    Returns a dict:
      {
        "chunks":  list[dict],   # retrieved chunks with scores and text
        "sources": list[str],    # unique source document names
      }
    """
    query = user_question
    if conversation_history:
        for turn in reversed(conversation_history):
            prior = turn.get("content")
            if turn.get("role") == "user" and prior and prior != user_question:
                query = f"{prior}\n{user_question}"
                break

    chunks = retrieve_chunks(query, k=RAG_TOP_K)

    sources = list({c["source"] for c in chunks})

    return {
        "chunks":  chunks,
        "sources": sources,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the RAG FAISS index from PDFs in PDF_DIR.")
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the index even if it already exists.",
    )
    args = parser.parse_args()
    build_index(force=args.force)