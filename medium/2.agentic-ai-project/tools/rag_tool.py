# tools/rag_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# RAG 도구: PDF 인제스트 → FAISS 벡터 스토어 → 청크 검색
#
# 임베딩 모델:
#   - 기본값: Voyage AI (상용, 예: voyage-4-lite)
#   - 선택  : sentence-transformers (로컬, 무료)
#   사용할 모델은 config.EMBEDDING_MODEL 값으로 결정되며,
#   값이 "voyage" 로 시작하면 Voyage AI, 그렇지 않으면 SentenceTransformer 가 사용된다.
#
# 벡터 스토어: FAISS (CPU)
#
# 참고: 이 도구는 관련 청크 검색만 담당하며,
#       최종 답변 합성은 중앙 synthesis 노드에서 수행된다.
#
# 사용법:
#   # 최초 1회 빌드 (또는 PDF 가 변경되었을 때마다 실행):
#   uv run python -m tools.rag_tool            # 인덱스가 없을 때만 빌드
#   uv run python -m tools.rag_tool --force    # 강제 재빌드
#
#   # 질의:
#   result = run_rag_tool("환불 정책이 어떻게 되나요?", history)
# ─────────────────────────────────────────────────────────────────────────────

import os
import pickle
import re
import numpy as np
import faiss
from pypdf import PdfReader
from observability import trace  # 관측성(트레이싱) 데코레이터


from config import (
    PDF_DIR, FAISS_INDEX_DIR, EMBEDDING_MODEL, VOYAGE_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K,
)

# ── 임베딩 헬퍼 ──────────────────────────────────────────────────────────────
#
# Voyage AI 와 SentenceTransformer 두 가지 provider 를 지원한다.
# EMBEDDING_MODEL 문자열이 "voyage" 로 시작하면 Voyage AI 를 사용하고,
# 그렇지 않으면 로컬 SentenceTransformer 모델을 사용한다.

_embedder = None          # 임베더 인스턴스 캐시 (provider 별 객체)
_provider: str | None = None  # "voyage" 또는 "sentence-transformers"


def _resolve_provider() -> str:
    """EMBEDDING_MODEL 값을 보고 사용할 임베딩 provider 를 결정한다."""
    return "voyage" if EMBEDDING_MODEL.lower().startswith("voyage") else "sentence-transformers"


def _get_embedder():
    """
    임베딩 클라이언트/모델을 lazy-load 하여 캐시한다.
    - Voyage: voyageai.Client 인스턴스 반환
    - 로컬  : sentence_transformers.SentenceTransformer 인스턴스 반환
    """
    global _embedder, _provider
    if _embedder is not None:
        return _embedder

    _provider = _resolve_provider()

    if _provider == "voyage":
        # 상용 Voyage AI 임베딩 사용
        import voyageai  # 지연 import (로컬 전용 환경에서 voyageai 미설치여도 동작)
        if not VOYAGE_API_KEY:
            raise RuntimeError(
                "[rag_tool] VOYAGE_API_KEY 가 설정되어 있지 않습니다. "
                ".env 에 키를 추가하거나, config.EMBEDDING_MODEL 을 로컬 모델로 변경하세요."
            )
        print(f"[rag_tool] Voyage AI 임베딩 클라이언트 초기화: {EMBEDDING_MODEL} …")
        _embedder = voyageai.Client(api_key=VOYAGE_API_KEY)
    else:
        # 로컬 SentenceTransformer 임베딩 사용 (선택 사항, 무료)
        from sentence_transformers import SentenceTransformer  # 지연 import
        print(f"[rag_tool] 로컬 임베딩 모델 로드: {EMBEDDING_MODEL} …")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    return _embedder


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화. FAISS IndexFlatIP 에서 코사인 유사도로 동작하도록 보장한다."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # 0-division 방지
    return vecs / norms


def embed_texts(texts: list[str], input_type: str = "document") -> np.ndarray:
    """
    텍스트 목록을 임베딩하여 (N, D) float32 배열로 반환한다.
    반환되는 벡터는 L2 정규화된 상태이므로 IndexFlatIP 와 함께 사용하면 코사인 유사도가 된다.

    Parameters
    ----------
    texts      : 임베딩 대상 문자열 리스트
    input_type : Voyage AI 전용 힌트 ("document" 또는 "query").
                 로컬 SentenceTransformer 에서는 무시된다.
    """
    model = _get_embedder()

    if _provider == "voyage":
        # Voyage AI: document/query 용도를 구분하면 검색 품질이 향상된다.
        result = model.embed(texts, model=EMBEDDING_MODEL, input_type=input_type)
        vecs = np.array(result.embeddings, dtype="float32")
        # Voyage 임베딩은 기본적으로 단위벡터가 아니므로 수동으로 정규화한다.
        vecs = _l2_normalize(vecs)
    else:
        # SentenceTransformer: normalize_embeddings=True 옵션으로 자체 정규화 지원
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        vecs = np.array(vecs, dtype="float32")

    return vecs

# ── PDF → 청크 ───────────────────────────────────────────────────────────────

def _extract_text_from_pdf(pdf_path: str) -> str:
    """PDF 파일에서 모든 페이지의 텍스트를 추출한다."""
    reader = PdfReader(pdf_path)
    pages  = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _chunk_text(text: str, source: str) -> list[dict]:
    """
    텍스트를 오버랩이 있는 청크들로 분할한다.
    각 청크는 dict 형태: {"text": str, "source": str, "chunk_id": int}
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

# ── 인덱스 빌드 & 로드 ───────────────────────────────────────────────────────

_INDEX_FILE = None   # FAISS_INDEX_DIR 가 결정된 뒤 설정됨
_META_FILE  = None


def _index_paths():
    """FAISS 인덱스 및 메타데이터 파일 경로를 반환한다 (디렉토리는 자동 생성)."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    idx_file  = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    meta_file = os.path.join(FAISS_INDEX_DIR, "metadata.pkl")
    return idx_file, meta_file


def build_index(force: bool = False) -> None:
    """
    PDF_DIR 의 모든 PDF 를 인제스트하여 FAISS Flat-IP 인덱스를 빌드한다.
    완성된 인덱스와 메타데이터는 FAISS_INDEX_DIR 에 저장된다.
    """
    os.makedirs(PDF_DIR, exist_ok=True)
    idx_file, meta_file = _index_paths()

    if os.path.exists(idx_file) and not force:
        print(f"[rag_tool] FAISS 인덱스가 이미 존재합니다: {idx_file}. 빌드를 건너뜁니다.")
        print("           강제로 재빌드하려면 build_index(force=True) 를 호출하세요.")
        return

    pdf_files = [
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"[rag_tool] 경고: {PDF_DIR} 에 PDF 파일이 없습니다.")
        print("           PDF 를 추가한 뒤 build_index() 를 다시 실행하세요.")
        # PDF 가 없더라도 빈 인덱스를 만들어 두어야 이후 코드가 깨지지 않는다.
        # 차원을 알기 위해 더미 텍스트 한 건을 임베딩한다.
        dummy = embed_texts(["dimension probe"], input_type="document")
        dim   = dummy.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, idx_file)
        with open(meta_file, "wb") as fh:
            pickle.dump([], fh)
        return

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        source = os.path.basename(pdf_path)
        print(f"[rag_tool] 파싱 중: {source} …")
        text   = _extract_text_from_pdf(pdf_path)
        chunks = _chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"           → {len(chunks)} 개 청크 생성")

    texts  = [c["text"] for c in all_chunks]
    # 인제스트 단계이므로 input_type="document" 로 임베딩한다.
    vecs   = embed_texts(texts, input_type="document")

    dim    = vecs.shape[1]
    index  = faiss.IndexFlatIP(dim)   # 정규화된 벡터에 대한 내적 = 코사인 유사도
    index.add(vecs)

    faiss.write_index(index, idx_file)
    with open(meta_file, "wb") as fh:
        pickle.dump(all_chunks, fh)

    print(f"[rag_tool] ✓ 인덱스 빌드 완료: {len(all_chunks)} 청크, 차원={dim}")


def _load_index():
    """디스크에서 FAISS 인덱스와 메타데이터를 로드한다. 인덱스가 없으면 빌드한다."""
    idx_file, meta_file = _index_paths()
    if not os.path.exists(idx_file):
        build_index()
    index = faiss.read_index(idx_file)
    with open(meta_file, "rb") as fh:
        metadata = pickle.load(fh)
    return index, metadata


# ── 검색 ─────────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, k: int = RAG_TOP_K) -> list[dict]:
    """
    질의를 임베딩한 뒤 가장 유사도가 높은 top-k 청크를 반환한다.
    반환되는 각 dict: {"text": str, "source": str, "chunk_id": int, "score": float}
    """
    index, metadata = _load_index()

    if index.ntotal == 0:
        return []

    # 검색 단계이므로 input_type="query" 로 임베딩한다 (Voyage 한정으로 품질에 영향).
    q_vec  = embed_texts([query], input_type="query")
    scores, idxs = index.search(q_vec, min(k, index.ntotal))

    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        chunk = dict(metadata[i])
        chunk["score"] = float(score)
        results.append(chunk)
    return results

@trace(span_type="RETRIEVER", attributes={  # 관측성(트레이싱) 어트리뷰트
    "retrieval.model": EMBEDDING_MODEL,
    "retrieval.top_k": RAG_TOP_K,
})
def run_rag_tool(user_question: str, conversation_history: list[dict]) -> dict:
    """
    RAG 도구의 메인 진입점.

    반환 형식:
      {
        "chunks":  list[dict],   # 점수와 본문이 포함된 검색된 청크 리스트
        "sources": list[str],    # 청크들이 유래한 고유 문서명 목록
      }
    """
    # 직전 사용자 발화가 있다면 query 를 보강하여 컨텍스트를 살린다.
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

    parser = argparse.ArgumentParser(description="PDF_DIR 의 PDF 들로부터 RAG FAISS 인덱스를 빌드한다.")
    parser.add_argument(
        "--force", action="store_true",
        help="인덱스가 이미 존재하더라도 강제로 재빌드한다.",
    )
    args = parser.parse_args()
    build_index(force=args.force)
