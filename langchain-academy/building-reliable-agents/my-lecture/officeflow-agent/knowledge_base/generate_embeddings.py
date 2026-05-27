"""Knowledge base의 임베딩을 미리 생성해서 캐시로 저장하는 스크립트.

에이전트가 시작될 때마다 임베딩을 새로 만들면 느리기 때문에,
이 스크립트로 한 번 생성해두면 이후 에이전트가 캐시 파일을 그대로 읽어와 빠르게 부팅됩니다.
"""
import asyncio
import json
import os
from pathlib import Path

import voyageai
from dotenv import load_dotenv

# .env 파일에서 환경 변수(VOYAGE_API_KEY 등)를 로드
load_dotenv()

# Voyage AI 비동기 클라이언트 초기화 (네트워크 I/O 대기 효율화를 위해 Async 사용)
client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))

# ===== 설정 =====
DOCS_DIR = "./documents"                 # 원문 마크다운 문서가 위치한 디렉터리
EMBEDDINGS_DIR = "./embeddings"          # 생성된 임베딩 캐시를 저장할 디렉터리
EMBEDDING_MODEL = "voyage-4-lite"        # 임베딩 모델 (200M 무료 토큰 제공)


async def generate_embeddings():
    """문서 단위(파일 하나 = 청크 하나)로 임베딩을 생성한다."""
    docs_path = Path(DOCS_DIR)
    embeddings_path = Path(EMBEDDINGS_DIR)

    # 1) documents 디렉터리에서 마크다운 파일들을 로드
    #    CHUNKING_NOTES.md는 메타 설명 문서이므로 인덱싱 대상에서 제외
    docs = []
    for file_path in docs_path.glob("*.md"):
        if file_path.name == "CHUNKING_NOTES.md":
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            docs.append((file_path.name, content))

    # 2) 모든 문서를 한 번의 요청으로 묶어서 임베딩 생성
    #    - Voyage 무료 등급은 3 RPM 제한이 있으므로, 문서마다 따로 호출하지 않고 배치로 보냄
    #    - 저장(indexing) 용도이므로 input_type="document" 지정 → 검색 정확도 향상
    #    - response.embeddings 는 입력 순서와 동일한 순서로 반환됨
    print(f"Generating embeddings for {len(docs)} documents using Voyage AI...")
    contents = [content for _, content in docs]
    response = await client.embed(
        contents,
        model=EMBEDDING_MODEL,
        input_type="document",
    )
    embeddings = response.embeddings
    for filename, _ in docs:
        print(f"  {filename}")

    # 3) 결과를 JSON 파일로 직렬화하여 캐시 저장
    #    agent_v0.py가 동일 경로(embeddings_voyage.json)에서 캐시를 읽어 사용함
    cache_data = {
        "docs": docs,
        "embeddings": embeddings
    }
    cache_path = embeddings_path / "embeddings_voyage.json"
    embeddings_path.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    print(f"\nSaved embeddings to {cache_path}")


async def main():
    await generate_embeddings()
    print("\nDone! Agents will now load embeddings from cache.")


if __name__ == "__main__":
    asyncio.run(main())
