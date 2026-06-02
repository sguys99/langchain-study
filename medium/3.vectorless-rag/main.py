"""
main.py
-------
Vectorless RAG — LangGraph Agent + PDF Tree (PageIndex 없이 구현)

[한글 설명]
이 파일은 "벡터 없는(vectorless) RAG" 데모의 진입점(entry point)이다.
임베딩 벡터/벡터 DB 없이, PDF를 계층적 문서 트리(DocumentTree)로 만들어 두고
LLM 에이전트가 트리를 탐색하며 질의에 답한다.

전체 흐름(Flow):
  1. Bigtable 논문 PDF를 내려받는다.
  2. PDF를 파싱해 DocumentTree로 만든다. (최초 1회만 수행, 결과는 JSON으로 캐시)
  3. 각 질문마다: 에이전트가 트리를 탐색 → 관련 섹션을 수집 → 답변을 생성한다.

설치(Install):
  pip install PyMuPDF pymupdf4llm anthropic langgraph pydantic python-dotenv

.env 예시:
  ANTHROPIC_API_KEY=...
  ANTHROPIC_MODEL=claude-sonnet-4-6
"""

import json
import os
import urllib.request
from pathlib import Path
from dataclasses import asdict

# anthropic: 이 프로젝트의 기본 LLM 클라이언트
import anthropic
# .env 파일의 환경 변수(API 키, 모델명)를 로딩
from dotenv import load_dotenv

# retriever: 트리 탐색 기반 검색/답변 생성 + 워크플로 시각화
from retriever import retrieve, generate_workflow_png
# tree: PDF → 문서 트리 파싱 및 트리 노드 자료구조
from tree import parse_pdf, TreeNode
# questions: 트리 탐색/검색 품질을 검증하기 위한 한글 샘플 질문 목록
from questions import QUESTIONS

load_dotenv()

# ── 설정(Config) ───────────────────────────────────────────────────────────────
# 데모로 사용할 Google Bigtable(OSDI'06) 논문 PDF URL
PDF_URL = (
    "https://static.googleusercontent.com/media/research.google.com"
    "/en//archive/bigtable-osdi06.pdf"
)
PDF_PATH        = Path("bigtable-osdi06.pdf")            # 내려받은 PDF가 저장될 로컬 경로
TREE_CACHE_PATH = Path("results/document_tree.json")    # 파싱한 문서 트리를 캐시할 JSON 경로

# 참고: 사용할 LLM 모델은 이 파일에서 지정하지 않는다.
# 모델은 retriever.DEFAULT_MODEL(= .env의 ANTHROPIC_MODEL, 기본값 "claude-sonnet-4-6")이
# 단일 출처(single source of truth)이므로, 여기서는 클라이언트만 만들어 넘긴다.

# ── 클라이언트 초기화 (tree.py / retriever.py와 공유되는 단일 인스턴스) ──────────────
# ANTHROPIC_API_KEY가 없으면 더 진행할 수 없으므로 친절한 안내와 함께 종료한다.
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "ANTHROPIC_API_KEY가 설정되지 않았습니다.\n"
        ".env 파일을 만들고 다음과 같이 입력하세요:\n"
        "  ANTHROPIC_API_KEY=...\n"
        "  ANTHROPIC_MODEL=claude-sonnet-4-6"
    )

# 여기서 만든 client 인스턴스는 ask() → retriever.retrieve()로 그대로 전달되어
# 모든 LLM 호출에서 재사용된다. (호출마다 새 클라이언트를 만들지 않도록 한 곳에서 생성)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── 1단계: PDF 다운로드 ──────────────────────────────────────────────────────────
def download_pdf() -> None:
    """Bigtable 논문 PDF를 내려받는다. 이미 존재하면 다시 받지 않는다."""
    if PDF_PATH.exists():
        print(f"[✓] PDF가 이미 존재합니다: {PDF_PATH}")
        return
    print("[↓] Bigtable 논문 다운로드 중 …")
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print(f"[✓] 저장 완료 → {PDF_PATH}")


# ── 2단계: 트리 빌드 / 로드 ─────────────────────────────────────────────────────
def dict_to_treenode(data: dict) -> TreeNode:
    """캐시된 dict로부터 TreeNode를 재귀적으로 복원한다.

    JSON 캐시는 중첩된 dict 형태로 저장되므로, 가장 깊은 자식부터 위로 올라오며
    TreeNode 객체로 다시 조립한다.
    """
    # 자식 노드들을 먼저 재귀적으로 복원
    children = [
        dict_to_treenode(child) for child in data.get("children", [])
    ]
    # 원본 dict를 복사한 뒤 children 필드만 복원된 TreeNode 리스트로 교체
    data_copy = data.copy()
    data_copy["children"] = children
    # 나머지 필드(id/title/level/... )는 키워드 인자로 그대로 펼쳐 TreeNode 생성
    return TreeNode(**data_copy)


def get_tree() -> TreeNode:
    """캐시된 트리를 JSON에서 로드하거나, 없으면 새로 빌드해 캐시한다.

    - 캐시(JSON)가 있으면: 파일을 읽어 dict_to_treenode로 TreeNode를 복원한다.
      (DocumentTree로 저장된 경우 그 안의 "root"를 꺼내고, TreeNode 단독이면 그대로 사용)
    - 캐시가 없으면: parse_pdf로 PDF를 파싱해 트리를 만들고, 그 결과를 JSON으로 캐시한다.
    """
    # 캐시 파일을 담을 디렉터리(results/)를 미리 생성
    TREE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TREE_CACHE_PATH.exists():
        print(f"[✓] 캐시된 트리를 불러옵니다: {TREE_CACHE_PATH}")
        with open(TREE_CACHE_PATH) as f:
            data = json.load(f)
        # 캐시 dict로부터 TreeNode 복원 (DocumentTree면 root를 추출)
        tree = dict_to_treenode(data.get("root", data))
        print(f"    {len(tree.children)}개 섹션을 불러왔습니다")
        return tree

    print("[~] 트리를 빌드합니다 (최초 실행 — PyMuPDF4LLM 기준 약 10~30초 소요) …")
    tree = parse_pdf(str(PDF_PATH))

    # 다음 실행부터 재사용할 수 있도록 트리를 JSON으로 직렬화해 저장
    # (dataclass → dict 변환은 asdict, 직렬화 불가 타입은 str로 변환)
    with open(TREE_CACHE_PATH, "w") as f:
        json.dump(asdict(tree), f, indent=2, default=str)
    print(f"[✓] 트리를 캐시했습니다 → {TREE_CACHE_PATH}")
    return tree


# ── 3단계: 질문하기 ─────────────────────────────────────────────────────────────
def ask(question: str, tree: TreeNode) -> dict:
    """하나의 질문에 대해 트리를 탐색해 답변하고, 그 과정을 보기 좋게 출력한다.

    retriever.retrieve()가 돌려주는 결과 dict의 주요 키:
      - reasoning  : 에이전트가 마지막에 내린 판단의 근거(한 문장)
      - confidence : 답을 담고 있다는 확신도(0~1)
      - path       : 루트부터 거쳐온 노드 경로
      - sources    : 답변의 근거가 된 본문 청크들(헤더 + 본문)
      - answer     : 최종 생성된 한글 답변
    """
    print(f"\n{'─'*70}")
    print(f"  Q: {question}")
    print(f"{'─'*70}")

    # 미리 만들어 둔 단일 LLM 클라이언트를 retrieve에 전달 (모델은 retriever 기본값 사용)
    result = retrieve(question, tree, client)

    print(f"\n  [판단 근거]  {result['reasoning']}")
    print(f"  [확신도]    {result['confidence']:.0%}")
    print(f"  [탐색 경로]  {' → '.join(result['path'])}")

    if result["sources"]:
        print(f"\n  [출처]")
        for src in result["sources"][:2]:
            print(f"    {src.splitlines()[0]}")   # 각 청크의 첫 줄(제목/페이지 헤더)만 표시

    print(f"\n  [답변]\n{result['answer']}")
    return result


# ── 메인(Main) ──────────────────────────────────────────────────────────────────
def main() -> None:
    """전체 파이프라인 실행: PDF 다운로드 → 트리 준비 → 질문별 답변 생성."""
    download_pdf()

    # 트리를 로드(또는 빌드)
    tree = get_tree()
    # 워크플로 구조를 PNG 이미지로 시각화해 저장
    TREE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    workflow_png_path = TREE_CACHE_PATH.parent / "workflow.png"
    generate_workflow_png(output_path=str(workflow_png_path))
    print(f"[✓] 워크플로 다이어그램 저장 → {workflow_png_path}")
    print(f"\n{'═'*70}")
    print("  Vectorless RAG — Google Bigtable (PageIndex 없이)")
    print(f"{'═'*70}")

    # 각 질문을 순회하며 답변 생성. 한 질문이 실패해도 나머지는 계속 진행한다.
    results = []
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}]")
        try:
            result = ask(question, tree)
            results.append({"question": question, "result": result, "ok": True})
        except Exception as e:
            print(f"  [오류] {e}")
            results.append({"question": question, "error": str(e), "ok": False})

    # 성공한 질문 수를 집계해 마지막에 요약 출력
    ok = sum(r["ok"] for r in results)
    print(f"\n{'═'*70}")
    print(f"  완료: 전체 {len(results)}개 중 {ok}개 질문 답변 성공")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
