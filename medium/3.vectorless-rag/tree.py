"""
tree.py
-------
PyMuPDF4LLM을 사용하여 PDF를 계층적 DocumentTree(문서 트리)로 파싱한다.

벡터 임베딩 없이, 레이아웃 인식(layout-aware) 기반의 PDF 파싱을 사용한다.
(즉, 별도의 벡터 DB나 임베딩 모델 없이 문서의 구조 자체를 활용하는 "vectorless RAG" 방식)

전략(Strategy):
1. PyMuPDF4LLM으로 레이아웃을 보존한 마크다운(markdown)을 추출한다.
2. 마크다운의 헤더(#, ##, ...)를 파싱하여 트리 계층 구조로 만든다.
3. page_chunks(페이지 단위 청크)를 사용하여 정확한 페이지 경계를 탐지한다.

설치(Install): pip install PyMuPDF pymupdf4llm
"""

import os
import re
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF (PDF 처리 핵심 라이브러리)
import pymupdf4llm  # 주 파서: PDF를 LLM 친화적인 마크다운으로 변환


# ── 데이터 모델(Data models) ───────────────────────────────────────────────────

@dataclass
class TreeNode:
    """계층적 문서 노드(Hierarchical document node).

    문서 트리를 구성하는 하나의 단위로, 챕터/섹션/하위 섹션 등을 표현한다.
    각 노드는 자식 노드들을 가질 수 있어 재귀적인 트리 구조를 형성한다.
    """
    id: str                 # 노드 고유 식별자 (제목 슬러그 + 라인 번호로 생성)
    title: str              # 헤딩(제목) 텍스트
    level: int              # 계층 레벨: 0=루트, 1=챕터, 2=섹션, 3=하위 섹션
    page_start: int         # 이 노드가 시작되는 페이지 번호
    page_end: int           # 이 노드가 끝나는 페이지 번호
    content: str            # 이 노드에 속한 본문 텍스트
    children: List['TreeNode'] = field(default_factory=list)  # 하위 노드 목록
    heading_type: Optional[str] = None  # 헤딩 종류: "numbered", "unnumbered", "page" 등
    summary: str = ""       # 노드 내용 요약 (보통 첫 문단)

    def to_dict(self) -> Dict:
        """노드를 직렬화 가능한 dict로 변환한다 (JSON 출력/디버깅용)."""
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "pages": f"{self.page_start}-{self.page_end}",
            "type": self.heading_type,
            "children_count": len(self.children),
            # 본문 미리보기: 200자를 넘으면 잘라서 "..." 를 덧붙인다
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content
        }


@dataclass
class DocumentTree:
    """메타데이터를 포함한 완성된 문서 트리(Complete document tree with metadata)."""
    document_name: str      # 문서 이름 (확장자 제외 파일명)
    root: TreeNode          # 트리의 최상위(루트) 노드
    total_pages: int        # 문서 전체 페이지 수
    source_path: str = ""   # 원본 PDF 파일 경로

    def print_tree(self, node: Optional[TreeNode] = None, indent: int = 0):
        """트리 구조를 보기 좋게(들여쓰기 포함) 콘솔에 출력한다.

        node를 지정하지 않으면 루트부터 재귀적으로 전체 트리를 출력한다.
        indent는 재귀 깊이에 따라 들여쓰기 수준을 결정한다.
        """
        # 최초 호출(node=None)이면 루트 노드부터 시작하고 헤더를 출력
        if node is None:
            node = self.root
            print(f"\n📄 {self.document_name} ({self.total_pages} pages)")

        # 들여쓰기 문자열 (깊이 1단계당 공백 2칸)
        prefix = "  " * indent
        # 레벨에 따라 다른 아이콘 사용 (루트/챕터/섹션/하위 섹션)
        icon = "📑" if node.level == 0 else "📖" if node.level == 1 else "📄" if node.level == 2 else "📝"
        print(f"{prefix}{icon} [{node.level}] {node.title} (p{node.page_start}-{node.page_end})")

        # 자식 노드들을 들여쓰기를 한 단계 늘려 재귀 출력
        for child in node.children:
            self.print_tree(child, indent + 1)




class PyMuPDF4LLMTreeBuilder:
    """
    PyMuPDF4LLM을 사용하여 계층적 문서 트리를 구축하는 빌더 클래스.

    GPU 기반 방식보다 약 10배 빠르며, 별도의 모델 다운로드가 필요 없다.
    """

    def __init__(self, max_content_length: int = 8000):
        # 노드 하나에 담을 본문 최대 길이 (너무 긴 내용 방지)
        self.max_content_length = max_content_length

        # ── 헤딩 탐지를 위한 정규표현식 패턴들 ───────────────────────────────
        # 다양한 형태의 제목(섹션 번호 체계)을 분류하기 위한 패턴 모음
        self.patterns = {
            # 숫자 번호 섹션: "1. Introduction", "2.3.1 Methods" 등
            'numbered_section': re.compile(r'^(?:\d+\.)+\s+(.+)$'),
            # 로마 숫자 섹션: "I. Introduction", "II.3" 등 (대소문자 무시)
            'roman_section': re.compile(r'^(?:[IVX]+)\.?\s+(.+)$', re.IGNORECASE),
            # 알파벳 섹션: "A. Methods", "B. Results" 등
            'letter_section': re.compile(r'^([A-Z])\.\s+(.+)$'),
            # 번호 없는 헤딩: "Abstract", "Conclusion" 등 (대문자로 시작하는 짧은 제목)
            'unnumbered_heading': re.compile(r'^([A-Z][a-zA-Z\s]{3,50})$'),
        }

    def parse_pdf(self, pdf_path: str) -> DocumentTree:
        """
        PDF를 계층적 트리 구조로 파싱한다.

        전략(Strategy):
        1. PyMuPDF4LLM으로 레이아웃을 보존한 마크다운을 추출한다.
        2. 마크다운 헤더를 트리 계층 구조로 파싱한다.
        3. page_chunks를 사용해 정확한 페이지 경계를 탐지한다.
        """
        pdf_path = Path(pdf_path)
        print(f"🔍 Parsing {pdf_path.name} with PyMuPDF4LLM...")

        # 파싱 소요 시간 측정 시작
        start_time = time.time()

        # 방법 1: 문서 구조 파악을 위해 전체를 하나의 마크다운으로 추출
        full_md = pymupdf4llm.to_markdown(str(pdf_path))

        # 방법 2: 정확한 페이지 구분을 위해 페이지 단위 청크로 추출
        # (이미지는 추출/임베딩하지 않아 속도를 높임)
        page_chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,      # 페이지별로 분리하여 반환
            write_images=False,    # 이미지 파일로 저장하지 않음
            embed_images=False     # 이미지를 마크다운에 임베딩하지 않음
        )

        # 페이지 번호(1부터 시작) → 페이지 텍스트 형태의 인덱스 생성
        # 이후 노드 내용이 어느 페이지에 있는지 찾을 때 사용한다
        page_contents = {i+1: chunk["text"] for i, chunk in enumerate(page_chunks)}
        total_pages = len(page_chunks)

        # 마크다운으로부터 트리 구조를 구축 (루트 노드 반환)
        root = self._build_tree_from_markdown(full_md, page_contents, pdf_path.name)

        # 소요 시간 및 결과 요약 출력
        elapsed = time.time() - start_time
        print(f"✅ Parsed in {elapsed:.2f}s: {total_pages} pages, {self._count_nodes(root)} nodes")

        # 최종 DocumentTree 객체 생성하여 반환
        return DocumentTree(
            document_name=pdf_path.stem,  # 확장자를 제외한 파일명
            root=root,
            total_pages=total_pages,
            source_path=str(pdf_path)
        )
    
    def _build_tree_from_markdown(
        self,
        markdown: str,
        page_contents: Dict[int, str],
        doc_name: str
    ) -> TreeNode:
        """
        마크다운 헤더를 계층적 트리로 파싱한다.
        번호가 있는 헤딩과 없는 헤딩을 모두 처리한다.

        핵심 아이디어: 스택(stack)을 사용해 현재 위치의 계층 경로를 추적한다.
        새 헤딩을 만나면 레벨을 비교하여 적절한 부모 아래에 노드를 붙인다.
        """
        # 마크다운을 줄 단위로 분리
        lines = markdown.split('\n')

        # 루트 노드 생성 (전체 문서를 대표하는 최상위 노드)
        root = TreeNode(
            id="root",
            title=doc_name,
            level=0,
            page_start=1,
            page_end=max(page_contents.keys()) if page_contents else 1,
            content="",
            heading_type="root"
        )

        # 스택은 현재 계층 경로를 (레벨, 노드) 튜플로 유지한다
        # 맨 위(stack[-1])가 현재 내용이 귀속될 노드
        stack = [(0, root)]
        current_content_lines = []  # 다음 헤딩 전까지 누적되는 본문 라인들
        current_start_page = 1

        def flush_content():
            """누적된 본문(current_content_lines)을 현재 노드에 붙인다."""
            if current_content_lines and stack:
                content = '\n'.join(current_content_lines).strip()
                if content:
                    # 스택 맨 위 노드(현재 노드)에 본문을 추가
                    stack[-1][1].content += "\n\n" + content
                    # 요약이 아직 없다면 첫 문단(최대 300자)으로 요약 생성
                    if not stack[-1][1].summary:
                        first_para = content.replace('#', '').strip()[:300]
                        stack[-1][1].summary = first_para
            # 다음 노드를 위해 누적 버퍼 초기화
            current_content_lines.clear()

        # 모든 줄을 순회하며 헤딩과 본문을 구분 처리
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 마크다운 헤더(#로 시작) 탐지
            if stripped.startswith('#'):
                # 새 헤딩을 만났으니 직전까지 누적된 본문을 먼저 저장
                flush_content()

                # '#' 개수를 세어 헤딩 레벨 계산 (# = 1, ## = 2, ...)
                level = len(stripped.split()[0]) if stripped.split() else 0
                # 앞쪽 '#'과 공백을 제거해 순수 제목 텍스트 추출
                title = stripped.lstrip('#').strip()

                # 헤딩 종류(numbered/roman/letter/unnumbered/unknown) 분류
                heading_type = self._classify_heading(title)

                # 본문 내 위치(라인 비율)를 기반으로 페이지 번호를 대략 추정
                # (이후 page_chunks 매칭으로 더 정확히 보정한다)
                page_num = self._estimate_page_number(i, len(lines), max(page_contents.keys()))

                # 새 노드 생성: 제목 슬러그 + 라인 번호로 고유 id 생성
                title_slug = '_'.join(re.findall(r'\w+', title))[:20]
                node_id = f"{title_slug}_{i}"
                new_node = TreeNode(
                    id=node_id,
                    title=title,
                    level=level,
                    page_start=page_num,
                    page_end=page_num,  # 이후 단계에서 갱신됨
                    content="",
                    heading_type=heading_type
                )

                # 현재 헤딩과 같거나 더 깊은(레벨 >= 새 레벨) 노드들을 스택에서 닫는다
                # → 새 헤딩의 올바른 부모를 찾기 위함
                while stack and stack[-1][0] >= level:
                    closed_node = stack.pop()[1]
                    # 닫힌 자식의 page_end를 부모에 반영
                    if stack:
                        stack[-1][1].page_end = max(stack[-1][1].page_end, closed_node.page_end)

                # 스택 맨 위 노드를 부모로 삼아 새 노드를 자식으로 연결
                if stack:
                    parent = stack[-1][1]
                    parent.children.append(new_node)
                    parent.page_end = max(parent.page_end, page_num)

                # 새 노드를 스택에 push하여 현재 경로로 설정
                stack.append((level, new_node))
                current_start_page = page_num

            else:
                # 헤딩이 아닌 줄은 본문으로 누적
                current_content_lines.append(line)

            i += 1

        # 마지막까지 남은 본문을 저장
        flush_content()

        # page_chunks 내용 매칭으로 페이지 경계를 정밀 보정
        self._refine_page_boundaries(root, page_contents)

        # 본문을 리프(말단) 노드로 적절히 분배
        self._distribute_content_to_leaves(root)

        return root
    
    def _classify_heading(self, title: str) -> str:
        """헤딩을 numbered / roman / letter / unnumbered 중 하나로 분류한다.

        어떤 패턴에도 맞지 않으면 "unknown"을 반환한다.
        """
        title = title.strip()

        # 정의된 패턴 순서대로 매칭하여 가장 먼저 일치하는 종류를 반환
        if self.patterns['numbered_section'].match(title):
            return "numbered"      # 숫자 번호 (예: 1. , 2.3.1)
        elif self.patterns['roman_section'].match(title):
            return "roman"         # 로마 숫자 (예: I. , II.)
        elif self.patterns['letter_section'].match(title):
            return "letter"        # 알파벳 (예: A. , B.)
        elif self.patterns['unnumbered_heading'].match(title):
            return "unnumbered"    # 번호 없는 제목 (예: Abstract)
        else:
            return "unknown"       # 분류 불가

    def _estimate_page_number(self, line_idx: int, total_lines: int, total_pages: int) -> int:
        """라인 위치(line_idx)를 기반으로 페이지 번호를 대략 추정한다.

        전체 라인 대비 현재 라인의 비율을 전체 페이지 수에 곱해 추정한다.
        (정확하지 않은 임시 추정값이며, 이후 _refine_page_boundaries에서 보정됨)
        """
        if total_pages == 0:
            return 1
        # 현재 라인이 문서 전체에서 차지하는 위치 비율 (0.0 ~ 1.0)
        ratio = line_idx / total_lines if total_lines > 0 else 0
        # 비율 × 전체 페이지 + 1 (1부터 시작), 단 전체 페이지 수를 넘지 않도록 제한
        return min(int(ratio * total_pages) + 1, total_pages)
    
    def _refine_page_boundaries(self, root: TreeNode, page_contents: Dict[int, str]):
        """
        노드 본문을 실제 페이지 청크와 매칭하여 페이지 경계를 정밀 보정한다.
        마크다운 파싱 단계의 대략적인 추정값을 실제 페이지 위치로 교정한다.
        """
        def find_page_for_content(content: str, start_search: int = 1) -> int:
            """주어진 본문이 어느 페이지에 들어있는지 찾는다."""
            # 본문 앞부분 100자만 사용해 매칭 (전체 비교는 비효율적)
            content_snippet = content[:100].strip()
            if not content_snippet:
                return start_search

            # start_search 페이지부터 순회하며 스니펫이 포함된 페이지를 탐색
            for page_num, page_text in page_contents.items():
                if page_num >= start_search and content_snippet in page_text:
                    return page_num
            # 못 찾으면 탐색 시작 페이지를 그대로 반환
            return start_search

        def refine_node(node: TreeNode, parent_start: int = 1):
            """노드를 재귀적으로 순회하며 페이지 시작/끝을 보정한다."""
            # 본문이 있으면 내용 매칭으로 시작 페이지를 갱신
            if node.content:
                matched_page = find_page_for_content(node.content, parent_start)
                node.page_start = matched_page
                node.page_end = matched_page

            # 자식 노드들을 처리 (앞선 자식의 끝 페이지부터 탐색 시작)
            prev_end = node.page_start
            for child in node.children:
                refine_node(child, prev_end)
                prev_end = max(prev_end, child.page_end)

            # 자식이 있으면 부모의 페이지 범위가 모든 자식을 포괄하도록 갱신
            if node.children:
                node.page_end = max(c.page_end for c in node.children)
                node.page_start = min(c.page_start for c in node.children)

        refine_node(root)

    def _distribute_content_to_leaves(self, node: TreeNode):
        """
        본문이 적절한 리프(말단) 노드에 저장되도록 정리한다.
        자식이 있는 노드(섹션 헤더 역할)의 본문은 요약으로 축약한다.
        """
        # 자식이 없는 리프 노드는 본문을 그대로 유지하고 종료
        if not node.children:
            return

        # 자식이 있는 노드의 본문이 너무 길면(500자 초과) 요약으로 잘라낸다
        # (해당 노드는 헤더 역할이므로 전체 본문을 보관할 필요가 없음)
        if len(node.content) > 500:
            node.summary = node.content[:500] + "..."
            node.content = node.summary

        # 모든 자식에 대해 재귀적으로 동일 처리
        for child in node.children:
            self._distribute_content_to_leaves(child)

    def _count_nodes(self, node: TreeNode) -> int:
        """트리의 전체 노드 개수를 센다 (자신 + 모든 자손)."""
        return 1 + sum(self._count_nodes(c) for c in node.children)


# 공개 API(Public API)
def parse_pdf(pdf_path: str) -> DocumentTree:
    """PyMuPDF4LLM을 사용해 PDF를 계층적 문서 트리로 파싱하는 진입점 함수.

    내부적으로 PyMuPDF4LLMTreeBuilder를 생성해 파싱을 위임한다.
    """
    builder = PyMuPDF4LLMTreeBuilder()
    return builder.parse_pdf(pdf_path)