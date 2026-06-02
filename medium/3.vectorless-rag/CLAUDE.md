# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 참고하는 가이드입니다.

## 개요

**Vectorless RAG** 데모입니다. 임베딩 벡터나 벡터 DB 없이, PDF를 계층적
문서 트리(`DocumentTree`)로 만들어 두고 LLM 에이전트가 트리를 탐색하며
질문에 답합니다. "어떤 텍스트가 유사한가?"(벡터 검색) 대신
"다음에 어디를 봐야 하는가?"(추론 기반 탐색)로 검색 문제를 푸는 것이
핵심 아이디어입니다. 데모 문서는 Google Bigtable(OSDI'06) 논문입니다.

## 실행 / 설정

- 패키지 매니저는 **uv**, Python은 **3.12** (`.python-version`)입니다.
- 실행:
  ```bash
  uv sync          # 의존성 설치 (uv.lock 기준)
  uv run main.py   # 실행
  ```
  pip 대안: `pip install -e .` 후 `python main.py`.
- 환경 변수: `.env.example`을 `.env`로 복사한 뒤 값을 채웁니다.
  - `ANTHROPIC_API_KEY` (필수) — 없으면 `main.py`가 안내 메시지와 함께 종료합니다.
  - `ANTHROPIC_MODEL` (선택) — 기본값 `claude-sonnet-4-6`.
- **최초 실행**: PDF 다운로드(`bigtable-osdi06.pdf`) + 트리 파싱(약 10~30초)
  후 `results/document_tree.json`에 캐시합니다.
  **이후 실행**: 캐시를 즉시 로드하므로 파싱을 건너뜁니다.

## 아키텍처 / 데이터 흐름

```
PDF → [tree.py] 문서 트리 → results/document_tree.json (캐시)
                                     ↓
질문 → [retriever.py] LangGraph 에이전트가 트리 탐색 → 답변 생성
                                     ↓
       [main.py] 전체 오케스트레이션 + 결과 출력
```

`retriever.py`의 LangGraph `StateGraph`는 4개 노드로 구성됩니다:

```
analyze ──┬─ (confidence < 0.3) ─────────────→ END
          ├─ (should_descend & 자식 존재) ── descend ──→ analyze (재귀)
          ├─ (depth >= MAX_DEPTH=5) ─────────→ retrieve
          └─ (그 외) ────────────────────────→ retrieve → generate → END
```

- **analyze**: 현재 노드와 자식들 중 어디에 답이 있는지 LLM이 판단(JSON 반환).
- **descend**: 선택한 자식으로 `current_node` 이동.
- **retrieve**: 최종 노드의 본문 + 제목/페이지 헤더 추출.
- **generate**: 수집한 섹션으로 최종 답변을 인용과 함께 생성.

## 파일 구조

- [main.py](main.py) — 진입점. PDF 다운로드 → 트리 로드/빌드 → 워크플로 PNG
  생성 → `QUESTIONS`를 순회하며 `retrieve()` 호출 및 결과 출력.
- [tree.py](tree.py) — `parse_pdf()` / `PyMuPDF4LLMTreeBuilder`. `pymupdf4llm`로
  PDF를 마크다운으로 변환한 뒤, 헤더를 스택 기반으로 파싱해
  `TreeNode`/`DocumentTree` 계층을 만듭니다 (임베딩 없음).
- [retriever.py](retriever.py) — LangGraph 에이전트. 공개 API는
  `retrieve(query, tree, client, model)`와 `generate_workflow_png()`.
  주요 상수: `DEFAULT_MODEL`(46행), `MAX_DEPTH=5`(433행).
- [questions.py](questions.py) — 샘플 질문 `QUESTIONS` 리스트(한국어,
  대부분 주석 처리됨).
- `results/` — 생성물. `document_tree.json`(트리 캐시), `workflow.png`(그래프 시각화).
- `retriever.log` — LLM 호출 전체 로그(프롬프트/응답/지연시간).

## 자주 하는 작업

- **질문 변경**: [questions.py](questions.py)에서 항목을 주석 해제하거나 수정.
- **트리 재파싱 강제**: `results/document_tree.json`을 삭제하면 다음 실행 시
  PDF를 다시 파싱합니다. 트리 빌드 로직을 바꾼 뒤에는 반드시 캐시를 지우세요.
- **탐색 깊이/프롬프트 조정**: `retriever.py`의 `MAX_DEPTH`, 또는
  `_make_analyze` 등의 프롬프트를 수정.
- **디버깅**: 전체 프롬프트·원문 응답·지연시간은 `retriever.log`(DEBUG)에 기록됩니다.

## 관례 / 주의사항

- LLM 클라이언트는 `main.py`에서 단일 `anthropic.Anthropic` 인스턴스를 만들어
  `retrieve()`로 전달합니다. 호출마다 새 클라이언트를 만들지 마세요.
- 사용 모델은 `retriever.DEFAULT_MODEL`(= 환경변수 `ANTHROPIC_MODEL`)이
  단일 출처입니다. 다른 곳에 모델명을 하드코딩하지 마세요.
- 핵심 의존성: `anthropic`, `langgraph`, `pymupdf4llm`/`pymupdf`, `pydantic`,
  `python-dotenv`. (`langchain`/`langchain-*`/`openai`는 설치되어 있으나
  vectorless 핵심 경로에는 쓰이지 않습니다.)
- `results/`는 생성물이므로 직접 수정하지 마세요.
- 코드 주석·README·질문이 모두 한국어로 작성된 프로젝트입니다.
