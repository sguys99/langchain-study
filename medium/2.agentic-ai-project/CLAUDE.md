# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 빠르게 컨텍스트를 잡기 위한 가이드입니다. 프로젝트 배경·이론 설명은 [README.md](README.md)를 참고하세요.

## 1. 프로젝트 개요

- **목적**: LangGraph 기반 멀티-라우트 에이전트(Text2SQL + RAG + Web Search)에 MLflow 가시성 추적을 결합한 학습/데모 프로젝트
- **흐름**: 사용자 질문 → 라우터(LLM 분류) → 도구 노드(SQL/RAG/Web) → 합성 LLM → 대화 이력 갱신
- **관측**: 모든 스팬·트레이스·비용을 MLflow로 자동 기록 (`@trace` 데코레이터)

## 2. Quick Start

```bash
# 최초 1회 — 데이터셋 다운로드 + SQLite 적재 + FAISS 인덱스 빌드
uv run setup.py

# CSV → SQLite만 재실행 (DB 스키마/데이터만 갱신)
uv run data_loader.py

# MLflow UI (별도 터미널에서 먼저 실행 권장)
mlflow server --host 0.0.0.0 --port 5001

# 에이전트 실행
uv run main.py           # 대화형 CLI
uv run main.py --demo    # 3턴 스크립트 데모 (SQL / Web / RAG 한 번씩)
```

- Python ≥3.12, `uv` 패키지 매니저 사용 ([pyproject.toml](pyproject.toml))
- 필수 환경변수(.env 또는 export): `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `VOYAGE_API_KEY`
- 선택: `MLFLOW_TRACKING_URI` (기본 `http://localhost:5001`), `MLFLOW_EXPERIMENT` (기본 `ecommerce-agent`), `MLFLOW_ENABLED=false`로 추적 비활성화 가능

## 3. 아키텍처 한눈에 보기

```
[START]
  │
  ▼
router_node
  │
  ├── "sql"        ──▶ sql_node        ──┐
  ├── "mcp_sql"    ──▶ mcp_sql_node    ──┤   (※ 현재 미구현 — Gotchas 참조)
  ├── "rag"        ──▶ rag_node        ──┤
  └── "web_search" ──▶ web_search_node ──┤
                                          ▼
                                  synthesise_node
                                          │
                                          ▼
                                 update_history_node
                                          │
                                        [END]
```

- 진입점: [session.py](session.py)의 `EcommerceSession.ask()`가 한 턴 전체를 `trace_span(name="turn_N")`으로 감싼 뒤 컴파일된 그래프를 `invoke()` 함
- 상태 객체: [agent/state.py](agent/state.py)의 `AgentState` — `user_message`, `route`, 4개 도구 결과 슬롯, `final_answer`, `turn_number`, `conversation_history` 보유

## 4. 핵심 파일 지도

| 영역 | 파일 | 책임 |
|---|---|---|
| 진입점 | [main.py](main.py) | CLI / `--demo` 모드 / 환경 변수·데이터 아티팩트 사전 검증 |
| 세션 | [session.py](session.py) | 멀티턴 대화 상태 보관, 그래프 호출, turn 스팬 생성 |
| 그래프 | [agent/graph.py](agent/graph.py) | LangGraph 노드 연결, 조건부 라우팅 (`_TOOL_NODES` 단일 진실) |
| 상태 | [agent/state.py](agent/state.py) | `AgentState` TypedDict |
| 노드 | [agent/nodes.py](agent/nodes.py) | router / sql / mcp_sql / rag / web_search / synthesise / history |
| 라우터 | [agent/router.py](agent/router.py) | Claude 기반 경로 분류 (JSON 출력) |
| 설정 | [config.py](config.py) | API 키, 모델, 경로, RAG 파라미터, `ROUTES` |
| 관측 | [observability.py](observability.py) | `@trace`, `trace_span`, `set_attrs`, 토큰·비용 자동 추정 |
| SQL 도구 | [tools/sql_tool.py](tools/sql_tool.py) | Text2SQL 생성 → sqlite3 실행 → 마크다운 테이블 |
| RAG 도구 | [tools/rag_tool.py](tools/rag_tool.py) | PDF 청킹 + 임베딩(Voyage/로컬) + FAISS 검색 |
| 웹 검색 | [tools/web_search_tool.py](tools/web_search_tool.py) | Serper API 호출 |
| MCP 서버 | [mcp_servers/sqlite_server.py](mcp_servers/sqlite_server.py) | FastMCP 기반 read-only SQLite 서버 (DDL 차단) |

## 5. Conventions & Gotchas

- **임베딩 기본값**: 상용 Voyage AI `voyage-4-lite` (config.py:25). 로컬 폴백이 필요하면 `config.py`에서 `EMBEDDING_MODEL`을 `sentence-transformers/all-MiniLM-L6-v2`로 교체 (주석으로 남겨져 있음). [tools/rag_tool.py](tools/rag_tool.py)가 모델 이름으로 provider를 자동 판별.
- **LLM 기본값**: `claude-sonnet-4-6`, temperature 0.0, max_tokens 1024 (config.py:19-21).
- **라우트 키**: `ROUTES = ["sql", "mcp_sql", "rag", "web_search"]` (config.py:48). 새 라우트 추가 시 반드시 [agent/graph.py](agent/graph.py)의 `_TOOL_NODES` 딕셔너리도 함께 갱신할 것 — 그래프 토폴로지의 단일 진실.
- **`mcp_sql_node`는 현재 `pass` 스텁** ([agent/nodes.py](agent/nodes.py) 참조). 라우터가 `mcp_sql`을 선택해도 결과 슬롯이 비어 합성 단계에서 빈 답이 나올 수 있음. MCP SQL 경로로 실험하려면 이 노드를 실제 MCP 클라이언트 호출로 구현해야 함.
- **데이터 아티팩트 누락 시 즉시 종료**: [main.py](main.py)의 `_check_env()`가 `data/ecommerce.db`와 `data/faiss_index/index.faiss` 존재를 확인. 없으면 `uv run setup.py`를 먼저 실행.
- **MLflow 추적**: HTTP URI인 경우 [observability.py](observability.py)가 사전 소켓 프로브를 수행해 서버 미기동 시 빠르게 폴백 (urllib3 재시도로 수십 초 멈추는 것을 방지).
- **`@trace` 자동 캡처**: `cost.usd`, `tokens.input/output`, `bytes.input/output`, `duration_ms` 그리고 결과 dict에서 `sql`/`chunks`/`results`/`sources`를 자동 추출. 새 LLM·임베딩 모델 추가 시 [observability.py](observability.py)의 `MODEL_PRICING` 테이블에 단가도 함께 등록해야 비용이 정확히 산출됨.
- **사용자 요청 자동 추출**: 도구/노드 함수의 파라미터 이름이 `question` / `message` / `prompt` / `query` / `input` 중 하나여야 `@trace`가 사용자 요청을 스팬 입력으로 기록 (`_extract_request_arg`). 다른 이름을 쓰면 트레이스가 비어 보일 수 있음.
- **두 가지 SQL 안전 모델**: 직접 sqlite3를 쓰는 [tools/sql_tool.py](tools/sql_tool.py)는 자연어→SQL 생성 후 그대로 실행하지만, [mcp_servers/sqlite_server.py](mcp_servers/sqlite_server.py)는 URI `mode=ro`로 read-only를 강제하고 DDL 키워드를 정규식으로 차단. 보안이 필요한 경로면 MCP 서버 쪽을 사용.

## 6. 코드 스타일

- **언어**: 한국어 주석/문서, 영어 식별자 — 기존 파일과 일관성 유지
- **노드 반환**: LangGraph 머지 규약에 따라 갱신할 키만 담은 `partial dict` 반환
- **로그 prefix**: `[router]`, `[sql_node]`, `[rag_node]`, `[web_node]`, `[synthesise_node]` 형식으로 stdout에 한 줄씩 — 동일 컨벤션 유지

## 7. 검증 방법

- **엔드투엔드**: `uv run main.py --demo` 실행 시 3턴 모두 정상 응답이어야 함 (SQL: 87,428건, Web: 브라질 트렌드, RAG: 반품 정책)
- **MLflow UI**: `http://localhost:5001` → `ecommerce-agent` 실험에서 각 턴이 `turn_N` 부모 스팬과 router/tool/synthesise 자식 스팬을 가져야 함
- **도구 단독 실행**:
  ```bash
  uv run python -m tools.rag_tool          # 인덱스 빌드/유사도 검색
  uv run python -m tools.rag_tool --force  # 강제 재빌드
  ```
- **그래프 시각화**: `main.py` 실행 시 `agent_graph.png`가 자동 저장됨
