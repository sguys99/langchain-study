#!/usr/bin/env python3
# main.py
# ─────────────────────────────────────────────────────────────────────────────
# E-Commerce 라우팅 에이전트의 대화형 CLI 진입점.
#
# 이 파일의 역할:
#   - 실행 전 필수 환경 변수 및 데이터 아티팩트 존재 여부 검증
#   - MLflow 관측(observability) 초기화
#   - EcommerceSession 생성 및 LangGraph 시각화 저장
#   - 사용자가 선택한 모드(대화형 / 데모)로 세션 실행
#
# 사전 준비:
#   1. python setup.py            (Olist 데이터셋 다운로드 + SQLite 적재 + FAISS 인덱스 빌드)
#   2. export ANTHROPIC_API_KEY=...   (기본 LLM 키, 또는 OPENAI_API_KEY를 폴백으로 사용)
#   3. export SERPER_API_KEY=...      (웹 검색 도구용 Serper API 키)
#
# 사용법:
#   python main.py               # 대화형(REPL) 모드
#   python main.py --demo        # 3턴 스크립트 데모 (SQL / Web / RAG 각 1회)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys

# 프로젝트 루트를 import 경로에 추가하여 어디서 실행해도 모듈을 찾을 수 있도록 보장
# (예: 다른 디렉토리에서 `python /path/to/main.py`로 호출하는 경우 대비)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 내부 모듈 import ──────────────────────────────────────────────────────────
from session import EcommerceSession                  # 멀티턴 대화 상태 + 그래프 호출 담당
from config import (
    ANTHROPIC_API_KEY,                                # Claude API 키 (기본 LLM)
    OPENAI_API_KEY,                                   # OpenAI 키 (Anthropic 폴백)
    SERPER_API_KEY,                                   # Serper 웹 검색 API 키
    DB_PATH,                                          # SQLite DB 경로 (data/ecommerce.db)
    FAISS_INDEX_DIR,                                  # FAISS 인덱스 디렉토리 (data/faiss_index)
)
from agent.graph import save_graph_visualization      # LangGraph 토폴로지를 PNG로 저장
from observability import init as init_observability  # MLflow 트래킹 초기화


def _check_env() -> None:
    """
    실행에 필요한 모든 사전 조건을 검증한 뒤, 누락 시 즉시 종료한다.

    검증 항목:
      1) 필수 파이썬 패키지(anthropic) 설치 여부
      2) 필수 환경 변수 (LLM API 키, Serper API 키)
      3) setup.py가 생성하는 데이터 아티팩트 (SQLite DB, FAISS 인덱스)

    하나라도 누락되면 사용자에게 안내 메시지를 출력하고 sys.exit(1)로 종료한다.
    """

    # ── 1) anthropic 패키지 설치 확인 ─────────────────────────────────────────
    # 기본 LLM이 Claude이므로 anthropic SDK가 반드시 필요함.
    try:
        import anthropic  # noqa: F401  # import 자체가 목적이므로 사용하지 않음
    except ImportError:
        print("[ERROR] 'anthropic' package not installed. Run: uv add anthropic")
        sys.exit(1)

    # ── 2) API 키 환경 변수 확인 ───────────────────────────────────────────────
    # ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 중 최소 하나는 있어야 함 (LLM 호출용).
    # SERPER_API_KEY는 웹 검색 라우트에서 사용하므로 필수.
    missing = []
    if not (ANTHROPIC_API_KEY or OPENAI_API_KEY):
        missing.append("ANTHROPIC_API_KEY (default) or OPENAI_API_KEY")
    if not SERPER_API_KEY:
        missing.append("SERPER_API_KEY")
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Set them before running, e.g.:")
        print("  export ANTHROPIC_API_KEY=your_key_here")
        print("  export SERPER_API_KEY=your_key_here")
        sys.exit(1)

    # ── 3) 데이터 아티팩트 확인 ────────────────────────────────────────────────
    # setup.py가 생성하는 두 파일이 모두 있어야 SQL/RAG 도구가 동작함.
    #   - SQLite DB: Text2SQL 도구가 쿼리하는 대상
    #   - FAISS 인덱스: RAG 도구가 유사도 검색을 수행하는 벡터 저장소
    missing_artifacts = []
    if not os.path.exists(DB_PATH):
        missing_artifacts.append(f"  - SQLite DB: {DB_PATH}")
    if not os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        missing_artifacts.append(f"  - FAISS index: {FAISS_INDEX_DIR}/index.faiss")
    if missing_artifacts:
        print("[ERROR] Required data artifacts are missing:")
        for line in missing_artifacts:
            print(line)
        print("Run setup first:  uv run setup.py")
        sys.exit(1)


def run_demo(session: EcommerceSession) -> None:
    """
    세 가지 라우팅 경로(SQL / Web / RAG)를 한 번씩 거치는 스크립트 데모를 실행.

    - 라우터가 각 질문을 올바른 도구로 분류하는지 검증하는 용도로도 활용 가능
    - 모든 턴이 동일한 session 객체를 공유하므로 대화 이력(conversation_history)이 누적됨
      → 멀티턴 컨텍스트가 필요한 후속 질문 테스트도 가능 (현재는 주석 처리됨)
    """

    # ── 데모용 질문 시나리오 ──────────────────────────────────────────────────
    # 주석 처리된 질문들은 추가 검증이 필요할 때 활성화하여 확장 테스트 가능.
    demo_turns = [
        # ── SQL 쿼리: 구조화된 데이터(SQLite) 집계 ─────────────────────────────
        "배송이 완료된 주문(order)은 몇건입니까?",
        #"What are the top 5 customer states by total number of orders?",
        #"What is the average payment value for credit card transactions?",
        #"Show me the top 5 product categories by total revenue.",
        #"Which payment type is most commonly used?",

        # ── 후속 질문 (멀티턴 컨텍스트 활용 테스트) ────────────────────────────
        # 이전 턴의 "top 5 categories" 결과를 참조하는 follow-up
        #"Now show me those same categories but only for delivered orders.",

        # ── 웹 검색: 최신 정보가 필요한 질문 (Serper API 사용) ─────────────────
        "2026년 대한민국의 최신 e-commerce 트렌드는 무엇인가요?",

        # ── RAG: 사내 PDF 문서(반품 정책 등) 검색 ──────────────────────────────
        "가전 제품(electronics)의 반품 정책(return policy)에 대해서 알려주세요.",
    ]

    # ── 헤더 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  데모모드 — Scripted Multi-Turn Conversation")
    print("═" * 60)

    # ── 각 턴을 순차 실행 ─────────────────────────────────────────────────────
    # enumerate(..., 1)로 1부터 시작하는 턴 번호를 사람이 읽기 좋게 출력.
    for i, question in enumerate(demo_turns, 1):
        print(f"\n{'─'*60}")
        print(f"[Turn {i}] USER: {question}")
        print("─" * 60)
        # session.ask()가 내부적으로 LangGraph 그래프를 invoke하고
        # 라우팅 → 도구 실행 → 합성 → 이력 갱신 전체 파이프라인을 수행함
        answer = session.ask(question)
        print(f"\nASSISTANT:\n{answer}")

    # ── 종료 메시지 + 전체 대화 이력 덤프 ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("  데모 종료.")
    print("═" * 60)
    session.print_history()  # 누적된 모든 turn의 user/assistant 메시지 출력


def run_interactive(session: EcommerceSession) -> None:
    """
    REPL(Read-Eval-Print-Loop) 방식의 대화형 세션.

    지원 특수 명령:
      - 'quit' / 'exit'  : 세션 종료
      - 'reset'          : 대화 이력 초기화 (멀티턴 컨텍스트 리셋)
      - 'history'        : 지금까지의 user/assistant 메시지를 모두 출력

    그 외 입력은 모두 일반 질문으로 간주하여 session.ask()에 전달.
    """

    # ── 환영 메시지 / 사용법 안내 ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  E-Commerce Analytics Agent")
    print("  Routes: SQL  |  RAG (PDFs)  |  Web Search")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'reset' to clear conversation history.")
    print("  Type 'history' to print the full conversation.")
    print("═" * 60 + "\n")

    # ── 메인 REPL 루프 ───────────────────────────────────────────────────────
    while True:
        # 사용자 입력 수신 (EOF/Ctrl+C를 안전하게 처리)
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D(EOF) 또는 Ctrl+C(KeyboardInterrupt) 시 깔끔하게 종료
            print("\nGoodbye!")
            break

        # 빈 입력은 무시하고 다음 프롬프트로
        if not user_input:
            continue

        # ── 특수 명령 처리 ────────────────────────────────────────────────────
        # 종료 명령
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # 대화 이력만 초기화하고 세션은 유지 (turn_number도 재설정됨)
        if user_input.lower() == "reset":
            session.reset()
            continue

        # 지금까지의 전체 대화 이력 출력
        if user_input.lower() == "history":
            session.print_history()
            continue

        # ── 일반 질문 처리 ────────────────────────────────────────────────────
        # 빈 줄 + 응답 + 빈 줄로 가독성을 높임
        print()
        answer = session.ask(user_input)  # 그래프 실행 (라우팅 → 도구 → 합성)
        print(f"\nAssistant:\n{answer}\n")


def main() -> None:
    """
    CLI 진입점.

    순서:
      1) 환경 검증 (_check_env)
      2) MLflow 관측 초기화 (init_observability)
      3) CLI 인자 파싱 (--demo 여부)
      4) EcommerceSession 생성 + 그래프 시각화 저장
      5) 모드에 따라 데모 / 대화형 실행
    """

    # ── 1) 사전 조건 검증 — 실패 시 즉시 종료 ─────────────────────────────────
    _check_env()

    # ── 2) MLflow 트래킹 서버 연결 및 실험 설정 초기화 ────────────────────────
    # HTTP URI라면 소켓 프로브 후 연결 가능 여부에 따라 자동 폴백 (observability.py 참고)
    init_observability()

    # ── 3) CLI 인자 파싱 ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="E-Commerce Routing Agent")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a scripted multi-turn demo instead of interactive mode."
    )
    args = parser.parse_args()

    # ── 4) 세션 객체 생성 ─────────────────────────────────────────────────────
    # EcommerceSession 생성자에서 LangGraph 그래프 컴파일과 session_id 발급이 수행됨.
    session = EcommerceSession()
    print(f"[main] Session ID: {session.session_id}")

    # 컴파일된 그래프 토폴로지를 PNG(agent_graph.png)로 저장 — 디버깅/문서화 용도
    # save_graph_visualization(session.graph)

    # ── 5) 실행 모드 분기 ─────────────────────────────────────────────────────
    if args.demo:
        # 스크립트 데모: 미리 정의된 3개 질문을 순차 실행
        run_demo(session)
    else:
        # 대화형 REPL: 사용자가 직접 입력하며 멀티턴 대화
        run_interactive(session)


# ── 모듈이 스크립트로 직접 실행될 때만 main() 호출 ────────────────────────────
# (다른 모듈에서 import 될 때는 main()이 자동으로 실행되지 않도록 가드)
if __name__ == "__main__":
    main()
