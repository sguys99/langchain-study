# session.py
# ─────────────────────────────────────────────────────────────────────────────
# Session 모듈
# ─────────────────────────────────────────────────────────────────────────────
# 이 파일은 컴파일된 LangGraph 에이전트를 감싸는 "세션(Session)" 추상화를 제공한다.
# LangGraph 그래프 자체는 상태가 없는(stateless) 단일 호출 단위이므로,
# 여러 턴(turn)에 걸친 대화를 이어가려면 외부에서 대화 이력을 보관해 줄 주체가 필요하다.
#
# EcommerceSession 클래스가 그 역할을 담당하며 다음을 책임진다.
#   1) 세션마다 고유 ID 부여 (MLflow 트레이스에서 세션 단위 묶음으로 보기 위함)
#   2) 그래프를 1회 빌드/컴파일해 두고, 매 턴마다 재사용
#   3) conversation_history(대화 이력) 누적 관리
#   4) 매 턴 전체를 `turn_N`이라는 부모 스팬으로 감싸 MLflow에 기록
# ─────────────────────────────────────────────────────────────────────────────

import uuid                                              # 세션 ID 생성을 위한 표준 라이브러리
from langgraph.graph.state import CompiledStateGraph     # 빌드된 그래프 객체의 타입 힌트용
from agent.graph import build_graph                      # 라우터+도구+합성 노드를 연결한 그래프 빌더
from agent.state import AgentState                       # 그래프 한 턴 동안 흘러다니는 상태 스키마(TypedDict)
from observability import trace_span                     # MLflow 컨텍스트 매니저 (turn_N 스팬 생성용)


class EcommerceSession:
    """
    이커머스 라우팅 에이전트와의 단일 사용자 세션을 표현하는 클래스.

    한 인스턴스 = 한 사용자의 한 대화 세션이며, ask()를 반복 호출해
    멀티턴 대화를 진행한다. 내부적으로 LangGraph는 매 호출마다 새 상태를
    받지만, 이 클래스가 conversation_history를 외부에서 보관/주입해 주므로
    이전 턴의 맥락이 유지된다.

    사용 예시:
        session = EcommerceSession()
        answer  = session.ask("지난달 총 매출은 얼마인가요?")
        answer2 = session.ask("그것을 고객 주(state)별로 분해해 주세요.")
        #         └── "그것" 이라는 대명사를 이전 턴의 맥락으로 해석 가능
    """

    def __init__(self):
        # ── 세션 식별자 ─────────────────────────────────────────────────────
        # uuid4의 앞 8자만 잘라 짧은 ID를 생성. MLflow UI/로그에서 세션을
        # 시각적으로 구분할 수 있을 정도면 충분하므로 굳이 전체 UUID는 쓰지 않음.
        self.session_id:           str                 = str(uuid.uuid4())[:8]

        # ── 컴파일된 LangGraph 그래프 ───────────────────────────────────────
        # build_graph()는 라우터/SQL/MCP_SQL/RAG/Web/합성/이력 노드를 엮어
        # CompiledStateGraph를 반환한다. 세션 생성 시 1번만 빌드해서 재사용
        # (그래프 빌드 비용이 크지는 않지만 매 턴 다시 만들 이유가 없음).
        self.graph:                CompiledStateGraph  = build_graph()

        # ── 대화 이력 ──────────────────────────────────────────────────────
        # [{"role": "user"|"assistant", "content": "..."}] 형태로 누적된다.
        # 매 턴이 끝나면 update_history_node가 이 리스트를 갱신한 새 버전을
        # 반환하고, ask()가 그것을 다시 self.conversation_history에 저장한다.
        self.conversation_history: list[dict]          = []

        # ── 턴 번호 카운터 ─────────────────────────────────────────────────
        # 1부터 시작하는 누적 카운터. trace 스팬 이름(turn_1, turn_2, ...)과
        # AgentState["turn_number"]에 함께 전파되어 관측 가능성(observability)에
        # 활용된다.
        self.turn_number:          int                 = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        사용자 질문 한 건을 처리하고 어시스턴트의 최종 답변을 반환한다.

        내부 동작:
            1) turn 번호를 +1 한 뒤 그 값으로 turn_N MLflow 스팬을 연다.
            2) 현재까지의 conversation_history를 포함해 초기 AgentState를 구성.
            3) self.graph.invoke()로 한 턴 전체(라우팅→도구→합성→이력갱신)를 실행.
            4) 그래프가 돌려준 final_state에서 갱신된 history와 답변을 추출하여
               인스턴스 상태에 반영하고 답변 문자열만 호출자에게 반환한다.
        """
        # 이번 턴 번호 계산. self.turn_number는 turn 완료가 확정된 뒤에만
        # 업데이트하므로, 예외로 인해 그래프 실행이 실패해도 다음 ask() 호출이
        # 같은 번호를 재시도하도록 의도되어 있다.
        next_turn = self.turn_number + 1

        # MLflow 스팬으로 한 턴 전체를 감싼다. 이 스팬이 부모가 되고,
        # 그래프 내부의 router_node / *_node / synthesise_node 등이 자식 스팬으로
        # 매달리게 된다. attributes는 MLflow UI에서 필터링/그룹핑할 때 쓰인다.
        with trace_span(
            name=f"turn_{next_turn}",
            span_type="CHAIN",
            attributes={
                "session.id": self.session_id,
                "session.turn_number": next_turn,
            }
        ):
            # ── 초기 상태 구성 ──────────────────────────────────────────────
            # LangGraph는 매 invoke마다 새 상태를 받는다. 이력만 외부에서
            # 가져와 주입하고 나머지 슬롯(라우팅 결과, 도구별 결과, 최종 답변)은
            # 빈 값으로 두어 그래프가 채워 넣도록 한다.
            #
            # 4개의 도구 결과 슬롯(sql/mcp_sql/rag/web_search)을 모두 None으로
            # 명시하는 이유: 라우터가 선택한 한 가지 경로만 채워지고 나머지는
            # None으로 남는 게 정상이므로, 합성 노드에서 "어느 슬롯이 채워졌나"를
            # None 검사로 분기하기 위함.
            initial_state: AgentState = {
                "user_message":         question,                  # 이번 턴 사용자 입력
                "conversation_history": self.conversation_history, # 누적된 이전 대화
                "route":                "",                        # 라우터가 채울 칸
                "route_reason":         "",                        # 라우팅 근거(로그/관측용)
                "sql_result":           None,                      # 직접 sqlite3 결과 슬롯
                "mcp_sql_result":       None,                      # MCP SQLite 서버 결과 슬롯
                "rag_result":           None,                      # FAISS RAG 결과 슬롯
                "web_search_result":    None,                      # Serper 웹 검색 결과 슬롯
                "final_answer":         "",                        # 합성 노드가 채울 최종 답변
                "turn_number":          next_turn,                 # 노드들이 트레이스에 활용
            }

            # 그래프 실행 — 라우팅→도구→합성→이력갱신이 한 번에 진행된다.
            # 반환되는 final_state는 모든 노드가 머지한 후의 최종 AgentState.
            final_state = self.graph.invoke(initial_state)

            # ── 다음 턴을 위한 상태 영속화 ─────────────────────────────────
            # update_history_node가 user/assistant 메시지를 양쪽 다 붙여 반환하므로
            # 그 결과 리스트를 그대로 받아 다음 ask() 호출에서 재사용한다.
            self.conversation_history = final_state["conversation_history"]
            self.turn_number          = next_turn

            # 호출자에게는 답변 문자열만 반환. 도구 결과/라우팅 근거 등은
            # 트레이스로만 노출되며 외부에서 직접 보지 않는다.
            return final_state["final_answer"]

    def reset(self) -> None:
        """
        대화 이력을 비우고 처음 상태로 되돌린다.
        세션 ID와 컴파일된 그래프는 그대로 유지되므로, 같은 세션에서
        새로운 주제로 다시 시작할 때 사용한다 (예: CLI의 reset 명령).
        """
        self.conversation_history = []
        self.turn_number          = 0
        print(f"[session:{self.session_id}] Conversation reset.")

    def get_history(self) -> list[dict]:
        """
        대화 이력의 사본(shallow copy)을 반환한다.

        내부 리스트를 직접 노출하지 않는 이유: 호출자가 받은 리스트를
        실수로 변경(append/pop 등)해도 세션 내부 상태가 오염되지 않도록
        하기 위한 방어적 복사.
        """
        return list(self.conversation_history)

    def print_history(self) -> None:
        """
        대화 이력을 사람이 읽기 좋은 형식으로 콘솔에 출력한다.
        디버깅/데모용이며, MLflow UI를 띄울 수 없는 환경에서 빠르게
        흐름을 확인할 때 유용하다.
        """
        # ═ 라인으로 헤더/푸터를 만들어 시각적으로 구분
        print(f"\n{'═'*60}")
        print(f"  Session {self.session_id}  |  {self.turn_number} turns")
        print(f"{'═'*60}")

        # 각 메시지를 [User] / [Assistant] 라벨과 함께 한 블록씩 출력
        for msg in self.conversation_history:
            role    = msg["role"].capitalize()   # "user" → "User"
            content = msg["content"]
            print(f"\n[{role}]\n{content}")

        print(f"\n{'═'*60}\n")