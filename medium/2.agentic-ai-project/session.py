# session.py
# ─────────────────────────────────────────────────────────────────────────────
# Session: wraps the compiled LangGraph agent and maintains conversation state
# across multiple user turns.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from langgraph.graph.state import CompiledStateGraph
from agent.graph import build_graph
from agent.state import AgentState
from observability import trace_span


class EcommerceSession:
    """
    A single user session with the e-commerce routing agent.

    Example usage:
        session = EcommerceSession()
        answer  = session.ask("What were total sales last month?")
        answer2 = session.ask("Break that down by customer state.")
    """

    def __init__(self):
        self.session_id:           str                 = str(uuid.uuid4())[:8]
        self.graph:                CompiledStateGraph  = build_graph()
        self.conversation_history: list[dict]          = []
        self.turn_number:          int                 = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        Process one user turn and return the assistant's final answer.
        Updates internal conversation_history automatically.
        """
        next_turn = self.turn_number + 1
        with trace_span(
            name=f"turn_{next_turn}",
            span_type="CHAIN",
            attributes={
                "session.id": self.session_id,
                "session.turn_number": next_turn,
            }
        ):
            initial_state: AgentState = {
                "user_message":         question,
                "conversation_history": self.conversation_history,
                "route":                "",
                "route_reason":         "",
                "sql_result":           None,
                "rag_result":           None,
                "web_search_result":    None,
                "final_answer":         "",
                "turn_number":          next_turn,
            }

            final_state = self.graph.invoke(initial_state)

            # Persist updated history for the next turn
            self.conversation_history = final_state["conversation_history"]
            self.turn_number          = next_turn

            return final_state["final_answer"]

    def reset(self) -> None:
        """Clear conversation history and start fresh."""
        self.conversation_history = []
        self.turn_number          = 0
        print(f"[session:{self.session_id}] Conversation reset.")

    def get_history(self) -> list[dict]:
        """Return a copy of the conversation history."""
        return list(self.conversation_history)

    def print_history(self) -> None:
        """Pretty-print the conversation history."""
        print(f"\n{'═'*60}")
        print(f"  Session {self.session_id}  |  {self.turn_number} turns")
        print(f"{'═'*60}")
        for msg in self.conversation_history:
            role    = msg["role"].capitalize()
            content = msg["content"]
            print(f"\n[{role}]\n{content}")
        print(f"\n{'═'*60}\n")