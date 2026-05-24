#!/usr/bin/env python3
# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI for the E-Commerce Routing Agent.
#
# Prerequisites:
#   1. python setup.py            (download data + build index)
#   2. export ANTHROPIC_API_KEY=...   (or OPENAI_API_KEY as fallback)
#   3. export SERPER_API_KEY=...
#
# Usage:
#   python main.py               # interactive mode
#   python main.py --demo        # run a scripted multi-turn demo
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from session import EcommerceSession
from config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    SERPER_API_KEY,
    DB_PATH,
    FAISS_INDEX_DIR,
)
from agent.graph import save_graph_visualization
from observability import init as init_observability


def _check_env() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("[ERROR] 'anthropic' package not installed. Run: uv add anthropic")
        sys.exit(1)

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
    Run a scripted multi-turn demo covering all three routing paths.
    """
    demo_turns = [
        # ── SQL queries ────────────────────────────────────────────────────────
        "How many orders were delivered successfully?",
        #"What are the top 5 customer states by total number of orders?",
        #"What is the average payment value for credit card transactions?",
        #"Show me the top 5 product categories by total revenue.",
        #"Which payment type is most commonly used?",

        # ── Follow-up (multi-turn context test) ────────────────────────────────
        #"Now show me those same categories but only for delivered orders.",

        # ── Web search ─────────────────────────────────────────────────────────
        "What are the latest e-commerce trends in Brazil for 2024?",

        # ── RAG (PDF documents) ────────────────────────────────────────────────
        "What does our return policy say about electronics?",
    ]

    print("\n" + "═" * 60)
    print("  DEMO MODE — Scripted Multi-Turn Conversation")
    print("═" * 60)

    for i, question in enumerate(demo_turns, 1):
        print(f"\n{'─'*60}")
        print(f"[Turn {i}] USER: {question}")
        print("─" * 60)
        answer = session.ask(question)
        print(f"\nASSISTANT:\n{answer}")

    print("\n" + "═" * 60)
    print("  Demo complete.")
    print("═" * 60)
    session.print_history()


def run_interactive(session: EcommerceSession) -> None:
    """
    REPL-style interactive session.
    """
    print("\n" + "═" * 60)
    print("  E-Commerce Analytics Agent")
    print("  Routes: SQL  |  RAG (PDFs)  |  Web Search")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'reset' to clear conversation history.")
    print("  Type 'history' to print the full conversation.")
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            session.reset()
            continue

        if user_input.lower() == "history":
            session.print_history()
            continue

        print()
        answer = session.ask(user_input)
        print(f"\nAssistant:\n{answer}\n")


def main() -> None:
    _check_env()
    init_observability()

    parser = argparse.ArgumentParser(description="E-Commerce Routing Agent")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a scripted multi-turn demo instead of interactive mode."
    )
    args = parser.parse_args()

    session = EcommerceSession()
    print(f"[main] Session ID: {session.session_id}")

    # Save graph visualization
    save_graph_visualization(session.graph)

    if args.demo:
        run_demo(session)
    else:
        run_interactive(session)


if __name__ == "__main__":
    main()