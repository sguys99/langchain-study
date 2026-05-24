# agent/graph.py
# ─────────────────────────────────────────────────────────────────────────────
# LangGraph graph definition.
#
# Graph topology:
#
#   [START]
#     │
#     ▼
#   router_node
#     │
#     ├── "sql"        ──▶ sql_node        ──┐
#     ├── "rag"        ──▶ rag_node        ──┤
#     └── "web_search" ──▶ web_search_node ──┤
#                                            │
#                                            ▼
#                                      synthesise_node
#                                            │
#                                            ▼
#                                     update_history_node
#                                            │
#                                          [END]
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from agent.state import AgentState
from agent.nodes import (
    router_node,
    sql_node,
    rag_node,
    web_search_node,
    synthesise_node,
    update_history_node,
)

# Single source of truth: route key → (node name, node function)
_TOOL_NODES: dict[str, tuple[str, callable]] = {
    "sql":        ("sql_node",        sql_node),
    "rag":        ("rag_node",        rag_node),
    "web_search": ("web_search_node", web_search_node),
}


def _route_selector(state: AgentState) -> str:
    # router.py guarantees state["route"] ∈ ROUTES (config.py:43).
    # Normalising here too is a safety net for direct/manual invocations.
    route = state.get("route", "web_search")
    return route if route in _TOOL_NODES else "web_search"


def build_graph() -> CompiledStateGraph:
    """
    Assemble and compile the LangGraph routing agent.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("router_node",         router_node)
    graph.add_node("synthesise_node",     synthesise_node)
    graph.add_node("update_history_node", update_history_node)
    for node_name, node_fn in _TOOL_NODES.values():
        graph.add_node(node_name, node_fn)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.set_entry_point("router_node")

    # ── Conditional routing: router → one of the tool nodes ───────────────────
    graph.add_conditional_edges(
        source="router_node",
        path=_route_selector,
        path_map={route: node_name for route, (node_name, _) in _TOOL_NODES.items()},
    )

    # ── All tool nodes feed into synthesis ────────────────────────────────────
    for node_name, _ in _TOOL_NODES.values():
        graph.add_edge(node_name, "synthesise_node")

    # ── Linear tail ───────────────────────────────────────────────────────────
    graph.add_edge("synthesise_node",      "update_history_node")
    graph.add_edge("update_history_node",  END)

    return graph.compile()


def save_graph_visualization(compiled_graph, filepath: str = "agent_graph.png") -> None:
    """
    Save the compiled graph as a Mermaid PNG visualization.
    
    Args:
        compiled_graph: The compiled StateGraph from build_graph()
        filepath: Path where to save the PNG file (default: agent_graph.png)
    """
    try:
        graph_obj = compiled_graph.get_graph()
        png_data = graph_obj.draw_mermaid_png()
        with open(filepath, "wb") as f:
            f.write(png_data)
        print(f"Graph visualization saved to {filepath}")
    except Exception as e:
        print(f"Error saving graph visualization: {e}")