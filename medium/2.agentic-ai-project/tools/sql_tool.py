# tools/sql_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# SQL Tool: generates and executes SQL queries against the local SQLite DB.
#
# Flow:
#   1. Build a system prompt that includes all table schemas + sample rows
#   2. Call Claude (LLM_MODEL) to translate the user question into SQLite SQL
#   3. Execute the SQL; return results as a markdown table string
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3
import json
from anthropic import Anthropic
from config import DB_PATH, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, ANTHROPIC_API_KEY
from observability import trace  # NEW: Import observability

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Schema documentation (hand-written, mirrors Kaggle dataset) ───────────────
SCHEMA_DESCRIPTION = """
You have access to a SQLite database with the following tables:

TABLE: orders
  order_id                      TEXT  (primary key)
  customer_id                   TEXT
  order_status                  TEXT  (delivered, cancelled, invoiced, processing, shipped, unavailable, created, approved)
  order_purchase_timestamp      TEXT  (ISO datetime)
  order_approved_at             TEXT  (ISO datetime)
  order_delivered_timestamp     TEXT  (ISO datetime, may be NULL)
  order_estimated_delivery_date TEXT  (ISO date)

TABLE: order_items
  order_id          TEXT
  order_item_id     INTEGER  (item sequence within an order)
  product_id        TEXT
  seller_id         TEXT
  price             REAL
  shipping_charges  REAL

TABLE: customers
  customer_id             TEXT  (primary key)
  customer_zip_code_prefix TEXT
  customer_city           TEXT
  customer_state          TEXT

TABLE: payments
  order_id              TEXT
  payment_sequential    INTEGER
  payment_type          TEXT  (credit_card, boleto, voucher, debit_card)
  payment_installments  INTEGER
  payment_value         REAL

TABLE: products
  product_id            TEXT  (primary key)
  product_category_name TEXT
  product_weight_g      REAL
  product_length_cm     REAL
  product_height_cm     REAL
  product_width_cm      REAL

JOIN HINTS:
  orders.customer_id       → customers.customer_id
  order_items.order_id     → orders.order_id
  order_items.product_id   → products.product_id
  payments.order_id        → orders.order_id

RULES:
  - Output ONLY the raw SQLite SQL query, no markdown, no explanation.
  - Use table aliases for readability.
  - For date filtering use: strftime('%Y-%m', order_purchase_timestamp) = '2018-01'
  - Always add LIMIT 50 unless the user asks for all rows.
  - Aggregate with SUM / COUNT / AVG / GROUP BY as appropriate.
  - Never use SELECT *; always name columns explicitly.
"""

SQL_SYSTEM_PROMPT = (
    "You are an expert SQLite query writer for an e-commerce analytics database.\n"
    + SCHEMA_DESCRIPTION
)


def _generate_sql(user_question: str, conversation_history: list[dict]) -> str:
    """Call Claude to translate a natural language question into SQLite SQL."""
    # Include recent conversation for multi-turn context (last 6 messages = 3 turns)
    messages = list(conversation_history[-6:])
    messages.append({"role": "user", "content": user_question})

    response = client.messages.create(
        model=LLM_MODEL,
        system=SQL_SYSTEM_PROMPT,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=512,
    )
    return response.content[0].text.strip()


def _execute_sql(sql: str) -> tuple[list[dict], str | None]:
    """
    Execute sql against the SQLite DB.
    Returns (rows_as_list_of_dicts, error_string_or_None).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows, None
    except sqlite3.Error as exc:
        return [], str(exc)


def _rows_to_markdown(rows: list[dict], max_rows: int = 20) -> str:
    """Convert a list of dicts into a compact markdown table string."""
    if not rows:
        return "_No rows returned._"

    headers  = list(rows[0].keys())
    display  = rows[:max_rows]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line    = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines  = [
        "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |"
        for r in display
    ]
    table = "\n".join([header_line, sep_line] + data_lines)

    if len(rows) > max_rows:
        table += f"\n\n_… and {len(rows) - max_rows} more rows (showing first {max_rows})_"
    return table

@trace(span_type="TOOL", model=LLM_MODEL, attributes={  # NEW: Add tracing
    "db.type": "sqlite",
    "db.path": DB_PATH,
})
def run_sql_tool(user_question: str, conversation_history: list[dict]) -> dict:
    """
    Main entry point for the SQL tool.

    Returns a dict:
      {
        "sql":      str,          # the generated SQL
        "rows":     list[dict],   # raw query results
        "table_md": str,          # markdown-formatted result table
        "error":    str | None,   # execution error, if any
      }
    """
    sql   = _generate_sql(user_question, conversation_history)
    rows, error = _execute_sql(sql)

    return {
        "sql":      sql,
        "rows":     rows,
        "table_md": _rows_to_markdown(rows) if not error else f"_SQL error: {error}_",
        "error":    error,
    }