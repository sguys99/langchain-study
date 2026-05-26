# observability.py
# ─────────────────────────────────────────────────────────────────────────────
# MLflow tracing with automatic cost calculation and size tracking.
#
# Auto-captured attributes:
#   - cost.usd          : Estimated cost based on tokens
#   - tokens.input      : Input token count (estimated)
#   - tokens.output     : Output token count (estimated)
#   - bytes.input       : Input size in bytes
#   - bytes.output      : Output size in bytes
#   - duration_ms       : Execution time
#
# Usage:
#   @trace(span_type="TOOL", model="claude-sonnet-4-6")  # Enables cost tracking
#   def run_sql_tool(...): ...
# ─────────────────────────────────────────────────────────────────────────────

import os
import functools
import socket
import time
import json
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import mlflow

# ── Pricing (per 1M tokens) ────────────────────────────────────────────────
MODEL_PRICING = {
    # Anthropic Claude
    "claude-opus-4-7":   {"input": 15.00, "output": 75.00},
    "claude-opus-4-6":   {"input": 15.00, "output": 75.00},
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input":  3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":  {"input":  1.00, "output":  5.00},

    # OpenAI
    "gpt-4o":      {"input":  2.50, "output": 10.00},
    "gpt-4o-mini": {"input":  0.15, "output":  0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},

    # Local embeddings (no API cost)
    "sentence-transformers/all-MiniLM-L6-v2": {"input": 0.00, "output": 0.00},
}

# Rough token estimation (1 token ≈ 4 chars for English)
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) // 4

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    # Fallback to Claude Sonnet pricing — the project's default LLM tier.
    pricing = MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)

# ── Setup ────────────────────────────────────────────────────────────────────

ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
_initialized = False

def _probe_tracking_server(uri: str, timeout: float = 1.0) -> bool:
    # Local backends (file:, sqlite:, …) need no probe; only HTTP needs one
    # because mlflow.set_experiment() calls urllib3 which otherwise retries
    # for tens of seconds before giving up.
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        return True
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def init():
    global _initialized, ENABLED
    if _initialized or not ENABLED:
        return

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "ecommerce-agent")

    if not _probe_tracking_server(uri):
        print(
            f"[observability] MLflow tracking server unreachable at {uri} — "
            "disabling observability for this session. Start the server "
            "(e.g. `mlflow server --host 0.0.0.0 --port 5001`) or set "
            "MLFLOW_ENABLED=false to silence this warning."
        )
        ENABLED = False
        _initialized = True
        return

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    _initialized = True

# ── Size Calculation ─────────────────────────────────────────────────────────

def calculate_size(obj: Any) -> int:
    """Calculate approximate byte size of an object."""
    try:
        if isinstance(obj, str):
            return len(obj.encode('utf-8'))
        elif isinstance(obj, (list, dict)):
            return len(json.dumps(obj).encode('utf-8'))
        else:
            return len(str(obj).encode('utf-8'))
    except Exception:
        return 0

# ── Request-arg extraction ───────────────────────────────────────────────────

_REQUEST_KEYWORDS = ("question", "message", "prompt", "query", "input")

def _extract_request_arg(arg_names, args, kwargs):
    """Pick the user-facing request string from a function's bound arguments.

    Prefers a string arg whose name contains a request-like keyword
    (question/message/prompt/query/input); falls back to the first string arg.
    """
    def _iter_string_args():
        for i, arg_name in enumerate(arg_names):
            if i < len(args):
                val = args[i]
            elif arg_name in kwargs:
                val = kwargs[arg_name]
            else:
                continue
            if isinstance(val, str):
                yield arg_name, val

    for arg_name, val in _iter_string_args():
        if any(kw in arg_name.lower() for kw in _REQUEST_KEYWORDS):
            return val
    for _, val in _iter_string_args():
        return val
    return None

# ── The Decorator (With Cost & Size Tracking) ────────────────────────────────

def trace(name=None, span_type="CHAIN", attributes=None, model=None):
    """
    Trace any function with automatic cost and size tracking.

    Args:
        name: Span name (defaults to function name)
        span_type: CHAIN, TOOL, RETRIEVER, PARSER, LLM
        attributes: Dict of static attributes
        model: Model name for cost tracking (e.g., "claude-sonnet-4-6")
    """
    static_attrs = attributes or {}

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLED:
                return func(*args, **kwargs)

            init()
            span_name = name or func.__name__

            # Build inputs and calculate input size
            inputs = {}
            input_text_parts = []
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]

            for i, arg_name in enumerate(arg_names):
                if i < len(args):
                    val = args[i]
                elif arg_name in kwargs:
                    val = kwargs[arg_name]
                else:
                    continue

                # Store truncated version for display
                display_val = val
                if isinstance(val, str):
                    input_text_parts.append(val)
                    if len(val) > 500:
                        display_val = val[:500] + "..."
                elif isinstance(val, (list, dict)):
                    input_text_parts.append(json.dumps(val))
                inputs[arg_name] = display_val

            # Resolve the user-facing request string once and reuse later.
            request_arg = _extract_request_arg(arg_names, args, kwargs)
            if request_arg:
                inputs["request"] = request_arg

            input_size = calculate_size(args) + calculate_size(kwargs)

            start = time.perf_counter()

            with mlflow.start_span(name=span_name, span_type=span_type) as span:
                # Set inputs
                if inputs:
                    span.set_inputs(inputs)

                # Static attributes
                for key, value in static_attrs.items():
                    span.set_attribute(key, value)

                # Function metadata
                span.set_attribute("func.name", func.__name__)
                span.set_attribute("func.module", func.__module__)

                # Input size tracking
                span.set_attribute("bytes.input", input_size)

                # Estimate input tokens if model specified
                if model:
                    input_text = " ".join(input_text_parts)
                    input_tokens = estimate_tokens(input_text)
                    span.set_attribute("tokens.input", input_tokens)
                    span.set_attribute("model.name", model)

                try:
                    result = func(*args, **kwargs)

                    # Timing
                    elapsed = round((time.perf_counter() - start) * 1000, 2)
                    span.set_attribute("duration_ms", elapsed)

                    # Output size
                    output_size = calculate_size(result)
                    span.set_attribute("bytes.output", output_size)

                    # Estimate output tokens and calculate cost
                    if model and isinstance(result, dict):
                        output_text = json.dumps(result) if result else ""
                        output_tokens = estimate_tokens(output_text)
                        span.set_attribute("tokens.output", output_tokens)

                        input_tokens = span.get_attribute("tokens.input") or 0
                        cost = calculate_cost(model, input_tokens, output_tokens)
                        span.set_attribute("cost.usd", cost)
                        span.set_attribute("tokens.total", input_tokens + output_tokens)

                    # Smart output extraction
                    if isinstance(result, dict):
                        outputs = {
                            "response": result  # Explicitly capture full response
                        }
                        if "error" in result:
                            outputs["error"] = result["error"]
                            span.set_attribute("error", True)
                        if "rows" in result:
                            outputs["row_count"] = len(result["rows"])
                            span.set_attribute("db.rows_returned", len(result["rows"]))
                        if "sql" in result:
                            outputs["sql"] = result["sql"]  # Capture full SQL
                        if "table_md" in result:
                            outputs["table_md"] = result["table_md"]  # Capture formatted results
                        if "chunks" in result:
                            outputs["chunk_count"] = len(result["chunks"])
                            outputs["chunks"] = result["chunks"]  # Capture full chunk details
                            span.set_attribute("retrieval.chunks", len(result["chunks"]))
                        if "sources" in result:
                            outputs["sources"] = result["sources"]  # Capture sources
                            span.set_attribute("retrieval.sources", len(result["sources"]))
                        if "results" in result:
                            outputs["result_count"] = len(result["results"])
                            # Extract top-k links for web search results
                            if result["results"] and isinstance(result["results"], list):
                                top_links = [item.get("url") for item in result["results"] if item.get("url")]
                                if top_links:
                                    outputs["top_links"] = top_links
                                    span.set_attribute("search.top_links_count", len(top_links))

                        span.set_outputs(outputs)
                    else:
                        # For non-dict results, still capture as response
                        span.set_outputs({"response": result})

                    # Reuse the request resolved at entry for the trace-level update.
                    request_str = request_arg

                    response_str = None
                    if isinstance(result, dict):
                        response_str = json.dumps(result)
                    elif isinstance(result, str):
                        response_str = result
                    else:
                        response_str = str(result)
                    
                    # Update the trace with request preview and response preview
                    # Only include parameters that are not None to avoid nullifying them
                    trace_update_kwargs = {}
                    if request_str:
                        trace_update_kwargs["request_preview"] = request_str[:500]
                    if response_str:
                        trace_update_kwargs["response_preview"] = response_str[:500]
                    
                    if trace_update_kwargs:
                        mlflow.update_current_trace(**trace_update_kwargs)

                    return result

                except Exception as e:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    raise

        return wrapper
    return decorator

# ── Context Manager (Also with cost/size tracking) ───────────────────────────

@contextmanager
def trace_span(name, span_type="CHAIN", attributes=None, model=None):
    """Trace a block of code with full tracking."""
    if not ENABLED:
        yield None
        return

    init()

    with mlflow.start_span(name=name, span_type=span_type) as span:
        start = time.perf_counter()

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        if model:
            span.set_attribute("model.name", model)

        try:
            yield span

            elapsed = round((time.perf_counter() - start) * 1000, 2)
            span.set_attribute("duration_ms", elapsed)

        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise

# ── Helper: Manual Attribute Updates ────────────────────────────────────────

def set_attr(key: str, value):
    """Set attribute on current span."""
    if ENABLED:
        current = mlflow.get_current_active_span()
        if current:
            current.set_attribute(key, value)

def set_attrs(attributes: dict):
    """Set multiple attributes."""
    if ENABLED:
        current = mlflow.get_current_active_span()
        if current:
            for key, value in attributes.items():
                current.set_attribute(key, value)

def log_cost(model: str, input_tokens: int, output_tokens: int):
    """Manually log cost for a span."""
    if ENABLED:
        cost = calculate_cost(model, input_tokens, output_tokens)
        set_attrs({
            "cost.usd": cost,
            "tokens.input": input_tokens,
            "tokens.output": output_tokens,
            "tokens.total": input_tokens + output_tokens,
            "model.name": model
        })