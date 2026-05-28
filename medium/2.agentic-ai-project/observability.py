# observability.py
# ─────────────────────────────────────────────────────────────────────────────
# MLflow 트레이싱: 비용 자동 계산 및 입출력 크기 추적
#
# 자동 수집 속성:
#   - cost.usd          : 토큰 기반 추정 비용 (USD)
#   - tokens.input      : 입력 토큰 수 (추정값)
#   - tokens.output     : 출력 토큰 수 (추정값)
#   - bytes.input       : 입력 크기 (바이트)
#   - bytes.output      : 출력 크기 (바이트)
#   - duration_ms       : 실행 시간 (밀리초)
#
# 사용 예:
#   @trace(span_type="TOOL", model="claude-sonnet-4-6")  # 비용 추적 활성화
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

# ── 모델 가격표 (100만 토큰당 USD) ──────────────────────────────────────────
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

    # Voyage AI 임베딩 (상용)
    # 임베딩 모델은 입력 토큰만 과금되며 출력 비용은 없음
    "voyage-3-large":        {"input": 0.18, "output": 0.00},
    "voyage-3.5":            {"input": 0.06, "output": 0.00},
    "voyage-3.5-lite":       {"input": 0.02, "output": 0.00},
    "voyage-3":              {"input": 0.06, "output": 0.00},
    "voyage-3-lite":         {"input": 0.02, "output": 0.00},
    "voyage-4-lite":         {"input": 0.02, "output": 0.00},
    "voyage-code-3":         {"input": 0.18, "output": 0.00},
    "voyage-finance-2":      {"input": 0.12, "output": 0.00},
    "voyage-law-2":          {"input": 0.12, "output": 0.00},
    "voyage-multilingual-2": {"input": 0.12, "output": 0.00},

    # 로컬 임베딩 (API 비용 없음)
    "sentence-transformers/all-MiniLM-L6-v2": {"input": 0.00, "output": 0.00},
}

# 토큰 수 대략 추정 (영어 기준 1 토큰 ≈ 4 문자)
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) // 4

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    # 가격표에 없는 모델은 프로젝트 기본 LLM(Claude Sonnet) 가격으로 폴백
    pricing = MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)

# ── 초기화 ──────────────────────────────────────────────────────────────────

ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
_initialized = False

def _probe_tracking_server(uri: str, timeout: float = 1.0) -> bool:
    # 로컬 백엔드(file:, sqlite: 등)는 프로브 불필요. HTTP만 확인하는 이유는
    # mlflow.set_experiment() 내부에서 urllib3가 연결 실패 시 수십 초간
    # 재시도하기 때문에 미리 차단하기 위함.
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

# ── 크기 계산 ───────────────────────────────────────────────────────────────

def calculate_size(obj: Any) -> int:
    """객체의 대략적인 바이트 크기를 계산."""
    try:
        if isinstance(obj, str):
            return len(obj.encode('utf-8'))
        elif isinstance(obj, (list, dict)):
            return len(json.dumps(obj).encode('utf-8'))
        else:
            return len(str(obj).encode('utf-8'))
    except Exception:
        return 0

# ── 요청 인자 추출 ──────────────────────────────────────────────────────────

_REQUEST_KEYWORDS = ("question", "message", "prompt", "query", "input")

def _extract_request_arg(arg_names, args, kwargs):
    """함수의 바인딩된 인자 중 사용자 요청 문자열을 선택.

    인자 이름에 요청 관련 키워드(question/message/prompt/query/input)가
    포함된 문자열 인자를 우선 사용하고, 없으면 첫 번째 문자열 인자로 폴백.
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

# ── 데코레이터 (비용 및 크기 추적 포함) ─────────────────────────────────────

def trace(name=None, span_type="CHAIN", attributes=None, model=None):
    """
    함수에 트레이싱을 적용하고 비용/크기를 자동 추적.

    Args:
        name: 스팬 이름 (기본값: 함수 이름)
        span_type: CHAIN, TOOL, RETRIEVER, PARSER, LLM
        attributes: 정적 속성 딕셔너리
        model: 비용 추적용 모델 이름 (예: "claude-sonnet-4-6")
    """
    static_attrs = attributes or {}

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLED:
                return func(*args, **kwargs)

            init()
            span_name = name or func.__name__

            # 입력값 구성 및 입력 크기 계산
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

                # 화면 표시용으로 잘라낸 버전을 저장
                display_val = val
                if isinstance(val, str):
                    input_text_parts.append(val)
                    if len(val) > 500:
                        display_val = val[:500] + "..."
                elif isinstance(val, (list, dict)):
                    input_text_parts.append(json.dumps(val))
                inputs[arg_name] = display_val

            # 진입 시점에 사용자 요청 문자열을 한 번 추출해 두고 이후 재사용
            request_arg = _extract_request_arg(arg_names, args, kwargs)
            if request_arg:
                inputs["request"] = request_arg

            input_size = calculate_size(args) + calculate_size(kwargs)

            start = time.perf_counter()

            with mlflow.start_span(name=span_name, span_type=span_type) as span:
                # 입력값 기록
                if inputs:
                    span.set_inputs(inputs)

                # 정적 속성 기록
                for key, value in static_attrs.items():
                    span.set_attribute(key, value)

                # 함수 메타데이터 기록
                span.set_attribute("func.name", func.__name__)
                span.set_attribute("func.module", func.__module__)

                # 입력 크기 추적
                span.set_attribute("bytes.input", input_size)

                # 모델이 지정된 경우 입력 토큰 추정
                if model:
                    input_text = " ".join(input_text_parts)
                    input_tokens = estimate_tokens(input_text)
                    span.set_attribute("tokens.input", input_tokens)
                    span.set_attribute("model.name", model)

                try:
                    result = func(*args, **kwargs)

                    # 실행 시간 기록
                    elapsed = round((time.perf_counter() - start) * 1000, 2)
                    span.set_attribute("duration_ms", elapsed)

                    # 출력 크기 기록
                    output_size = calculate_size(result)
                    span.set_attribute("bytes.output", output_size)

                    # 출력 토큰 추정 및 비용 계산
                    if model and isinstance(result, dict):
                        output_text = json.dumps(result) if result else ""
                        output_tokens = estimate_tokens(output_text)
                        span.set_attribute("tokens.output", output_tokens)

                        input_tokens = span.get_attribute("tokens.input") or 0
                        cost = calculate_cost(model, input_tokens, output_tokens)
                        span.set_attribute("cost.usd", cost)
                        span.set_attribute("tokens.total", input_tokens + output_tokens)

                    # 결과 유형별 출력 추출
                    if isinstance(result, dict):
                        outputs = {
                            "response": result  # 전체 응답을 명시적으로 캡처
                        }
                        if "error" in result:
                            outputs["error"] = result["error"]
                            span.set_attribute("error", True)
                        if "rows" in result:
                            outputs["row_count"] = len(result["rows"])
                            span.set_attribute("db.rows_returned", len(result["rows"]))
                        if "sql" in result:
                            outputs["sql"] = result["sql"]  # SQL 쿼리 전문 캡처
                        if "table_md" in result:
                            outputs["table_md"] = result["table_md"]  # 마크다운 결과 캡처
                        if "chunks" in result:
                            outputs["chunk_count"] = len(result["chunks"])
                            outputs["chunks"] = result["chunks"]  # 검색된 청크 상세 캡처
                            span.set_attribute("retrieval.chunks", len(result["chunks"]))
                        if "sources" in result:
                            outputs["sources"] = result["sources"]  # 출처 정보 캡처
                            span.set_attribute("retrieval.sources", len(result["sources"]))
                        if "results" in result:
                            outputs["result_count"] = len(result["results"])
                            # 웹 검색 결과에서 상위 링크 추출
                            if result["results"] and isinstance(result["results"], list):
                                top_links = [item.get("url") for item in result["results"] if item.get("url")]
                                if top_links:
                                    outputs["top_links"] = top_links
                                    span.set_attribute("search.top_links_count", len(top_links))

                        span.set_outputs(outputs)
                    else:
                        # dict가 아닌 결과도 response 키로 캡처
                        span.set_outputs({"response": result})

                    # 진입 시 추출한 요청 문자열을 트레이스 레벨 업데이트에 재사용
                    request_str = request_arg

                    response_str = None
                    if isinstance(result, dict):
                        response_str = json.dumps(result)
                    elif isinstance(result, str):
                        response_str = result
                    else:
                        response_str = str(result)

                    # 트레이스에 요청/응답 미리보기 업데이트
                    # None 값은 기존 값을 덮어쓰지 않도록 제외
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

# ── 컨텍스트 매니저 (비용/크기 추적 포함) ───────────────────────────────────

@contextmanager
def trace_span(name, span_type="CHAIN", attributes=None, model=None):
    """코드 블록을 트레이싱하며 비용/크기를 추적."""
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

# ── 헬퍼: 수동 속성 업데이트 ────────────────────────────────────────────────

def set_attr(key: str, value):
    """현재 스팬에 속성 하나를 설정."""
    if ENABLED:
        current = mlflow.get_current_active_span()
        if current:
            current.set_attribute(key, value)

def set_attrs(attributes: dict):
    """현재 스팬에 여러 속성을 한 번에 설정."""
    if ENABLED:
        current = mlflow.get_current_active_span()
        if current:
            for key, value in attributes.items():
                current.set_attribute(key, value)

def log_cost(model: str, input_tokens: int, output_tokens: int):
    """스팬에 비용을 수동으로 기록."""
    if ENABLED:
        cost = calculate_cost(model, input_tokens, output_tokens)
        set_attrs({
            "cost.usd": cost,
            "tokens.input": input_tokens,
            "tokens.output": output_tokens,
            "tokens.total": input_tokens + output_tokens,
            "model.name": model
        })
