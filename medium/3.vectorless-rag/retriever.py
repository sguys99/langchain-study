"""
retriever.py
------------
Agent-based retrieval that navigates a DocumentTree to answer a query,
with detailed logging of every LLM call and decision.

[한글 설명]
이 모듈은 "벡터 없는(vectorless) RAG"를 구현한다.
일반적인 RAG는 문서를 임베딩 벡터로 변환해 유사도 검색을 하지만,
여기서는 문서를 트리(DocumentTree) 구조로 만들어 두고,
LLM 에이전트가 루트 노드부터 시작해 질의와 가장 관련 있는
자식 노드로 "내려가며(descend)" 답을 찾는다.

핵심 흐름:
  1. analyze   : 현재 노드를 보고 더 내려갈지/여기서 내용을 가져올지 LLM이 판단
  2. descend   : 더 내려가기로 했다면 가장 관련 있는 자식 노드로 이동
  3. retrieve  : 답을 찾을 노드에 도달하면 그 노드의 본문을 수집
  4. generate  : 수집한 본문만 근거로 최종 답변을 LLM이 생성

LangGraph의 StateGraph로 이 흐름을 상태 기계(state machine)처럼 구성한다.
또한 모든 LLM 호출과 의사결정을 콘솔/파일에 상세히 로깅한다.
"""

import os
import json
import re
import time
import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator

# LangGraph: 노드/엣지 기반 워크플로(상태 기계)를 만드는 라이브러리
from langgraph.graph import StateGraph, END
# Anthropic: LLM 호출 클라이언트
import anthropic
# .env 파일에서 환경 변수(API 키, 모델명)를 로딩
from dotenv import load_dotenv

# tree.py에서 정의한 문서 트리 자료구조
from tree import TreeNode, DocumentTree

# .env의 ANTHROPIC_API_KEY / ANTHROPIC_MODEL을 환경 변수로 로딩
load_dotenv()

# 기본 모델: .env의 ANTHROPIC_MODEL을 우선 사용, 없으면 폴백 값 사용
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ── Logger setup ──────────────────────────────────────────────────────────────
#
# Two handlers:
#   console  — INFO and above, human-readable with colour-coded prefixes
#   file     — DEBUG and above, full detail including raw prompts/responses
#
# Usage from outside:
#   import logging
#   logging.getLogger("retriever").setLevel(logging.DEBUG)  # show raw prompts too
#
# [한글 설명] 로거 설정
# 두 개의 핸들러(출력 경로)를 둔다:
#   - 콘솔 핸들러 : INFO 레벨 이상만 출력 → 사람이 읽기 좋은 요약 로그
#   - 파일 핸들러 : DEBUG 레벨 이상 출력 → 프롬프트/응답 원문까지 전부 기록
# 외부에서 raw 프롬프트까지 콘솔에 보고 싶다면 위 Usage처럼 레벨을 DEBUG로 낮추면 된다.

logger = logging.getLogger("retriever")       # "retriever"라는 이름의 전용 로거 생성
logger.setLevel(logging.DEBUG)                 # 로거 자체는 가장 낮은 DEBUG까지 통과시킴(핸들러가 다시 필터링)
logger.propagate = False   # don't bubble up to root logger
                            # 상위(root) 로거로 메시지가 전파되어 중복 출력되는 것을 방지

# 핸들러가 아직 없을 때만 추가 → 모듈이 여러 번 import돼도 핸들러가 중복 등록되지 않게 함
if not logger.handlers:
    # ── Console handler (INFO) ────────────────────────────────────────────
    # 콘솔(표준 스트림) 출력용 핸들러
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)                            # INFO 이상만 콘솔에 표시
    _ch.setFormatter(logging.Formatter("%(message)s"))   # raw message only
                                                          # 시간/레벨 없이 메시지 본문만 깔끔하게 출력
    logger.addHandler(_ch)

    # ── File handler (DEBUG) ─────────────────────────────────────────────
    # 파일(retriever.log) 출력용 핸들러. mode="a"는 기존 로그에 이어쓰기(append)
    _fh = logging.FileHandler("retriever.log", mode="a", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)                            # DEBUG 이상 전부 파일에 기록
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",       # 시각 + 레벨 + 메시지 형식
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_fh)


# ── Visual helpers ────────────────────────────────────────────────────────────
# [한글 설명] 로그를 보기 좋게 꾸미기 위한 시각적 헬퍼들

_DIVIDER   = "─" * 65   # 가는 구분선 (호출 단위 구분용)
_SEPARATOR = "═" * 65   # 굵은 구분선 (큰 섹션 구분용)

def _box(title: str) -> str:
    """제목을 굵은 구분선으로 위아래 감싸 박스처럼 강조한 문자열을 반환한다."""
    return f"\n{_SEPARATOR}\n  {title}\n{_SEPARATOR}"

def _indent(text: str, n: int = 4) -> str:
    """여러 줄 텍스트의 각 줄 앞에 공백 n칸을 붙여 들여쓰기한다 (프롬프트/응답 가독성용)."""
    pad = " " * n
    return "\n".join(pad + line for line in str(text).splitlines())


# ── State ─────────────────────────────────────────────────────────────────────
# [한글 설명] 그래프 전체에서 공유되는 "상태(State)" 정의
#
# LangGraph의 각 노드는 이 상태 딕셔너리를 입력으로 받고, 변경분(dict)을 반환한다.
# Annotated[..., operator.add]로 표시된 필드는 노드가 반환한 값이 "덮어쓰기"가 아니라
# 기존 값에 "누적(append/합치기)"된다. (예: 리스트끼리 + 연산으로 이어붙임)
# 그 외 일반 필드는 노드가 반환하면 그 값으로 그냥 덮어써진다.

class RetrievalState(TypedDict):
    query: str                                            # 사용자의 질문(질의)
    current_node: Optional[TreeNode]                      # 지금 탐색 중인 트리 노드 (시작 시 None)
    tree: TreeNode                                        # 전체 문서 트리 (또는 루트 노드)
    path_taken: Annotated[List[str], operator.add]        # 거쳐온 노드 id들의 경로 (누적)
    retrieved_content: Annotated[List[str], operator.add] # 수집된 본문 청크들 (누적)
    reasoning: str                                        # LLM이 내린 마지막 판단의 근거(한 문장)
    confidence: float                                     # 현재 노드가 답을 가질 확신도 (0~1)
    should_descend: bool                                  # 자식 노드로 더 내려갈지 여부
    target_child_id: Optional[str]                        # 내려갈 대상 자식 노드의 id
    depth: int                                            # 현재 탐색 깊이 (루트=0)
    final_answer: Optional[str]                           # 최종 생성된 답변
    call_log: Annotated[List[dict], operator.add]   # full log of every LLM call
                                                    # 모든 LLM 호출 기록 (누적)


# ── Core LLM caller with logging ──────────────────────────────────────────────

def _call_llm(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    call_type: str,          # "navigate" | "answer"
    call_number: int,
    max_tokens: int = 1024,  # Anthropic Messages API는 max_tokens가 필수 인자
) -> tuple[str, float]:
    """
    Call the LLM and log everything:
      - call number and type
      - full prompt (DEBUG / file only)
      - raw response (DEBUG / file only)
      - latency
    Returns (response_text, elapsed_seconds).

    [한글 설명]
    LLM을 1회 호출하고 관련 정보를 모두 로깅하는 공통 함수.
      - 호출 번호와 종류(navigate/answer)
      - 전체 프롬프트 (DEBUG → 파일에만 기록, 콘솔에는 너무 길어서 생략)
      - LLM 원문 응답 (DEBUG → 파일에만)
      - 응답 지연 시간(latency)
    반환값: (응답 텍스트, 소요 시간(초))
    """
    # 호출 헤더 로깅: 몇 번째 호출이고 어떤 종류인지 표시
    logger.info(f"\n{_DIVIDER}")
    logger.info(f"  LLM Call #{call_number}  [{call_type.upper()}]")
    logger.info(_DIVIDER)

    # Full prompt goes to file (DEBUG) only — too verbose for console
    # 전체 프롬프트는 너무 길어서 파일(DEBUG)에만 남김
    logger.debug(f"PROMPT:\n{_indent(prompt)}")

    t0 = time.perf_counter()                  # 호출 직전 시각 기록 (지연 시간 측정 시작)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,                 # 생성할 최대 토큰 수 (Anthropic은 필수)
        temperature=0.0,                       # 0.0 → 결정론적 출력(매번 같은 답) 유도, 탐색 일관성 확보
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - t0         # 호출에 걸린 시간(초) 계산
    raw = response.content[0].text.strip()     # 응답 본문(첫 텍스트 블록)만 추출하고 앞뒤 공백 제거

    # 응답 원문은 파일(DEBUG)에만, 모델/지연/토큰 사용량은 콘솔(INFO)에도 기록
    logger.debug(f"RAW RESPONSE:\n{_indent(raw)}")
    logger.info(f"  Model    : {model}")
    logger.info(f"  Latency  : {elapsed:.2f}s")
    logger.info(f"  Tokens   : {response.usage.input_tokens} in / "     # 입력 토큰 수
                f"{response.usage.output_tokens} out")                   # 출력 토큰 수

    return raw, elapsed


def _strip_fences(text: str) -> str:
    """
    LLM이 JSON을 ```json ... ``` 같은 마크다운 코드펜스로 감싸 응답하는 경우가 많다.
    이 함수는 펜스로 분리된 조각들 중 실제로 JSON 파싱이 되는 부분을 찾아 반환한다.
    펜스가 없거나 유효한 JSON 조각을 못 찾으면 원본 텍스트를 그대로 반환한다.
    """
    if "```" in text:                                  # 코드펜스가 있는 경우만 처리
        for part in text.split("```"):                 # ```를 기준으로 조각냄
            part = part.strip().lstrip("json").strip() # 앞쪽의 "json" 언어 표기와 공백 제거
            try:
                json.loads(part)                       # 이 조각이 유효한 JSON인지 시도
                return part                            # 성공하면 그 조각 반환
            except json.JSONDecodeError:
                continue                               # 실패하면 다음 조각 시도
    return text                                        # 못 찾으면 원본 그대로 반환


# ── Graph nodes ───────────────────────────────────────────────────────────────
# [한글 설명] 그래프를 구성하는 각 노드(단계)들.
# 각 _make_xxx 함수는 client/model을 클로저로 가둔 뒤 실제 노드 함수를 반환하는
# "팩토리(factory)" 패턴이다. 이렇게 하면 노드 함수가 LangGraph 시그니처
# (state -> dict)를 유지하면서도 client/model에 접근할 수 있다.

def _make_analyze(client: anthropic.Anthropic, model: str):
    """[분석 노드 생성] 현재 노드를 보고 '더 내려갈지 / 여기서 답을 찾을지'를 LLM에게 판단시킨다."""
    def analyze_node(state: RetrievalState) -> dict:
        # Handle both TreeNode and DocumentTree objects
        # current_node가 있으면 그 노드를, 없으면(최초 진입) 트리의 루트를 시작점으로 삼는다.
        if state["current_node"]:
            node: TreeNode = state["current_node"]
        else:
            # If tree is a DocumentTree, get its root; otherwise use it directly
            # tree가 DocumentTree 객체면 .root를, TreeNode면 그 자체를 사용 (둘 다 호환)
            tree_obj = state["tree"]
            node = tree_obj.root if hasattr(tree_obj, "root") else tree_obj

        call_num = len(state["call_log"]) + 1   # 이번이 몇 번째 LLM 호출인지 (로그 길이 + 1)
        depth = state["depth"]                   # 현재 탐색 깊이

        # 자식 노드 정보를 LLM에게 보여주기 위해 (id/제목/요약 일부) 형태로 추림
        # 요약은 150자까지만 잘라 프롬프트가 과도하게 길어지지 않게 함
        children_info = (
            [{"id": c.id, "title": c.title,
              "summary": getattr(c, "summary", "")[:150]}
             for c in node.children]
            if node.children else []             # 자식이 없으면 빈 리스트(리프 노드)
        )

        # ── Log: entering this node ───────────────────────────────────────
        # 현재 진입한 노드 정보를 로깅. depth만큼 들여쓰기해 트리 구조를 시각적으로 표현
        logger.info(f"\n{'  ' * depth}┌─ Depth {depth} | Node: \"{node.title}\"")
        logger.info(f"{'  ' * depth}│  id={node.id}  pages={node.page_start}-{node.page_end}")
        logger.info(f"{'  ' * depth}│  children={[c.title for c in node.children] or 'none (leaf)'}")

        # ── LLM에게 보낼 네비게이션(탐색) 판단용 프롬프트 구성 ──
        # 현재 노드의 메타데이터/본문 미리보기와 자식 목록을 주고,
        # 확신도/내려갈지/대상 자식/근거를 JSON으로 답하라고 지시한다.
        prompt = f"""당신은 질의에 답하기 위해 연구 논문 트리를 탐색하고 있습니다.

질의: "{state['query']}"

현재 노드:
  id      : {node.id}
  제목     : {node.title}
  요약     : {getattr(node, 'summary', '')[:300]}
  페이지   : {node.page_start}–{node.page_end}
  미리보기 : {node.content[:500] if node.content else 'N/A'}

자식 노드: {json.dumps(children_info, ensure_ascii=False, indent=2) if children_info else "없음 (리프 노드)"}

다음을 판단하세요:
1. confidence      : 0~1 사이 값. 이 노드(또는 그 자식)가 답을 포함할 가능성은 얼마나 되는가?
2. should_descend  : 특정 자식 노드가 현재 노드의 본문보다 더 관련 있을 때에만 true
3. target_child_id : 방문할 가장 적합한 자식 노드의 id (should_descend가 false면 null)
4. reasoning       : 당신의 판단을 설명하는 한 문장 (한글로 작성)

반드시 마크다운 코드펜스 없이 유효한 JSON으로만 응답하세요:
{{
  "confidence": 0.85,
  "should_descend": true,
  "target_child_id": "1_Introduction_12",
  "reasoning": "서론(Introduction) 섹션이 Bigtable이 무엇인지 직접 설명한다."
}}"""

        # LLM 호출 → 응답에서 코드펜스 제거 (짧은 JSON 응답이므로 max_tokens는 작게)
        raw, elapsed = _call_llm(client, model, prompt, "navigate", call_num, max_tokens=512)
        raw = _strip_fences(raw)

        # 응답을 JSON으로 파싱. 실패하면 안전한 기본값(fallback)으로 대체
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"  [!] JSON parse failed — using fallback decision")
            # 파싱 실패 시: 자식이 있으면 첫 자식으로 내려가도록 보수적으로 설정
            decision = {
                "confidence": 0.5,
                "should_descend": bool(node.children),
                "target_child_id": node.children[0].id if node.children else None,
                "reasoning": "Fallback: could not parse LLM response",
            }

        # 판단 결과를 각 변수로 분해. .get(키, 기본값)으로 키 누락에도 안전하게 처리
        conf      = float(decision.get("confidence", 0.5))      # 확신도 (0~1)
        descend   = bool(decision.get("should_descend", False)) # 더 내려갈지 여부
        child_id  = decision.get("target_child_id")             # 내려갈 자식 id
        reasoning = decision.get("reasoning", "")               # 판단 근거

        # ── Log: decision ─────────────────────────────────────────────────
        # 판단 결과를 로깅. 내려갈지(↓)/여기서 수집할지(→)를 화살표로 표시
        arrow = "↓ descend" if (descend and node.children) else "→ retrieve"
        logger.info(f"{'  ' * depth}│")
        logger.info(f"{'  ' * depth}│  Decision   : {arrow}")
        logger.info(f"{'  ' * depth}│  Confidence : {conf:.0%}")
        if child_id and descend:
            logger.info(f"{'  ' * depth}│  Next node  : {child_id}")
        logger.info(f"{'  ' * depth}│  Reasoning  : {reasoning}")
        logger.info(f"{'  ' * depth}└─ ({elapsed:.2f}s)")

        # 이번 호출 기록을 call_log에 누적할 형태로 구성
        entry = {
            "call_number":    call_num,
            "call_type":      "navigate",
            "node_id":        node.id,
            "node_title":     node.title,
            "depth":          depth,
            "confidence":     conf,
            "should_descend": descend,
            "target_child":   child_id,
            "reasoning":      reasoning,
            "latency_s":      round(elapsed, 3),
        }

        # 상태 변경분을 반환. operator.add가 붙은 필드(path_taken, call_log)는 누적됨
        return {
            "path_taken":      [node.id],     # LangGraph appends automatically
                                              # 경로에 현재 노드 id 추가(누적)
            "current_node":    node,          # 현재 노드 확정(루트 진입 보정 반영)
            "confidence":      conf,
            "should_descend":  descend,
            "target_child_id": child_id,
            "reasoning":       reasoning,
            "depth":           depth + 1,     # 깊이 1 증가
            "call_log":        [entry],        # LangGraph appends automatically
                                               # 호출 기록 추가(누적)
        }

    return analyze_node


def _make_descend():
    """[하강 노드 생성] analyze가 고른 자식 노드로 current_node를 이동시킨다."""
    def descend(state: RetrievalState) -> dict:
        current: TreeNode = state["current_node"]
        target_id: Optional[str] = state.get("target_child_id")
        depth = state["depth"]

        # 대상 id와 일치하는 자식을 찾는다. 못 찾으면 첫 번째 자식으로 fallback
        target = next(
            (c for c in current.children if c.id == target_id),
            current.children[0],
        )

        logger.info(f"\n{'  ' * depth}➜  Descending into: \"{target.title}\"")

        # current_node를 자식으로 교체 → 다음 analyze는 이 자식을 분석하게 됨
        return {"current_node": target}

    return descend


def _make_retrieve():
    """[수집 노드 생성] 답을 찾을 노드에 도달했을 때 그 노드의 본문을 수집한다."""
    def retrieve(state: RetrievalState) -> dict:
        node: TreeNode = state["current_node"]
        depth = state["depth"]

        logger.info(f"\n{'  ' * depth}✦  Retrieving content from: \"{node.title}\"")
        logger.info(f"{'  ' * depth}   Pages {node.page_start}–{node.page_end} | "
                    f"{len(node.content)} chars")

        # 노드 제목/페이지 범위를 헤더로 붙인 본문 청크 생성 (출처 표시용)
        chunk = (
            f"=== **{node.title}** "
            f"(Pages {node.page_start}-{node.page_end}) ===\n"
            f"{node.content}"
        )
        return {"retrieved_content": [chunk]}   # LangGraph appends automatically
                                                # 수집 본문 목록에 추가(누적)

    return retrieve


def _make_generate(client: anthropic.Anthropic, model: str):
    """[답변 생성 노드 생성] 수집한 본문만 근거로 최종 답변을 LLM에게 생성시킨다."""
    def generate_answer(state: RetrievalState) -> dict:
        call_num  = len(state["call_log"]) + 1
        # 수집된 청크들을 구분선으로 이어 하나의 컨텍스트로 합침
        context   = "\n\n---\n\n".join(state["retrieved_content"])
        # 각 청크의 첫 줄(=== 제목/페이지 헤더)만 뽑아 출처 목록으로 사용
        sources   = [s.splitlines()[0] for s in state["retrieved_content"]]

        logger.info(f"\n{_DIVIDER}")
        logger.info(f"  Generating answer from {len(state['retrieved_content'])} "
                    f"retrieved section(s):")
        for s in sources:
            logger.info(f"    • {s}")

        # 답변 생성 프롬프트: 오직 수집된 본문만 근거로, 주장마다 출처를 인용하고,
        # 근거가 부족하면 추측하지 말고 그렇다고 명시하라고 지시
        prompt = f"""당신은 분산 시스템과 데이터베이스 엔지니어링 전문가입니다.
아래에 수집된 문서 구간만을 근거로 질문에 답하세요.
당신이 제시하는 모든 주장에는 섹션 제목과 페이지 범위를 인용하세요.
근거가 부족하면 추측하지 말고 그렇다고 명확히 밝히세요.
답변은 한글로 작성하세요.

질문: {state['query']}

수집된 구간:
{context}

답변:"""

        # LLM 호출 → 최종 답변 텍스트 획득 (긴 답변이므로 max_tokens는 넉넉히)
        raw, elapsed = _call_llm(client, model, prompt, "answer", call_num, max_tokens=2048)

        # 답변 생성 호출 기록
        entry = {
            "call_number": call_num,
            "call_type":   "answer",
            "sources":     sources,
            "latency_s":   round(elapsed, 3),
        }

        logger.info(f"\n  Answer generated in {elapsed:.2f}s")

        # 최종 답변을 상태에 기록하고, 호출 로그에 누적
        return {
            "final_answer": raw,
            "call_log":     [entry],
        }

    return generate_answer


# ── Routing ───────────────────────────────────────────────────────────────────
# [한글 설명] analyze 노드 이후 어디로 갈지 결정하는 분기 함수.
# 반환하는 문자열("end"/"retrieve"/"descend")이 다음에 실행할 노드를 가리킨다.

MAX_DEPTH = 5   # 무한 하강을 막기 위한 최대 탐색 깊이

def _route(state: RetrievalState) -> str:
    # 1) 확신도가 너무 낮으면(0.3 미만) 탐색을 중단하고 종료
    if state["confidence"] < 0.3:
        logger.info(f"\n  ✗  Low confidence ({state['confidence']:.0%}) — stopping traversal")
        return "end"
    # 2) 최대 깊이에 도달하면 더 내려가지 않고 현재 노드에서 본문 수집
    if state["depth"] >= MAX_DEPTH:
        logger.info(f"\n  ⚠  Max depth ({MAX_DEPTH}) reached — retrieving current node")
        return "retrieve"
    # 3) 더 내려가기로 했고 실제로 자식이 있으면 하강
    if state["should_descend"] and state["current_node"].children:
        return "descend"
    # 4) 그 외에는 현재 노드에서 본문 수집
    return "retrieve"


# ── Graph assembly ────────────────────────────────────────────────────────────
# [한글 설명] 위에서 만든 노드들과 분기 규칙을 엮어 실행 가능한 그래프로 조립한다.

def _build_graph(client: anthropic.Anthropic, model: str) -> Any:
    workflow = StateGraph(RetrievalState)   # 상태 타입을 지정해 그래프 생성

    # 4개의 노드 등록
    workflow.add_node("analyze",  _make_analyze(client, model))
    workflow.add_node("descend",  _make_descend())
    workflow.add_node("retrieve", _make_retrieve())
    workflow.add_node("generate", _make_generate(client, model))

    workflow.set_entry_point("analyze")     # 시작 노드는 analyze
    # analyze 이후에는 _route의 반환값에 따라 분기 (조건부 엣지)
    workflow.add_conditional_edges(
        "analyze", _route,
        {"descend": "descend", "retrieve": "retrieve", "end": END},
    )
    # 고정 엣지들:
    workflow.add_edge("descend",  "analyze")   # 하강 후 다시 분석 (재귀적 탐색)
    workflow.add_edge("retrieve", "generate")  # 수집 후 답변 생성
    workflow.add_edge("generate", END)         # 답변 생성 후 종료

    return workflow.compile()   # 실행 가능한 형태로 컴파일


# ── Visualization ─────────────────────────────────────────────────────────────

def generate_workflow_png(output_path: str = "workflow.png") -> str:
    """
    Generate a PNG visualization of the LangGraph workflow structure.

    Args:
        output_path: Path where the PNG will be saved (default: "workflow.png")

    Returns:
        Path to the generated PNG file

    [한글 설명]
    워크플로 구조를 PNG 이미지로 시각화한다. 실제 동작이 아닌 '구조'만 그리면 되므로
    각 노드는 상태를 그대로 돌려주는 더미(lambda) 함수로 채운다.
    """
    workflow = StateGraph(RetrievalState)

    # Add nodes (dummy functions for structure visualization)
    # 구조 시각화 목적이라 노드 동작은 더미(state 그대로 반환)로 둠
    workflow.add_node("analyze",  lambda state: state)
    workflow.add_node("descend",  lambda state: state)
    workflow.add_node("retrieve", lambda state: state)
    workflow.add_node("generate", lambda state: state)

    # 실제 _build_graph와 동일한 엣지 구조로 구성 (분기 함수도 더미로 대체)
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges(
        "analyze", lambda state: "retrieve",
        {"descend": "descend", "retrieve": "retrieve", "end": END},
    )
    workflow.add_edge("descend",  "analyze")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    graph = workflow.compile()
    # Mermaid 다이어그램을 PNG 바이트로 렌더링
    graph_image = graph.get_graph().draw_mermaid_png()

    # 바이너리 모드로 파일에 저장
    with open(output_path, "wb") as f:
        f.write(graph_image)

    logger.info(f"Workflow visualization saved to: {output_path}")
    return output_path


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    tree: TreeNode,
    client: Optional[anthropic.Anthropic] = None,
    model: Optional[str] = None,
) -> Dict:
    """
    Navigate the document tree and answer a query, logging every LLM call.

    Console output (INFO):
        Shows the traversal path, each decision, confidence, and reasoning.

    File output (DEBUG → retriever.log):
        Additionally logs the full prompt and raw LLM response for every call.

    [한글 설명] 외부에서 호출하는 공개 진입점(public API).
    문서 트리를 탐색해 질의에 답하고, 모든 LLM 호출을 로깅한다.
      - 콘솔(INFO) : 탐색 경로/각 판단/확신도/근거 요약
      - 파일(DEBUG → retriever.log) : 위에 더해 매 호출의 전체 프롬프트와 원문 응답
    반환값(dict): answer/path/reasoning/confidence/sources/call_log
    """
    # client 미전달 시 .env의 ANTHROPIC_API_KEY로 Anthropic 클라이언트를 자동 생성
    if client is None:
        client = anthropic.Anthropic()
    # model 미전달 시 .env의 ANTHROPIC_MODEL(기본값)을 사용
    if model is None:
        model = DEFAULT_MODEL

    logger.info(_box(f"New Query"))
    logger.info(f"\n  Q: {query}\n")

    graph = _build_graph(client, model)   # 실행 그래프 빌드

    t_start = time.perf_counter()         # 전체 소요 시간 측정 시작

    # 초기 상태를 넣고 그래프 실행. graph.invoke가 END에 도달할 때까지 노드들을 순회
    result = graph.invoke({
        "query":             query,
        "current_node":      None,
        "tree":              tree,
        "path_taken":        [],
        "retrieved_content": [],
        "reasoning":         "",
        "confidence":        0.0,
        "should_descend":    True,
        "target_child_id":   None,
        "depth":             0,
        "final_answer":      None,
        "call_log":          [],
    })

    total_s = time.perf_counter() - t_start    # 전체 소요 시간
    call_log = result.get("call_log", [])
    # 호출 종류별 횟수 집계 (탐색 호출 / 답변 호출)
    nav_calls = sum(1 for c in call_log if c["call_type"] == "navigate")
    ans_calls = sum(1 for c in call_log if c["call_type"] == "answer")

    # ── Final summary ──────────────────────────────────────────────────────
    # 탐색 결과 요약 로깅: 총 호출 수, 거쳐온 경로, 총 지연 시간
    logger.info(f"\n{_SEPARATOR}")
    logger.info("  RETRIEVAL SUMMARY")
    logger.info(_SEPARATOR)
    logger.info(f"  Total LLM calls : {len(call_log)}  "
                f"({nav_calls} navigate + {ans_calls} answer)")
    logger.info(f"  Path taken      : {' → '.join(result.get('path_taken', []))}")
    logger.info(f"  Total latency   : {total_s:.2f}s")
    logger.info(_SEPARATOR)

    # 호출자에게 돌려줄 결과 딕셔너리 구성
    # final_answer가 없으면(확신도 낮아 종료된 경우) 안내 문구로 대체
    return {
        "answer":     result.get("final_answer") or "No answer generated (low confidence).",
        "path":       result.get("path_taken", []),        # 거쳐온 노드 경로
        "reasoning":  result.get("reasoning", ""),         # 마지막 판단 근거
        "confidence": result.get("confidence", 0.0),       # 마지막 확신도
        "sources":    result.get("retrieved_content", []), # 답변 근거가 된 본문들
        "call_log":   call_log,                            # 전체 호출 로그
    }