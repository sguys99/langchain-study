# Anthropic(Claude) SDK 기반 '계획-실행-재계획(Plan-Execute-Replan)' AI 에이전트
#
# 이 파일은 AIAgent.py(정적 Plan-and-Execute)의 개선 버전입니다.
#
# [기존 AIAgent.py의 문제점]
#   - 1단계에서 계획을 '한 번' 세우고(Plan), 그 계획 전체를 그대로 실행(Execute)한 뒤,
#     마지막에 정리(Synthesize)합니다.
#   - 따라서 초기 계획이 잘못되거나 무의미하면, 이후 실행/정리 단계가 그대로 진행되어
#     의미 없는 결과가 도출됩니다. 도중에 계획을 고칠 방법이 없습니다.
#
# [개선 방식 - 첨부 이미지의 워크플로우와 동일, LangGraph plan-and-execute 패턴]
#   Plan(계획 분해) -> Execute(한 단계만 실행) -> Replan(정보 충분 여부 판단)
#     - 정보가 충분하면      -> 답변을 생성하고 종료
#     - 정보가 충분하지 않으면 -> 계획을 갱신하고 다시 Execute (반복)
#   그리고 '반복 횟수 상한(MAX_ITERATIONS)'을 두어 무한 루프를 방지합니다.
#
#   핵심은 '한 단계 실행 -> 즉시 Replan'이라는 점입니다. 잘못된 계획을 1단계 직후에
#   포착해 바로잡을 수 있어, 정적 계획 방식의 약점을 직접적으로 해결합니다.

import logging          # 실행 과정을 콘솔에 기록(로깅)하기 위한 표준 라이브러리
import json             # LLM이 만든 JSON 문자열을 파이썬 객체로 변환하기 위한 라이브러리
import os               # 환경 변수(API 키 등)에 접근하기 위한 라이브러리

import anthropic        # Anthropic(Claude) 공식 파이썬 SDK
from dotenv import load_dotenv  # .env 파일에 저장한 환경 변수를 읽어오기 위한 라이브러리

from tools import tools_schema, tools  # 에이전트가 사용할 도구 정의(스키마)와 실제 함수 모음

# .env 파일에 적어둔 ANTHROPIC_API_KEY 등의 환경 변수를 현재 프로세스로 불러옵니다.
load_dotenv()

# Anthropic 클라이언트를 생성합니다.
# 별도로 키를 넘기지 않으면 환경 변수 ANTHROPIC_API_KEY 값을 자동으로 사용합니다.
client = anthropic.Anthropic()

# 사용할 Claude 모델 이름을 상수로 정의합니다.
MODEL_NAME = "claude-sonnet-4-6"

# Replan 루프의 최대 반복 횟수. 무한 루프를 방지하는 안전장치입니다.
MAX_ITERATIONS = 5

# 로깅 설정: 시각 / 로그 레벨 / 메시지 형식을 지정합니다.
logging.basicConfig(
    level=logging.INFO,                                   # INFO 레벨 이상만 출력
    format='%(asctime)s - %(levelname)s - %(message)s',   # 출력 형식
    datefmt='%H:%M:%S'                                    # 시각 형식 (시:분:초)
)
logger = logging.getLogger(__name__)


class Agent():
    """계획(Plan) -> 한 단계 실행(Execute) -> 재계획(Replan)을 반복하는 에이전트.

    각 반복마다 계획의 첫 단계만 실행하고, 곧바로 'Replan'으로 지금까지의 결과가
    답변에 충분한지 판단합니다. 충분하면 답변을 출력하고, 부족하면 남은 계획을
    갱신해 다시 실행합니다. 반복은 max_iterations로 상한이 걸립니다.
    """

    def __init__(self, available_tools_dict, max_iterations: int = MAX_ITERATIONS):
        # available_tools_dict: {"함수이름": 실제_함수} 형태의 딕셔너리
        # 에이전트가 LLM이 고른 도구 이름으로 실제 함수를 찾아 호출할 때 사용합니다.
        self.plan = False                                 # 계획 수립 여부를 추적하는 플래그
        self.system_prompt = "당신은 유능한 비서입니다."   # 기본 시스템 프롬프트(한글)
        self.available_tools_dict = available_tools_dict
        self.max_iterations = max_iterations              # Replan 반복 상한

    # ---------------------------------------------------------------------
    # 전체 흐름을 지휘하는 오케스트레이터(orchestrator)
    # ---------------------------------------------------------------------
    def run(self, user_prompt: str, tools: list) -> str:
        """사용자 요청을 받아 Plan -> (Execute -> Replan) 반복 순으로 처리합니다."""
        print(f"\n--- 1단계: 작업 계획 수립 (PLAN TASKS) ---")
        # 초기 계획: 남은 단계들의 목록(list[str])
        plan = self.__plan_tasks(user_prompt)["tasks"]

        # 지금까지 실행을 마친 단계들의 정보/결과를 누적합니다.
        # 각 항목은 {task, function, properties, result} 형태의 dict 입니다.
        past_steps = []

        # ----- Execute -> Replan 반복 루프 (반복 횟수 상한 적용) -----
        for i in range(1, self.max_iterations + 1):
            print(f"\n=============== 반복 {i}/{self.max_iterations} ===============")

            # 남은 계획이 없으면 더 실행할 것이 없으므로 지금까지 결과로 답변을 정리합니다.
            if not plan:
                logger.info("남은 계획이 없습니다. 지금까지 결과로 답변을 정리합니다.")
                return self.__synthesize_answer(user_prompt, past_steps)

            # --- 2단계: Execute (계획의 첫 단계 '하나만' 실행) ---
            print(f"\n--- 2단계: 작업 실행 (EXECUTE STEP) ---")
            step = plan[0]
            result = self.__execute_step(step, past_steps, tools)
            past_steps.append(result)

            # --- 3단계: Replan (정보가 충분한지 판단하고 계획을 갱신) ---
            print(f"\n--- 3단계: 재계획 (REPLAN) ---")
            decision = self.__replan(user_prompt, past_steps, plan)

            if decision.get("action") == "respond":
                # 정보가 충분 -> 최종 답변을 출력하고 워크플로우 종료
                logger.info("Replan 판단: 정보가 충분합니다 -> 답변을 생성합니다.")
                return decision.get("response", "")

            # 정보가 부족 -> 갱신된 '남은 계획'으로 다시 실행
            plan = decision.get("plan", [])
            logger.info(f"Replan 판단: 정보가 부족합니다 -> 갱신된 계획: {plan}")

        # ----- 반복 상한 도달: 무한 루프 방지 -----
        # 여기까지 왔다는 것은 max_iterations 안에 'respond' 결정을 받지 못했다는 뜻입니다.
        # 지금까지 수집한 결과로 최선의(best-effort) 답변을 정리해 반환합니다.
        logger.warning(
            f"최대 반복({self.max_iterations}) 도달 — 지금까지 결과로 답변을 정리합니다."
        )
        return self.__synthesize_answer(user_prompt, past_steps)

    # ---------------------------------------------------------------------
    # LLM 호출을 담당하는 핵심 함수 (추론, Reasoning)
    # ---------------------------------------------------------------------
    @staticmethod  # 클래스 내부 데이터에 접근할 필요가 없으므로 정적 메서드로 정의
    def call_llm(system_prompt: str, user_prompt: str,
                 model: str = MODEL_NAME,
                 temperature: float = 0.7,
                 json_output: bool = False):
        """Claude 모델을 호출하고, 필요하면 응답을 JSON으로 파싱해서 돌려줍니다.

        - system_prompt: 모델의 역할/규칙을 정의하는 시스템 프롬프트
        - user_prompt: 실제 사용자(또는 에이전트)의 요청
        - model: 사용할 Claude 모델 이름
        - temperature: 응답의 창의성 정도(0에 가까울수록 일관적)
        - json_output: True면 응답을 json.loads로 파싱해서 dict/list로 반환
        """
        max_retries = 3   # 최대 재시도 횟수
        attempts = 0      # 현재까지 시도한 횟수

        while attempts < max_retries:
            try:
                # Claude 메시지 API 호출
                #  - OpenAI와 달리 system 프롬프트는 messages가 아니라 system 파라미터로 전달합니다.
                #  - max_tokens(최대 출력 토큰 수)는 Anthropic API에서 필수 항목입니다.
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_prompt,                       # 시스템 프롬프트(별도 파라미터)
                    messages=[
                        {"role": "user", "content": user_prompt}  # 사용자 메시지
                    ],
                )

                # 응답(response.content)은 여러 개의 '블록' 목록입니다.
                # 그중 텍스트 타입(block.type == "text") 블록들의 내용만 모아 하나의 문자열로 만듭니다.
                raw_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )

                # JSON 출력을 요구하지 않았다면 순수 텍스트를 그대로 반환합니다.
                if not json_output:
                    return raw_text

                # JSON 출력을 요구한 경우: 모델이 코드블록(```json ... ```)으로 감싸는 경우가 있어
                # 이를 제거한 뒤 파싱을 시도합니다.
                cleaned_text = Agent.__strip_code_fence(raw_text)
                try:
                    return json.loads(cleaned_text)           # JSON 문자열 -> 파이썬 객체
                except json.JSONDecodeError as e:             # JSON 형식이 아니면 재시도
                    attempts += 1
                    logger.warning(f"{attempts}번째 시도: JSON 파싱 실패 ({e}). 다시 시도합니다...")

            except Exception as e:  # API 오류(타임아웃, 5xx, 요청 한도 초과 등)를 잡아 재시도
                attempts += 1
                logger.error(f"{attempts}번째 시도: API 호출 실패 ({e}). 다시 시도합니다...")

        # 최대 재시도 횟수를 모두 소진하면 예외를 발생시켜 실패를 명확히 알립니다.
        raise RuntimeError(f"{max_retries}번 시도했지만 유효한 응답을 받지 못했습니다.")

    # ---------------------------------------------------------------------
    # 보조 함수: 모델 응답에서 마크다운 코드블록 표시(```)를 제거
    # ---------------------------------------------------------------------
    @staticmethod
    def __strip_code_fence(text: str) -> str:
        """모델이 JSON을 ```json ... ``` 으로 감싸 반환할 때 그 표시를 제거합니다."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # 첫 줄(```json 또는 ```)과 마지막 줄(```)을 잘라냅니다.
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]            # 첫 줄 제거
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]           # 마지막 줄 제거
            cleaned = "\n".join(lines).strip()
        return cleaned

    # ---------------------------------------------------------------------
    # 두뇌(The Brain): 복잡한 요청을 단순한 작업들로 분해(계획 수립)
    # ---------------------------------------------------------------------
    def __plan_tasks(self, user_prompt: str) -> dict:
        """LLM에게 사용자 요청을 순차적인 단순 작업 목록으로 나누도록 요청합니다."""
        planner_system_prompt = (
            "당신은 정교한 '계획 수립 에이전트'입니다. "
            "당신의 임무는 복잡한 사용자 질문을 순차적이고 단순한 작업들로 분해하는 것입니다. "
            "반드시 다음 형식의 JSON 객체 하나만 반환하세요. "
            "다른 설명이나 텍스트는 절대 포함하지 마세요. "
            " - 'tasks': list[string]  # 작업 문자열들의 목록"
        )

        # json_output=True 로 호출하여 결과를 파이썬 dict로 받습니다.
        action_plan = self.call_llm(planner_system_prompt, user_prompt, json_output=True)
        logger.info(f'수립된 작업 계획: {action_plan}')
        self.plan = True
        return action_plan

    # ---------------------------------------------------------------------
    # 환경(The Environment): 계획의 '한 단계'에 대해 도구를 골라 실제로 실행
    # ---------------------------------------------------------------------
    def __execute_step(self, task: str, past_steps: list, tools: list) -> dict:
        """주어진 단일 작업(task)에 적절한 도구를 고르고, 가능하면 실제로 함수를 실행합니다.

        AIAgent.py의 __plan_tools 루프 1회분에 해당합니다. 전체 계획을 한꺼번에
        실행하지 않고, '한 단계'만 실행해 곧바로 Replan으로 넘어가는 것이 핵심입니다.
        """
        execution_system_prompt = (
            "당신은 정교한 '도구 선택 에이전트'입니다. "
            "주어진 작업을 해결하기 위해 시스템이 제공한 도구 중 올바른 것을 찾는 것이 임무입니다. "
            "사용 가능한 도구 목록과 작업은 사용자 메시지로 전달됩니다. "
            "적합한 도구가 없으면 function 값으로 'None'을 사용하세요. "
            "반드시 다음 형식의 JSON 객체 하나만 반환하세요. 다른 텍스트는 포함하지 마세요. "
            " - 'id': int          # 작업을 식별하는 고유 번호 "
            " - 'task': string     # 해당 작업 설명 "
            " - 'function': string # 사용할 도구(함수)의 이름 "
            " - 'properties': dict # 함수를 실행할 때 넘길 인자들 "
            " - 'dependencies': list # 이 작업이 의존하는 다른 작업들의 id 목록"
        )

        logger.info(f"****************** 작업 실행: {task} *******************")

        # 도구 선택을 위한 사용자 프롬프트(한글)를 구성합니다.
        # 이미 완료된 단계들의 결과(past_steps)를 함께 전달해 의존 관계를 참고하게 합니다.
        user_prompt = (
            f"수행해야 할 작업은 다음과 같습니다: {task}\n"
            f"사용할 수 있는 도구 목록은 다음과 같습니다: {tools}\n"
            "이 작업에 사용할 올바른 도구를 알려주세요.\n"
            f"이미 완료된 작업들의 실행 결과는 다음과 같습니다: {past_steps}\n"
            "의존 관계가 있는 작업은 위 실행 결과를 참고하세요."
        )

        # 도구 선택 결과를 JSON(dict)으로 받습니다.
        response = self.call_llm(execution_system_prompt, user_prompt, json_output=True)
        kwargs = response.get("properties", {})   # 함수에 넘길 인자
        func_name = response.get("function")      # 호출할 함수 이름

        # 선택된 함수가 실제로 사용 가능한 도구 목록에 있으면 호출합니다.
        if func_name in self.available_tools_dict:
            logger.info(f"함수 실행: {func_name} (인자: {kwargs})...")
            function_to_call = self.available_tools_dict[func_name]
            result = function_to_call(kwargs)   # 실제 도구(함수) 실행
            response["result"] = result          # 결과를 작업 정보에 추가
            logger.info(f"{func_name}의 실행 결과: {result}")
        else:
            # 적합한 도구가 없는 경우에도 단계 정보는 남겨, Replan이 상황을 판단하게 합니다.
            logger.info(f"사용할 도구가 없습니다 (function={func_name}).")
            response["result"] = None

        return response

    # ---------------------------------------------------------------------
    # 재계획(Replan): 지금까지 결과를 보고 '종료(답변)' 또는 '계획 갱신'을 결정
    # ---------------------------------------------------------------------
    def __replan(self, user_prompt: str, past_steps: list, current_plan: list) -> dict:
        """지금까지의 실행 결과가 답변에 충분한지 판단하고, 다음 행동을 결정합니다.

        반환값(dict)은 둘 중 하나입니다.
          - {"action": "respond",  "response": "<최종 답변>"}  # 정보가 충분 -> 종료
          - {"action": "continue", "plan": ["<남은/수정된 단계>", ...]}  # 부족 -> 반복
        """
        replan_system_prompt = (
            "당신은 정교한 '재계획(Replan) 에이전트'입니다. "
            "사용자의 원래 목표, 지금까지 실행한 단계와 그 결과, 그리고 현재 남은 계획을 보고 "
            "다음 행동을 결정하는 것이 임무입니다.\n"
            "판단 기준:\n"
            "1) 지금까지의 결과만으로 사용자 질문에 충분히 답할 수 있다면, "
            "   다음 형식의 JSON을 반환하세요: "
            '{"action": "respond", "response": "<사용자에게 줄 최종 답변>"}\n'
            "2) 아직 정보가 부족하다면, 목표 달성을 위해 '앞으로 해야 할' 단계들의 목록을 "
            "   다음 형식의 JSON으로 반환하세요: "
            '{"action": "continue", "plan": ["<단계1>", "<단계2>", ...]}\n'
            "주의: 이미 완료한 단계는 plan에 다시 넣지 마세요. 앞으로 필요한 단계만 남기세요. "
            "계획이 잘못되었다면 새로운 단계로 자유롭게 수정해도 됩니다. "
            "반드시 위 두 형식 중 하나의 JSON 객체 하나만 반환하고, 다른 텍스트는 포함하지 마세요."
        )

        replan_user_prompt = (
            f"사용자의 원래 목표: {user_prompt}\n"
            f"현재 남아있는 계획: {current_plan}\n"
            f"지금까지 실행한 단계와 결과: {past_steps}\n"
            "위 정보를 바탕으로 'respond' 또는 'continue' 결정을 JSON으로 반환하세요."
        )

        decision = self.call_llm(replan_system_prompt, replan_user_prompt, json_output=True)
        logger.info(f"Replan 결정: {decision}")

        # ---- 방어 처리: 예상치 못한 형식이면 루프가 안전하게 종료되도록 보정 ----
        if not isinstance(decision, dict) or "action" not in decision:
            logger.warning("Replan 응답 형식이 올바르지 않습니다. 답변 생성으로 폴백합니다.")
            return {"action": "respond",
                    "response": self.__synthesize_answer(user_prompt, past_steps)}

        if decision["action"] == "continue" and not isinstance(decision.get("plan"), list):
            # continue인데 plan이 없으면 더 진행할 수 없으므로 빈 계획으로 처리 -> 다음 루프에서 정리
            decision["plan"] = []

        return decision

    # ---------------------------------------------------------------------
    # 정리(Synthesize): 실행 결과를 종합해 최종 답변 작성
    # ---------------------------------------------------------------------
    def __synthesize_answer(self, user_prompt, past_steps) -> str:
        """도구 실행 결과들을 바탕으로 사용자에게 줄 최종 답변을 생성합니다.

        반복 상한에 도달했거나 남은 계획이 모두 소진된 경우 호출되는 폴백 경로입니다.
        """
        synthesis_prompt = (
            "당신은 유능한 비서입니다. 사용자 질문과 여러 도구의 실행 결과가 주어집니다. "
            "이 결과들을 바탕으로 간결하고 명확한 최종 답변을 작성하세요. "
            "사용자의 요청이 충족되었는지 확인하세요. "
            "만약 충족되지 않았다면, 해당 작업을 수행할 도구가 없다는 점을 사용자에게 알려주세요."
        )

        # 사용자 질문과 실행 결과를 하나의 문맥(context) 문자열로 합칩니다.
        context = f"사용자 질문: {user_prompt}\n실행 결과: {past_steps}"
        return self.call_llm(synthesis_prompt, context)


# =========================================================================
# 실행 예시
# =========================================================================
user_prompt = "지구와 목성의 질량을 합치면 얼마인가요?"
# 다른 예시: "2026년 2월 22일 뮌헨에서 런던으로 가는 항공편을 예약하고,
#            도심 근처에 헬스장이 있는 호텔도 예약해 주세요."

if os.environ.get("ANTHROPIC_API_KEY"):
    my_Agent = Agent(tools)                          # 에이전트 생성(도구 딕셔너리 전달)
    response = my_Agent.run(user_prompt, tools_schema)  # 에이전트 실행
    logger.info(f"최종 출력: {response}")
else:
    # ANTHROPIC_API_KEY가 설정되지 않은 경우 안내 메시지를 출력합니다.
    print(
        "ANTHROPIC_API_KEY가 설정되어 있지 않습니다. "
        "https://console.anthropic.com/ 에서 API 키를 발급받아 .env 파일에 설정하세요."
    )
